"""Wear blend on the GPU (Dr.Jit CUDA): the same maps as ``field_blend._blend_stack_cpu``, to float32 rounding.

Every texel is blended (no active-texel gather) and the unworn base is selected outside the worn mask, so a 1024^2
body costs one kernel launch per channel instead of numpy passes over the worn texels. A stack's slices are uploaded
once and kept on the device (keyed by the stack object), so a frame uploads only its wear field. ``blend_device`` keeps
the blended maps on the device (the renderer encodes and assigns them there); ``blend_stack`` downloads them. Burnish
and edge wear (FFT blur, percentile) stay CPU-only: ``unsupported`` names them and ``field_blend.resolve_device`` falls
back. Scratches are drawn on the CPU after a GPU blend (``field_blend.blend_stack``).

Needs the render venv (drjit with CUDA); ``unavailable`` says why it cannot run anywhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

from .field_blend import (
    LOCALITY_EPS,
    PLAIN,
    WEAR_EPS,
    WEAR_TINT_RGB,
    BlendParams,
    MapSet,
    VariantStack,
    hermite_tangents,
)


def unsupported(p: BlendParams) -> str:
    """Why this blend needs the CPU path ('' when the GPU can do all of it)."""
    terms = [name for name, on in (("burnish", p.burnish > 0.0), ("edge_wear", p.edge_wear > 0.0)) if on]
    return f"{' and '.join(terms)} are CPU-only" if terms else ""


@cache
def unavailable() -> str:
    """Why the GPU blend cannot run in this process ('' when it can)."""
    try:
        import drjit as dr
        from drjit.cuda import Float

        dr.eval(Float(1.0) + 1.0)
    except Exception as exc:  # noqa: BLE001 -- ImportError, no CUDA device, driver errors
        return f"Dr.Jit CUDA unavailable ({type(exc).__name__}: {exc})"
    return ""


def _types():
    import drjit as dr
    from drjit.cuda import Array2f, Array3f, Float, UInt32

    return dr, Float, Array2f, Array3f, UInt32


# ---- device arrays ------------------------------------------------------------------------------------------------


def _f1(a: np.ndarray):
    _, Float, *_ = _types()
    return Float(np.ascontiguousarray(a, np.float32).ravel())


def _f3(a: np.ndarray):
    _, Float, _, Array3f, _ = _types()
    a = np.asarray(a, np.float32)
    return Array3f(*(Float(np.ascontiguousarray(a[..., c]).ravel()) for c in range(3)))


def _np1(x, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(x, np.float32).reshape(shape)


def _np3(v, shape: tuple[int, int]) -> np.ndarray:
    return np.stack([np.asarray(v[c], np.float32).reshape(shape) for c in range(3)], axis=-1)


_STACKS: dict[int, tuple[VariantStack, dict]] = {}  # id(stack) -> (stack, device slices); the stack keeps the id valid


def _device_stack(stack: VariantStack) -> dict:
    hit = _STACKS.get(id(stack))
    if hit is not None and hit[0] is stack:
        return hit[1]
    dev = {
        "albedo": [_f3(a) for a in stack.albedo],
        "normal": [_f3(n) for n in stack.normal],
        "roughness": [_f1(r) for r in stack.roughness],
        "height": [_f1(h) for h in stack.height],
        "metallic": [_f1(np.broadcast_to(m, stack.albedo[0].shape[:2])) for m in stack.metallic],
    }
    _STACKS[id(stack)] = (stack, dev)
    return dev


def clear_cache() -> None:
    _STACKS.clear()


# ---- the shader functions, on device arrays -------------------------------------------------------------------------


def _hermite_ramp(dr, e0: float, e1: float, x, m0: float, m1: float):
    e1 = max(e1, e0 + 1e-4)
    t = dr.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    return dr.clip((-2.0 * t3 + 3.0 * t2) + (t3 - 2.0 * t2 + t) * m0 + (t3 - t2) * m1, 0.0, 1.0)


def _normalize(dr, v, Array3f):
    n = dr.norm(v)
    out = v / dr.maximum(n, 1e-12)
    return dr.select(n < 1e-6, Array3f(0.0, 0.0, 1.0), out)


def _blur_wear(dr, Float, UInt32, wear, h: int, w: int, px: float):
    w0 = dr.clip(wear, 0.0, 1.0)
    if px <= 0.01:
        return w0
    d = min(px, 4.0)
    idx = dr.arange(UInt32, h * w)
    row = idx // w
    col = idx - row * w
    taps = Float(0.0)
    for s in (d, -d):
        k = math.floor(s)
        f = s - k
        for axis in (0, 1):
            pos, n = (row, h) if axis == 0 else (col, w)
            i0 = dr.clip(Float(pos) + k, 0.0, n - 1.0)
            i1 = dr.clip(Float(pos) + k + 1.0, 0.0, n - 1.0)
            if axis == 0:
                j0, j1 = UInt32(i0) * w + col, UInt32(i1) * w + col
            else:
                j0, j1 = row * w + UInt32(i0), row * w + UInt32(i1)
            taps += (1.0 - f) * dr.gather(Float, w0, j0) + f * dr.gather(Float, w0, j1)
    return dr.clip(0.4 * w0 + 0.15 * taps, 0.0, 1.0)


def _slerp2(dr, Array3f, a, b, t):
    a = _normalize(dr, a, Array3f)
    b = _normalize(dr, b, Array3f)
    t = dr.clip(t, 0.0, 1.0)
    c = dr.clip(dr.dot(a, b), -1.0, 1.0)
    omega = dr.acos(c)
    sin_omega = dr.maximum(dr.sin(omega), 1e-7)
    out = (dr.sin((1.0 - t) * omega) / sin_omega) * a + (dr.sin(t * omega) / sin_omega) * b
    out = dr.select(c > 1.0 - 1e-6, (1.0 - t) * a + t * b, out)
    return _normalize(dr, out, Array3f)


def _slerp4(dr, Float, Array2f, Array3f, normals: list, weights: list):
    n = [_normalize(dr, v, Array3f) for v in normals]
    w = [dr.maximum(x, 0.0) for x in weights]
    total = w[0] + w[1] + w[2] + w[3]
    w = [x / dr.maximum(total, 1e-7) for x in w]
    ref, wmax = Array3f(n[0]), Float(w[0])
    for k in (1, 2, 3):  # first maximum, like the shader's strict comparisons
        better = w[k] > wmax
        ref = dr.select(better, n[k], ref)
        wmax = dr.select(better, w[k], wmax)
    axis = dr.select(dr.abs(ref.z) < 0.999, Array3f(0.0, 0.0, 1.0), Array3f(1.0, 0.0, 0.0))
    tan = _normalize(dr, dr.cross(axis, ref), Array3f)
    bit = _normalize(dr, dr.cross(ref, tan), Array3f)
    acc = Array2f(0.0, 0.0)
    for k in range(4):
        c = dr.clip(dr.dot(n[k], ref), -1.0, 1.0)
        perp = n[k] - c * ref
        p_len = dr.norm(perp)
        d = perp / dr.maximum(p_len, 1e-12)
        log = Array2f(dr.dot(d, tan), dr.dot(d, bit)) * dr.acos(c)
        log = dr.select(p_len < 1e-7, Array2f(math.pi, 0.0), log)
        log = dr.select(c > 1.0 - 1e-6, Array2f(0.0, 0.0), log)
        log = dr.select(w[k] > 1e-7, log, Array2f(0.0, 0.0))
        acc += w[k] * log
    ang = dr.norm(acc)
    unit = acc / dr.maximum(ang, 1e-12)
    direction = _normalize(dr, tan * unit.x + bit * unit.y, Array3f)
    out = _normalize(dr, dr.cos(ang) * ref + dr.sin(ang) * direction, Array3f)
    out = dr.select(ang < 1e-7, ref, out)
    return dr.select(total < 1e-7, n[0], out)


def _mix(slices: list, t, w4):
    if len(slices) == 2:
        return (1.0 - t) * slices[0] + t * slices[1]
    return w4[0] * slices[0] + w4[1] * slices[1] + w4[2] * slices[2] + w4[3] * slices[3]


@dataclass
class DeviceMaps:
    """A blended body on the device (Dr.Jit CUDA arrays of H x W texels, row-major): the ``MapSet`` channels, the
    uploaded field, the worn mask and the blend ``t``. ``height`` and ``metallic`` are None when the ``MapSet`` would
    hold None. ``to_mapset`` downloads it; the renderer encodes it in place (mitsuba_video.scene.encode_maps_device)."""

    shape: tuple[int, int]
    albedo: object  # Array3f, sRGB-encoded
    roughness: object  # Float
    normal: object  # Array3f, unit vectors
    height: object | None  # Float
    metallic: object | None  # Float
    field: object  # Float: the wear field as given (before the blur)
    active: object  # Bool: worn texels (the CPU path's mask)
    t: object  # Float: blend t (meaningful where ``active``)
    any_active: bool

    def to_mapset(self) -> MapSet:
        return MapSet(
            albedo=_np3(self.albedo, self.shape),
            roughness=_np1(self.roughness, self.shape),
            normal=_np3(self.normal, self.shape),
            height=None if self.height is None else _np1(self.height, self.shape),
            metallic=None if self.metallic is None else _np1(self.metallic, self.shape),
        )

    def active_np(self) -> np.ndarray:
        if not self.any_active:
            return np.zeros(self.shape, bool)
        return np.asarray(self.active, bool).reshape(self.shape)


def blend_device(wear: np.ndarray, stack: VariantStack, p: BlendParams = PLAIN, texel_scale: float = 1.0,
                 tint: bool = True) -> DeviceMaps:  # fmt: skip
    """The CPU path's maps, computed and kept on the device: every texel is blended and the unworn base is selected
    outside the worn mask, so nothing is downloaded. ``tint`` False leaves ``wear_tint`` to the caller."""
    if why := unsupported(p) or unavailable():
        raise RuntimeError(why)
    dr, Float, Array2f, Array3f, UInt32 = _types()
    h, w = wear.shape
    dev = _device_stack(stack)
    field = _f1(wear)
    # The unworn base, as ``stack.base`` (with the roughness floor): what the CPU path leaves outside the mask.
    base_r = dr.maximum(dev["roughness"][0], 0.04) if p.roughness_floor else dev["roughness"][0]
    base_m = dev["metallic"][0] if stack.metallic_varies else None
    base_h = dev["height"][0] if stack.height else None
    out = {"albedo": dev["albedo"][0], "roughness": base_r, "normal": dev["normal"][0], "height": base_h,
           "metallic": base_m}  # fmt: skip

    mask = _blur_wear(dr, Float, UInt32, field, h, w, p.wear_blur_px * texel_scale) if p.wear_blur_px > 0.0 else field
    active = mask > (LOCALITY_EPS if p.locality > 0.0 else WEAR_EPS)
    if not dr.any(active):
        return DeviceMaps((h, w), **out, field=field, active=active, t=dr.zeros(Float, h * w), any_active=False)

    m0, m1 = hermite_tangents(p.tension, p.bias)
    e0, e1 = (0.0, 1.0) if p.edge0 == 0.0 and p.edge1 == 0.0 else (p.edge0, p.edge1)
    shaped = dr.power(dr.maximum(mask, 0.0), p.wear_gamma) if p.wear_gamma != 1.0 else mask  # BlendParams.wear_gamma
    t = _hermite_ramp(dr, e0, e1, shaped, m0, m1)
    if len(stack.height) >= 2:
        hs = dev["height"]
        worn_idx = min(3 if stack.stages >= 4 else 1, len(hs) - 1)
        s = float(np.clip(p.height_strength, 0.0, 1.0))
        c = max(p.height_contrast, 1e-4)
        h_delta = dr.clip((hs[worn_idx] - hs[0]) * c, -1.0, 1.0)
        wt = dr.clip(t, 0.0, 1.0)
        t = dr.clip(wt + h_delta * s * 0.5 * wt * wt * wt, 0.0, 1.0)
    if p.locality > 0.0:
        x = dr.clip((mask - LOCALITY_EPS) / p.locality, 0.0, 1.0)
        t = t * (x * x * (3.0 - 2.0 * x))
    w4 = None
    if max(stack.stages, len(stack.height)) >= 4:
        tc = dr.clip(t, 0.0, 1.0)
        r0 = _hermite_ramp(dr, 0.0, 1.0 / 3.0, tc, m0, m1)
        r1 = _hermite_ramp(dr, 1.0 / 3.0, 2.0 / 3.0, tc, m0, m1)
        r2 = _hermite_ramp(dr, 2.0 / 3.0, 1.0, tc, m0, m1)
        w4 = [1.0 - r0, r0 - r1, r1 - r2, r2]

    if len(stack.albedo) >= 2:
        albedo = _mix(dev["albedo"], t, w4)
        if tint and p.wear_tint > 0.0:
            k = float(np.clip(p.wear_tint, 0.0, 1.0)) * t
            albedo = (1.0 - k) * albedo + k * Array3f(*(float(c) for c in WEAR_TINT_RGB))
        out["albedo"] = dr.select(active, albedo, out["albedo"])
        if base_m is not None:
            out["metallic"] = dr.select(active, _mix(dev["metallic"], t, w4), base_m)
    elif tint and p.wear_tint > 0.0:
        k = float(np.clip(p.wear_tint, 0.0, 1.0)) * t
        tinted = (1.0 - k) * out["albedo"] + k * Array3f(*(float(c) for c in WEAR_TINT_RGB))
        out["albedo"] = dr.select(active, tinted, out["albedo"])
    if len(stack.roughness) >= 2:
        r = _mix(dev["roughness"], t, w4)
        if p.roughness_floor:
            r = dr.maximum(r, 0.04 + 0.05 * dr.minimum(1.0, mask * 2.0) * t)
        out["roughness"] = dr.select(active, r, base_r)
    if len(stack.normal) >= 2:
        if len(stack.normal) == 2:
            n = _slerp2(dr, Array3f, dev["normal"][0], dev["normal"][1], t)
            polish_w = t
        else:
            n = _slerp4(dr, Float, Array2f, Array3f, dev["normal"], w4)
            polish_w = w4[3]
        if p.polish > 0.0:
            s = 1.0 - polish_w * p.polish
            n = _normalize(dr, Array3f(n.x * s, n.y * s, n.z), Array3f)
        out["normal"] = dr.select(active, n, out["normal"])
    if base_h is not None and len(stack.height) >= 2:
        out["height"] = dr.select(active, _mix(dev["height"], t, w4), base_h)
    return DeviceMaps((h, w), **out, field=field, active=active, t=t, any_active=True)


def blend_stack(wear: np.ndarray, stack: VariantStack, p: BlendParams = PLAIN, texel_scale: float = 1.0,
                tint: bool = True) -> tuple[MapSet, np.ndarray, np.ndarray]:  # fmt: skip
    """(maps, worn mask (H, W), blend t of the worn texels in row-major order): what the CPU path computes, plus
    the two values its scratch pass needs. ``tint`` False leaves ``wear_tint`` to the caller (after scratches)."""
    dm = blend_device(wear, stack, p, texel_scale, tint)
    active = dm.active_np()
    t = _np1(dm.t, dm.shape)[active] if dm.any_active else np.zeros(0, np.float32)
    return dm.to_mapset(), active, t
