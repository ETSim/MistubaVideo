"""Field-driven material blend: unworn -> worn maps, weighted per texel by the field value.

The math is TextureFriction's live viewer (``glsl/scene/basicShader.frag`` with ``glsl/common/basicCommon.glsl``),
ported one function at a time so a render matches what the viewer shows:

* ``wear_remap_t``          <- ``wearRemapT`` (Hermite ramp between edge0/edge1 with tension/bias tangents)
* ``height_blend_t``        <- ``wearHeightBlendT``
* ``wear_blend_weights4``   <- ``wearBlendWeights4`` (four stages, cut at 1/3 and 2/3)
* ``slerp_normals``         <- ``slerpGeodesic`` (two-variant normal blend)
* ``slerp4_normals``        <- ``slerp4Normals`` (log/exp map around the heaviest stage)
* ``blur_wear``             <- ``sampleWearMask`` (5-tap cross, ``u_wearMapBlurPx``)
* ``locality``              <- ``wearMaskBlendLocality``

A material is a ``VariantStack``: for each channel, 1, 2 or 4 slices ordered unworn -> fully worn. Those are the
images the viewer packs side by side (2) or into a 2x2 grid (4) for its variant atlases. A channel with one slice
does not change with wear, which is also what the viewer does.

``BlendParams()`` is the plain blend other sources get. ``VIEWER_BLEND`` adds the viewer's presentation terms (wear
mask blur, a feathered start, flattening of worn normals, a roughness floor); TextureFriction recordings use it, with
the values the viewer had when the run was recorded. The viewer's direction-oriented worn normals are not ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .maps import MapSet

WEAR_EPS = 1e-6
# basicCommon.glsl WEAR_VARIANT_BLEND_EPS: wear at or below this keeps the unworn slice when locality is on.
LOCALITY_EPS = 1e-3


@dataclass(frozen=True)
class BlendParams:
    """Remap, height shaping and the viewer's optional presentation terms.

    The first six match ``SimViewer`` (``include/SimViewer.h``). The last four are off here and on in
    ``VIEWER_BLEND``: ``wear_blur_px`` (cross blur radius in field texels), ``locality`` (feather width above
    ``LOCALITY_EPS``; 0 keeps the hard ``WEAR_EPS`` cut), ``polish`` (worn normals keep ``1 - polish`` of their
    detail) and ``roughness_floor`` (worn texels never get smoother than 0.04 to 0.09, every texel at least 0.04).
    """

    edge0: float = 0.05
    edge1: float = 0.95
    tension: float = 0.65
    bias: float = 0.0
    height_strength: float = 0.6
    height_contrast: float = 1.25
    wear_blur_px: float = 0.0
    locality: float = 0.0
    polish: float = 0.0
    roughness_floor: bool = False


VIEWER_BLEND = BlendParams(wear_blur_px=0.75, locality=0.02, polish=0.7, roughness_floor=True)


@dataclass
class VariantStack:
    """Per-channel slices at one resolution, ordered unworn -> fully worn (1, 2 or 4 each).

    ``height`` holds the height-atlas slices the viewer shapes the blend with; it may be empty. Arrays follow
    ``MapSet``: albedo HxWx3 sRGB-encoded, normal HxWx3 unit vectors, roughness and height HxW in [0, 1].
    """

    albedo: list[np.ndarray]
    normal: list[np.ndarray]
    roughness: list[np.ndarray]
    height: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("albedo", "normal", "roughness"):
            if len(getattr(self, name)) not in (1, 2, 4):
                raise ValueError(f"{name}: {len(getattr(self, name))} slices, expected 1, 2 or 4")
        if len(self.height) not in (0, 1, 2, 4):
            raise ValueError(f"height: {len(self.height)} slices, expected 0, 1, 2 or 4")

    @classmethod
    def pair(cls, base: MapSet, worn: MapSet) -> VariantStack:
        """Two-slice stack from a base and a worn ``MapSet`` (how plain providers describe a material)."""
        heights = [h for h in (base.height, worn.height) if h is not None]
        return cls(
            albedo=[base.albedo, worn.albedo],
            normal=[base.normal, worn.normal],
            roughness=[base.roughness, worn.roughness],
            height=heights if len(heights) == 2 else heights[:1],
        )

    def _slice(self, last: bool) -> MapSet:
        def pick(slices: list[np.ndarray]) -> np.ndarray:
            return slices[-1] if last else slices[0]

        return MapSet(
            albedo=pick(self.albedo),
            roughness=pick(self.roughness),
            normal=pick(self.normal),
            height=pick(self.height) if self.height else None,
        )

    @property
    def base(self) -> MapSet:
        return self._slice(last=False)

    @property
    def worn(self) -> MapSet:
        """The fully worn slice of every channel."""
        return self._slice(last=True)

    @property
    def stages(self) -> int:
        return max(len(self.albedo), len(self.normal), len(self.roughness))

    def describe(self) -> str:
        """Slices per channel, e.g. ``"normal 2, albedo 1, roughness 2, height 2"``."""
        return ", ".join(f"{k} {len(getattr(self, k))}" for k in ("normal", "albedo", "roughness", "height"))


def hermite_tangents(tension: float, bias: float) -> tuple[float, float]:
    tension = float(np.clip(tension, -1.0, 1.0))
    bias = float(np.clip(bias, -1.0, 1.0))
    c = max(1.0 - tension, 0.0)
    return 0.5 * c * (1.0 + bias), 0.5 * c * (1.0 - bias)


def hermite01(x: np.ndarray, m0: float, m1: float) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    return (-2.0 * t3 + 3.0 * t2) + (t3 - 2.0 * t2 + t) * m0 + (t3 - t2) * m1


def hermite_ramp(e0: float, e1: float, x: np.ndarray, m0: float, m1: float) -> np.ndarray:
    """``hermiteRamp``: 0 at or below ``e0``, 1 at or above ``e1``, the clamped Hermite curve between."""
    e1 = max(e1, e0 + 1e-4)
    return np.clip(hermite01((x - e0) / (e1 - e0), m0, m1), 0.0, 1.0)


def wear_remap_t(wear: np.ndarray, p: BlendParams = BlendParams()) -> np.ndarray:
    m0, m1 = hermite_tangents(p.tension, p.bias)
    e0, e1 = (0.0, 1.0) if p.edge0 == 0.0 and p.edge1 == 0.0 else (p.edge0, p.edge1)
    return hermite_ramp(e0, e1, wear, m0, m1)


def height_blend_t(
    wear_t: np.ndarray, h_base: np.ndarray, h_worn: np.ndarray, p: BlendParams = BlendParams()
) -> np.ndarray:
    s = float(np.clip(p.height_strength, 0.0, 1.0))
    c = max(p.height_contrast, 1e-4)
    h_delta = np.clip((h_worn - h_base) * c, -1.0, 1.0)
    wt = np.clip(wear_t, 0.0, 1.0)
    return np.clip(wt + h_delta * s * 0.5 * wt * wt * wt, 0.0, 1.0)


def wear_blend_weights4(t: np.ndarray, p: BlendParams = BlendParams()) -> np.ndarray:
    """(..., 4) stage weights for ``t`` in [0, 1]; they sum to 1 and are one-hot at t = 0 and t = 1."""
    t = np.clip(t, 0.0, 1.0)
    m0, m1 = hermite_tangents(p.tension, p.bias)
    r0 = hermite_ramp(0.0, 1.0 / 3.0, t, m0, m1)
    r1 = hermite_ramp(1.0 / 3.0, 2.0 / 3.0, t, m0, m1)
    r2 = hermite_ramp(2.0 / 3.0, 1.0, t, m0, m1)
    return np.stack([1.0 - r0, r0 - r1, r1 - r2, r2], axis=-1)


def locality(wear: np.ndarray, width: float) -> np.ndarray:
    """``wearMaskBlendLocality``: smoothstep from ``LOCALITY_EPS`` over ``width``."""
    x = np.clip((wear - LOCALITY_EPS) / width, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _shift(a: np.ndarray, d: float, axis: int) -> np.ndarray:
    """Bilinear sample of ``a`` offset by ``d`` texels along ``axis``, clamped at the edges."""
    k = int(np.floor(d))
    f = d - k
    n = a.shape[axis]
    idx = np.arange(n)
    i0 = np.clip(idx + k, 0, n - 1)
    i1 = np.clip(idx + k + 1, 0, n - 1)
    return (1.0 - f) * np.take(a, i0, axis=axis) + f * np.take(a, i1, axis=axis)


def blur_wear(wear: np.ndarray, px: float) -> np.ndarray:
    """``sampleWearMask``: 0.4 x centre + 0.15 x four bilinear taps ``px`` texels away (at most 4)."""
    w0 = np.clip(wear, 0.0, 1.0)
    if px <= 0.01:
        return w0
    d = min(px, 4.0)
    taps = sum(_shift(w0, s, axis) for axis in (0, 1) for s in (d, -d))
    return np.clip(0.4 * w0 + 0.15 * taps, 0.0, 1.0)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.maximum(n, 1e-12)
    out[(n[..., 0] < 1e-6)] = (0.0, 0.0, 1.0)
    return out


def slerp_normals(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Geodesic interpolation of unit vectors ``a`` -> ``b`` (..., 3) by ``t`` (...)."""
    a = _normalize(a)
    b = _normalize(b)
    t = np.clip(t, 0.0, 1.0)[..., None]
    c = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
    omega = np.arccos(c)
    sin_omega = np.maximum(np.sin(omega), 1e-7)
    w0 = np.sin((1.0 - t) * omega) / sin_omega
    w1 = np.sin(t * omega) / sin_omega
    out = w0 * a + w1 * b
    near = c > 1.0 - 1e-6
    out = np.where(near, (1.0 - t) * a + t * b, out)
    return _normalize(out)


def _tangent_frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``tangentFrameAt``: (t, b) completing unit ``n`` (..., 3) to a right-handed frame."""
    axis = np.where((np.abs(n[..., 2]) < 0.999)[..., None], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    t = _normalize(np.cross(axis, n))
    return t, _normalize(np.cross(n, t))


def slerp4_normals(normals: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """``slerp4Normals``: weighted mean of four normals (..., 3) in the log map of the heaviest one."""
    n = _normalize(np.stack(normals, axis=-2))  # (..., 4, 3)
    w = np.maximum(weights, 0.0)
    total = w.sum(axis=-1, keepdims=True)
    w = w / np.maximum(total, 1e-7)
    ref_idx = np.argmax(w, axis=-1)  # first maximum, like the shader's strict comparisons
    ref = np.take_along_axis(n, ref_idx[..., None, None], axis=-2)[..., 0, :]
    t, b = _tangent_frame(ref)
    c = np.clip(np.einsum("...kc,...c->...k", n, ref), -1.0, 1.0)
    perp = n - c[..., None] * ref[..., None, :]
    p_len = np.linalg.norm(perp, axis=-1)
    d = perp / np.maximum(p_len, 1e-12)[..., None]
    log = np.stack([np.einsum("...kc,...c->...k", d, t), np.einsum("...kc,...c->...k", d, b)], axis=-1)
    log = log * np.arccos(c)[..., None]
    log = np.where((p_len < 1e-7)[..., None], [np.pi, 0.0], log)  # antipodal: a fixed direction
    log = np.where((c > 1.0 - 1e-6)[..., None], 0.0, log)
    log = np.where((w > 1e-7)[..., None], log, 0.0)
    acc = np.einsum("...k,...kc->...c", w, log)
    ang = np.linalg.norm(acc, axis=-1)
    unit = acc / np.maximum(ang, 1e-12)[..., None]
    direction = _normalize(t * unit[..., :1] + b * unit[..., 1:])
    out = _normalize(np.cos(ang)[..., None] * ref + np.sin(ang)[..., None] * direction)
    out = np.where((ang < 1e-7)[..., None], ref, out)
    return np.where(total < 1e-7, n[..., 0, :], out)


def _detail_strength(n: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``applyDetailStrength``: scale the tangential part and renormalize (0 = flat, 1 = as authored)."""
    out = n.copy()
    out[..., :2] *= s[..., None]
    return _normalize(out)


def _mix(slices: list[np.ndarray], t: np.ndarray, w4: np.ndarray | None) -> np.ndarray:
    """Blend scalar (N,) or colour (N, 3) slices: lerp for two, stage weights for four."""
    if len(slices) == 2:
        tt = t[:, None] if slices[0].ndim == 2 else t
        return (1.0 - tt) * slices[0] + tt * slices[1]
    ww = w4 if slices[0].ndim == 1 else w4[..., None]
    return sum(ww[:, k] * slices[k] for k in range(4))


def blend_stack(
    wear: np.ndarray, stack: VariantStack, p: BlendParams = BlendParams(), texel_scale: float = 1.0
) -> MapSet:
    """The worn appearance for one field image; ``wear`` and the stack share one resolution.

    ``texel_scale`` converts ``p.wear_blur_px`` from field texels to this resolution (render width / field width).
    """
    out = stack.base.copy()
    if p.roughness_floor:
        out.roughness = np.maximum(out.roughness, 0.04)
    mask = blur_wear(wear, p.wear_blur_px * texel_scale) if p.wear_blur_px > 0.0 else wear
    active = mask > (LOCALITY_EPS if p.locality > 0.0 else WEAR_EPS)
    if not active.any():
        return out
    m = mask[active]
    t = wear_remap_t(m, p)
    if len(stack.height) >= 2:
        h = stack.height
        worn_idx = min(3 if stack.stages >= 4 else 1, len(h) - 1)
        t = height_blend_t(t, h[0][active], h[worn_idx][active], p)
    if p.locality > 0.0:
        t = t * locality(m, p.locality)
    w4 = wear_blend_weights4(t, p) if max(stack.stages, len(stack.height)) >= 4 else None

    if len(stack.albedo) >= 2:
        out.albedo[active] = _mix([a[active] for a in stack.albedo], t, w4)
    if len(stack.roughness) >= 2:
        r = _mix([s[active] for s in stack.roughness], t, w4)
        if p.roughness_floor:
            r = np.maximum(r, 0.04 + 0.05 * np.minimum(1.0, m * 2.0) * t)
        out.roughness[active] = r
    if len(stack.normal) >= 2:
        if len(stack.normal) == 2:
            n = slerp_normals(stack.normal[0][active], stack.normal[1][active], t)
            polish_w = t
        else:
            n = slerp4_normals([s[active] for s in stack.normal], w4)
            polish_w = w4[:, 3]
        if p.polish > 0.0:
            n = _detail_strength(n, 1.0 - polish_w * p.polish)
        out.normal[active] = n
    if out.height is not None and len(stack.height) >= 2:
        out.height[active] = _mix([h[active] for h in stack.height], t, w4)
    return out


def blend_maps(wear: np.ndarray, base: MapSet, worn: MapSet, p: BlendParams = BlendParams()) -> MapSet:
    """Two-variant blend of a base and a worn ``MapSet`` (``blend_stack`` on ``VariantStack.pair``)."""
    return blend_stack(wear, VariantStack.pair(base, worn), p)
