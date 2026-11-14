"""Scalar-field material interpolation on CPU, with optional CUDA acceleration.

A normalized field selects one, two or four stages of albedo, roughness, normals,
height and metallic maps. Optional locality, filtering and surface effects apply
only to active texels. Legacy wear-named parameters retain their serialized names.
See docs/field-blend-origin.md in the repository for implementation provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class MapSet:
    """Per-body material maps at one shared resolution, image convention (row 0 = V = 1)."""

    albedo: np.ndarray  # HxWx3 float32, sRGB-encoded [0, 1]
    roughness: np.ndarray  # HxW float32 [0, 1]
    normal: np.ndarray  # HxWx3 float32 unit tangent-space vectors
    height: np.ndarray | None  # HxW float32 [0, 1]
    # HxW float32 [0, 1], or None for the body's constant metallic. Set when the stages differ (enamel over steel),
    # so a worn track shades as bare metal instead of a light-grey dielectric.
    metallic: np.ndarray | None = None

    def copy(self) -> MapSet:
        return replace(
            self,
            albedo=self.albedo.copy(),
            roughness=self.roughness.copy(),
            normal=self.normal.copy(),
            height=None if self.height is None else self.height.copy(),
            metallic=None if self.metallic is None else self.metallic.copy(),
        )


WEAR_EPS = 1e-6
# basicCommon.glsl WEAR_VARIANT_BLEND_EPS: wear at or below this keeps the unworn slice when locality is on.
LOCALITY_EPS = 1e-3
WEAR_TINT_RGB = np.array([1.0, 0.55, 0.12], dtype=np.float32)


@dataclass(frozen=True)
class BlendParams:
    """Remap, height shaping and the viewer's optional presentation terms.

    The plain preset leaves presentation effects disabled. Optional controls include ``wear_blur_px`` (cross blur radius in field texels), ``locality`` (feather width above
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
    # Presentation terms the viewer does not have (0 = off). ``scratches``: strength of fine striations along the
    # recorded slide direction inside worn texels (``blend_stack``'s ``direction``). ``burnish`` in [0, 1]: a wider,
    # smooth wear mask (Gaussian, 10 texels at 1) with weaker height gating, so a stroke reads as one burnished track
    # rather than the speckle of the individual asperity contacts. ``scratch_density``: the fraction of texels that
    # seed a grit streak (lower = fewer, more separate scratches). ``edge_wear``: worn texels on steep relief of the base
    # height (hole rims, ridges, grain edges) reach the worn stage sooner, as exposed edges do; 0 = off.
    scratches: float = 0.0
    scratch_length_px: float = 24.0
    scratch_density: float = 0.05
    burnish: float = 0.0
    edge_wear: float = 0.0
    # ``wear_tint`` in [0, 1]: worn texels' colour pulled toward a warm amber by that fraction of the blend ``t``, so a
    # track reads at a glance from far away (an annotation, not a material property); 0 = off.
    wear_tint: float = 0.0
    # ``wear_gamma``: the normalised wear is raised to this power before the edge window (1 = linear). Below 1 it lifts
    # shallow tracks without flattening deep ones, so a sweep spanning 70x in depth keeps its order on screen (0.5 =
    # square root). A display mapping like the window itself; the colour bar names it.
    wear_gamma: float = 1.0

    @property
    def wide(self) -> bool:
        """True when the blend reaches further than the viewer's 5-texel wear blur (tile updates cannot be used)."""
        return self.scratches > 0.0 or self.burnish > 0.0 or self.edge_wear > 0.0


PLAIN = BlendParams()  # the default blend (no viewer presentation terms)


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
    # Per-stage metallic, aligned with ``albedo`` (unworn -> worn). Empty or all equal: the body's constant applies.
    metallic: list[float | np.ndarray] = field(default_factory=list)

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
            metallic=[base.metallic if base.metallic is not None else 0.0,
                      worn.metallic if worn.metallic is not None else 0.0]
            if base.metallic is not None or worn.metallic is not None else [],
        )

    def _slice(self, last: bool) -> MapSet:
        def pick(slices: list[np.ndarray]) -> np.ndarray:
            return slices[-1] if last else slices[0]

        return MapSet(
            albedo=pick(self.albedo),
            roughness=pick(self.roughness),
            normal=pick(self.normal),
            height=pick(self.height) if self.height else None,
            metallic=np.full(self.albedo[0].shape[:2], self.metallic[-1 if last else 0], np.float32)
            if self.metallic_varies
            else None,
        )

    @property
    def metallic_varies(self) -> bool:
        """True when the stages carry different metallic values, so the blend needs a per-texel metallic map."""
        return len(self.metallic) == len(self.albedo) and bool(self.metallic) and (
            any(np.ndim(m) > 0 for m in self.metallic) or max(self.metallic) - min(self.metallic) > 1e-3)

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


def wear_remap_t(wear: np.ndarray, p: BlendParams = PLAIN) -> np.ndarray:
    m0, m1 = hermite_tangents(p.tension, p.bias)
    e0, e1 = (0.0, 1.0) if p.edge0 == 0.0 and p.edge1 == 0.0 else (p.edge0, p.edge1)
    if p.wear_gamma != 1.0:
        wear = np.power(np.clip(wear, 0.0, None), p.wear_gamma)
    return hermite_ramp(e0, e1, wear, m0, m1)


def height_blend_t(wear_t: np.ndarray, h_base: np.ndarray, h_worn: np.ndarray, p: BlendParams = PLAIN) -> np.ndarray:
    s = float(np.clip(p.height_strength, 0.0, 1.0))
    c = max(p.height_contrast, 1e-4)
    h_delta = np.clip((h_worn - h_base) * c, -1.0, 1.0)
    wt = np.clip(wear_t, 0.0, 1.0)
    return np.clip(wt + h_delta * s * 0.5 * wt * wt * wt, 0.0, 1.0)


def wear_blend_weights4(t: np.ndarray, p: BlendParams = PLAIN) -> np.ndarray:
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


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur (FFT, edges reflected by padding) of a 2-D array; ``sigma`` in texels."""
    if sigma <= 0.05:
        return a
    pad = int(np.ceil(3.0 * sigma))
    p = np.pad(a, pad, mode="reflect")
    h, w = p.shape
    ky = np.fft.fftfreq(h).reshape(-1, 1)
    kx = np.fft.fftfreq(w).reshape(1, -1)
    g = np.exp(-2.0 * (np.pi * sigma) ** 2 * (kx * kx + ky * ky))
    out = np.real(np.fft.ifft2(np.fft.fft2(p) * g))
    return out[pad:-pad, pad:-pad].astype(a.dtype, copy=False)


_EDGE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def edge_field(height: np.ndarray) -> np.ndarray:
    """Steepness of a height map in [0, 1]: the slope magnitude (lightly smoothed) over its 98th percentile, so the
    steepest few percent of texels (rims, ridges) read 1. Cached per array."""
    hit = _EDGE_CACHE.get(id(height))
    if hit is not None and hit[0] is height:
        return hit[1]
    gy, gx = np.gradient(gaussian_blur(np.asarray(height, np.float32), 0.75))
    mag = np.hypot(gx, gy)
    top = float(np.percentile(mag, 98.0))
    out = np.clip(mag / top, 0.0, 1.0).astype(np.float32) if top > 1e-9 else np.zeros_like(mag, np.float32)
    _EDGE_CACHE[id(height)] = (height, out)
    return out


def scratch_field(direction: np.ndarray, region: np.ndarray, length_px: float, seed: int = 0,
                  density: float = 0.05) -> np.ndarray:  # fmt: skip
    """Fine striations along a slide-direction field, in [0, 1] (0 outside ``region``).

    ``direction`` is (H, W, 2) in image axes (+col, +row), zero where nothing slid; ``region`` (H, W) bool marks the
    texels to fill. Directions are spread a little (the slide footprint) and normalised, then sparse grit noise is
    averaged along them over +-``length_px`` texels: a line-integral convolution, which draws each grain as a short
    streak parallel to the slide, the way abrasive particles plough grooves. ``density`` is the fraction of texels
    holding a grain."""
    out = np.zeros(region.shape, np.float32)
    if not region.any():
        return out
    rows, cols = np.nonzero(region)
    reach = int(np.ceil(length_px)) + 4
    r0, r1 = max(int(rows.min()) - reach, 0), min(int(rows.max()) + reach + 1, region.shape[0])
    c0, c1 = max(int(cols.min()) - reach, 0), min(int(cols.max()) + reach + 1, region.shape[1])
    d = direction[r0:r1, c0:c1].astype(np.float32)
    d = np.stack([gaussian_blur(d[..., 0], 2.0), gaussian_blur(d[..., 1], 2.0)], axis=-1)
    norm = np.linalg.norm(d, axis=-1)
    known = norm > 0.05
    d = np.where(known[..., None], d / np.maximum(norm, 1e-6)[..., None], 0.0)
    h, w = norm.shape
    rng = np.random.default_rng(seed + 7919 * r0 + c0)
    grit = ((rng.random((h, w)) < float(np.clip(density, 0.0, 1.0))) * rng.uniform(0.3, 1.0, (h, w))).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    steps = max(round(length_px), 1)
    acc = np.zeros((h, w), np.float32)
    for k in range(-steps, steps + 1):
        yy = np.clip(np.rint(ys + k * d[..., 1]), 0, h - 1).astype(np.intp)
        xx = np.clip(np.rint(xs + k * d[..., 0]), 0, w - 1).astype(np.intp)
        acc += grit[yy, xx]
    acc /= 2 * steps + 1
    inside = region[r0:r1, c0:c1] & known
    if inside.any():
        top = float(np.percentile(acc[inside], 99.0))
        acc = np.clip(acc / max(top, 1e-6), 0.0, 1.0)
    out[r0:r1, c0:c1] = np.where(inside, acc, 0.0)
    return out


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


BLEND_DEVICES = ("auto", "cpu", "gpu")


def resolve_device(device: str, p: BlendParams = PLAIN) -> tuple[str, str]:
    """(``"cpu"`` or ``"gpu"``, why) for a requested ``device`` (auto | cpu | gpu). ``auto`` takes the GPU when Dr.Jit
    has CUDA and the blend has no CPU-only term; ``gpu`` raises instead of falling back."""
    if device not in BLEND_DEVICES:
        raise ValueError(f"blend device {device!r}; expected one of {', '.join(BLEND_DEVICES)}")
    if device == "cpu":
        return "cpu", "requested"
    from . import field_blend_gpu as gpu

    why_not = gpu.unsupported(p) or gpu.unavailable()
    if why_not:
        if device == "gpu":
            raise RuntimeError(f"GPU wear blend not possible: {why_not}")
        return "cpu", why_not
    return "gpu", "Dr.Jit CUDA"


def blend_stack(
    wear: np.ndarray,
    stack: VariantStack,
    p: BlendParams = PLAIN,
    texel_scale: float = 1.0,
    direction: np.ndarray | None = None,
    device: str = "cpu",
) -> MapSet:
    """The worn appearance for one field image; ``wear`` and the stack share one resolution.

    ``texel_scale`` converts ``p.wear_blur_px`` from field texels to this resolution (render width / field width).
    ``direction`` (H, W, 2), the slide direction in image axes, drives ``p.scratches``; without it there are none.
    ``device``: cpu (numpy), gpu (Dr.Jit CUDA, ``wear_blend_gpu``) or auto (see ``resolve_device``). Both give the
    same maps to float32 rounding.
    """
    if resolve_device(device, p)[0] == "gpu":
        from . import field_blend_gpu as gpu

        scratches = p.scratches > 0.0 and direction is not None and direction.shape[:2] == wear.shape
        out, active, t = gpu.blend_stack(wear, stack, p, texel_scale, tint=not scratches)
        if scratches and active.any():
            _apply_scratches(out, active, t, direction, p, texel_scale)
            if p.wear_tint > 0.0:
                k = (float(np.clip(p.wear_tint, 0.0, 1.0)) * t)[:, None]
                out.albedo[active] = (1.0 - k) * out.albedo[active] + k * WEAR_TINT_RGB
        return out
    return _blend_stack_cpu(wear, stack, p, texel_scale, direction)


def _blend_stack_cpu(
    wear: np.ndarray,
    stack: VariantStack,
    p: BlendParams,
    texel_scale: float,
    direction: np.ndarray | None,
) -> MapSet:
    out = stack.base.copy()
    if p.roughness_floor:
        out.roughness = np.maximum(out.roughness, 0.04)
    mask = blur_wear(wear, p.wear_blur_px * texel_scale) if p.wear_blur_px > 0.0 else wear
    if p.burnish > 0.0:
        b = float(np.clip(p.burnish, 0.0, 1.0))
        smooth = gaussian_blur(np.clip(wear, 0.0, 1.0).astype(np.float32), 10.0 * b * texel_scale)
        mask = (1.0 - b) * mask + b * smooth
    active = mask > (LOCALITY_EPS if p.locality > 0.0 else WEAR_EPS)
    if not active.any():
        return out
    m = mask[active]
    t = wear_remap_t(m, p)
    if len(stack.height) >= 2:
        h = stack.height
        worn_idx = min(3 if stack.stages >= 4 else 1, len(h) - 1)
        hp = replace(p, height_strength=p.height_strength * (1.0 - float(np.clip(p.burnish, 0.0, 1.0))))
        t = height_blend_t(t, h[0][active], h[worn_idx][active], hp)
    if p.edge_wear > 0.0 and stack.height and stack.height[0] is not None:
        e = edge_field(stack.height[0])[active]
        t = 1.0 - (1.0 - t) ** (1.0 + float(p.edge_wear) * e)  # only worn texels move: t = 0 stays 0
    if p.locality > 0.0:
        t = t * locality(m, p.locality)
    w4 = wear_blend_weights4(t, p) if max(stack.stages, len(stack.height)) >= 4 else None

    if len(stack.albedo) >= 2:
        out.albedo[active] = _mix([a[active] for a in stack.albedo], t, w4)
        if stack.metallic_varies and out.metallic is not None:
            out.metallic[active] = _mix([np.broadcast_to(m, wear.shape)[active].astype(np.float32)
                                       for m in stack.metallic], t, w4)
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
    if p.scratches > 0.0 and direction is not None and direction.shape[:2] == wear.shape:
        _apply_scratches(out, active, t, direction, p, texel_scale)
    if p.wear_tint > 0.0:
        k = (float(np.clip(p.wear_tint, 0.0, 1.0)) * t)[:, None]
        out.albedo[active] = (1.0 - k) * out.albedo[active] + k * WEAR_TINT_RGB
    return out


def _apply_scratches(out: MapSet, active: np.ndarray, t: np.ndarray, direction: np.ndarray, p: BlendParams,
                     texel_scale: float) -> None:  # fmt: skip
    """Striations along the slide inside worn texels, weighted by the blend ``t``: brighter, smoother lines in the
    colour and roughness, and shallow grooves in the normal (the line field as a height, across its gradient)."""
    s = scratch_field(direction, active, p.scratch_length_px * max(texel_scale, 0.25), density=p.scratch_density)
    if not s.any():
        return
    k = float(p.scratches)
    w = np.clip(k * t, 0.0, 1.0)
    sa = s[active]
    # Stronger albedo lift so tracks stay readable under glossy enamel/lacquer at grazing angles.
    out.albedo[active] = np.clip(out.albedo[active] * (1.0 + 1.8 * (w * (sa - 0.30)))[:, None], 0.0, 1.0)
    out.roughness[active] = np.clip(out.roughness[active] * (1.0 - 0.65 * w * sa), 0.03, 1.0)
    gy, gx = np.gradient(gaussian_blur(s, 0.6))
    n = out.normal[active].copy()
    # Grooves as height -s in image coordinates: n = (-dh/dcol, -dh/drow, 1).
    n[:, 0] += 3.0 * w * gx[active]
    n[:, 1] += 3.0 * w * gy[active]
    out.normal[active] = _normalize(n)


def blend_maps(wear: np.ndarray, base: MapSet, worn: MapSet, p: BlendParams = PLAIN) -> MapSet:
    """Two-variant blend of a base and a worn ``MapSet`` (``blend_stack`` on ``VariantStack.pair``)."""
    return blend_stack(wear, VariantStack.pair(base, worn), p)
