"""NumPy port of the flat -> worn appearance blend (``WearAtlasBake`` in ``src/MaterialAtlas.cpp``).

The C++ bake runs once at the end of a recording. Porting it here lets every rendered frame show the worn
appearance that matches that frame's wear texture. The math mirrors the C++ helpers one to one:

* ``wear_remap_t``      <- ``wearRemapT`` (Hermite ramp between edge0/edge1 with tension/bias tangents)
* ``height_blend_t``    <- ``wearHeightBlendT``
* ``slerp_normals``     <- ``slerpGeodesic`` (2-variant normal blend)

Texels with ``wear <= 1e-6`` keep the base maps exactly, as in the C++ bake. Direction-oriented normal rotation is
not ported: the recorded direction PNGs are an HSV preview, not the raw RG direction encoding it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

WEAR_EPS = 1e-6


@dataclass(frozen=True)
class BlendParams:
    """Defaults match ``SimViewer`` (``include/SimViewer.h``): normal-wear edges, Hermite T/B, height blend."""

    edge0: float = 0.05
    edge1: float = 0.95
    tension: float = 0.65
    bias: float = 0.0
    height_strength: float = 0.6
    height_contrast: float = 1.25


@dataclass
class MapSet:
    """Per-body material maps at one shared resolution, image convention (row 0 = V = 1)."""

    albedo: np.ndarray  # HxWx3 float32, sRGB-encoded [0, 1]
    roughness: np.ndarray  # HxW float32 [0, 1]
    normal: np.ndarray  # HxWx3 float32 unit tangent-space vectors
    height: np.ndarray | None  # HxW float32 [0, 1]

    def copy(self) -> MapSet:
        return replace(
            self,
            albedo=self.albedo.copy(),
            roughness=self.roughness.copy(),
            normal=self.normal.copy(),
            height=None if self.height is None else self.height.copy(),
        )


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


def wear_remap_t(wear: np.ndarray, p: BlendParams = BlendParams()) -> np.ndarray:
    m0, m1 = hermite_tangents(p.tension, p.bias)
    e1 = max(p.edge1, p.edge0 + 1e-6)
    return np.clip(hermite01((wear - p.edge0) / (e1 - p.edge0), m0, m1), 0.0, 1.0)


def height_blend_t(
    wear_t: np.ndarray, h_base: np.ndarray, h_worn: np.ndarray, p: BlendParams = BlendParams()
) -> np.ndarray:
    s = float(np.clip(p.height_strength, 0.0, 1.0))
    c = max(p.height_contrast, 1e-4)
    h_delta = np.clip((h_worn - h_base) * c, -1.0, 1.0)
    wt = np.clip(wear_t, 0.0, 1.0)
    return np.clip(wt + h_delta * s * 0.5 * wt * wt * wt, 0.0, 1.0)


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


def blend_maps(wear: np.ndarray, base: MapSet, worn: MapSet, p: BlendParams = BlendParams()) -> MapSet:
    """Worn appearance for one wear texture; ``wear``, ``base`` and ``worn`` share one resolution."""
    out = base.copy()
    mask = wear > WEAR_EPS
    if not mask.any():
        return out
    w = wear[mask]
    t = wear_remap_t(w, p)
    if base.height is not None and worn.height is not None:
        t = height_blend_t(t, base.height[mask], worn.height[mask], p)
    t3 = t[:, None]
    out.albedo[mask] = (1.0 - t3) * base.albedo[mask] + t3 * worn.albedo[mask]
    out.roughness[mask] = (1.0 - t) * base.roughness[mask] + t * worn.roughness[mask]
    out.normal[mask] = slerp_normals(base.normal[mask], worn.normal[mask], t)
    if out.height is not None and worn.height is not None:
        out.height[mask] = (1.0 - t) * base.height[mask] + t * worn.height[mask]
    return out


# Inferno (matplotlib), 11 stops. Used only for the labelled "normalized wear" preview colouring.
_INFERNO = np.array(
    [
        [0.001462, 0.000466, 0.013866],
        [0.087411, 0.044556, 0.224813],
        [0.258234, 0.038571, 0.406485],
        [0.416331, 0.090203, 0.432943],
        [0.578304, 0.148039, 0.404411],
        [0.735683, 0.215906, 0.330245],
        [0.865006, 0.316822, 0.226055],
        [0.954506, 0.468744, 0.099874],
        [0.987622, 0.645320, 0.039886],
        [0.964394, 0.843848, 0.273391],
        [0.988362, 0.998364, 0.644924],
    ],
    dtype=np.float32,
)


def colormap(values: np.ndarray, lo: float = 0.0, hi: float = 1.0, floor: float = 0.15) -> np.ndarray:
    """Map ``values`` linearly onto inferno (sRGB-encoded). ``floor`` skips the near-black start of the map."""
    x = np.clip((np.asarray(values, dtype=np.float32) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    x = floor + (1.0 - floor) * x
    pos = x * (len(_INFERNO) - 1)
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, len(_INFERNO) - 2)
    f = (pos - i0)[..., None]
    return (1.0 - f) * _INFERNO[i0] + f * _INFERNO[i0 + 1]


def heatmap_albedo(albedo: np.ndarray, wear: np.ndarray, opacity: float = 0.9) -> np.ndarray:
    """Overlay the wear colormap on ``albedo`` wherever wear was written (> one 8-bit step)."""
    out = albedo.copy()
    mask = wear > (0.5 / 255.0)
    if mask.any():
        out[mask] = (1.0 - opacity) * albedo[mask] + opacity * colormap(wear[mask])
    return out


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(np.float32)
