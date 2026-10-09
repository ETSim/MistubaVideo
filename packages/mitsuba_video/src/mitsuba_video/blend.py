"""Field-driven material blend: base -> affected ("worn") maps, weighted per texel by the field value.

Ported from TextureFriction's ``WearAtlasBake`` (``src/MaterialAtlas.cpp``), which bakes the flat -> worn
appearance once at the end of a recording; here every rendered frame gets its own blend. The math mirrors the C++
helpers one to one:

* ``wear_remap_t``      <- ``wearRemapT`` (Hermite ramp between edge0/edge1 with tension/bias tangents)
* ``height_blend_t``    <- ``wearHeightBlendT``
* ``slerp_normals``     <- ``slerpGeodesic`` (2-variant normal blend)

Texels with a field value <= 1e-6 keep the base maps exactly, as in the C++ bake. The defaults are
TextureFriction's (``SimViewer.h``); other sources pass their own ``BlendParams``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .maps import MapSet

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
