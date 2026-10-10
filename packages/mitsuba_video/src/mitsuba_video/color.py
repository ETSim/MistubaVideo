"""Colour helpers: sRGB transfer functions, the inferno field colormap and the heatmap overlay (display only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Inferno (matplotlib), 11 stops. Used only for the labelled field preview colouring.
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


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(np.float32)


def colormap(values: np.ndarray, lo: float = 0.0, hi: float = 1.0, floor: float = 0.15) -> np.ndarray:
    """Map ``values`` linearly onto inferno (sRGB-encoded). ``floor`` skips the near-black start of the map."""
    x = np.clip((np.asarray(values, dtype=np.float32) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    x = floor + (1.0 - floor) * x
    pos = x * (len(_INFERNO) - 1)
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, len(_INFERNO) - 2)
    f = (pos - i0)[..., None]
    return (1.0 - f) * _INFERNO[i0] + f * _INFERNO[i0 + 1]


@dataclass(frozen=True)
class FieldDisplay:
    """Display-only mapping of a normalized field to colour; never fed back into the data.

    max:     top of the colormap. Below 1 it stretches low values across the full ramp; the colour bar then says
             "display range 0-max (preview)".
    ramp:    overlay opacity rises from 0 at value 0 to ``opacity`` at value = ramp, so small values read as a faint
             tint and large ones as solid colour. 0 keeps a flat ``opacity`` on every covered texel.
    opacity: overlay opacity at full ramp.
    """

    max: float = 1.0
    ramp: float = 0.0
    opacity: float = 0.9


WearDisplay = FieldDisplay  # TextureFriction name


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def heatmap_albedo(
    albedo: np.ndarray, value: np.ndarray, display: FieldDisplay = FieldDisplay(), threshold: float = 0.5 / 255.0
) -> np.ndarray:
    """Overlay the field colormap on ``albedo`` wherever the field exceeds ``threshold``."""
    out = albedo.copy()
    mask = value > threshold
    if mask.any():
        w = value[mask]
        if display.ramp > 0.0:
            alpha = (display.opacity * _smoothstep(0.0, display.ramp, w))[:, None]
        else:
            alpha = display.opacity
        out[mask] = (1.0 - alpha) * albedo[mask] + alpha * colormap(w, hi=display.max)
    return out
