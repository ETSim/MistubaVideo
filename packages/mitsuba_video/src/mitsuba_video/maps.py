"""Material maps at one render resolution: container, resizing, normal-map decoding/conditioning, albedo tint.

Every array uses image convention (row 0 = V = 1). Albedo is sRGB-encoded in [0, 1]; roughness/height in [0, 1];
normals are unit tangent-space vectors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .color import linear_to_srgb, srgb_to_linear

Image.MAX_IMAGE_PIXELS = None

DEFAULT_ROUGHNESS = 0.6
# Real materials reflect at most ~90% of light; flat "white" placeholder textures are clamped to this.
MAX_LINEAR_ALBEDO = 0.85


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


def resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize uint8 or float32 HxW[xC] to ``size`` = (width, height)."""
    h, w = arr.shape[:2]
    if (w, h) == tuple(size):
        return arr
    resample = Image.Resampling.BOX if w > size[0] else Image.Resampling.BILINEAR
    if arr.dtype == np.uint8:
        return np.asarray(Image.fromarray(arr).resize(size, resample))
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize(size, resample))
    return np.stack([resize(arr[..., c], size) for c in range(arr.shape[2])], axis=-1)


def resize_scalar(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return resize(arr.astype(np.float32), size)


def fit_size(shape_hw: tuple[int, int] | None, max_size: int, fallback: tuple[int, int] = (512, 512)) -> tuple[int, int]:
    """(width, height) for a (height, width) source shape, capped so the longer side is at most ``max_size``."""
    h, w = shape_hw or fallback
    s = min(1.0, max_size / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


@dataclass(frozen=True)
class NormalOptions:
    """Appearance-only normal-map conditioning.

    recenter_above_deg: a tangent-space normal map should average to +Z over the atlas. When its mean normal is
    tilted more than this, the map is mis-encoded; every texel is then rotated so the mean lands on +Z, keeping the
    micro-variation. Without it, grazing views see shading normals facing away from the camera, which Mitsuba
    renders black. <= 0 disables.
    strength: scales the tangential (x, y) part before renormalizing; 1 = as authored, 0 = flat.
    """

    recenter_above_deg: float = 5.0
    strength: float = 1.0


def _rotation_to_z(m: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``m`` onto +Z (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(m, z)
    s, c = np.linalg.norm(axis), float(np.dot(m, z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)


def normalize(n: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    out = n / np.maximum(norm, 1e-6)
    out[norm[..., 0] < 1e-6] = (0.0, 0.0, 1.0)
    return out


def decode_normal(
    img: np.ndarray | None,
    size: tuple[int, int],
    opts: NormalOptions = NormalOptions(),
    label: str = "",
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    """uint8 RGB tangent-space normal map -> unit vectors at ``size``; flat +Z when ``img`` is None."""
    if img is None:
        out = np.zeros((size[1], size[0], 3), np.float32)
        out[..., 2] = 1.0
        return out
    n = normalize(resize(img[..., :3], size).astype(np.float32) / 127.5 - 1.0)
    mean = n.reshape(-1, 3).mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-9)
    tilt = float(np.degrees(np.arccos(np.clip(mean[2], -1.0, 1.0))))
    if 0.0 < opts.recenter_above_deg < tilt:
        n = (n.reshape(-1, 3) @ _rotation_to_z(mean).T).reshape(n.shape).astype(np.float32)
        if log:
            log(f"normals   {label}: mean normal tilted {tilt:.0f} deg from +Z (mis-encoded map); recentred")
    if opts.strength != 1.0:
        n = n.copy()
        n[..., :2] *= opts.strength
        n = normalize(n)
    return n.astype(np.float32)


def scalar_map(img: np.ndarray | None, fallback: float, size: tuple[int, int]) -> np.ndarray:
    """uint8 grey map -> float [0, 1] at ``size``; constant ``fallback`` when ``img`` is None."""
    if img is None:
        return np.full((size[1], size[0]), fallback, np.float32)
    return resize(img, size).astype(np.float32) / 255.0


def tinted(texture: np.ndarray | None, colour: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """MTL semantics: diffuse = texture * colour (in linear space); sRGB-encoded, clamped to a real albedo.

    ``texture`` is sRGB-encoded float [0, 1] at ``size`` (or None for a flat colour).
    """
    linear = np.broadcast_to(colour, (size[1], size[0], 3)) if texture is None else srgb_to_linear(texture) * colour
    return linear_to_srgb(np.minimum(linear, MAX_LINEAR_ALBEDO))


def load_image(path: Path, mode: str) -> np.ndarray:
    """An image file as uint8 in ``mode`` ("RGB" or "L"), image convention."""
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))
