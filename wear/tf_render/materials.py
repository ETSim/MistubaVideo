"""Build each body's base and worn ``MapSet`` at one render resolution.

Sources:

* base normal / roughness / height: the body's HDF5 ``material_atlas`` variant 0 (byte-identical to the ``.mtl``
  ``map_Bump`` / ``map_Pr`` / ``map_disp`` files once un-flipped), falling back to those files.
* worn normal / roughness / height: the ``.mtl`` worn maps OBJLoader reads (``map_Bump_Worn``, ``map_Pr_worn``,
  ``map_disp_worn``), falling back to the atlas worn variant. The atlas variants are exported from each atlas
  material's own ``img_*`` images, so they do not carry ``map_Pr_worn``: on ``plane_concrete`` variant 1 has the
  base roughness while ``concrete_roughness_worn.png`` is the real worn target.
* albedo: ``map_Kd`` / ``map_Kd_worn`` times ``Kd`` (not stored in the atlas), falling back to the ``Kd`` colour.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .recording import Body, MaterialVariant
from .wear_blend import MapSet, linear_to_srgb, srgb_to_linear

DEFAULT_ROUGHNESS = 0.6
# Real materials reflect at most ~90% of light; the scenario "flat_diffuse" placeholders are pure white.
MAX_LINEAR_ALBEDO = 0.85


def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize uint8 or float32 HxW[xC] to ``size`` = (width, height)."""
    h, w = arr.shape[:2]
    if (w, h) == tuple(size):
        return arr
    resample = Image.Resampling.BOX if w > size[0] else Image.Resampling.BILINEAR
    if arr.dtype == np.uint8:
        return np.asarray(Image.fromarray(arr).resize(size, resample))
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize(size, resample))
    return np.stack([_resize(arr[..., c], size) for c in range(arr.shape[2])], axis=-1)


def resize_scalar(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return _resize(arr.astype(np.float32), size)


@dataclass(frozen=True)
class NormalOptions:
    """Appearance-only normal-map conditioning (normal maps do not feed any physics here).

    recenter_above_deg: a tangent-space normal map should average to +Z over the atlas. When its mean normal is
    tilted more than this, the map is mis-encoded (``plane_concrete``'s ``test_4mm.png`` averages ~47 degrees off
    +Z); every texel is then rotated so the mean lands on +Z, keeping the micro-variation. Without it, grazing
    views see shading normals facing away from the camera, which Mitsuba renders black. <= 0 disables.
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


def _normalize(n: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    out = n / np.maximum(norm, 1e-6)
    out[norm[..., 0] < 1e-6] = (0.0, 0.0, 1.0)
    return out


def _decode_normal(
    img: np.ndarray | None,
    size: tuple[int, int],
    opts: NormalOptions = NormalOptions(),
    label: str = "",
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    if img is None:
        out = np.zeros((size[1], size[0], 3), np.float32)
        out[..., 2] = 1.0
        return out
    n = _normalize(_resize(img[..., :3], size).astype(np.float32) / 127.5 - 1.0)
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
        n = _normalize(n)
    return n.astype(np.float32)


def _scalar_map(img: np.ndarray | None, fallback: float, size: tuple[int, int]) -> np.ndarray:
    if img is None:
        return np.full((size[1], size[0]), fallback, np.float32)
    return _resize(img, size).astype(np.float32) / 255.0


def _albedo(body: Body, key: str, size: tuple[int, int]) -> np.ndarray | None:
    path = body.resolve_texture(key)
    if path is None:
        return None
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    return _resize(arr, size).astype(np.float32) / 255.0


def _mtl_image(body: Body, key: str, mode: str) -> np.ndarray | None:
    """A ``.mtl`` map as uint8 (mode "RGB" or "L"), image convention (row 0 = V = 1), or None."""
    path = body.resolve_texture(key)
    if path is None:
        return None
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def _kd(body: Body, default: float = 0.6) -> np.ndarray:
    kd = body.mtl.get("Kd", [default] * 3)
    kd = kd if isinstance(kd, list) and len(kd) >= 3 else [float(kd)] * 3
    return np.asarray(kd[:3], np.float32)


def _tinted(texture: np.ndarray | None, kd: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """MTL semantics: diffuse = map_Kd * Kd (in linear space); returns sRGB-encoded, clamped to a real albedo."""
    linear = np.broadcast_to(kd, (size[1], size[0], 3)) if texture is None else srgb_to_linear(texture) * kd
    return linear_to_srgb(np.minimum(linear, MAX_LINEAR_ALBEDO))


def _roughness_fallback(body: Body, variant: MaterialVariant | None) -> float:
    if variant is not None and variant.roughness_value > 0.0:
        return variant.roughness_value
    pr = body.mtl.get("Pr")
    return float(pr) if isinstance(pr, float) and pr > 0.0 else DEFAULT_ROUGHNESS


def metallic(body: Body) -> float:
    variants = body.variants
    if variants and variants[0].metallic > 0.0:
        return variants[0].metallic
    pm = body.mtl.get("Pm")
    return float(pm) if isinstance(pm, float) else 0.0


def texture_size(body: Body, wear_shape: tuple[int, int] | None, max_size: int) -> tuple[int, int]:
    """Render texture (width, height): the wear atlas resolution (or largest material map), capped."""
    if wear_shape is not None:
        h, w = wear_shape
    else:
        shapes = [v.normal.shape[:2] for v in body.variants if v.normal is not None] or [(512, 512)]
        h, w = max(shapes)
    s = min(1.0, max_size / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


def build_mapsets(
    body: Body,
    size: tuple[int, int],
    normals: NormalOptions = NormalOptions(),
    log: Callable[[str], None] | None = None,
) -> tuple[MapSet, MapSet]:
    """Return (base, worn) material maps for ``body`` at ``size`` = (width, height)."""
    variants = body.variants
    base_v = variants[0] if variants else None
    worn_v = variants[body.worn_variant_index] if variants else None

    base_tex = _albedo(body, "map_Kd", size)
    kd = _kd(body, default=0.6 if base_tex is None else 1.0)
    base_albedo = _tinted(base_tex, kd, size)
    worn_tex = _albedo(body, "map_Kd_worn", size)
    worn_albedo = base_albedo if worn_tex is None else _tinted(worn_tex, kd, size)

    def pick(primary: np.ndarray | None, fallback: np.ndarray | None) -> np.ndarray | None:
        return primary if primary is not None else fallback

    def mapset(normal, roughness, height, albedo: np.ndarray, rough_default: float, which: str) -> MapSet:
        return MapSet(
            albedo=np.array(albedo, dtype=np.float32),
            roughness=_scalar_map(roughness, rough_default, size),
            normal=_decode_normal(normal, size, normals, f"body {body.index} {which}", log),
            height=None if height is None else _scalar_map(height, 0.0, size),
        )

    base_n = pick(base_v.normal if base_v else None, _mtl_image(body, "map_Bump", "RGB"))
    base_r = pick(base_v.roughness if base_v else None, _mtl_image(body, "map_Pr", "L"))
    base_h = pick(base_v.height if base_v else None, _mtl_image(body, "map_disp", "L"))
    worn_n = pick(_mtl_image(body, "map_Bump_Worn", "RGB"), worn_v.normal if worn_v else None)
    worn_r = pick(_mtl_image(body, "map_Pr_worn", "L"), worn_v.roughness if worn_v else None)
    worn_h = pick(_mtl_image(body, "map_disp_worn", "L"), worn_v.height if worn_v else None)

    base = mapset(base_n, base_r, base_h, base_albedo, _roughness_fallback(body, base_v), "base")
    worn = mapset(
        pick(worn_n, base_n), pick(worn_r, base_r), worn_h, worn_albedo, _roughness_fallback(body, worn_v), "worn"
    )
    return base, worn
