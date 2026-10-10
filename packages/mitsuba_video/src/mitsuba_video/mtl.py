"""Wavefront ``.mtl`` parsing and a material provider built from ``.mtl`` texture maps.

Recognised entries (case-insensitive keys): ``Kd``, ``Pr``, ``Pm``, ``map_Kd``, ``map_Bump`` (tangent-space normal),
``map_Pr`` (roughness), ``map_disp`` (height), and their ``*_worn`` variants (``map_Kd_worn``, ``map_Bump_Worn``,
``map_Pr_worn``, ``map_disp_worn``) for the affected appearance. Missing worn maps fall back to the base maps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .maps import (
    DEFAULT_ROUGHNESS,
    MapSet,
    NormalOptions,
    decode_normal,
    load_image,
    resize,
    scalar_map,
    tinted,
)


def parse_mtl_text(text: str) -> dict[str, dict[str, object]]:
    """Map ``newmtl`` names to their scalar and ``map_*`` entries (last value wins, keys case-preserved)."""
    materials: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        rest = rest.strip()
        if key == "newmtl":
            current = materials.setdefault(rest, {})
        elif current is not None and rest:
            if key.startswith("map_"):
                # Texture options (-bm 1.0 ...) precede the filename; the filename is the last token.
                current[key] = rest.split()[-1]
            else:
                values = rest.split()
                try:
                    floats = [float(v) for v in values]
                    current[key] = floats[0] if len(floats) == 1 else floats
                except ValueError:
                    current[key] = rest
    return materials


def mtl_get(mtl: dict[str, object], key: str) -> object | None:
    """Case-insensitive lookup (assets mix spellings such as ``map_Bump_Worn`` / ``map_Pr_worn``)."""
    return next((v for k, v in mtl.items() if k.lower() == key.lower()), None)


def resolve_map(mtl: dict[str, object], key: str, base_dir: Path | None) -> Path | None:
    """Resolve a ``map_*`` entry relative to the ``.mtl`` directory to an existing file, or None."""
    rel = mtl_get(mtl, key)
    if not isinstance(rel, str) or base_dir is None:
        return None
    path = (base_dir / rel.replace("\\", "/")).resolve()
    return path if path.is_file() else None


def kd_colour(mtl: dict[str, object], default: float = 0.6) -> np.ndarray:
    kd = mtl_get(mtl, "Kd")
    if kd is None:
        return np.full(3, default, np.float32)
    kd = kd if isinstance(kd, list) and len(kd) >= 3 else [float(kd)] * 3  # type: ignore[arg-type]
    return np.asarray(kd[:3], np.float32)


def scalar_entry(mtl: dict[str, object], key: str) -> float | None:
    v = mtl_get(mtl, key)
    return float(v) if isinstance(v, float) else None


@dataclass
class MapImages:
    """Raw uint8 maps for one appearance (base or worn); any may be None."""

    albedo: np.ndarray | None = None  # HxWx3 uint8 sRGB
    normal: np.ndarray | None = None  # HxWx3 uint8
    roughness: np.ndarray | None = None  # HxW uint8
    height: np.ndarray | None = None  # HxW uint8


def build_mapset(
    images: MapImages,
    colour: np.ndarray,
    roughness_default: float,
    size: tuple[int, int],
    normals: NormalOptions,
    label: str,
    log: Callable[[str], None] | None,
) -> MapSet:
    albedo_tex = None if images.albedo is None else resize(images.albedo, size).astype(np.float32) / 255.0
    return MapSet(
        albedo=np.array(tinted(albedo_tex, colour, size), dtype=np.float32),
        roughness=scalar_map(images.roughness, roughness_default, size),
        normal=decode_normal(images.normal, size, normals, label, log),
        height=None if images.height is None else scalar_map(images.height, 0.0, size),
    )


class MtlMaterial:
    """MaterialProvider reading the maps an ``.mtl`` material references (see module docstring)."""

    def __init__(self, mtl: dict[str, object], base_dir: Path | None, label: str = "") -> None:
        self.mtl = mtl
        self.base_dir = base_dir
        self.label = label
        pm = scalar_entry(mtl, "Pm")
        self.metallic = pm if pm is not None else 0.0

    def _image(self, key: str, mode: str) -> np.ndarray | None:
        path = resolve_map(self.mtl, key, self.base_dir)
        return None if path is None else load_image(path, mode)

    def images(self, worn: bool) -> MapImages:
        suffix = "_worn" if worn else ""
        return MapImages(
            albedo=self._image("map_Kd" + suffix, "RGB"),
            normal=self._image("map_Bump" + suffix, "RGB"),
            roughness=self._image("map_Pr" + suffix, "L"),
            height=self._image("map_disp" + suffix, "L"),
        )

    def preferred_size(self) -> tuple[int, int] | None:
        shapes = [img.shape[:2] for img in vars(self.images(False)).values() if img is not None]
        return max(shapes) if shapes else None

    def build(
        self, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
    ) -> tuple[MapSet, MapSet]:
        base_imgs, worn_imgs = self.images(False), self.images(True)
        for key in ("albedo", "normal", "roughness", "height"):
            if getattr(worn_imgs, key) is None:
                setattr(worn_imgs, key, getattr(base_imgs, key))
        colour = kd_colour(self.mtl, default=0.6 if base_imgs.albedo is None else 1.0)
        pr = scalar_entry(self.mtl, "Pr")
        rough = pr if pr is not None and pr > 0.0 else DEFAULT_ROUGHNESS
        base = build_mapset(base_imgs, colour, rough, size, normals, f"{self.label} base", log)
        worn = build_mapset(worn_imgs, colour, rough, size, normals, f"{self.label} worn", log)
        return base, worn
