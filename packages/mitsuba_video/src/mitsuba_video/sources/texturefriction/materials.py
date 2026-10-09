"""TextureFriction material provider: base/worn maps for a recorded body.

Sources:

* base normal / roughness / height: the body's HDF5 ``material_atlas`` variant 0 (byte-identical to the ``.mtl``
  ``map_Bump`` / ``map_Pr`` / ``map_disp`` files once un-flipped), falling back to those files.
* worn normal / roughness / height: the ``.mtl`` worn maps TextureFriction's OBJLoader reads (``map_Bump_Worn``,
  ``map_Pr_worn``, ``map_disp_worn``), falling back to the atlas worn variant. The atlas variants are exported from
  each atlas material's own images, so they do not carry ``map_Pr_worn``: on ``plane_concrete`` variant 1 has the
  base roughness while ``concrete_roughness_worn.png`` is the real worn target.
* albedo: ``map_Kd`` / ``map_Kd_worn`` times ``Kd`` (not stored in the atlas), falling back to the ``Kd`` colour.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...maps import DEFAULT_ROUGHNESS, MapSet, NormalOptions, load_image
from ...mtl import MapImages, build_mapset, kd_colour, scalar_entry
from .recording import MaterialVariant, RecordedBody


class TextureFrictionMaterial:
    def __init__(self, body: RecordedBody) -> None:
        self.body = body
        variants = body.variants
        if variants and variants[0].metallic > 0.0:
            self.metallic = variants[0].metallic
        else:
            pm = scalar_entry(body.mtl, "Pm")
            self.metallic = pm if pm is not None else 0.0

    def _mtl_image(self, key: str, mode: str) -> np.ndarray | None:
        path = self.body.resolve_texture(key)
        return None if path is None else load_image(path, mode)

    def _roughness_fallback(self, variant: MaterialVariant | None) -> float:
        if variant is not None and variant.roughness_value > 0.0:
            return variant.roughness_value
        pr = scalar_entry(self.body.mtl, "Pr")
        return pr if pr is not None and pr > 0.0 else DEFAULT_ROUGHNESS

    def preferred_size(self) -> tuple[int, int] | None:
        shapes = [v.normal.shape[:2] for v in self.body.variants if v.normal is not None]
        return max(shapes) if shapes else None

    def build(
        self, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
    ) -> tuple[MapSet, MapSet]:
        variants = self.body.variants
        base_v = variants[0] if variants else None
        worn_v = variants[self.body.worn_variant_index] if variants else None

        def pick(primary: np.ndarray | None, fallback: np.ndarray | None) -> np.ndarray | None:
            return primary if primary is not None else fallback

        base_albedo = self._mtl_image("map_Kd", "RGB")
        worn_albedo = self._mtl_image("map_Kd_worn", "RGB")
        base = MapImages(
            albedo=base_albedo,
            normal=pick(base_v.normal if base_v else None, self._mtl_image("map_Bump", "RGB")),
            roughness=pick(base_v.roughness if base_v else None, self._mtl_image("map_Pr", "L")),
            height=pick(base_v.height if base_v else None, self._mtl_image("map_disp", "L")),
        )
        worn = MapImages(
            albedo=pick(worn_albedo, base_albedo),
            normal=pick(pick(self._mtl_image("map_Bump_Worn", "RGB"), worn_v.normal if worn_v else None), base.normal),
            roughness=pick(pick(self._mtl_image("map_Pr_worn", "L"), worn_v.roughness if worn_v else None),
                           base.roughness),  # fmt: skip
            height=pick(self._mtl_image("map_disp_worn", "L"), worn_v.height if worn_v else None),
        )
        colour = kd_colour(self.body.mtl, default=0.6 if base_albedo is None else 1.0)
        label = f"body {self.body.index}"
        return (
            build_mapset(base, colour, self._roughness_fallback(base_v), size, normals, f"{label} base", log),
            build_mapset(worn, colour, self._roughness_fallback(worn_v), size, normals, f"{label} worn", log),
        )
