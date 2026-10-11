"""TextureFriction material provider: for each channel, the images the viewer blends with wear.

The viewer packs up to four images per channel into a variant atlas (``MeshAssets::ensureNormalAtlasTexture``,
``ensurePbrAtlasTextures`` and ``ensureHeightAtlasTexture``). ``viewer_channel_sources`` applies the same choice to a
recording, ordered unworn -> fully worn:

* normal: the first four ``material_atlas`` variants when the atlas has four or more, the first two when it has two
  or three; otherwise the ``.mtl`` ``map_Bump`` / ``map_Bump_Worn`` pair.
* roughness: those same variants when every one has a roughness map; otherwise ``map_Pr`` / ``map_Pr_worn``.
* albedo: those same variants when every one has an albedo image. There is no ``.mtl`` fallback, because the viewer
  does not use ``map_Kd_worn``; recordings made before the serializer exported variant albedo get the base only.
* height (it shapes the blend and is not shaded): the variants' heights for the normal channel's count, or
  ``map_disp`` / ``map_disp_worn`` for two.

A channel left with one image does not change with wear. When the recording says the viewer's variant blend was off,
every channel keeps its first image. Atlas images are stored GL-flipped and flipped back on read (``recording``);
``.mtl`` images are read from the asset files. Albedo keeps MTL semantics, texture times ``Kd``: the base material's
``Kd`` tints every variant, since a variant's own ``Kd`` is not recorded.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...blend import VariantStack
from ...maps import DEFAULT_ROUGHNESS, MapSet, NormalOptions, decode_normal, load_image, resize, scalar_map, tinted
from ...mtl import kd_colour, scalar_entry
from .recording import MaterialVariant, RecordedBody

Image = np.ndarray | None
MtlImage = Callable[[str, str], Image]


def _atlas_channel(variants: list[MaterialVariant], attr: str, every: bool) -> list[Image]:
    """The variants the viewer packs for one channel: 4 when there are 4+, 2 when 2 or 3; ``every`` requires all."""
    n = len(variants)
    count = 4 if n >= 4 else 2 if n >= 2 else 0
    images = [getattr(v, attr) for v in variants[:count]]
    if not images or (every and any(img is None for img in images)):
        return []
    return images


def _mtl_pair(base: Image, worn_key: str, mode: str, mtl_image: MtlImage) -> list[Image]:
    """``MaterialAtlas`` base/worn registry pair: needs both images."""
    worn = mtl_image(worn_key, mode)
    return [base, worn] if base is not None and worn is not None else []


def viewer_channel_sources(variants: list[MaterialVariant], mtl_image: MtlImage) -> dict[str, list[Image]]:
    """Per channel, the uint8 images (unworn -> worn) the viewer blends; one entry means the channel is static."""
    first = variants[0] if variants else None

    def own(attr: str, key: str, mode: str) -> Image:
        img = getattr(first, attr) if first is not None else None
        return img if img is not None else mtl_image(key, mode)

    base_normal = own("normal", "map_Bump", "RGB")
    base_rough = own("roughness", "map_Pr", "L")
    base_height = own("height", "map_disp", "L")
    base_albedo = mtl_image("map_Kd", "RGB")
    if base_albedo is None and first is not None:
        base_albedo = first.albedo

    normal = _atlas_channel(variants, "normal", every=False)
    normal = normal or _mtl_pair(base_normal, "map_Bump_Worn", "RGB", mtl_image)
    rough = _atlas_channel(variants, "roughness", every=True)
    rough = rough or _mtl_pair(base_rough, "map_Pr_worn", "L", mtl_image)
    albedo = _atlas_channel(variants, "albedo", every=True)

    height: list[Image] = []
    if len(normal) >= 2:
        height = _atlas_channel(variants, "height", every=True)
        if len(height) != len(normal):
            height = []
        if not height and len(normal) == 2:
            pair = _mtl_pair(base_height, "map_disp_worn", "L", mtl_image)
            height = pair if len(pair) == 2 and pair[0].shape == pair[1].shape else []

    return {
        "normal": normal or [base_normal],
        "roughness": rough or [base_rough],
        "albedo": albedo or [base_albedo],
        "height": height or ([base_height] if base_height is not None else []),
    }


class TextureFrictionMaterial:
    def __init__(self, body: RecordedBody, blend_enabled: bool = True) -> None:
        self.body = body
        self.blend_enabled = blend_enabled
        variants = body.variants
        if variants and variants[0].metallic > 0.0:
            self.metallic = variants[0].metallic
        else:
            pm = scalar_entry(body.mtl, "Pm")
            self.metallic = pm if pm is not None else 0.0

    def _mtl_image(self, key: str, mode: str) -> Image:
        path = self.body.resolve_texture(key)
        return None if path is None else load_image(path, mode)

    def _roughness_fallback(self) -> float:
        variants = self.body.variants
        if variants and variants[0].roughness_value > 0.0:
            return variants[0].roughness_value
        pr = scalar_entry(self.body.mtl, "Pr")
        return pr if pr is not None and pr > 0.0 else DEFAULT_ROUGHNESS

    def preferred_size(self) -> tuple[int, int] | None:
        shapes = [v.normal.shape[:2] for v in self.body.variants if v.normal is not None]
        return max(shapes) if shapes else None

    def build_stack(
        self, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
    ) -> VariantStack:
        sources = viewer_channel_sources(self.body.variants, self._mtl_image)
        if not self.blend_enabled:
            sources = {k: v[:1] for k, v in sources.items()}
        colour = kd_colour(self.body.mtl, default=0.6 if sources["albedo"][0] is None else 1.0)
        rough = self._roughness_fallback()
        label = f"body {self.body.index}"

        def albedo(img: Image) -> np.ndarray:
            tex = None if img is None else resize(img[..., :3], size).astype(np.float32) / 255.0
            return np.array(tinted(tex, colour, size), dtype=np.float32)

        return VariantStack(
            albedo=[albedo(img) for img in sources["albedo"]],
            normal=[decode_normal(img, size, normals, f"{label} stage {k}", log)
                    for k, img in enumerate(sources["normal"])],  # fmt: skip
            roughness=[scalar_map(img, rough, size) for img in sources["roughness"]],
            height=[scalar_map(img, 0.0, size) for img in sources["height"]],
        )

    def build(
        self, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
    ) -> tuple[MapSet, MapSet]:
        """(unworn, fully worn) for consumers that take two maps, such as the manifest export."""
        stack = self.build_stack(size, normals, log)
        return stack.base, stack.worn
