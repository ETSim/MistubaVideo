"""Source-agnostic data model: bodies (mesh + material), frames, and the per-frame scalar field being visualised.

A *field* is any per-texel scalar in [0, 1] stored in a body's UV atlas: wear depth, sliding distance, temperature,
damage, coverage... It drives both the material blend (base -> "worn"/affected look) and the heatmap overlay.
Image convention everywhere: row 0 of an atlas is the top of the texture (V = 1), as in ordinary image files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .maps import MapSet, NormalOptions


@dataclass
class Frame:
    key: str  # source-specific identifier (HDF5 group name, manifest index...)
    index: int  # simulation frame number shown to people
    time: float  # seconds
    position: int  # 0-based order within the source


@dataclass(frozen=True)
class FieldSpec:
    """How a field is named, thresholded and labelled. Sources describe their fields with it."""

    name: str = "field"
    label: str = "field, normalized"  # colour-bar caption
    threshold: float = 1e-6  # values above count as "covered" (worn area, panel crop)
    video_prefix: str = "field"  # output video stem: <prefix>_<look>_<camera>.mp4
    area_csv: str = "covered_area.csv"
    area_title: str = "covered area"


class MaterialProvider(Protocol):
    """Builds a body's base and affected ("worn") material maps at a requested resolution."""

    metallic: float

    def preferred_size(self) -> tuple[int, int] | None:
        """(height, width) of the native material maps, used when the body has no field image."""
        ...

    def build(
        self, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
    ) -> tuple[MapSet, MapSet]:
        """(base, worn) at ``size`` = (width, height)."""
        ...


@dataclass
class Body:
    index: int
    name: str
    vertices: np.ndarray  # (N, 3) float32, mesh-local units (before scale)
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32, OBJ convention (v up)
    faces: np.ndarray  # (M, 3) uint32
    scale: float
    fixed: bool
    material: MaterialProvider
    display_name: str = ""  # short name for panels; defaults to ``name``
    notes: dict[str, str] = field(default_factory=dict)  # extra `inspect` columns

    @property
    def label(self) -> str:
        return self.display_name or self.name

    def area_per_texel(self, atlas_shape: tuple[int, int]) -> float:
        """World area (scaled mesh units squared) covered by one texel of an ``atlas_shape`` texture.

        Assumes uniform texel density: total triangle area divided by the UV area those triangles cover. Exact for
        planar boxes and planes; an average for meshes with uneven UV stretch.
        """
        tri = (self.vertices * self.scale)[self.faces.astype(np.int64)]
        world = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        uv = self.uvs[self.faces.astype(np.int64)]
        e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
        h, w = atlas_shape
        return float(world / max(uv_area * w * h, 1e-12))

    @cached_property
    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.vertices * self.scale
        return v.min(axis=0), v.max(axis=0)
