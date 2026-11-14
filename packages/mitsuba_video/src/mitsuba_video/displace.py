"""Geometric displacement from the material height maps (opt-in: ``tf-render render --displace MM``).

A normal map shades relief but leaves silhouettes, self-shadowing and contact occlusion flat. With displacement on,
each body mesh is subdivided once (midpoint 1 -> 4 split, up to a triangle budget) and every frame its vertices move
inward along the welded vertex normal by ``amplitude * (h - h_top)``: ``h`` is that frame's blended height map (base
-> worn stages, so worn tracks sink with the wear) and ``h_top`` the first frame's 99.5th percentile. The asperity
tops stay on the collider surface the simulation used and the valleys go into the body, so resting contacts neither
float nor interpenetrate. Vertices that share a position (UV seams, hard edges) move together (mean height, mean
normal), so the mesh stays closed. Shading normals stay the mesh's own: the normal map still carries the fine detail.

The amplitude is a render choice: the simulator's height maps are dimensionless (mid-grey is the mean plane), so the
millimetres given here are the full 0..1 height range drawn in the render, not a measured roughness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Subdivision stops before a body's mesh would pass this many triangles, or its share of the scene's budget.
DEFAULT_MAX_TRIANGLES = 1_500_000
SCENE_MAX_TRIANGLES = 6_000_000
MIN_BODY_TRIANGLES = 100_000


def body_budget(displaced_bodies: int) -> int:
    """Per-body triangle cap when ``displaced_bodies`` bodies share the scene budget."""
    return min(DEFAULT_MAX_TRIANGLES, max(MIN_BODY_TRIANGLES, SCENE_MAX_TRIANGLES // max(displaced_bodies, 1)))
# Never draw relief deeper than this fraction of the body's bounding-box diagonal (a 0.5 mm relief on a 3 mm bead).
MAX_AMPLITUDE_FRACTION = 0.02
# Percentile of the first frame's height taken as the asperity tops (the surface the collider stands for).
TOP_PERCENTILE = 99.5


def subdivide(vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """One midpoint 1 -> 4 split. Edges are shared by vertex index, so the mesh's own seams (split UVs, hard
    normals) stay split; positions, UVs and normals are interpolated linearly (normals renormalised)."""
    f = faces.astype(np.int64)
    n = len(vertices)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    lo, hi = edges.min(axis=1), edges.max(axis=1)
    keys, inverse = np.unique(lo * n + hi, return_inverse=True)
    a, b = keys // n, keys % n
    mid = (n + inverse.reshape(-1)).reshape(3, -1)
    m01, m12, m20 = mid
    nv = 0.5 * (normals[a] + normals[b])
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
    out_v = np.concatenate([vertices, 0.5 * (vertices[a] + vertices[b])])
    out_n = np.concatenate([normals, nv.astype(normals.dtype)])
    out_uv = np.concatenate([uvs, 0.5 * (uvs[a] + uvs[b])])
    v0, v1, v2 = f.T
    out_f = np.concatenate([np.stack([v0, m01, m20], 1), np.stack([m01, v1, m12], 1), np.stack([m20, m12, v2], 1),
                            np.stack([m01, m12, m20], 1)])  # fmt: skip
    return out_v, out_n, out_uv, out_f.astype(np.uint32)


def subdivision_levels(vertices: np.ndarray, faces: np.ndarray, target_edge: float,
                       max_triangles: int = DEFAULT_MAX_TRIANGLES) -> int:  # fmt: skip
    """How many 1 -> 4 splits bring the median edge to ``target_edge`` without passing ``max_triangles``."""
    f = faces.astype(np.int64)
    if len(f) == 0 or target_edge <= 0.0:
        return 0
    edge = float(np.median(np.linalg.norm(vertices[f[:, 1]] - vertices[f[:, 0]], axis=1)))
    levels = 0
    while len(f) * 4 ** (levels + 1) <= max_triangles and edge / 2**levels > target_edge:
        levels += 1
    return levels


def weld_groups(vertices: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertices at the same position (to 1e-7 of the extent): (group of each vertex, unit mean normal per group,
    members per group). A group whose normals cancel (a two-sided sheet) keeps its first member's normal."""
    extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) if len(vertices) else 1.0
    cells = np.round(vertices / max(extent * 1e-7, 1e-12)).astype(np.int64)
    _, first, group = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    group = group.reshape(-1)
    count = np.bincount(group).astype(np.float64)
    summed = np.stack([np.bincount(group, normals[:, c], len(count)) for c in range(3)], axis=1)
    length = np.linalg.norm(summed, axis=1, keepdims=True)
    mean = np.where(length > 1e-6, summed / np.maximum(length, 1e-12), normals[first])
    return group, mean, count


def sample_height(height: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Bilinear, wrap-around sample of an H x W map (image convention: row 0 = V = 1) at OBJ UVs (v up)."""
    h, w = height.shape[:2]
    x = np.asarray(uvs[:, 0], np.float64) * w - 0.5
    y = (1.0 - np.asarray(uvs[:, 1], np.float64)) * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    x0, y0 = x0.astype(np.int64) % w, y0.astype(np.int64) % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    m = np.asarray(height, np.float32).reshape(h, w)
    top = m[y0, x0] * (1.0 - fx) + m[y0, x1] * fx
    bottom = m[y1, x0] * (1.0 - fx) + m[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


@dataclass
class DisplacedMesh:
    """A body's subdivided mesh (scaled, body-local) and how to displace it."""

    vertices: np.ndarray  # (N, 3) float64, undisplaced
    normals: np.ndarray  # (N, 3) float64, shading normals (left as they are)
    uvs: np.ndarray  # (N, 2) OBJ convention
    faces: np.ndarray  # (M, 3) uint32
    group: np.ndarray  # (N,) weld group
    group_normal: np.ndarray  # (G, 3) displacement direction
    group_count: np.ndarray  # (G,)
    amplitude: float  # scene units for the full 0..1 height range
    levels: int
    top: float | None = None  # h_top, fixed by the first height map seen

    @classmethod
    def build(cls, vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray,
              amplitude: float, texture_size: tuple[int, int],
              max_triangles: int = DEFAULT_MAX_TRIANGLES) -> DisplacedMesh:  # fmt: skip
        """Subdivide toward edges of two height texels (from the mesh's mean texel density at ``texture_size`` =
        (width, height)) and cap ``amplitude`` at ``MAX_AMPLITUDE_FRACTION`` of the bounding-box diagonal."""
        v = np.asarray(vertices, np.float64)
        nrm = np.asarray(normals, np.float64)
        uv = np.asarray(uvs, np.float64)
        f = np.asarray(faces, np.uint32)
        target = 2.0 * np.sqrt(_area_per_texel(v, uv, f, texture_size))
        levels = subdivision_levels(v, f, target, max_triangles)
        for _ in range(levels):
            v, nrm, uv, f = subdivide(v, nrm, uv, f)
        diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) if len(v) else 0.0
        group, gn, count = weld_groups(v, nrm)
        return cls(v, nrm, uv, f, group, gn, count, min(float(amplitude), MAX_AMPLITUDE_FRACTION * diag), levels)

    def offsets(self, height: np.ndarray) -> np.ndarray:
        """Per-vertex inward offsets (<= 0, scene units) for this frame's blended height map."""
        h = sample_height(height, self.uvs)
        if self.top is None:
            self.top = float(np.percentile(np.asarray(height, np.float32), TOP_PERCENTILE))
        d = np.minimum(self.amplitude * (h - self.top), 0.0)
        return np.bincount(self.group, d, len(self.group_count)) / self.group_count

    def displaced(self, height: np.ndarray | None) -> np.ndarray:
        """Displaced vertex positions (N, 3); undisplaced without a height map."""
        if height is None or self.amplitude <= 0.0:
            return self.vertices
        g = self.offsets(height)
        return self.vertices + self.group_normal[self.group] * g[self.group][:, None]


def _area_per_texel(v: np.ndarray, uv: np.ndarray, f: np.ndarray, size: tuple[int, int]) -> float:
    """Mean world area of one texel of a (width, height) texture (``model.Body.area_per_texel``)."""
    fi = f.astype(np.int64)
    tri, t = v[fi], uv[fi]
    world = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    return float(world / max(uv_area * size[0] * size[1], 1e-12))
