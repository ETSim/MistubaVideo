"""Minimal Wavefront OBJ loader for renderable triangle meshes.

Returns a triangle soup indexed per unique (position, texcoord, normal) corner, so UV seams stay split. Polygons are
fan-triangulated. Missing normals are computed per face; missing texcoords are zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjMesh:
    vertices: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32 (OBJ convention, v up)
    faces: np.ndarray  # (M, 3) uint32
    mtllib: str | None
    materials: list[str]  # usemtl names in order of appearance


def _index(token: str, count: int) -> int | None:
    if not token:
        return None
    i = int(token)
    return i - 1 if i > 0 else count + i


def load_obj(path: Path) -> ObjMesh:
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals_in: list[list[float]] = []
    corner_ids: dict[tuple[int, int | None, int | None], int] = {}
    out_v: list[list[float]] = []
    out_t: list[list[float]] = []
    out_n: list[list[float] | None] = []
    faces: list[tuple[int, int, int]] = []
    mtllib, used = None, []

    def corner(token: str) -> int:
        parts = (token.split("/") + ["", ""])[:3]
        key = (
            _index(parts[0], len(positions)),
            _index(parts[1], len(texcoords)),
            _index(parts[2], len(normals_in)),
        )
        if key not in corner_ids:
            vi, ti, ni = key
            corner_ids[key] = len(out_v)
            out_v.append(positions[vi])  # type: ignore[index]
            out_t.append(texcoords[ti][:2] if ti is not None else [0.0, 0.0])
            out_n.append(normals_in[ni] if ni is not None else None)
        return corner_ids[key]

    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tag, _, rest = line.partition(" ")
            vals = rest.split()
            if tag == "v":
                positions.append([float(x) for x in vals[:3]])
            elif tag == "vt":
                texcoords.append([float(x) for x in vals[:2]] + ([0.0] if len(vals) < 2 else []))
            elif tag == "vn":
                normals_in.append([float(x) for x in vals[:3]])
            elif tag == "f":
                ids = [corner(t) for t in vals]
                for k in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[k], ids[k + 1]))
            elif tag == "mtllib":
                mtllib = rest.strip()
            elif tag == "usemtl":
                used.append(rest.strip())

    v = np.asarray(out_v, np.float32).reshape(-1, 3)
    f = np.asarray(faces, np.uint32).reshape(-1, 3)
    n = np.zeros_like(v)
    have = np.array([x is not None for x in out_n], bool)
    if have.any():
        n[have] = np.asarray([x for x in out_n if x is not None], np.float32)
    if not have.all() and len(f):
        tri = v[f.astype(np.int64)]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        acc = np.zeros_like(v)
        for k in range(3):
            np.add.at(acc, f[:, k], fn)
        n[~have] = acc[~have]
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(norm > 1e-12, n / np.maximum(norm, 1e-12), np.float32([0, 1, 0]))
    return ObjMesh(v, n.astype(np.float32), np.asarray(out_t, np.float32).reshape(-1, 2), f, mtllib, used)


def write_obj(path: Path, vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray,
              mtllib: str | None = None, material: str | None = None) -> Path:  # fmt: skip
    """Write a corner-indexed triangle mesh (each vertex carries its own uv/normal)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if mtllib:
            fh.write(f"mtllib {mtllib}\n")
        for x, y, z in vertices:
            fh.write(f"v {x:.7g} {y:.7g} {z:.7g}\n")
        for u, w in uvs:
            fh.write(f"vt {u:.7g} {w:.7g}\n")
        for x, y, z in normals:
            fh.write(f"vn {x:.7g} {y:.7g} {z:.7g}\n")
        if material:
            fh.write(f"usemtl {material}\n")
        for a, b, c in np.asarray(faces, np.int64) + 1:
            fh.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    return path
