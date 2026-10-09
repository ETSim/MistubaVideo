"""Tiny synthetic manifest (plane + sliding box) for tests and smoke renders; needs no h5py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .writer import ManifestWriter

TEX = 64


def _plane() -> tuple[np.ndarray, ...]:
    v = np.float32([[-1, 0, -0.5], [1, 0, -0.5], [1, 0, 0.5], [-1, 0, 0.5]])
    uv = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    n = np.tile(np.float32([0, 1, 0]), (4, 1))
    return v, n, uv, np.uint32([[0, 2, 1], [0, 3, 2]])


def _cube(half: float = 0.1) -> tuple[np.ndarray, ...]:
    verts, norms, uvs, faces = [], [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3, np.float32)
            normal[axis] = sign
            u_axis, v_axis = [a for a in range(3) if a != axis]
            base = len(verts)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3, np.float32)
                p[axis], p[u_axis], p[v_axis] = sign * half, du * half, dv * half
                verts.append(p)
                norms.append(normal)
                uvs.append(((du + 1) / 2, (dv + 1) / 2))
            quad = np.array(verts[base : base + 4])
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), normal) >= 0:
                faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            else:
                faces += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    return np.array(verts, np.float32), np.array(norms, np.float32), np.array(uvs, np.float32), np.array(faces, np.uint32)


def write_synthetic_manifest(root: Path, frames: int = 6, box_scale: float = 1.5) -> Path:
    """Box sliding along +x at z = +0.2 across a 2 x 1 plane, wearing a stripe into the plane and a disc into itself."""
    w = ManifestWriter(Path(root), name="synthetic: box sliding on plane", travel_label="slid")
    w.add_field("wear", label="wear, normalized", threshold=0.5 / 255.0, area_title="worn area")
    pv, pn, puv, pf = _plane()
    w.add_body(0, "synthetic_plane", pv, pn, puv, pf, fixed=True,
               material={"base": {"albedo": [0.35, 0.34, 0.33], "roughness": 0.7},
                         "worn": {"roughness": 0.3}})  # fmt: skip
    cv, cn, cuv, cf = _cube()
    w.add_body(1, "synthetic_box", cv, cn, cuv, cf, scale=box_scale,
               material={"base": {"albedo": [0.62, 0.45, 0.30], "roughness": 0.6}})  # fmt: skip
    half, z = 0.1 * box_scale, 0.2
    plane_wear = np.zeros((TEX, TEX), np.float32)
    box_wear = np.zeros((TEX, TEX), np.float32)
    r = np.hypot(*(np.mgrid[0:TEX, 0:TEX] / TEX - 0.5))
    for k in range(frames):
        s = k / max(frames - 1, 1)
        x, angle = -0.6 + 1.2 * s, 0.6 * s
        u0, u1 = (x - half + 1.0) / 2.0, (x + half + 1.0) / 2.0
        v0, v1 = z - half + 0.5, z + half + 0.5  # plane v = z + 0.5; image row 0 = V = 1
        cols = slice(int(u0 * TEX), max(int(u1 * TEX), int(u0 * TEX) + 1))
        rows = slice(int((1 - v1) * TEX), int((1 - v0) * TEX))
        plane_wear[rows, cols] = np.minimum(plane_wear[rows, cols] + 0.35, 1.0)
        box_wear = np.maximum(box_wear, np.clip(0.9 * s - r, 0.0, 1.0))
        q = [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]
        w.add_frame(k * 0.01, [[0, 0, 0], [x, half, z]], [[1, 0, 0, 0], q],
                    fields={"wear": {0: plane_wear, 1: box_wear}})  # fmt: skip
    return w.close()
