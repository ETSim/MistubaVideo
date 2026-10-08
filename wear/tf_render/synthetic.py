"""Write a tiny synthetic ``--record`` HDF5 (plane + sliding box) for tests and the Docker/act smoke render.

It reproduces only the schema ``recording.Recording`` reads, mirroring ``SimulationSerializer``:
``metadata/bodies/body_k/{mesh,mtl,material_atlas}`` (atlas images GL-flipped, as ``OBJLoader`` stores them),
``frames/frame_i/{positions,orientations,time,textures/body_k/wear}`` and the ``scale`` body attribute.

    python tools/tf_render/synthetic.py out_dir/FrictionTexture_synthetic.h5
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

TEX = 64


def _png(arr: np.ndarray) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _write_png(group: h5py.Group, name: str, arr: np.ndarray) -> None:
    ds = group.create_dataset(name, data=_png(arr))
    ds.attrs["format"] = "PNG"
    ds.attrs["width"] = np.int32(arr.shape[1])
    ds.attrs["height"] = np.int32(arr.shape[0])


def _plane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.float32([[-1, 0, -0.5], [1, 0, -0.5], [1, 0, 0.5], [-1, 0, 0.5]])
    uv = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    tri = [0, 2, 1, 0, 3, 2]
    v = corners[tri]
    return v, np.tile(np.float32([0, 1, 0]), (len(v), 1)), uv[tri]


def _cube(half: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts, norms, uvs = [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3, np.float32)
            n[axis] = sign
            u_axis, v_axis = [a for a in range(3) if a != axis]
            quad = []
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3, np.float32)
                p[axis] = sign * half
                p[u_axis] = du * half
                p[v_axis] = dv * half
                quad.append((p, ((du + 1) / 2, (dv + 1) / 2)))
            order = [0, 1, 2, 0, 2, 3]
            e1 = quad[1][0] - quad[0][0]
            e2 = quad[2][0] - quad[0][0]
            if np.dot(np.cross(e1, e2), n) < 0:
                order = [0, 2, 1, 0, 3, 2]
            for k in order:
                verts.append(quad[k][0])
                norms.append(n)
                uvs.append(quad[k][1])
    return np.array(verts, np.float32), np.array(norms, np.float32), np.array(uvs, np.float32)


def _atlas(group: h5py.Group, worn: bool, rng: np.random.Generator) -> None:
    yy, xx = np.mgrid[0:TEX, 0:TEX] / TEX
    if worn:
        nx = 0.25 * np.sin(xx * 6 * math.pi)
        normal = np.stack([nx, np.zeros_like(nx), np.ones_like(nx)], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        rough = np.full((TEX, TEX), 0.3)
        height = 0.4 + 0.05 * rng.random((TEX, TEX))
    else:
        normal = np.tile([0.0, 0.0, 1.0], (TEX, TEX, 1))
        rough = 0.7 + 0.1 * rng.random((TEX, TEX))
        height = 0.5 + 0.05 * rng.random((TEX, TEX))
    enc = lambda a: np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)  # noqa: E731
    # Stored GL-flipped (row 0 = V = 0), exactly like OBJLoader's imageFlippedForOpenGL atlas images.
    _write_png(group, "normal", enc(normal * 0.5 + 0.5)[::-1].copy())
    _write_png(group, "roughness", enc(rough)[::-1].copy())
    _write_png(group, "height", enc(height)[::-1].copy())


def write_synthetic(path: Path, frames: int = 6, box_scale: float = 1.5) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    meshes = [_plane(), _cube()]
    names = ["synthetic_plane", "synthetic_box"]
    kd = [(0.35, 0.34, 0.33), (0.62, 0.45, 0.30)]
    dt = 0.01
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.create_dataset("num_bodies", data=np.uint64(2))
        bodies = meta.create_group("bodies")
        for i, (v, n, uv) in enumerate(meshes):
            g = bodies.create_group(f"body_{i}")
            g.attrs["scale"] = np.float32(1.0 if i == 0 else box_scale)
            g.attrs["material_name"] = f"Material.{i:03d}"
            g.attrs["material_mtl_filename"] = f"{names[i]}.mtl"
            g.attrs["normal_variant_count"] = np.int32(2)
            m = g.create_group("mesh")
            m.create_dataset("vertices", data=v)
            m.create_dataset("normals", data=n)
            m.create_dataset("uvs", data=uv)
            m.create_dataset("face_vertex_indices", data=np.arange(len(v), dtype=np.int32))
            m.create_dataset("face_vertex_counts", data=np.full(len(v) // 3, 3, np.int32))
            m.attrs["mesh_name"] = names[i]
            m.attrs["mesh_filename"] = ""
            mtl = g.create_group("mtl")
            text = f"newmtl Material.{i:03d}\nKd {kd[i][0]} {kd[i][1]} {kd[i][2]}\nPr 0.6\nPm 0.0\n"
            mtl.create_dataset("mtl_content", data=text, dtype=h5py.string_dtype())
            atlas = g.create_group("material_atlas")
            for k, worn in enumerate((False, True)):
                vg = atlas.create_group(f"variant_{k:03d}")
                vg.attrs["name"] = f"Material.{i:03d}_{'alt' if worn else 'atlasUnworn'}"
                vg.attrs["roughness"] = np.float32(0.0)
                vg.attrs["metallic"] = np.float32(0.0)
                _atlas(vg, worn, rng)
        scene = meta.create_group("scene")
        scene.attrs["scenario_name"] = "synthetic: box sliding on plane"
        ic = scene.create_group("initial_conditions")
        ic.create_dataset("fixed", data=np.array([True, False]))
        sp = scene.create_group("simulation_parameters")
        sp.create_dataset("time_step", data=np.float32(dt))
        sem = meta.create_group("texture_semantics")
        sem.attrs["wear"] = "cumulative_wear_metric"

        frames_group = f.create_group("frames")
        plane_wear = np.zeros((TEX, TEX), np.float32)
        box_wear = np.zeros((TEX, TEX), np.float32)
        half = 0.1 * box_scale
        z = 0.2  # off the plane's centre line, so a V flip would put the trail on the wrong side
        for k in range(frames):
            s = k / max(frames - 1, 1)
            x = -0.6 + 1.2 * s
            angle = 0.6 * s
            g = frames_group.create_group(f"frame_{k}")
            g.attrs["frame_index"] = np.int32(k)
            g.create_dataset("time", data=np.float32(k * dt))
            g.create_dataset("positions", data=np.float32([[0, 0, 0], [x, half, z]]))
            q = [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]  # (w, x, y, z), about +y
            g.create_dataset("orientations", data=np.float32([[1, 0, 0, 0], q]))
            # Plane wear: footprint under the box, in image convention (row 0 = V = 1).
            u0, u1 = (x - half + 1.0) / 2.0, (x + half + 1.0) / 2.0
            v0, v1 = (z - half + 0.5), (z + half + 0.5)  # plane v = z + 0.5
            cols = slice(int(u0 * TEX), max(int(u1 * TEX), int(u0 * TEX) + 1))
            rows = slice(int((1 - v1) * TEX), int((1 - v0) * TEX))
            plane_wear[rows, cols] = np.minimum(plane_wear[rows, cols] + 0.35, 1.0)
            r = np.hypot(*(np.mgrid[0:TEX, 0:TEX] / TEX - 0.5))
            box_wear = np.maximum(box_wear, np.clip(0.9 * s - r, 0.0, 1.0))
            tex = g.create_group("textures")
            for b, wear in enumerate((plane_wear, box_wear)):
                tb = tex.create_group(f"body_{b}")
                gray = np.clip(wear * 255.0, 0, 255).astype(np.uint8)
                _write_png(tb, "wear", np.dstack([gray, gray, gray, np.full_like(gray, 255)]))
        mf = meta.create_group("frames")
        mf.create_dataset("frame_count", data=np.int32(frames))
        mf.create_dataset("frame_indices", data=np.arange(frames, dtype=np.int32))
        mf.create_dataset("frame_times", data=np.arange(frames, dtype=np.float32) * dt)
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("FrictionTexture_synthetic.h5")
    print(write_synthetic(target))
