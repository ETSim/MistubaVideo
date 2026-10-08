"""Read a TextureFriction ``--record`` HDF5 file for offline rendering.

Only what an offline renderer needs is exposed: each body's local mesh and uniform scale, its material atlas
variants (flat and worn), and per-frame poses plus the wear texture.

Image convention: every image returned here has row 0 at the top of the texture (V = 1), the same convention as
the PNG files referenced by the scenario ``.mtl`` files and the per-frame wear PNGs. The ``material_atlas``
images stored in the HDF5 were run through ``imageFlippedForOpenGL`` by ``OBJLoader`` (row 0 = V = 0), so they are
flipped back on read.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ASSET_ROOT_ENV = "TF_ASSET_ROOT"


def ensure_windows_hdf5_locking() -> None:
    """Disable HDF5 file locking on Windows so a read-only open works while the simulator holds the file."""
    if os.name == "nt":
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def decode_png(dataset) -> np.ndarray:
    """Decode a PNG byte dataset (uint8 1-D) into an HxW or HxWxC uint8 array."""
    blob = dataset[()]
    raw = blob.tobytes() if hasattr(blob, "tobytes") else bytes(blob)
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img)


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


@dataclass
class MaterialVariant:
    """One entry of a body's ``material_atlas`` (index 0 = unworn base)."""

    name: str
    normal: np.ndarray | None  # HxWx3 uint8, tangent-space normal map
    roughness: np.ndarray | None  # HxW uint8
    height: np.ndarray | None  # HxW uint8
    roughness_value: float
    metallic: float


@dataclass
class Body:
    index: int
    name: str
    vertices: np.ndarray  # (N, 3) float32, mesh-local units (before scale)
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32, OBJ convention (v up)
    faces: np.ndarray  # (M, 3) uint32
    scale: float
    scale_recorded: bool
    fixed: bool
    material_name: str
    mtl: dict[str, object]
    mtl_dir: Path | None
    normal_variant_count: int
    _group: h5py.Group = field(repr=False)

    @cached_property
    def variants(self) -> list[MaterialVariant]:
        if "material_atlas" not in self._group:
            return []
        out = []
        atlas = self._group["material_atlas"]
        for key in sorted(atlas.keys()):
            g = atlas[key]

            def img(name: str) -> np.ndarray | None:
                if name not in g:
                    return None
                arr = decode_png(g[name])
                if arr.ndim == 3 and name != "normal":
                    arr = arr[..., 0]
                if arr.ndim == 3:
                    arr = arr[..., :3]
                return np.ascontiguousarray(arr[::-1])  # undo imageFlippedForOpenGL

            out.append(
                MaterialVariant(
                    name=str(g.attrs.get("name", key)),
                    normal=img("normal"),
                    roughness=img("roughness"),
                    height=img("height"),
                    roughness_value=float(g.attrs.get("roughness", 0.0)),
                    metallic=float(g.attrs.get("metallic", 0.0)),
                )
            )
        return out

    @property
    def worn_variant_index(self) -> int:
        """Same rule as ``WearAtlasBake::tryExportWornBlendedToBodyDir``: variant 3 for 4-way atlases, else 1."""
        n = len(self.variants)
        if n < 2:
            return 0
        return min(3 if self.normal_variant_count >= 4 else 1, n - 1)

    def resolve_texture(self, key: str) -> Path | None:
        """Resolve an ``.mtl`` ``map_*`` entry to an existing file, or None."""
        # OBJLoader matches map_* keys case-sensitively but the assets mix spellings (map_Bump_Worn, map_Pr_worn).
        rel = next((v for k, v in self.mtl.items() if k.lower() == key.lower()), None)
        if not isinstance(rel, str) or self.mtl_dir is None:
            return None
        path = (self.mtl_dir / rel.replace("\\", "/")).resolve()
        return path if path.is_file() else None

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


def _resolve_asset_dir(mesh_filename: str, roots: list[Path]) -> Path | None:
    """Directory of the source OBJ (where its .mtl textures live).

    The HDF5 stores the absolute OBJ path of the machine that recorded it (often a build tree such as
    ``build-ninja-msvc/resources/...``). When that path does not exist here, its ``resources/...`` suffix is looked up
    under each of ``roots``.
    """
    if not mesh_filename:
        return None
    path = Path(mesh_filename)
    if path.parent.is_dir():
        return path.parent
    parts = Path(mesh_filename.replace("\\", "/")).parts
    if "resources" not in parts:
        return None
    suffix = Path(*parts[parts.index("resources") :]).parent
    for root in roots:
        candidate = root / suffix
        if candidate.is_dir():
            return candidate
    return None


def default_asset_roots(recording_dir: Path, extra: list[Path] | None = None) -> list[Path]:
    """Explicit roots, then ``$TF_ASSET_ROOT``, then every ancestor of the recording, then the working directory."""
    roots = [Path(p) for p in (extra or [])]
    env = os.environ.get(ASSET_ROOT_ENV)
    if env:
        roots += [Path(p) for p in env.split(os.pathsep) if p]
    roots += list(recording_dir.resolve().parents) + [Path.cwd()]
    return roots


@dataclass
class Frame:
    key: str
    index: int
    time: float
    position: int  # recording order, matches textures/body_N/<kind>/NNNN.png on disk


class Recording:
    """Lazy reader over one ``FrictionTexture_*.h5`` file."""

    def __init__(
        self,
        source: Path,
        scale_overrides: dict[int, float] | None = None,
        asset_roots: list[Path] | None = None,
    ):
        ensure_windows_hdf5_locking()
        source = Path(source)
        if source.is_dir():
            files = sorted(source.glob("FrictionTexture_*.h5")) or sorted(source.glob("*.h5"))
            if not files:
                raise FileNotFoundError(f"no .h5 recording in {source}")
            source = files[-1]
        self.path = source
        self.root = source.parent
        self.file = h5py.File(source, "r")
        self._scale_overrides = scale_overrides or {}
        self.asset_roots = default_asset_roots(self.root, asset_roots)
        self.bodies = self._read_bodies()
        self.frames = self._read_frames()

    def body(self, index: int) -> Body:
        return next(b for b in self.bodies if b.index == index)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> Recording:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def scenario_name(self) -> str:
        scene = self.file.get("metadata/scene")
        name = scene.attrs.get("scenario_name", "") if scene is not None else ""
        return name.decode(errors="replace") if isinstance(name, bytes) else str(name)

    def _read_bodies(self) -> list[Body]:
        meta = self.file["metadata"]
        fixed = np.zeros(0, dtype=bool)
        if "scene/initial_conditions/fixed" in meta:
            fixed = np.asarray(meta["scene/initial_conditions/fixed"][()], dtype=bool)
        bodies = []
        group = meta.get("bodies")
        if group is None:
            return bodies
        for key in sorted(group.keys(), key=lambda k: int(k.split("_")[-1])):
            g = group[key]
            index = int(key.split("_")[-1])
            if "mesh/vertices" not in g:
                print(f"[render] body {index}: no embedded mesh, skipped")
                continue
            vertices = np.asarray(g["mesh/vertices"][()], dtype=np.float32)
            normals = (
                np.asarray(g["mesh/normals"][()], dtype=np.float32)
                if "mesh/normals" in g
                else np.tile(np.float32([0, 1, 0]), (len(vertices), 1))
            )
            uvs = (
                np.asarray(g["mesh/uvs"][()], dtype=np.float32)
                if "mesh/uvs" in g
                else np.zeros((len(vertices), 2), np.float32)
            )
            counts = np.asarray(g["mesh/face_vertex_counts"][()])
            if counts.size and not np.all(counts == 3):
                raise ValueError(f"body {index}: only triangle meshes are supported")
            faces = np.asarray(g["mesh/face_vertex_indices"][()], dtype=np.uint32).reshape(-1, 3)

            mtl_text = ""
            if "mtl/mtl_content" in g:
                raw = g["mtl/mtl_content"][()]
                mtl_text = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
            material_name = str(g.attrs.get("material_name", ""))
            materials = parse_mtl_text(mtl_text)
            mtl = materials.get(material_name) or (next(iter(materials.values())) if materials else {})

            scale_recorded = "scale" in g.attrs
            scale = float(g.attrs["scale"]) if scale_recorded else 1.0
            if index in self._scale_overrides:
                scale = self._scale_overrides[index]

            mesh_attrs = g["mesh"].attrs
            bodies.append(
                Body(
                    index=index,
                    name=str(mesh_attrs.get("mesh_name", key)),
                    vertices=vertices,
                    normals=normals,
                    uvs=uvs,
                    faces=faces,
                    scale=scale,
                    scale_recorded=scale_recorded,
                    fixed=bool(fixed[index]) if index < fixed.size else False,
                    material_name=material_name,
                    mtl=mtl,
                    mtl_dir=_resolve_asset_dir(str(mesh_attrs.get("mesh_filename", "")), self.asset_roots),
                    normal_variant_count=int(g.attrs.get("normal_variant_count", 0)),
                    _group=g,
                )
            )
        return bodies

    def _read_frames(self) -> list[Frame]:
        frames_group = self.file.get("frames")
        if frames_group is None:
            return []
        keys = sorted(frames_group.keys(), key=lambda k: int(k.split("_")[-1]))
        frames = []
        for position, key in enumerate(keys):
            g = frames_group[key]
            if "positions" not in g or "orientations" not in g:
                continue
            index = int(g.attrs.get("frame_index", key.split("_")[-1]))
            time = float(g["time"][()]) if "time" in g else float(index)
            frames.append(Frame(key=key, index=index, time=time, position=position))
        return frames

    def pose(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions [N,3], orientations [N,4] as (w, x, y, z))`` for every body."""
        g = self.file["frames"][frame.key]
        return (
            np.asarray(g["positions"][()], dtype=np.float64),
            np.asarray(g["orientations"][()], dtype=np.float64),
        )

    def texture(self, frame: Frame, body_index: int, kind: str = "wear") -> np.ndarray | None:
        """Per-frame 8-bit texture (wear/sliding) as float32 HxW in [0, 1], or None when absent.

        The value is the exported channel divided by 255; 1.0 is the exporter's clamp, not a physical depth.
        """
        g = self.file["frames"][frame.key]
        name = f"textures/body_{body_index}/{kind}"
        arr = None
        if name in g:
            arr = decode_png(g[name])
        else:
            disk = self.root / "textures" / f"body_{body_index}" / kind / f"{frame.position:04d}.png"
            if disk.is_file():
                with Image.open(disk) as img:
                    arr = np.asarray(img)
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32) / 255.0


def quaternion_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix for a (w, x, y, z) quaternion (Eigen storage order used by the serializer)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def body_world_transform(position: np.ndarray, q_wxyz: np.ndarray, scale: float) -> np.ndarray:
    """4x4 model matrix ``T(x) * R(q) * S(scale)``, matching ``RigidBodyRendererDetail`` and ``RigidBody``."""
    m = np.eye(4)
    m[:3, :3] = quaternion_to_matrix(q_wxyz) * scale
    m[:3, 3] = position
    return m
