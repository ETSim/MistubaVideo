"""Read a TextureFriction ``--record`` HDF5 file for offline rendering.

Only what an offline renderer needs is exposed: each body's local mesh and uniform scale, its material atlas
variants (flat and worn), and per-frame poses plus the wear texture.

Image convention: every image returned here has row 0 at the top of the texture (V = 1), the same convention as
the PNG files referenced by the scenario ``.mtl`` files and the per-frame wear PNGs. The ``material_atlas``
images stored in the HDF5 were run through ``imageFlippedForOpenGL`` by ``OBJLoader`` (row 0 = V = 0), so they are
flipped back on read.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from ...model import Frame
from ...mtl import mtl_get, parse_mtl_text

Image.MAX_IMAGE_PIXELS = None

ASSET_ROOT_ENV = "TF_ASSET_ROOT"


def ensure_windows_hdf5_locking() -> None:
    """Disable HDF5 file locking on Windows so a read-only open works while the simulator holds the file."""
    if os.name == "nt":
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _png_bytes(dataset) -> bytes:
    blob = dataset[()]
    return blob.tobytes() if hasattr(blob, "tobytes") else bytes(blob)


def _png_digest(raw: bytes) -> bytes:
    return hashlib.blake2b(raw, digest_size=16).digest()


def decode_png(dataset) -> np.ndarray:
    """Decode a PNG byte dataset (uint8 1-D) into an HxW or HxWxC uint8 array."""
    with Image.open(io.BytesIO(_png_bytes(dataset))) as img:
        return np.asarray(img)


# Bodies that share a material store byte-identical material_atlas PNGs (twenty cubes, four materials), and a render
# opens the recording more than once, so atlas decodes are shared by content hash. Entries are read-only.
_ATLAS_IMAGES: dict[tuple[bytes, bool], np.ndarray] = {}


def atlas_image(dataset, colour: bool) -> np.ndarray:
    """A ``material_atlas`` PNG as uint8 with row 0 = V = 1: HxWx3 when ``colour`` (normal, albedo) and the PNG has
    channels, else HxW (first channel)."""
    raw = _png_bytes(dataset)
    key = (_png_digest(raw), colour)
    cached = _ATLAS_IMAGES.get(key)
    if cached is None:
        with Image.open(io.BytesIO(raw)) as img:
            arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[..., :3] if colour else arr[..., 0]
        cached = np.ascontiguousarray(arr[::-1])  # undo imageFlippedForOpenGL
        cached.setflags(write=False)
        _ATLAS_IMAGES[key] = cached
    return cached


@dataclass
class MaterialVariant:
    """One entry of a body's ``material_atlas`` (index 0 = unworn base)."""

    name: str
    normal: np.ndarray | None  # HxWx3 uint8, tangent-space normal map
    roughness: np.ndarray | None  # HxW uint8
    height: np.ndarray | None  # HxW uint8
    roughness_value: float
    metallic: float
    albedo: np.ndarray | None = None  # HxWx3 uint8 sRGB; recorded since the serializer exports it


@dataclass(frozen=True)
class VariantInfo:
    """A ``material_atlas`` entry's scalar attributes, readable without decoding its images."""

    name: str
    roughness_value: float
    metallic: float


@dataclass
class RecordedBody:
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
    def variant_info(self) -> list[VariantInfo]:
        """Names, roughness and metallic of the atlas variants (cheap: no image decode)."""
        if "material_atlas" not in self._group:
            return []
        atlas = self._group["material_atlas"]
        return [
            VariantInfo(name=str(atlas[key].attrs.get("name", key)),
                        roughness_value=float(atlas[key].attrs.get("roughness", 0.0)),
                        metallic=float(atlas[key].attrs.get("metallic", 0.0)))  # fmt: skip
            for key in sorted(atlas.keys())
        ]

    @cached_property
    def material_key(self) -> bytes:
        """Content hash of everything the material is built from (atlas PNGs, .mtl entries and folder), so bodies
        that share a material can share its decoded, resized stack."""
        h = hashlib.blake2b(digest_size=16)
        h.update(repr((self.material_name, sorted((k, str(v)) for k, v in self.mtl.items()), str(self.mtl_dir),
                       self.normal_variant_count)).encode())  # fmt: skip
        if "material_atlas" in self._group:
            atlas = self._group["material_atlas"]
            for key in sorted(atlas.keys()):
                for name in sorted(atlas[key].keys()):
                    h.update(f"{key}/{name}".encode())
                    h.update(_png_digest(_png_bytes(atlas[key][name])))
        return h.digest()

    @cached_property
    def variants(self) -> list[MaterialVariant]:
        """The atlas variants with their images (decoded on first use, shared between bodies with one material)."""
        if "material_atlas" not in self._group:
            return []
        out = []
        atlas = self._group["material_atlas"]
        for key, info in zip(sorted(atlas.keys()), self.variant_info, strict=True):
            g = atlas[key]

            def img(name: str) -> np.ndarray | None:
                return atlas_image(g[name], colour=name in ("normal", "albedo")) if name in g else None

            out.append(
                MaterialVariant(
                    name=info.name,
                    normal=img("normal"),
                    roughness=img("roughness"),
                    height=img("height"),
                    albedo=img("albedo"),
                    roughness_value=info.roughness_value,
                    metallic=info.metallic,
                )
            )
        return out

    def resolve_texture(self, key: str) -> Path | None:
        """Resolve an ``.mtl`` ``map_*`` entry to an existing file, or None."""
        # OBJLoader matches map_* keys case-sensitively but the assets mix spellings (map_Bump_Worn, map_Pr_worn).
        rel = mtl_get(self.mtl, key)
        if not isinstance(rel, str) or self.mtl_dir is None:
            return None
        path = (self.mtl_dir / rel.replace("\\", "/")).resolve()
        return path if path.is_file() else None


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
        self.skipped: list[str] = []
        self._last_texture: dict[tuple[int, str], tuple[bytes, np.ndarray]] = {}
        self.bodies = self._read_bodies()
        self.frames = self._read_frames()
        self.variant_blend = self._read_variant_blend()

    def body(self, index: int) -> RecordedBody:
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

    def _read_variant_blend(self) -> dict[str, float] | None:
        """``metadata/scene/variant_blend``: the viewer's blend settings at record time (None before they were)."""
        group = self.file.get("metadata/scene/variant_blend")
        if group is None:
            return None
        return {key: float(value) for key, value in group.attrs.items()}

    def _read_bodies(self) -> list[RecordedBody]:
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
                self.skipped.append(f"body {index}: no embedded mesh")
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
                RecordedBody(
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
            # Most bodies' wear does not change from one frame to the next: reuse the last decode when the PNG
            # bytes are identical (the returned array is read-only and shared).
            raw = _png_bytes(g[name])
            digest = _png_digest(raw)
            last = self._last_texture.get((body_index, kind))
            if last is not None and last[0] == digest:
                return last[1]
            with Image.open(io.BytesIO(raw)) as img:
                arr = np.asarray(img)
            arr = (arr[..., 0] if arr.ndim == 3 else arr).astype(np.float32) / 255.0
            arr.setflags(write=False)
            self._last_texture[(body_index, kind)] = (digest, arr)
            return arr
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
