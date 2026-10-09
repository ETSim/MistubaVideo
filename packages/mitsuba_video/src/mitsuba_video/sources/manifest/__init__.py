"""Manifest source: a versioned JSON file any project can write (see ``schema/manifest-v1.schema.json``).

    manifest.json            bodies (OBJ meshes + .mtl or inline materials), fields, frames
    poses.npz                (frames_file) arrays index[F], time[F], position[F,N,3], orientation[F,N,4] (w,x,y,z)
    fields/<field>/body_<b>/<frame>.npy|png   (field_pattern) per-frame atlases, row 0 = V = 1

Use ``ManifestWriter`` (``writer.py``) to produce one from Python, and ``mitsuba-video validate`` to check it.
Bodies are indexed by ``id``; pose arrays are indexed the same way (``position[f, id]``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import ClassVar

import numpy as np

from ...blend import BlendParams
from ...maps import DEFAULT_ROUGHNESS, MapSet, NormalOptions, load_image
from ...model import Body, FieldSpec, Frame
from ...mtl import MapImages, MtlMaterial, build_mapset, parse_mtl_text
from ...obj import load_obj

FORMAT = "mitsuba-video.manifest"
MANIFEST_NAME = "manifest.json"
DEFAULT_PATTERN = "fields/{field}/body_{body}/{frame:06d}.npy"


@lru_cache(maxsize=1)
def schema() -> dict:
    text = resources.files(__package__).joinpath("schema/manifest-v1.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def validate(data: dict) -> list[str]:
    """Schema errors as readable strings (empty when valid)."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    ]


def _manifest_path(path: Path) -> Path:
    path = Path(path)
    return path / MANIFEST_NAME if path.is_dir() else path


class InlineMaterial:
    """MaterialProvider for ``{"base": {...}, "worn": {...}}`` with colours/values or image paths."""

    def __init__(self, spec: dict, base_dir: Path, label: str) -> None:
        self.spec, self.base_dir, self.label = spec, base_dir, label
        self.metallic = float(spec.get("base", {}).get("metallic", 0.0))

    def _images(self, appearance: dict) -> tuple[MapImages, np.ndarray, float]:
        imgs = MapImages()
        colour = np.ones(3, np.float32)
        albedo = appearance.get("albedo", [0.6, 0.6, 0.6])
        if isinstance(albedo, str):
            imgs.albedo = load_image(self.base_dir / albedo, "RGB")
        else:
            colour = np.asarray(albedo, np.float32)
        rough = appearance.get("roughness", DEFAULT_ROUGHNESS)
        if isinstance(rough, str):
            imgs.roughness = load_image(self.base_dir / rough, "L")
            rough = DEFAULT_ROUGHNESS
        if "normal" in appearance:
            imgs.normal = load_image(self.base_dir / appearance["normal"], "RGB")
        if "height" in appearance:
            imgs.height = load_image(self.base_dir / appearance["height"], "L")
        return imgs, colour, float(rough)

    def preferred_size(self) -> tuple[int, int] | None:
        imgs, _, _ = self._images(self.spec.get("base", {}))
        shapes = [img.shape[:2] for img in vars(imgs).values() if img is not None]
        return max(shapes) if shapes else None

    def build(
        self, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
    ) -> tuple[MapSet, MapSet]:
        base_spec = self.spec.get("base", {})
        worn_spec = {**base_spec, **self.spec.get("worn", {})}  # unspecified worn entries keep the base
        b_imgs, b_col, b_rough = self._images(base_spec)
        w_imgs, w_col, w_rough = self._images(worn_spec)
        return (
            build_mapset(b_imgs, b_col, b_rough, size, normals, f"{self.label} base", log),
            build_mapset(w_imgs, w_col, w_rough, size, normals, f"{self.label} worn", log),
        )


class ManifestSource:
    kind: ClassVar[str] = "manifest"

    def __init__(self, path: Path, data: dict) -> None:
        self.path = path
        self.root = path.parent
        self.data = data
        self.title = data.get("name", path.parent.name)
        self.up_axis = data.get("up_axis", "y")
        units = data.get("units", {})
        self.length_unit = units.get("length", "m")
        self.travel_label = data.get("travel_label", "travel")
        self.fields = {
            name: FieldSpec(
                name=name,
                label=spec.get("label", f"{name}, normalized"),
                threshold=float(spec.get("threshold", 1e-6)),
                video_prefix=spec.get("video_prefix", name),
                area_csv=f"{name}_area.csv",
                area_title=spec.get("area_title", f"covered area ({name})"),
            )
            for name, spec in data["fields"].items()
        }
        self._ranges = {name: tuple(spec.get("range", [0.0, 1.0])) for name, spec in data["fields"].items()}
        self.primary_field = data.get("primary_field", next(iter(self.fields)))
        self.capabilities = frozenset()
        blend = data.get("blend")
        self.blend = BlendParams(**blend) if blend else None
        self.pattern = data.get("field_pattern", DEFAULT_PATTERN)
        self.bodies = [self._body(b) for b in sorted(data["bodies"], key=lambda b: b["id"])]
        self._index = {b.index: i for i, b in enumerate(self.bodies)}
        self.frames, self._positions, self._orientations, self._inline_fields = self._frames()

    # --- construction ----------------------------------------------------------------------------------------
    def _body(self, spec: dict) -> Body:
        mesh_path = self.root / spec["mesh"]
        mesh = load_obj(mesh_path)
        label = spec.get("name", mesh_path.stem)
        material_spec = spec.get("material")
        if material_spec is None and mesh.mtllib:
            material_spec = {"mtl": str((mesh_path.parent / mesh.mtllib).relative_to(self.root))}
            if mesh.materials:
                material_spec["name"] = mesh.materials[0]
        if material_spec and "mtl" in material_spec:
            mtl_path = self.root / material_spec["mtl"]
            mats = parse_mtl_text(mtl_path.read_text(encoding="utf-8", errors="replace"))
            chosen = mats.get(material_spec.get("name", ""), next(iter(mats.values()), {}))
            material = MtlMaterial(chosen, mtl_path.parent, label=label)
        else:
            material = InlineMaterial(material_spec or {"base": {}}, self.root, label=label)
        return Body(
            index=int(spec["id"]),
            name=label,
            vertices=mesh.vertices,
            normals=mesh.normals,
            uvs=mesh.uvs,
            faces=mesh.faces,
            scale=float(spec.get("scale", 1.0)),
            fixed=bool(spec.get("fixed", False)),
            material=material,
            notes={"mesh": spec["mesh"]},
        )

    def _frames(self) -> tuple[list[Frame], np.ndarray, np.ndarray, list[dict]]:
        n = max(b.index for b in self.bodies) + 1
        if "frames_file" in self.data:
            arrays = np.load(self.root / self.data["frames_file"])
            times = np.asarray(arrays["time"], np.float64)
            index = np.asarray(arrays["index"], np.int64) if "index" in arrays else np.arange(len(times))
            positions = np.asarray(arrays["position"], np.float64)
            orientations = np.asarray(arrays["orientation"], np.float64)
            inline: list[dict] = [{} for _ in range(len(times))]
        else:
            entries = self.data["frames"]
            times = np.array([float(e["time"]) for e in entries])
            index = np.array([int(e.get("index", i)) for i, e in enumerate(entries)])
            positions = np.zeros((len(entries), n, 3))
            orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(entries), n, 1))
            for f, entry in enumerate(entries):
                if f > 0:  # bodies not listed keep their previous pose
                    positions[f], orientations[f] = positions[f - 1], orientations[f - 1]
                for body_id, pose in entry.get("poses", {}).items():
                    positions[f, int(body_id)] = pose["position"]
                    orientations[f, int(body_id)] = pose["orientation"]
            inline = [entry.get("fields", {}) for entry in entries]
        frames = [Frame(key=str(i), index=int(index[i]), time=float(times[i]), position=i) for i in range(len(times))]
        return frames, positions, orientations, inline

    # --- protocol --------------------------------------------------------------------------------------------
    @classmethod
    def probe(cls, path: Path) -> bool:
        target = _manifest_path(Path(path))
        if not target.is_file() or target.suffix.lower() != ".json":
            return False
        try:
            return json.loads(target.read_text(encoding="utf-8")).get("format") == FORMAT
        except (OSError, ValueError):
            return False

    @classmethod
    def open(cls, path: Path, **options: object) -> ManifestSource:
        if options:
            raise ValueError(f"the manifest source takes no options (got {', '.join(sorted(options))})")
        target = _manifest_path(Path(path))
        data = json.loads(target.read_text(encoding="utf-8"))
        errors = validate(data)
        if errors:
            raise ValueError(f"{target} is not a valid manifest:\n  " + "\n  ".join(errors[:10]))
        return cls(target, data)

    def pose(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        return self._positions[frame.position], self._orientations[frame.position]

    def field(self, frame: Frame, body_index: int, name: str | None = None) -> np.ndarray | None:
        name = name or self.primary_field
        rel = self._inline_fields[frame.position].get(name, {}).get(str(body_index))
        if rel is None:
            rel = self.pattern.format(field=name, body=body_index, frame=frame.index)
        path = self.root / rel
        if not path.is_file():
            return None
        lo, hi = self._ranges.get(name, (0.0, 1.0))
        raw = np.load(path).astype(np.float32) if path.suffix.lower() == ".npy" else _load_scalar_png(path)
        return np.clip((raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)

    def describe(self) -> list[tuple[str, str]]:
        return [("manifest", str(self.path)), ("name", self.title), ("fields", ", ".join(self.fields))]

    def close(self) -> None:
        pass


def _load_scalar_png(path: Path) -> np.ndarray:
    """8- or 16-bit grey PNG (or the first channel of RGB/RGBA) -> float in [0, 1]."""
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    scale = 65535.0 if arr.dtype == np.uint16 or arr.max(initial=0) > 255 else 255.0
    return arr.astype(np.float32) / scale
