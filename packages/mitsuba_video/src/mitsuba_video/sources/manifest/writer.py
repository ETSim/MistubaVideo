"""Write a manifest from any project (simulator, experiment logger, post-processing script).

    from mitsuba_video.sources.manifest.writer import ManifestWriter

    w = ManifestWriter("out/run", name="pin on disc")
    w.add_field("wear", label="wear depth, normalized", threshold=0.002)
    w.add_body(0, "disc", vertices, normals, uvs, faces, fixed=True, material={"base": {"albedo": [0.5, 0.5, 0.55]}})
    w.add_body(1, "pin", mesh="pin.obj", material={"mtl": "pin.mtl"})
    for i, (t, pos, quat, wear0) in enumerate(steps):          # pos [N,3], quat [N,4] (w, x, y, z)
        w.add_frame(t, pos, quat, fields={"wear": {0: wear0}})  # HxW float in [0, 1], row 0 = V = 1
    w.close()                                                   # writes poses.npz + manifest.json

Fields are stored as 8-bit PNG (``field_format="png"``, compact, exact for 8-bit data) or float32 ``.npy``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ...obj import write_obj
from . import FORMAT, MANIFEST_NAME, validate


class ManifestWriter:
    def __init__(
        self,
        root: Path,
        name: str = "",
        up_axis: str = "y",
        length_unit: str = "m",
        field_format: str = "png",
        travel_label: str | None = None,
    ) -> None:
        if field_format not in ("png", "npy"):
            raise ValueError("field_format is 'png' or 'npy'")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.field_format = field_format
        self.data: dict = {
            "format": FORMAT,
            "version": 1,
            "name": name or self.root.name,
            "units": {"length": length_unit, "time": "s"},
            "up_axis": up_axis,
            "fields": {},
            "bodies": [],
            "frames_file": "poses.npz",
            "field_pattern": "fields/{field}/body_{body}/{frame:06d}." + field_format,
        }
        if travel_label:
            self.data["travel_label"] = travel_label
        self._times: list[float] = []
        self._index: list[int] = []
        self._positions: list[np.ndarray] = []
        self._orientations: list[np.ndarray] = []

    def add_field(self, name: str, label: str | None = None, threshold: float = 0.5 / 255.0, **extra: object) -> None:
        self.data["fields"][name] = {"label": label or f"{name}, normalized", "threshold": threshold, **extra}
        self.data.setdefault("primary_field", name)

    def add_body(
        self,
        body_id: int,
        name: str,
        vertices: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        *,
        mesh: str | Path | None = None,
        material: dict | None = None,
        scale: float = 1.0,
        fixed: bool = False,
    ) -> None:
        """Either pass arrays (written to ``meshes/<name>.obj``) or ``mesh`` (copied next to the manifest)."""
        if mesh is None:
            if vertices is None or faces is None:
                raise ValueError("add_body needs mesh=... or vertices/faces arrays")
            n = len(vertices)
            rel = f"meshes/body_{body_id}.obj"
            write_obj(
                self.root / rel,
                vertices,
                normals if normals is not None else np.tile(np.float32([0, 1, 0]), (n, 1)),
                uvs if uvs is not None else np.zeros((n, 2), np.float32),
                faces,
            )
        else:
            src = Path(mesh)
            rel = f"meshes/{src.name}"
            if src.resolve() != (self.root / rel).resolve():
                (self.root / "meshes").mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, self.root / rel)
        entry = {"id": int(body_id), "name": name, "mesh": rel, "scale": float(scale), "fixed": bool(fixed)}
        if material is not None:
            entry["material"] = material
        self.data["bodies"].append(entry)

    def write_image(self, rel: str, image: np.ndarray) -> str:
        """Save a material image (uint8 or float in [0, 1]) under the manifest; returns its relative path."""
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = image if image.dtype == np.uint8 else np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
        return rel

    def add_frame(
        self,
        time: float,
        positions: np.ndarray,
        orientations: np.ndarray,
        fields: dict[str, dict[int, np.ndarray]] | None = None,
        index: int | None = None,
    ) -> int:
        frame_index = len(self._times) if index is None else int(index)
        self._times.append(float(time))
        self._index.append(frame_index)
        self._positions.append(np.asarray(positions, np.float64))
        self._orientations.append(np.asarray(orientations, np.float64))
        for name, per_body in (fields or {}).items():
            if name not in self.data["fields"]:
                self.add_field(name)
            for body, value in per_body.items():
                rel = self.data["field_pattern"].format(field=name, body=body, frame=frame_index)
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                v = np.clip(np.asarray(value, np.float32), 0.0, 1.0)
                if self.field_format == "npy":
                    np.save(path, v)
                else:
                    Image.fromarray((v * 255.0 + 0.5).astype(np.uint8)).save(path)
        return frame_index

    def close(self) -> Path:
        np.savez_compressed(
            self.root / "poses.npz",
            index=np.asarray(self._index, np.int64),
            time=np.asarray(self._times, np.float64),
            position=np.stack(self._positions) if self._positions else np.zeros((0, 0, 3)),
            orientation=np.stack(self._orientations) if self._orientations else np.zeros((0, 0, 4)),
        )
        errors = validate(self.data)
        if errors:
            raise ValueError("manifest does not validate:\n  " + "\n  ".join(errors))
        path = self.root / MANIFEST_NAME
        path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return path
