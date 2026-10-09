"""TextureFriction source: a ``--record`` run (``simulation_<ts>/`` folder or ``FrictionTexture_*.h5``).

Fields: ``wear`` (primary) and ``sliding``, the per-frame 8-bit atlas PNGs the serializer writes, divided by 255
(1.0 is the exporter clamp, not a physical depth). Options (``-O`` on the CLI, keyword arguments in Python):

* ``scale``: body scale overrides for recordings made before scale was exported, ``"1=0.5,2=0.3"`` or a dict.
* ``assets``: extra roots holding ``resources/`` for the ``.mtl`` textures (``os.pathsep``-separated or a list).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np

from ...model import Body, FieldSpec, Frame
from .materials import TextureFrictionMaterial
from .recording import Recording

WEAR = FieldSpec(
    name="wear",
    label="wear, normalized (1.0 = clamp)",
    threshold=0.5 / 255.0,
    video_prefix="wear",
    area_csv="worn_area.csv",
    area_title="worn area",
)
SLIDING = FieldSpec(
    name="sliding",
    label="sliding distance, normalized",
    threshold=0.5 / 255.0,
    video_prefix="sliding",
    area_csv="slid_area.csv",
    area_title="slid area",
)


def _scales(value: object) -> dict[int, float]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    out = {}
    for item in str(value).split(","):
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"scale override {item!r} is not BODY=SCALE")
        out[int(key.strip().replace("body_", ""))] = float(val)
    return out


def _roots(value: object) -> list[Path]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [Path(v) for v in value]
    return [Path(v) for v in str(value).split(os.pathsep) if v]


class TextureFrictionSource:
    kind: ClassVar[str] = "texturefriction"
    travel_label: ClassVar[str] = "slid"  # header: "body 1 slid 3.20 m"
    length_unit: ClassVar[str] = "m"

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.path = recording.path
        self.root = recording.root
        self.title = recording.scenario_name or recording.path.parent.name
        self.up_axis = "y"
        self.fields = {"wear": WEAR, "sliding": SLIDING}
        self.primary_field = "wear"
        self.capabilities = frozenset({"reset_detection"})
        self.frames: list[Frame] = recording.frames
        self.bodies: list[Body] = []
        for rb in recording.bodies:
            self.bodies.append(
                Body(
                    index=rb.index,
                    name=rb.name,
                    vertices=rb.vertices,
                    normals=rb.normals,
                    uvs=rb.uvs,
                    faces=rb.faces,
                    scale=rb.scale,
                    fixed=rb.fixed,
                    material=TextureFrictionMaterial(rb),
                    display_name=rb.name.split("_Material")[0],
                    notes={
                        "scale": f"{rb.scale:g}" + ("" if rb.scale_recorded else "?"),
                        "atlas": str(len(rb.variants)),
                        "mtl tex": "yes" if rb.mtl_dir else "no",
                    },
                )
            )

    @classmethod
    def probe(cls, path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() in (".h5", ".hdf5")
        return path.is_dir() and any(path.glob("FrictionTexture_*.h5"))

    @classmethod
    def open(cls, path: Path, **options: object) -> TextureFrictionSource:
        unknown = set(options) - {"scale", "assets"}
        if unknown:
            raise ValueError(f"texturefriction source options are scale, assets (got {', '.join(sorted(unknown))})")
        rec = Recording(path, scale_overrides=_scales(options.get("scale")), asset_roots=_roots(options.get("assets")))
        return cls(rec)

    def pose(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        return self.recording.pose(frame)

    def field(self, frame: Frame, body_index: int, name: str | None = None) -> np.ndarray | None:
        return self.recording.texture(frame, body_index, kind=name or self.primary_field)

    def describe(self) -> list[tuple[str, str]]:
        rows = [("recording", str(self.path)), ("scenario", self.title)]
        if any(not rb.scale_recorded for rb in self.recording.bodies):
            rows.append(("note", "scale not recorded (marked ?): pass -O scale=BODY=SCALE if a body was scaled"))
        rows += [("skipped", s) for s in self.recording.skipped]
        return rows

    def close(self) -> None:
        self.recording.close()

    def __enter__(self) -> TextureFrictionSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
