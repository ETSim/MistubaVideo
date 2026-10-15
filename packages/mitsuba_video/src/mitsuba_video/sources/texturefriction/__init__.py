"""TextureFriction source: a ``--record`` run (``simulation_<ts>/`` folder or ``FrictionTexture_*.h5``).

Fields: ``wear`` (primary) and ``sliding``, the per-frame 8-bit atlas PNGs the serializer writes, divided by 255
(1.0 is the exporter clamp, not a physical depth). Options (``-O`` on the CLI, keyword arguments in Python):

* ``scale``: body scale overrides for recordings made before scale was exported, ``"1=0.5,2=0.3"`` or a dict.
* ``assets``: extra roots holding ``resources/`` for the ``.mtl`` textures (``os.pathsep``-separated or a list).

Materials blend like the viewer (``blend.VIEWER_BLEND``), with the settings the viewer had when the run was recorded
(``metadata/scene/variant_blend``); older recordings get the viewer defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace
from typing import ClassVar

import numpy as np

from ...blend import VIEWER_BLEND, BlendParams
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


# metadata/scene/variant_blend attribute -> BlendParams field
_RECORDED_BLEND = {
    "normal_wear_edge0": "edge0",
    "normal_wear_edge1": "edge1",
    "hermite_tension": "tension",
    "hermite_bias": "bias",
    "height_blend_strength": "height_strength",
    "height_blend_contrast": "height_contrast",
    "wear_map_blur_px": "wear_blur_px",
}


def recorded_blend(values: dict[str, float] | None) -> BlendParams:
    """``VIEWER_BLEND`` with the recorded viewer settings substituted."""
    if not values:
        return VIEWER_BLEND
    return replace(VIEWER_BLEND, **{f: values[k] for k, f in _RECORDED_BLEND.items() if k in values})


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
        recorded = recording.variant_blend
        self.blend = recorded_blend(recorded)
        blend_enabled = not recorded or recorded.get("enabled", 1.0) != 0.0
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
                    material=TextureFrictionMaterial(rb, blend_enabled),
                    display_name=rb.name.split("_Material")[0],
                    notes={
                        "scale": f"{rb.scale:g}" + ("" if rb.scale_recorded else "?"),
                        "atlas": str(len(rb.variant_info)),
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

    def direction(self, frame: Frame, body_index: int) -> np.ndarray | None:
        """Slide direction per atlas texel, (H, W, 2) in image axes (+col, +row), zero where nothing slid.

        The serializer exports the direction atlas as a flow map (ImageUtils::directionImageToFlowMapHsv): hue =
        atan2(dz, dx) / 2 pi + 0.5 of the world XZ slide direction, full saturation, black where nothing was written.
        It is turned into the body frame with the frame's orientation and mapped onto the atlas with the UV Jacobian
        of the body's upward-facing triangles (a flat contact face: the ground tiles' tops)."""
        rgb = self.recording.texture_rgb(frame, body_index, "direction")
        if rgb is None:
            return None
        mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
        written = (mx > 0.2) & (mx - mn > 0.1)
        if not written.any():
            return None
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        delta = np.maximum(mx - mn, 1e-6)
        hue = np.where(mx == r, ((g - b) / delta) % 6.0, np.where(mx == g, (b - r) / delta + 2.0, (r - g) / delta + 4.0))
        angle = (hue / 6.0 - 0.5) * 2.0 * np.pi
        d = np.stack([np.cos(angle), np.sin(angle)], axis=-1)  # world (x, z)
        jac = self._uv_jacobian(body_index)
        if jac is None:
            return None
        _, orientations = self.recording.pose(frame)
        w, x, y, z = (float(c) for c in orientations[body_index])
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip
        world = np.stack([d[..., 0], np.zeros_like(d[..., 0]), d[..., 1]], axis=-1)
        local = world @ rot  # R^T applied to row vectors
        img = local[..., [0, 2]] @ jac  # local (x, z) -> (col, row)
        norm = np.linalg.norm(img, axis=-1, keepdims=True)
        out = np.where(written[..., None] & (norm > 1e-6), img / np.maximum(norm, 1e-6), 0.0)
        return out.astype(np.float32)

    def _uv_jacobian(self, body_index: int) -> np.ndarray | None:
        """2x2 map from a body-local (x, z) step to an atlas (col, row) step, fitted on the upward-facing triangles."""
        cache = self.__dict__.setdefault("_jacobians", {})
        if body_index in cache:
            return cache[body_index]
        rb = self.recording.body(body_index)
        tri = rb.faces.astype(np.intp)
        p = rb.vertices[tri]  # (M, 3, 3)
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        up = n[:, 1] > 0.9 * np.linalg.norm(n, axis=1)
        jac = None
        if up.any():
            uv = rb.uvs[tri[up]]
            pos = p[up][:, :, [0, 2]]
            e_pos = np.concatenate([pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]])
            e_uv = np.concatenate([uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]])
            a, *_ = np.linalg.lstsq(e_pos, e_uv, rcond=None)  # e_pos @ a = e_uv
            jac = a * np.array([1.0, -1.0])  # (u, v) -> (col, row): rows run down, v up
        cache[body_index] = jac
        return jac

    def describe(self) -> list[tuple[str, str]]:
        rows = [("recording", str(self.path)), ("scenario", self.title)]
        if self.recording.variant_blend is None:
            rows.append(("blend", "viewer defaults (recording predates metadata/scene/variant_blend)"))
        elif self.recording.variant_blend.get("enabled", 1.0) == 0.0:
            rows.append(("blend", "off in the viewer when recorded: unworn materials only"))
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
