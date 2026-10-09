"""Convert any source into a manifest (e.g. a TextureFriction recording, to hand to people without h5py).

Materials are baked to images at ``material_size`` (base/worn albedo, roughness, normal, height), fields to 8-bit
PNG. Poses and fields round-trip exactly for 8-bit sources; baked materials are 8-bit quantised.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...maps import MapSet, NormalOptions, fit_size
from ...source import Source
from .writer import ManifestWriter


def _bake(writer: ManifestWriter, prefix: str, maps: MapSet) -> dict:
    out = {
        "albedo": writer.write_image(f"{prefix}_albedo.png", maps.albedo),
        "roughness": writer.write_image(f"{prefix}_roughness.png", maps.roughness),
        "normal": writer.write_image(f"{prefix}_normal.png", maps.normal * 0.5 + 0.5),
    }
    if maps.height is not None:
        out["height"] = writer.write_image(f"{prefix}_height.png", maps.height)
    return out


def export_manifest(
    source: Source,
    root: Path,
    material_size: int = 1024,
    fields: list[str] | None = None,
    log: Callable[[str], None] | None = None,
    progress: bool = True,
) -> Path:
    names = fields or list(source.fields)
    writer = ManifestWriter(
        Path(root),
        name=source.title,
        up_axis=source.up_axis,
        length_unit=getattr(source, "length_unit", "m"),
        travel_label=getattr(source, "travel_label", None),
    )
    for name in names:
        spec = source.fields[name]
        writer.add_field(name, label=spec.label, threshold=spec.threshold, video_prefix=spec.video_prefix,
                         area_title=spec.area_title)  # fmt: skip
    writer.data["primary_field"] = source.primary_field if source.primary_field in names else names[0]
    for body in source.bodies:
        size = fit_size(body.material.preferred_size(), material_size)
        base, worn = body.material.build(size, NormalOptions(recenter_above_deg=0.0), log)
        material = {"base": _bake(writer, f"materials/body_{body.index}_base", base),
                    "worn": _bake(writer, f"materials/body_{body.index}_worn", worn)}  # fmt: skip
        material["base"]["metallic"] = float(body.material.metallic)
        writer.add_body(body.index, body.name, body.vertices, body.normals, body.uvs, body.faces,
                        scale=body.scale, fixed=body.fixed, material=material)  # fmt: skip
    for frame in tqdm(source.frames, desc="export", unit="frame", disable=not progress):
        pos, quat = source.pose(frame)
        values = {name: {} for name in names}
        for name in names:
            for body in source.bodies:
                value = source.field(frame, body.index, name)
                if value is not None:
                    values[name][body.index] = value
        writer.add_frame(frame.time, np.asarray(pos), np.asarray(quat), fields=values, index=frame.index)
    return writer.close()
