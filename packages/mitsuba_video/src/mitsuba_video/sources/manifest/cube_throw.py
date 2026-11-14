"""A rotating cube following a prescribed ballistic arc; no wear data or external images."""

import math
from pathlib import Path

import numpy as np

from .fixture import _cube
from .writer import ManifestWriter


def write_cube_throw(root: Path, frames: int = 24) -> Path:
    if frames < 1:
        raise ValueError("frames must be positive")
    writer = ManifestWriter(root, name="Cube toss", length_unit="m")
    writer.add_body(0, "cube", *_cube(0.22), material={
        "base": {"albedo": [0.035, 0.24, 0.48], "roughness": 0.3, "metallic": 0.15}})
    axis = np.array([1.0, 0.6, 0.3])
    axis /= np.linalg.norm(axis)
    for t in np.linspace(0, 0.6, frames):
        position = [-1.2 + 4 * t, 0.55 + 2.943 * t - 4.905 * t * t, 0]
        angle = 4 * t
        quaternion = [math.cos(angle / 2), *(axis * math.sin(angle / 2))]
        writer.add_frame(float(t), [position], [quaternion])
    return writer.close()
