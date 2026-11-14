"""Static displacement uses world units and image-row UV convention."""

from dataclasses import replace

import numpy as np
from PIL import Image

from mitsuba_video.geometry import displace_body
from mitsuba_video.sources.manifest import InlineMaterial, ManifestSource
from mitsuba_video.sources.manifest.cube_throw import write_cube_throw


def test_cube_throw_has_motion_without_wear(tmp_path):
    src = ManifestSource.open(write_cube_throw(tmp_path, frames=12))
    assert len(src.frames) == 12
    assert all(src.field(f, 0) is None for f in src.frames)
    positions = np.array([src.pose(f)[0][0] for f in src.frames])
    assert positions[-1, 0] > positions[0, 0]
    assert positions[5, 1] > positions[0, 1]
    src.close()


def test_displacement_world_scale_and_uv_flip(manifest_dir, tmp_path):
    # Image top row is V=1. One extreme displaces down and the other up.
    Image.fromarray(np.uint8([[255, 255], [0, 0]])).save(tmp_path / "height.png")
    src = ManifestSource.open(manifest_dir)
    body = replace(src.bodies[0], scale=2, material=InlineMaterial(
        {"base": {"height": "height.png"}}, tmp_path, "height"))
    displaced = displace_body(body, 0.2)
    world_delta = (displaced.vertices - body.vertices) * body.scale
    expected = np.where(body.uvs[:, 1] > 0.5, 0.1, -0.1)
    np.testing.assert_allclose(world_delta[:, 1], expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(displaced.normals, axis=1), 1, atol=1e-6)
    assert displace_body(body, 0) is body
    src.close()
