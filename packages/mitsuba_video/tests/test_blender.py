"""The Blender backends (mitsuba_video.blender): the package written for Blender and the pure-numpy half of the scene
script. Rendering itself needs Blender: set MITSUBA_VIDEO_TEST_BLENDER=1 to run the end-to-end EEVEE test."""

from __future__ import annotations

import json
import os
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from mitsuba_video import blender, blender_scene, video
from mitsuba_video.camera import CameraSettings, camera_path
from mitsuba_video.pipeline import (
    Look,
    RenderSettings,
    _signature,
    resolve_field,
    texture_sizes,
)
from mitsuba_video.source import open_source
from PIL import Image


def test_scene_script_tonemap_is_the_mitsuba_tonemap_byte_for_byte():
    rng = np.random.default_rng(0)
    hdr = np.concatenate([rng.uniform(0, 1, (64, 64, 3)), rng.uniform(0, 40, (64, 64, 3))]).astype(np.float32)
    for exposure in (0.0, -1.5, 2.0):
        assert np.array_equal(blender_scene.tonemap(hdr, exposure), video.tonemap(hdr.copy(), exposure))


def test_scene_script_png_round_trips():
    rgb = np.random.default_rng(1).integers(0, 256, (7, 13, 3), dtype=np.uint8)
    data = blender_scene.png_bytes(rgb)
    assert np.array_equal(np.asarray(Image.open(BytesIO(data)).convert("RGB")), rgb)
    assert zlib.decompress(data[data.index(b"IDAT") + 4:-16])  # one valid IDAT chunk


@pytest.mark.parametrize("up", [[0, 1, 0], [0, 0, 1]])
def test_camera_looks_down_its_local_minus_z(up):
    eye, target = np.array([3.0, 2.0, -1.0]), np.array([0.5, 0.0, 0.25])
    rot = blender_scene.look_rotation(eye, target, up)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)
    forward = (target - eye) / np.linalg.norm(target - eye)
    assert np.allclose(-rot[:, 2], forward)
    assert rot[:, 1] @ np.asarray(up, float) > 0  # the image's up is the world's up side


def test_up_basis_matches_the_camera_module():
    from mitsuba_video.camera import up_basis

    for axis in "xyz":
        for a, b in zip(blender_scene.up_basis(axis), up_basis(axis), strict=True):
            assert np.allclose(a, b)


def test_find_blender_prefers_the_explicit_path(tmp_path, monkeypatch):
    exe = tmp_path / ("blender.exe" if os.name == "nt" else "blender")
    exe.write_text("")
    monkeypatch.setenv("MITSUBA_VIDEO_BLENDER", str(tmp_path / "missing"))
    assert blender.find_blender(str(exe)) == exe
    assert blender.find_blender(str(tmp_path)) == exe  # a folder holding blender(.exe)
    with pytest.raises(FileNotFoundError):
        blender.find_blender(None)  # MITSUBA_VIDEO_BLENDER names a missing file: an error, not a silent fallback


def test_signature_names_the_blender_backend_only_when_used(manifest_dir):
    source = open_source(manifest_dir)
    try:
        spec = resolve_field(source, None)
        mitsuba = _signature(RenderSettings(source=manifest_dir), source, spec, source.frames)
        cycles = _signature(RenderSettings(source=manifest_dir, backend="cycles"), source, spec, source.frames)
    finally:
        source.close()
    assert "backend" not in mitsuba  # Mitsuba renders made before the option still resume
    assert cycles["backend"] == "cycles"
    assert RenderSettings(source=manifest_dir, backend="eevee").renderer_label == "Blender EEVEE"


def _package(manifest_dir: Path, out: Path, **settings) -> dict:
    s = RenderSettings(source=manifest_dir, backend="eevee", look=Look.worn, width=64, height=36, out=out,
                       camera=CameraSettings(up="y"), **settings)  # fmt: skip
    source = open_source(manifest_dir)
    try:
        spec = resolve_field(source, None)
        frames = source.frames
        sizes, _ = texture_sizes(s, source, spec, frames[0])
        poses = camera_path(source, frames, s.camera, s.width / s.height)
        path = blender.write_package(s, source, spec, frames, [0, len(frames) - 1], sizes, poses, "cpu", out,
                                     log=lambda _m: None, progress=False)  # fmt: skip
    finally:
        source.close()
    return json.loads(path.read_text(encoding="utf-8"))


def test_package_holds_meshes_poses_camera_and_texture_sequences(manifest_dir, tmp_path):
    pkg = _package(manifest_dir, tmp_path / "out")
    root = tmp_path / "out" / "blender"
    n = len(pkg["frames"])
    assert pkg["engine"] == "eevee" and pkg["render_frames"] == [0, n - 1]
    assert len(pkg["camera"]) == n
    for body in pkg["bodies"]:
        assert len(body["poses"]) == n and len(body["poses"][0]) == 7  # x y z w qx qy qz
        mesh = np.load(root / body["mesh"])
        assert mesh["faces"].max() < len(mesh["vertices"]) == len(mesh["uvs"])
        assert {"base_color", "roughness", "normalmap"} <= set(body["textures"])
        assert "height" not in body["textures"] and "displacement" not in body  # --displace off
        for rel in body["textures"].values():
            assert (root / rel).is_file()
        if body["sequence"]:  # one file per Blender frame (1-based), linked to the distinct textures
            for key in body["textures"]:
                files = sorted((root / "seq" / f"body_{body['index']}").glob(f"{key}_*.png"))
                assert [f.name for f in files] == [f"{key}_{i + 1:06d}.png" for i in range(n)]
    assert any(b["sequence"] for b in pkg["bodies"])  # the fixture's wear grows




@pytest.mark.skipif(os.environ.get("MITSUBA_VIDEO_TEST_BLENDER") != "1", reason="set MITSUBA_VIDEO_TEST_BLENDER=1 to render with Blender")
def test_eevee_renders_the_fixture_end_to_end(manifest_dir, tmp_path):
    import subprocess
    import sys

    out = tmp_path / "eevee"
    result = subprocess.run([sys.executable, "-m", "mitsuba_video", "render", str(manifest_dir),
                             "--backend", "eevee", "--res", "96x54", "--spp", "4", "--frames", "::4",
                             "--out", str(out), "--no-chart", "--no-encode"],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    beauty = sorted((out / "beauty").glob("frame_*.png"))
    assert beauty and all(np.asarray(Image.open(p)).std() > 1.0 for p in beauty)
    assert (out / "blender" / "scene.blend").is_file()
    config = json.loads((out / "render_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "done" and config["backend"] == "eevee"
