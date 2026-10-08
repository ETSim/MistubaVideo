"""Tests for ``tf_render`` (the ``tf-render`` Mitsuba wear renderer).

Run from ``tools/mitsuba_render``: ``python -m pytest``. Render/encode smoke tests skip without Mitsuba or ffmpeg.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from tf_render import wear_blend as wb  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from tf_render import pipeline  # noqa: E402
from tf_render.camera import CameraSettings, camera_path  # noqa: E402
from tf_render.cli import app  # noqa: E402
from tf_render.materials import build_mapsets  # noqa: E402
from tf_render.recording import Recording, body_world_transform, parse_mtl_text, quaternion_to_matrix  # noqa: E402
from tf_render.synthetic import TEX, write_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    path = write_synthetic(tmp_path_factory.mktemp("rec") / "FrictionTexture_synthetic.h5")
    rec = Recording(path.parent)
    yield rec
    rec.close()


def test_quaternion_is_wxyz():
    # 90 degrees about +y, stored (w, x, y, z): +x maps to -z.
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    np.testing.assert_allclose(quaternion_to_matrix(q) @ [1, 0, 0], [0, 0, -1], atol=1e-12)


def test_model_matrix_is_translate_rotate_scale():
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    m = body_world_transform(np.array([1.0, 2.0, 3.0]), q, 2.0)
    np.testing.assert_allclose(m @ [1, 0, 0, 1], [1, 2, 1, 1], atol=1e-12)


def test_hermite_remap_matches_cpp_reference():
    # tension 0.65, bias 0 -> m0 = m1 = 0.175; values hand-evaluated from WearAtlasBake::hermite01.
    p = wb.BlendParams()
    assert wb.wear_remap_t(np.float32(0.0), p) == 0.0
    assert wb.wear_remap_t(np.float32(1.0), p) == 1.0
    mid = float(wb.wear_remap_t(np.float32(0.5), p))
    assert mid == pytest.approx(0.5, abs=1e-6)  # symmetric tangents -> midpoint stays put
    m0, m1 = wb.hermite_tangents(0.65, 0.0)
    assert (m0, m1) == pytest.approx((0.175, 0.175))


def test_blend_endpoints():
    rng = np.random.default_rng(1)
    shape = (8, 8)

    def mapset(seed):
        r = np.random.default_rng(seed)
        n = r.normal(size=shape + (3,))
        n[..., 2] = np.abs(n[..., 2]) + 1.0
        return wb.MapSet(
            albedo=r.random(shape + (3,)).astype(np.float32),
            roughness=r.random(shape).astype(np.float32),
            normal=(n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32),
            height=None,
        )

    base, worn = mapset(2), mapset(3)
    unworn = wb.blend_maps(np.zeros(shape, np.float32), base, worn)
    np.testing.assert_array_equal(unworn.albedo, base.albedo)
    np.testing.assert_array_equal(unworn.normal, base.normal)
    full = wb.blend_maps(np.ones(shape, np.float32), base, worn)
    np.testing.assert_allclose(full.albedo, worn.albedo, atol=1e-6)
    np.testing.assert_allclose(full.roughness, worn.roughness, atol=1e-6)
    np.testing.assert_allclose(full.normal, worn.normal, atol=1e-5)
    half = wb.blend_maps(np.full(shape, 0.5, np.float32), base, worn)
    np.testing.assert_allclose(np.linalg.norm(half.normal, axis=-1), 1.0, atol=1e-5)
    del rng


def test_heatmap_only_touches_worn_texels():
    albedo = np.full((4, 4, 3), 0.5, np.float32)
    wear = np.zeros((4, 4), np.float32)
    wear[1, 2] = 1.0
    out = wb.heatmap_albedo(albedo, wear)
    assert np.array_equal(out[0, 0], albedo[0, 0])
    assert not np.allclose(out[1, 2], albedo[1, 2])


def test_parse_mtl_text_takes_last_token_for_maps():
    mtl = parse_mtl_text("newmtl A\nKd 0.1 0.2 0.3\nmap_Bump -bm 0.5 tex/n.png\nPr 0.4\n")
    assert mtl["A"]["Kd"] == [0.1, 0.2, 0.3]
    assert mtl["A"]["map_Bump"] == "tex/n.png"
    assert mtl["A"]["Pr"] == 0.4


def test_recording_reader(recording):
    assert [b.index for b in recording.bodies] == [0, 1]
    plane, box = recording.bodies
    assert plane.fixed and not box.fixed
    assert box.scale == pytest.approx(1.5) and box.scale_recorded
    assert len(recording.frames) == 6
    assert box.worn_variant_index == 1
    # Atlas images are flipped back from GL order: the synthetic worn normal varies along x only.
    assert box.variants[1].normal.shape == (TEX, TEX, 3)
    pos, quat = recording.pose(recording.frames[-1])
    assert pos.shape == (2, 3) and quat.shape == (2, 4)


def test_wear_texture_orientation(recording):
    # The box slides at z = +0.2 (plane v = z + 0.5 = 0.7). Image convention puts V = 1 at row 0, so the footprint
    # must sit in the upper half of the texture; a V flip would land it in the lower half.
    wear = recording.texture(recording.frames[-1], 0)
    rows = np.nonzero(wear.max(axis=1) > 0)[0]
    assert rows.max() < TEX // 2
    assert wear.max() <= 1.0


def test_mapsets_and_camera(recording):
    box = recording.bodies[1]
    base, worn = build_mapsets(box, (32, 32))
    assert base.albedo.shape == (32, 32, 3) and worn.normal.shape == (32, 32, 3)
    # Kd tint applied (box Kd is orange-ish, so red > blue).
    assert base.albedo[..., 0].mean() > base.albedo[..., 2].mean()
    for mode in ("fixed", "track", "orbit"):
        poses = camera_path(recording, recording.frames, CameraSettings(mode=mode), 16 / 9)
        assert len(poses) == len(recording.frames)
        assert all(np.linalg.norm(p.eye - p.target) > 0 for p in poses)


def test_area_per_texel(recording):
    plane, box = recording.bodies
    # Synthetic plane is 2 x 1 with UVs covering [0,1]^2 once.
    assert plane.area_per_texel((TEX, TEX)) == pytest.approx(2.0 / (TEX * TEX))
    # Each cube face (0.2 x 0.2, scaled 1.5) maps to the full UV square, so a texel covers one face's share.
    assert box.area_per_texel((TEX, TEX)) == pytest.approx((0.2 * 1.5) ** 2 / (TEX * TEX))


def test_chart_and_title_card_sizes():
    from tf_render import video

    series = video.AreaSeries("worn area (m²)", np.linspace(0, 2.4, 25), np.linspace(0, 0.5, 25) ** 0.5, "m²")
    for index in (0, 12, 24):
        assert video.area_chart(series, index, 300, 160).size == (300, 160)
    assert video.title_card((640, 360), "Title", ["line"]).size == (640, 360)


def test_parse_helpers():
    assert list(pipeline.parse_frames("::2", 6)) == [0, 2, 4]
    assert list(pipeline.parse_frames("-2:", 6)) == [4, 5]
    assert pipeline.parse_scales(["1=0.5", "body_2=3"]) == {1: 0.5, 2: 3.0}
    with pytest.raises(ValueError):
        pipeline.parse_scales(["0.5"])


def test_travel_counts_resets(recording):
    mover, travel = pipeline.travel_by_frame(recording)
    assert mover is not None and mover.index == 1
    slid, passes = travel[recording.frames[-1].key]
    assert passes == 1  # the synthetic box never teleports
    assert slid == pytest.approx(1.2, rel=1e-3)  # x from -0.6 to 0.6


def test_resume_refuses_different_settings(tmp_path):
    (tmp_path / pipeline.CONFIG_NAME).write_text(json.dumps({"signature": {"spp": 64, "look": "heatmap"}}))
    pipeline._check_resume(tmp_path, {"spp": 64, "look": "heatmap"})
    with pytest.raises(RuntimeError, match="spp"):
        pipeline._check_resume(tmp_path, {"spp": 128, "look": "heatmap"})


def test_cli_inspect_and_fixture(tmp_path):
    runner = CliRunner()
    h5 = tmp_path / "rec" / "FrictionTexture_synthetic.h5"
    result = runner.invoke(app, ["fixture", str(h5), "--frames", "4"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["inspect", str(h5.parent)])
    assert result.exit_code == 0, result.output
    assert "synthetic_box" in result.output and "4  (t = 0.000" in result.output


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cli_encode_reuses_frames(tmp_path):
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(4):
        Image.new("RGB", (65, 37), (40 * i, 80, 120)).save(frames / f"frame_{i:06d}.png")  # odd size on purpose
    result = CliRunner().invoke(app, ["encode", str(tmp_path), "--fps", "8", "--title", "T", "--title-seconds", "1",
                                      "--hold", "0.5", "--name", "clip"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (tmp_path / "clip.mp4").stat().st_size > 0 and (tmp_path / "title_card.png").is_file()


@pytest.mark.skipif(importlib.util.find_spec("mitsuba") is None, reason="mitsuba not installed")
def test_render_cli_smoke(recording, tmp_path):
    out = tmp_path / "render"
    cmd = [sys.executable, "-m", "tf_render", str(recording.path), "--out", str(out), "--res", "96x54",
           "--spp", "4", "--variant", "llvm", "--no-encode", "--frames", "0:6:5"]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(sorted((out / "frames").glob("frame_*.png"))) == 2
    config = json.loads((out / "render_config.json").read_text())
    assert config["status"] == "done" and config["signature"]["resolution"] == [96, 54]
    assert (out / "worn_area.csv").read_text().count("\n") == 3  # header + 2 frames
    # --resume with identical settings reuses both frames.
    result = subprocess.run(cmd + ["--resume"], capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reusing 2" in result.stdout + result.stderr
