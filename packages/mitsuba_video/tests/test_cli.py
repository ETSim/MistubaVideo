"""CLI and pipeline: registry, inspect/fixture/validate/encode, resume safety, and real render smokes."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import SRC
from mitsuba_video import pipeline
from mitsuba_video.cli import app
from mitsuba_video.source import available_sources, detect_kind, parse_options


def test_registry_has_builtins_and_parses_options(manifest_dir):
    assert {"texturefriction", "manifest"} <= set(available_sources())
    assert detect_kind(manifest_dir) == "manifest"
    assert parse_options(["scale=1=0.5", "assets=a"]) == {"scale": "1=0.5", "assets": "a"}
    with pytest.raises(ValueError):
        parse_options(["novalue"])
    with pytest.raises(ValueError, match="no source recognises"):
        detect_kind(Path(__file__))


def test_parse_frames():
    assert list(pipeline.parse_frames("::2", 6)) == [0, 2, 4]
    assert list(pipeline.parse_frames("-2:", 6)) == [4, 5]


def test_resume_refuses_different_settings(tmp_path):
    (tmp_path / pipeline.CONFIG_NAME).write_text(json.dumps({"signature": {"spp": 64, "look": "heatmap"}}))
    pipeline._check_resume(tmp_path, {"spp": 64, "look": "heatmap"})
    with pytest.raises(RuntimeError, match="spp"):
        pipeline._check_resume(tmp_path, {"spp": 128, "look": "heatmap"})


def test_cli_fixture_validate_inspect_sources(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["fixture", str(tmp_path / "m"), "--frames", "4"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["validate", str(tmp_path / "m")])
    assert result.exit_code == 0 and "4 frames" in result.output, result.output
    result = runner.invoke(app, ["inspect", str(tmp_path / "m")])
    assert result.exit_code == 0, result.output
    assert "synthetic_box" in result.output and "4  (t = 0.000" in result.output
    result = runner.invoke(app, ["sources"])
    assert "manifest" in result.output and "texturefriction" in result.output


def test_cli_inspect_texturefriction(tf_recording):
    result = CliRunner().invoke(app, ["inspect", str(tf_recording)])
    assert result.exit_code == 0, result.output
    assert "slid" in result.output and "atlas" in result.output


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


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "mitsuba_video", *args], capture_output=True, text=True,
                          timeout=600, env=env)  # fmt: skip


@pytest.mark.skipif(importlib.util.find_spec("mitsuba") is None, reason="mitsuba not installed")
def test_render_smoke_manifest_and_resume(manifest_dir, tmp_path):
    out = tmp_path / "render"
    args = [str(manifest_dir), "--out", str(out), "--res", "96x54", "--spp", "4", "--variant", "llvm",
            "--no-encode", "--frames", "0:6:5", "--ramp", "0.2"]  # fmt: skip
    result = _run(args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(sorted((out / "frames").glob("frame_*.png"))) == 2
    config = json.loads((out / pipeline.CONFIG_NAME).read_text())
    assert config["status"] == "done" and config["signature"]["source_kind"] == "manifest"
    assert config["video_stem"] == "wear_heatmap_fixed"
    assert (out / "wear_area.csv").read_text().count("\n") == 3  # header + 2 frames
    result = _run(args + ["--resume"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reusing 2" in result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec("mitsuba") is None, reason="mitsuba not installed")
def test_render_smoke_tf_alias(tf_recording, tmp_path):
    out = tmp_path / "render_tf"
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    code = ("import sys; from mitsuba_video.sources.texturefriction.cli import main; "
            "main(sys.argv[1:])")  # fmt: skip
    result = subprocess.run([sys.executable, "-c", code, str(tf_recording), "--out", str(out), "--res", "96x54",
                             "--spp", "4", "--variant", "llvm", "--no-encode", "--frames", "-1:"],
                            capture_output=True, text=True, timeout=600, env=env)  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((out / pipeline.CONFIG_NAME).read_text())
    assert config["video_stem"] == "wear_heatmap_fixed" and config["signature"]["source_kind"] == "texturefriction"
    assert (out / "worn_area.csv").is_file()


def test_tf_render_without_h5py_prints_install_hint(monkeypatch, capsys):
    from mitsuba_video import cli

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):
        return None if name == "h5py" else real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    with pytest.raises(SystemExit) as exc:
        cli.tf_main(["--help"])
    assert exc.value.code == 2
    assert "mitsuba-video[texturefriction]" in capsys.readouterr().err


def test_tf_render_fixture_defaults_to_a_recording(tmp_path):
    from mitsuba_video.cli import build_app

    tf_app = build_app(default_source="texturefriction", prog="tf-render")
    h5 = tmp_path / "rec" / "FrictionTexture_synthetic.h5"
    result = CliRunner().invoke(tf_app, ["fixture", str(h5), "--frames", "2"])
    assert result.exit_code == 0, result.output
    assert h5.is_file()
