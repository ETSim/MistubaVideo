from typer.testing import CliRunner

from mitsuba_video.cli import app


def test_wizard_cancellation_leaves_no_output(manifest_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["tui"], input=f"{manifest_dir}\n\n\n\n\n\nn\n")
    assert result.exit_code == 0, result.output
    assert "Command:" in result.output
    assert not (tmp_path / "output").exists()


def test_bad_resolution_has_actionable_error(manifest_dir):
    result = CliRunner().invoke(app, ["render", str(manifest_dir), "--spp", "0"])
    assert result.exit_code != 0
    assert "positive" in result.output
