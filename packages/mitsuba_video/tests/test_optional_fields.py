"""Geometry-only inputs and independent scalar-field examples share the same renderer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from mitsuba_video.cli import app
from mitsuba_video.pipeline import Look, RenderSettings, resolve_field
from mitsuba_video.source import available_fixtures, field_value, open_source
from mitsuba_video.sources.manifest.cube_throw import write_cube_throw
from mitsuba_video.sources.manifest.export import export_manifest


def test_geometry_only_round_trip_without_atlas(tmp_path):
    path = write_cube_throw(tmp_path / "cube", frames=3)
    data = json.loads(path.read_text())
    data.pop("fields")
    path.write_text(json.dumps(data))
    source = open_source(path)
    assert source.fields == {} and source.primary_field == ""
    assert resolve_field(source, None).name == ""
    with pytest.raises(ValueError, match="unknown field"):
        resolve_field(source, "temperature")
    exported = export_manifest(source, tmp_path / "copy", material_size=4, progress=False)
    copy = open_source(exported)
    assert copy.fields == {} and len(copy.frames) == 3
    for a, b in zip(source.frames, copy.frames):
        np.testing.assert_allclose(source.pose(a)[0], copy.pose(b)[0])
    assert RenderSettings(source=path).look is Look.plain
    source.close()
    copy.close()


def test_geometry_only_source_needs_no_field_reader():
    assert field_value(SimpleNamespace(fields={}), None, 0) is None


@pytest.mark.parametrize("name", ["temperature", "damage"])
def test_scalar_fixtures_and_material_look(tmp_path, name):
    writer = available_fixtures()[name]()
    path = writer(tmp_path / name, frames=2)
    source = open_source(path)
    assert list(source.fields) == [name]
    assert source.field(source.frames[-1], 0, name).max() > 0
    source.close()


def test_general_help_has_no_recording_options():
    result = CliRunner().invoke(app, ["render", "--help"])
    assert result.exit_code == 0
    assert "--field" in result.output and "--source" in result.output
    assert "--scratches" not in result.output and "--scale" not in result.output
