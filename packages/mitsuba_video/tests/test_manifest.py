"""Manifest schema and writer round trips for optional scalar fields."""

from __future__ import annotations

import json

import numpy as np

from mitsuba_video.maps import NormalOptions
from mitsuba_video.source import open_source
from mitsuba_video.sources.manifest import ManifestSource, validate
from mitsuba_video.sources.manifest.writer import ManifestWriter


def test_fixture_opens_and_auto_detects(manifest_dir):
    src = open_source(manifest_dir)
    assert src.kind == "manifest"
    assert [b.name for b in src.bodies] == ["synthetic_plane", "synthetic_box"]
    assert len(src.frames) == 6
    wear = src.field(src.frames[-1], 0)
    assert wear.shape == (64, 64) and 0.0 < wear.max() <= 1.0
    rows = np.nonzero(wear.max(axis=1) > 0)[0]
    assert rows.max() < 32  # same V convention as every other source
    base, worn = src.bodies[0].material.build((16, 16), NormalOptions(), None)
    assert worn.roughness.mean() < base.roughness.mean()  # worn roughness 0.3 vs base 0.7


def test_schema_rejects_bad_manifests():
    good = {"format": "mitsuba-video.manifest", "version": 1, "fields": {"wear": {}},
            "bodies": [{"id": 0, "mesh": "a.obj"}], "frames_file": "p.npz"}  # fmt: skip
    assert validate(good) == []
    assert validate({**good, "version": 2})
    assert validate({**good, "bodies": [{"id": 0}]})  # mesh required
    no_frames = dict(good)
    no_frames.pop("frames_file")
    assert validate(no_frames)  # frames_file or frames required
    assert validate({**good, "colour": "red"})  # unknown keys rejected


def test_writer_round_trip_inline_frames(tmp_path):
    v = np.float32([[0, 0, 0], [1, 0, 0], [1, 0, 1]])
    w = ManifestWriter(tmp_path / "m", field_format="npy")
    w.add_field("temperature", label="temperature, normalized", threshold=0.01)
    w.add_body(0, "tri", v, None, np.float32([[0, 0], [1, 0], [1, 1]]), np.uint32([[0, 1, 2]]))
    field = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    w.add_frame(0.0, [[0, 0, 0]], [[1, 0, 0, 0]], fields={"temperature": {0: field}})
    w.add_frame(0.5, [[0, 1, 0]], [[1, 0, 0, 0]])
    path = w.close()
    data = json.loads(path.read_text())
    assert data["primary_field"] == "temperature"
    src = ManifestSource.open(path)
    np.testing.assert_allclose(src.field(src.frames[0], 0), field)
    assert src.field(src.frames[1], 0) is None
    np.testing.assert_allclose(src.pose(src.frames[1])[0][0], [0, 1, 0])
    assert src.fields["temperature"].threshold == 0.01
