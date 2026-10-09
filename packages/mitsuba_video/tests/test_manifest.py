"""Manifest source: schema, writer round trip, and parity with the TextureFriction source after export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mitsuba_video.camera import CameraSettings, camera_path
from mitsuba_video.maps import NormalOptions
from mitsuba_video.pipeline import covered_area
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


def test_texturefriction_export_parity(tf_recording, tmp_path):
    pytest.importorskip("h5py")
    from mitsuba_video.sources.manifest.export import export_manifest

    tf = open_source(tf_recording, "texturefriction")
    path = export_manifest(tf, tmp_path / "exported", material_size=64, progress=False)
    man = open_source(path.parent)
    try:
        assert [b.index for b in man.bodies] == [b.index for b in tf.bodies]
        assert [f.index for f in man.frames] == [f.index for f in tf.frames]
        for f_tf, f_man in zip(tf.frames, man.frames):
            for a, b in zip(tf.pose(f_tf), man.pose(f_man)):
                np.testing.assert_allclose(a, b, atol=1e-6)
            for body in tf.bodies:
                np.testing.assert_array_equal(tf.field(f_tf, body.index), man.field(f_man, body.index))
        for b_tf, b_man in zip(tf.bodies, man.bodies):
            assert b_man.scale == b_tf.scale and b_man.fixed == b_tf.fixed
            np.testing.assert_allclose(b_man.vertices, b_tf.vertices, atol=1e-6)
            m_tf = b_tf.material.build((32, 32), NormalOptions(recenter_above_deg=0.0), None)
            m_man = b_man.material.build((32, 32), NormalOptions(recenter_above_deg=0.0), None)
            for a, b in zip(m_tf, m_man):
                np.testing.assert_allclose(a.albedo, b.albedo, atol=2 / 255)
                np.testing.assert_allclose(a.roughness, b.roughness, atol=2 / 255)
                np.testing.assert_allclose(a.normal, b.normal, atol=0.02)
            spec = tf.fields["wear"]
            c_tf, _ = covered_area(tf, tf.frames, b_tf, spec, progress=False)
            c_man, _ = covered_area(man, man.frames, b_man, man.fields["wear"], progress=False)
            np.testing.assert_array_equal(c_tf, c_man)
        for mode in ("fixed", "orbit"):
            a = camera_path(tf, tf.frames, CameraSettings(mode=mode), 16 / 9)
            b = camera_path(man, man.frames, CameraSettings(mode=mode), 16 / 9)
            np.testing.assert_allclose([p.eye for p in a], [p.eye for p in b], atol=1e-5)
    finally:
        tf.close()
        man.close()
