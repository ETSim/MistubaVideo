"""TextureFriction source: reader conventions and a golden comparison against the pre-refactor renderer."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import DATA
from mitsuba_video.camera import CameraSettings, camera_path
from mitsuba_video.maps import NormalOptions
from mitsuba_video.motion import travel_by_frame
from mitsuba_video.pipeline import covered_area
from mitsuba_video.source import open_source

pytest.importorskip("h5py")

TEX = 64


@pytest.fixture(scope="module")
def tf(tf_recording):
    src = open_source(tf_recording, "texturefriction")
    yield src
    src.close()


def test_auto_detects_texturefriction(tf_recording):
    src = open_source(tf_recording)
    assert src.kind == "texturefriction"
    src.close()


def test_reader(tf):
    assert [b.index for b in tf.bodies] == [0, 1]
    plane, box = tf.bodies
    assert plane.fixed and not box.fixed
    assert box.scale == pytest.approx(1.5) and box.notes["scale"] == "1.5"
    assert len(tf.frames) == 6
    rec_box = tf.recording.body(1)
    assert rec_box.worn_variant_index == 1
    assert rec_box.variants[1].normal.shape == (TEX, TEX, 3)
    pos, quat = tf.pose(tf.frames[-1])
    assert pos.shape == (2, 3) and quat.shape == (2, 4)
    assert set(tf.fields) == {"wear", "sliding"} and tf.primary_field == "wear"
    assert "reset_detection" in tf.capabilities


def test_wear_texture_orientation(tf):
    # The box slides at z = +0.2 (plane v = 0.7): image convention puts V = 1 at row 0, so the footprint sits in the
    # upper half of the texture; a V flip would land it in the lower half.
    wear = tf.field(tf.frames[-1], 0)
    rows = np.nonzero(wear.max(axis=1) > 0)[0]
    assert rows.max() < TEX // 2
    assert wear.max() <= 1.0


def test_area_per_texel(tf):
    plane, box = tf.bodies
    assert plane.area_per_texel((TEX, TEX)) == pytest.approx(2.0 / (TEX * TEX))
    assert box.area_per_texel((TEX, TEX)) == pytest.approx((0.2 * 1.5) ** 2 / (TEX * TEX))


def test_matches_pre_refactor_golden(tf):
    """Materials, worn area, camera paths and travel equal what tf_render produced before the generic split."""
    golden = np.load(DATA / "tf_synthetic_golden.npz")
    for body in tf.bodies:
        base, worn = body.material.build((32, 32), NormalOptions(), None)
        for tag, ms in (("base", base), ("worn", worn)):
            np.testing.assert_allclose(ms.albedo, golden[f"b{body.index}_{tag}_albedo"], atol=1e-6)
            np.testing.assert_allclose(ms.roughness, golden[f"b{body.index}_{tag}_roughness"], atol=1e-6)
            np.testing.assert_allclose(ms.normal, golden[f"b{body.index}_{tag}_normal"], atol=1e-6)
        assert body.material.metallic == pytest.approx(float(golden[f"b{body.index}_metallic"]))
        counts, area = covered_area(tf, tf.frames, body, tf.fields["wear"], progress=False)
        np.testing.assert_array_equal(counts, golden[f"b{body.index}_worn_counts"])
        np.testing.assert_allclose(area, golden[f"b{body.index}_worn_area"])
    for mode in ("fixed", "track", "orbit"):
        poses = camera_path(tf, tf.frames, CameraSettings(mode=mode), 16 / 9)
        np.testing.assert_allclose([p.eye for p in poses], golden[f"cam_{mode}_eye"], atol=1e-9)
        np.testing.assert_allclose([p.target for p in poses], golden[f"cam_{mode}_target"], atol=1e-9)
    _, travel = travel_by_frame(tf, None, detect_resets=True)
    np.testing.assert_allclose([travel[f.key] for f in tf.frames], golden["travel"])


def test_rejects_unknown_option(tf_recording):
    with pytest.raises(ValueError, match="scale, assets"):
        open_source(tf_recording, "texturefriction", colour="red")


def test_scale_override(tf_recording):
    src = open_source(tf_recording, "texturefriction", scale="1=3")
    assert src.bodies[1].scale == 3.0
    src.close()
