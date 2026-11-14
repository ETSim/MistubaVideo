import json

import numpy as np
from PIL import Image

from mitsuba_video.annotate import Annotations, Callout, callouts, draw_callouts, project, track_depth
from mitsuba_video.camera import CameraPose
from mitsuba_video.video import font


def _pose():
    # Looking down -z from z = 10 with +y up: world +x is image right, +y image up.
    return CameraPose(eye=np.array([0.0, 0.0, 10.0]), target=np.zeros(3), up=np.array([0.0, 1.0, 0.0]))


def test_project_matches_the_perspective_sensor():
    w, h, fov = 200, 100, 90.0  # tan(45 deg) = 1: x = +-10 at depth 10 spans the full width
    pts = project(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-5.0, -2.5, 0.0]]), _pose(), fov, w, h)
    assert np.allclose(pts[0], [100, 50, 10])
    assert np.allclose(pts[1, :2], [200, 50])  # right edge
    assert np.allclose(pts[2, :2], [100, 0])  # top edge (vertical half-extent is 5 at this aspect)
    assert np.allclose(pts[3, :2], [50, 75])


class _Body:
    def __init__(self, index, fixed):
        self.index, self.fixed = index, fixed
        self.local_bounds = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))


def test_callouts_tag_labelled_bodies_in_view_only():
    bodies = [_Body(0, True), _Body(1, False), _Body(2, False)]
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 20.0]])  # body 2 is behind the camera
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    items = callouts(bodies, positions, orientations, _pose(), 90.0, (200, 100), {0: "tile", 2: "behind"})
    assert [i.text for i in items] == ["tile"] and items[0].fixed
    ax, ay = items[0].anchor  # a quarter of the way from the topmost projected corner toward the centre (100, 50)
    x0, y0, _, _ = items[0].box
    assert y0 < ay < 50 and x0 <= ax <= 100
    edge = np.array([[9.5, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 20.0]])  # body 0's centre just inside the right edge
    assert [i.text for i in callouts(bodies, edge, orientations, _pose(), 90.0, (200, 100), {0: "tile"})] == ["tile"]
    outside = np.array([[10.5, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 20.0]])  # centre off frame, box still touching
    assert callouts(bodies, outside, orientations, _pose(), 90.0, (200, 100), {0: "tile"}) == []
    image = Image.new("RGB", (200, 100), (0, 120, 0))
    draw_callouts(image, [Callout("cube", (90, 60, 110, 80), False)], font(12))
    changed = np.argwhere(np.any(np.asarray(image) != (0, 120, 0), axis=2))
    assert len(changed) and changed[:, 0].max() < 60  # the moving body's tag sits above it
    assert image.getpixel((2, 98)) == (0, 120, 0)  # nothing drawn far from the tag


def test_fixed_tag_moves_off_a_body_that_covers_its_corner():
    bodies = [_Body(0, True), _Body(1, False)]
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    big = _Body(0, True)
    big.local_bounds = (np.array([-4.0, -2.0, -0.1]), np.array([4.0, 2.0, 0.1]))  # a plate seen from above
    bodies[0] = big
    free = np.array([[0.0, 0.0, 0.0], [9.0, -4.5, 0.0]])  # the cube far from the plate's top-left corner
    (tag,) = callouts(bodies, free, orientations, _pose(), 90.0, (200, 100), {0: "plate"})
    top_left = tag.anchor
    assert top_left[1] < 50  # the usual spot: near the topmost corner
    on_corner = np.array([[0.0, 0.0, 0.0], [-3.5, 1.5, 0.0]])  # the cube stopped on that corner
    (tag,) = callouts(bodies, on_corner, orientations, _pose(), 90.0, (200, 100), {0: "plate"})
    assert tag.anchor != top_left and tag.anchor[1] > 50  # moved to the bottom corner
    # Where the cube ends up counts from the first frame on, so the tag does not jump when it arrives there.
    (tag,) = callouts(bodies, free, orientations, _pose(), 90.0, (200, 100), {0: "plate"},
                      settled=(on_corner, orientations))  # fmt: skip
    assert tag.anchor[1] > 50


def test_annotations_load_and_panel_titles(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"body_labels": {"0": "H 100"}, "panel_titles": {"1": "sheet"},
                                "field_label": "wear depth", "field_max": "5 mm"}))  # fmt: skip
    notes = Annotations.load(path)
    assert notes.body_labels == {0: "H 100"} and notes.field_max == "5 mm"
    assert notes.panel_title(0, "wear depth") == "H 100"
    assert notes.panel_title(1, "wear depth") == "sheet"
    assert notes.panel_title(2, "wear depth") is None
    assert Annotations.load(None) is None
    path.write_text(json.dumps({"bogus": 1}))
    try:
        Annotations.load(path)
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("unknown keys must fail")


def test_track_depth_is_a_robust_deep_value_from_the_field():
    field = np.zeros((64, 64))
    assert track_depth(field, 5.0) is None  # nothing worn yet
    field[10:20, 10:40] = np.linspace(0.1, 0.5, 30)
    field[0, 0] = 1.0  # one stray texel does not set the depth
    depth = track_depth(field, 5.0)
    assert 2.0 < depth < 2.5
    assert track_depth(field, 0.0) is None
    image = Image.new("RGB", (200, 100), (0, 120, 0))
    draw_callouts(image, [Callout("H 100", (20, 20, 120, 80), True, "2.31 mm")], font(14))
    green = np.argwhere(np.all(np.asarray(image) == (28, 132, 72), axis=2))
    assert len(green)  # the measured value is drawn in the accent green


def test_motion_arrows_follow_the_recorded_velocity():
    from mitsuba_video.annotate import draw_arrows, motion_arrows

    bodies = [_Body(0, True), _Body(1, False)]
    q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    now = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    before = np.array([[0.0, 0.0, 0.0], [-0.01, 0.0, 0.0]])  # body 1 moves +x at 1 m/s
    arrows = motion_arrows(bodies, now, q, before, 0.01, _pose(), 90.0, (200, 100))
    assert len(arrows) == 1  # the fixed body gets none
    (x0, y0), (x1, y1) = arrows[0].start, arrows[0].end
    assert x1 > x0 and abs(y1 - y0) < 1e-6  # +x is image right for this camera
    assert motion_arrows(bodies, now, q, now, 0.01, _pose(), 90.0, (200, 100)) == []  # at rest
    crowd = [_Body(i, False) for i in range(10)]
    many = np.zeros((10, 3)), np.tile([1.0, 0, 0, 0], (10, 1)), np.full((10, 3), -0.01)
    assert motion_arrows(crowd, many[0], many[1], many[2], 0.01, _pose(), 90.0, (200, 100)) == []  # too many
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    draw_arrows(image, arrows)
    assert np.any(np.all(np.asarray(image) == (28, 132, 72), axis=2))


def test_cluster_boxes_project_each_clusters_world_aabb():
    from mitsuba_video.annotate import MIN_BOX_HALF_EXTENT, cluster_aabbs, cluster_box_segments, draw_cluster_boxes

    # A face contact as the serializer writes it: axes one per row (t1, t2, normal), 1 mm thick along the normal,
    # turned 45 degrees about the normal (+z here): its world AABB is sqrt(2) wider than the patch.
    r = np.sqrt(0.5)
    axes = np.array([[r, r, 0.0], [-r, r, 0.0], [0.0, 0.0, 1.0]])
    box = (np.array([1.0, 0.0, 0.0]), axes, np.array([0.1, 0.1, 0.001]))
    ((lo, hi),) = cluster_aabbs([box])
    assert np.allclose(hi - lo, [2 * 0.1 * 2 * r, 2 * 0.1 * 2 * r, 2 * MIN_BOX_HALF_EXTENT])
    assert np.allclose(0.5 * (lo + hi), [1.0, 0.0, 0.0])
    segments = cluster_box_segments([box], _pose(), 40.0, (400, 300))
    assert len(segments) == 12  # every edge of the box, all in front of the camera
    xs = [p[0] for seg in segments for p in seg]
    assert min(xs) > 200  # world +x is image right: the box sits right of centre
    # Behind the camera: nothing drawn.
    behind = (np.array([0.0, 0.0, 20.0]), np.eye(3), np.array([0.1, 0.1, 0.1]))
    assert cluster_box_segments([behind], _pose(), 40.0, (400, 300)) == []
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    draw_cluster_boxes(image, segments)
    assert np.asarray(image).min() < 200  # the lines are drawn


def test_annotations_switch_cluster_boxes_on(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"cluster_boxes": True}))
    assert Annotations.load(path).cluster_boxes
    path.write_text(json.dumps({}))
    assert not Annotations.load(path).cluster_boxes
