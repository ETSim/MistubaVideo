"""Source-independent pieces: transforms, blend, colour, maps, MTL/OBJ, motion."""

from __future__ import annotations

import numpy as np
import pytest

from mitsuba_video import blend as bl
from mitsuba_video import color
from mitsuba_video.maps import MapSet, NormalOptions, decode_normal, fit_size
from mitsuba_video.mtl import mtl_get, parse_mtl_text
from mitsuba_video.obj import load_obj, write_obj
from mitsuba_video.transforms import body_world_transform, quaternion_to_matrix


def test_quaternion_is_wxyz():
    # 90 degrees about +y, stored (w, x, y, z): +x maps to -z.
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    np.testing.assert_allclose(quaternion_to_matrix(q) @ [1, 0, 0], [0, 0, -1], atol=1e-12)


def test_model_matrix_is_translate_rotate_scale():
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    m = body_world_transform(np.array([1.0, 2.0, 3.0]), q, 2.0)
    np.testing.assert_allclose(m @ [1, 0, 0, 1], [1, 2, 1, 1], atol=1e-12)


def test_hermite_remap_matches_cpp_reference():
    p = bl.BlendParams()
    assert bl.wear_remap_t(np.float32(0.0), p) == 0.0
    assert bl.wear_remap_t(np.float32(1.0), p) == 1.0
    assert float(bl.wear_remap_t(np.float32(0.5), p)) == pytest.approx(0.5, abs=1e-6)
    assert bl.hermite_tangents(0.65, 0.0) == pytest.approx((0.175, 0.175))


def _random_mapset(seed: int, shape=(8, 8)) -> MapSet:
    r = np.random.default_rng(seed)
    n = r.normal(size=shape + (3,))
    n[..., 2] = np.abs(n[..., 2]) + 1.0
    return MapSet(
        albedo=r.random(shape + (3,)).astype(np.float32),
        roughness=r.random(shape).astype(np.float32),
        normal=(n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32),
        height=None,
    )


def test_blend_endpoints():
    shape = (8, 8)
    base, worn = _random_mapset(2), _random_mapset(3)
    unworn = bl.blend_maps(np.zeros(shape, np.float32), base, worn)
    np.testing.assert_array_equal(unworn.albedo, base.albedo)
    np.testing.assert_array_equal(unworn.normal, base.normal)
    full = bl.blend_maps(np.ones(shape, np.float32), base, worn)
    np.testing.assert_allclose(full.albedo, worn.albedo, atol=1e-6)
    np.testing.assert_allclose(full.roughness, worn.roughness, atol=1e-6)
    np.testing.assert_allclose(full.normal, worn.normal, atol=1e-5)
    half = bl.blend_maps(np.full(shape, 0.5, np.float32), base, worn)
    np.testing.assert_allclose(np.linalg.norm(half.normal, axis=-1), 1.0, atol=1e-5)


def test_heatmap_touches_only_covered_texels_and_ramps():
    albedo = np.full((1, 3, 3), 0.5, np.float32)
    value = np.array([[0.02, 0.2, 0.0]], np.float32)
    flat = color.heatmap_albedo(albedo, value)
    ramped = color.heatmap_albedo(albedo, value, color.FieldDisplay(ramp=0.2))
    assert np.abs(ramped[0, 0] - albedo[0, 0]).max() < np.abs(flat[0, 0] - albedo[0, 0]).max()
    np.testing.assert_allclose(ramped[0, 1], flat[0, 1], atol=1e-6)
    np.testing.assert_array_equal(ramped[0, 2], albedo[0, 2])
    np.testing.assert_allclose(color.colormap(np.float32(0.5), hi=0.5), color.colormap(np.float32(1.0)), atol=1e-6)


def test_normal_map_recentring():
    rng = np.random.default_rng(3)
    tilted = np.array([0.506, 0.506, 0.663]) + rng.normal(0, 0.02, (32, 32, 3))
    img = np.clip((tilted / np.linalg.norm(tilted, axis=-1, keepdims=True)) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    raw = decode_normal(img, (32, 32), NormalOptions(recenter_above_deg=0.0))
    fixed = decode_normal(img, (32, 32), NormalOptions())
    assert raw.reshape(-1, 3).mean(0)[2] < 0.75
    mean = fixed.reshape(-1, 3).mean(0)
    assert mean[2] / np.linalg.norm(mean) > 0.999
    np.testing.assert_allclose(np.linalg.norm(fixed, axis=-1), 1.0, atol=1e-5)
    flat = decode_normal(img, (32, 32), NormalOptions(strength=0.0))
    np.testing.assert_allclose(flat[..., 2], 1.0, atol=1e-6)


def test_fit_size_caps_longer_side():
    assert fit_size((4096, 2048), 1024) == (512, 1024)
    assert fit_size(None, 2048) == (512, 512)


def test_parse_mtl_and_case_insensitive_lookup():
    mtl = parse_mtl_text("newmtl A\nKd 0.1 0.2 0.3\nmap_Bump -bm 0.5 tex/n.png\nmap_Pr_worn r.png\nPr 0.4\n")
    assert mtl["A"]["Kd"] == [0.1, 0.2, 0.3]
    assert mtl["A"]["map_Bump"] == "tex/n.png"
    assert mtl_get(mtl["A"], "MAP_PR_WORN") == "r.png"


def test_obj_round_trip_and_quads(tmp_path):
    v = np.float32([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    n = np.tile(np.float32([0, 0, 1]), (3, 1))
    uv = np.float32([[0, 0], [1, 0], [1, 1]])
    write_obj(tmp_path / "t.obj", v, n, uv, np.uint32([[0, 1, 2]]))
    mesh = load_obj(tmp_path / "t.obj")
    np.testing.assert_allclose(mesh.vertices, v)
    np.testing.assert_allclose(mesh.uvs, uv)
    assert mesh.faces.tolist() == [[0, 1, 2]]
    quad = tmp_path / "q.obj"
    quad.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n")
    mesh = load_obj(quad)
    assert len(mesh.faces) == 2
    np.testing.assert_allclose(mesh.normals, np.tile([0, 0, 1], (4, 1)), atol=1e-6)  # computed when absent


def test_obj_splits_uv_seams(tmp_path):
    # Two faces share position 2 but use different texcoords there: the corner must be duplicated.
    path = tmp_path / "seam.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 0.5\n"
                    "f 1/1 2/2 3/3\nf 1/1 3/4 4/3\n")  # fmt: skip
    mesh = load_obj(path)
    assert len(mesh.vertices) == 5
