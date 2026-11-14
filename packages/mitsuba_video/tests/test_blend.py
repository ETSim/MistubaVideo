"""The viewer-port blend: four-stage weights, slerp4, the viewer's presentation terms and its channel choice."""

from __future__ import annotations

import numpy as np
import pytest

from mitsuba_video import blend as bl
from mitsuba_video.maps import MapSet

PLAIN = bl.BlendParams()


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _const_stack(shape, albedos, normals=None, roughness=None) -> bl.VariantStack:
    full = lambda value, ch: np.broadcast_to(np.asarray(value, np.float32), shape + ch).copy()  # noqa: E731
    normals = normals or [(0.0, 0.0, 1.0)] * len(albedos)
    roughness = roughness or [0.5] * len(albedos)
    return bl.VariantStack(
        albedo=[full(a, (3,)) for a in albedos],
        normal=[full(_unit(n), (3,)) for n in normals],
        roughness=[full(r, ()) for r in roughness],
    )


def test_weights4_partition_unity_and_endpoints():
    t = np.linspace(0.0, 1.0, 101)
    w = bl.wear_blend_weights4(t, PLAIN)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)
    assert (w >= -1e-12).all()
    np.testing.assert_allclose(w[0], [1, 0, 0, 0])
    np.testing.assert_allclose(w[-1], [0, 0, 0, 1])
    np.testing.assert_allclose(bl.wear_blend_weights4(np.float64(0.5), PLAIN), [0, 0.5, 0.5, 0], atol=1e-12)
    assert (np.diff(w[:, 3]) >= -1e-12).all()


def test_slerp4_one_hot_and_two_stage_match_slerp():
    rng = np.random.default_rng(1)
    n = [_unit(rng.normal(size=(50, 3)) + [0, 0, 2]) for _ in range(4)]
    for k in range(4):
        w = np.zeros((50, 4))
        w[:, k] = 1.0
        np.testing.assert_allclose(bl.slerp4_normals(n, w), n[k], atol=1e-6)
    for t in (0.2, 0.5, 0.8):
        w = np.zeros((50, 4))
        w[:, 0], w[:, 1] = 1.0 - t, t
        np.testing.assert_allclose(bl.slerp4_normals(n, w), bl.slerp_normals(n[0], n[1], np.full(50, t)), atol=1e-6)


def test_four_stage_blend_follows_weights():
    stack = _const_stack((4, 4), [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
    out = bl.blend_stack(np.full((4, 4), 0.5, np.float32), stack, PLAIN)  # remap(0.5) = 0.5 -> stages 1 and 2
    np.testing.assert_allclose(out.albedo, np.full((4, 4, 3), [0, 0.5, 0.5]), atol=1e-6)
    out = bl.blend_stack(np.ones((4, 4), np.float32), stack, PLAIN)
    np.testing.assert_allclose(out.albedo, 1.0, atol=1e-6)


def test_static_channel_keeps_its_slice():
    stack = _const_stack((2, 2), [(0.3, 0.3, 0.3)], normals=[(0, 0, 1), (0.6, 0, 0.8)])
    stack.roughness = stack.roughness[:1]
    out = bl.blend_stack(np.ones((2, 2), np.float32), stack, PLAIN)
    np.testing.assert_allclose(out.albedo, 0.3, atol=1e-6)
    np.testing.assert_allclose(out.normal[0, 0], [0.6, 0, 0.8], atol=1e-6)


def test_viewer_terms():
    stack = _const_stack((3, 3), [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)], normals=[(0, 0, 1), (0.6, 0, 0.8)],
                         roughness=[0.5, 0.0])  # fmt: skip
    wear = np.zeros((3, 3), np.float32)
    wear[0, 0] = 5e-4  # below the viewer's 1e-3 start: untouched even though > WEAR_EPS
    wear[2, 2] = 1.0
    p = bl.BlendParams(locality=0.02, polish=0.7, roughness_floor=True)
    out = bl.blend_stack(wear, stack, p)
    np.testing.assert_allclose(out.albedo[0, 0], 0.2, atol=1e-6)
    np.testing.assert_allclose(out.albedo[2, 2], 0.8, atol=1e-6)
    np.testing.assert_allclose(out.normal[2, 2], _unit([0.6 * 0.3, 0, 0.8]), atol=1e-6)  # 70% of the detail polished
    assert out.roughness[2, 2] == pytest.approx(0.09)  # 0.04 + 0.05 * min(1, 2 * wear) * t
    assert out.roughness.min() >= 0.04
    # The plain blend has none of these terms.
    plain = bl.blend_stack(wear, stack, PLAIN)
    np.testing.assert_allclose(plain.normal[2, 2], _unit([0.6, 0, 0.8]), atol=1e-6)
    assert plain.roughness[2, 2] == pytest.approx(0.0)


def test_wear_blur_is_the_viewer_cross():
    wear = np.zeros((5, 5))
    wear[2, 2] = 1.0
    out = bl.blur_wear(wear, 1.0)
    assert out[2, 2] == pytest.approx(0.4)
    assert out[2, 3] == pytest.approx(0.15) and out[1, 2] == pytest.approx(0.15)
    np.testing.assert_array_equal(bl.blur_wear(wear, 0.0), wear)


def test_pair_stack_keeps_two_slice_semantics():
    rng = np.random.default_rng(4)
    base = MapSet(rng.random((4, 4, 3)).astype(np.float32), rng.random((4, 4)).astype(np.float32),
                  np.tile(np.float32([0, 0, 1]), (4, 4, 1)), None)  # fmt: skip
    worn = MapSet(rng.random((4, 4, 3)).astype(np.float32), rng.random((4, 4)).astype(np.float32),
                  np.tile(np.float32([0, 0, 1]), (4, 4, 1)), None)  # fmt: skip
    stack = bl.VariantStack.pair(base, worn)
    assert stack.describe() == "normal 2, albedo 2, roughness 2, height 0"
    np.testing.assert_array_equal(stack.worn.albedo, worn.albedo)














def _random_stack(rng, shape, stages=4) -> bl.VariantStack:
    return bl.VariantStack(
        albedo=[rng.random(shape + (3,), dtype=np.float32) for _ in range(stages)],
        normal=[_unit(rng.normal([0, 0, 3], 1, shape + (3,))).astype(np.float32) for _ in range(stages)],
        roughness=[rng.random(shape, dtype=np.float32) for _ in range(stages)],
        height=[rng.random(shape, dtype=np.float32) for _ in range(stages)],
    )


@pytest.mark.parametrize("look", ["worn", "heatmap"])
def test_incremental_textures_match_a_full_rebuild(look, monkeypatch):
    from mitsuba_video import pipeline

    monkeypatch.setattr(pipeline.BodyTextures, "TILE", 32)  # several tiles on a small, non-multiple atlas
    rng = np.random.default_rng(3)
    shape = (150, 97)
    stack = _random_stack(rng, shape)
    params = bl.BlendParams(wear_blur_px=2.5)
    args = (pipeline.Look(look), stack, params, pipeline.FieldDisplay(), 0.5 / 255.0, 1.0)
    textures = pipeline.BodyTextures(*args)
    wear = np.zeros(shape, np.float32)
    textures.update(wear.copy())
    for step in range(4):  # scratches appear and deepen, including on tile and atlas borders
        y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        wear[max(y - 3, 0):y + 3, max(x - 9, 0):x + 9] += 0.3
        wear[-2:, 40 + step] = 0.8
        got = textures.update(np.clip(wear, 0, 1))
        want = pipeline.body_textures(args[0], np.clip(wear, 0, 1), stack, params, args[3], args[4], args[5])
        for key in want:
            np.testing.assert_allclose(got[key], want[key], atol=1e-6, err_msg=f"{key} after step {step}")
    # Shared unworn textures were copied before the first tile write, not modified in place.
    unworn = pipeline.body_textures(args[0], np.zeros(shape, np.float32), stack, params, args[3], args[4], args[5])
    full_unworn = pipeline._blended_textures(args[0], np.zeros(shape, np.float32), stack, params, *args[3:])
    for key in unworn:
        np.testing.assert_array_equal(unworn[key], full_unworn[key])


def test_metallic_follows_the_blend_between_stages():
    """Enamel (dielectric) over steel (metal): a worn texel's metallic moves toward the worn stage's, an unworn one
    keeps the base's, and the encoded textures carry the map."""
    from dataclasses import replace

    from mitsuba_video.scene import encode_maps

    stack = replace(_const_stack((4, 4), [(0.1, 0.4, 0.2), (0.6, 0.6, 0.6)]), metallic=[0.0, 1.0])
    wear = np.zeros((4, 4), np.float32)
    wear[:, 2:] = 1.0
    out = bl.blend_stack(wear, stack, PLAIN)
    np.testing.assert_allclose(out.metallic[:, :2], 0.0)
    np.testing.assert_allclose(out.metallic[:, 2:], 1.0, atol=1e-6)
    assert encode_maps(out)["metallic"].shape == (4, 4, 1)


def test_equal_stage_metallic_needs_no_map():
    from dataclasses import replace

    stack = replace(_const_stack((4, 4), [(0.1, 0.4, 0.2), (0.6, 0.6, 0.6)]), metallic=[0.0, 0.0])
    out = bl.blend_stack(np.ones((4, 4), np.float32), stack, PLAIN)
    assert out.metallic is None
