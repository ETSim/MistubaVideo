"""The viewer-port blend: four-stage weights, slerp4, the viewer's presentation terms and its channel choice."""

from __future__ import annotations

import numpy as np
import pytest

from mitsuba_video import blend as bl
from mitsuba_video.maps import MapSet
from mitsuba_video.sources.texturefriction import recorded_blend
from mitsuba_video.sources.texturefriction.materials import viewer_channel_sources
from mitsuba_video.sources.texturefriction.recording import MaterialVariant

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


def _variant(name, normal=True, rough=True, height=True, albedo=True) -> MaterialVariant:
    img3 = np.zeros((4, 4, 3), np.uint8)
    img1 = np.zeros((4, 4), np.uint8)
    return MaterialVariant(name=name, normal=img3 if normal else None, roughness=img1 if rough else None,
                           height=img1 if height else None, roughness_value=0.0, metallic=0.0,
                           albedo=img3 if albedo else None)  # fmt: skip


def _mtl(*keys):
    def image(key: str, mode: str):
        if key not in keys:
            return None
        return np.zeros((4, 4, 3) if mode == "RGB" else (4, 4), np.uint8)

    return image


def _counts(sources) -> dict[str, int]:
    return {k: len(v) for k, v in sources.items()}


def test_channel_choice_multi_material_atlas():
    # createBunnyInclinedPlaneMultiMaterial: smooth + metal + rock, the variant loader records no roughness or
    # height for the two appended materials. Normals and albedo come from variants 0 and 1, roughness falls back to
    # the base material's map_Pr / map_Pr_worn, and no height pair exists.
    variants = [_variant("smooth"), _variant("metal", rough=False, height=False),
                _variant("rock", rough=False, height=False)]  # fmt: skip
    sources = viewer_channel_sources(variants, _mtl("map_Kd", "map_Pr_worn"))
    assert _counts(sources) == {"normal": 2, "roughness": 2, "albedo": 2, "height": 1}
    assert sources["normal"][1] is variants[1].normal and sources["albedo"][1] is variants[1].albedo


def test_channel_choice_four_variants_and_single_material():
    variants = [_variant(f"stage{k}") for k in range(4)]
    assert _counts(viewer_channel_sources(variants, _mtl())) == {"normal": 4, "roughness": 4, "albedo": 4, "height": 4}
    # One atlas entry: the viewer's base/worn registry pairs from the .mtl, and albedo stays static.
    single = viewer_channel_sources([_variant("only")], _mtl("map_Kd", "map_Bump_Worn", "map_Pr_worn"))
    assert _counts(single) == {"normal": 2, "roughness": 2, "albedo": 1, "height": 1}
    # Recorded before variant albedo was exported: the base map_Kd only.
    old = viewer_channel_sources([_variant("a", albedo=False), _variant("b", albedo=False)], _mtl("map_Kd"))
    assert _counts(old)["albedo"] == 1


def test_recorded_blend_overrides_viewer_defaults():
    assert recorded_blend(None) == bl.VIEWER_BLEND
    p = recorded_blend({"enabled": 1.0, "normal_wear_edge0": 0.1, "hermite_tension": 0.5, "wear_map_blur_px": 0.0})
    assert (p.edge0, p.tension, p.wear_blur_px) == (0.1, 0.5, 0.0)
    assert p.polish == bl.VIEWER_BLEND.polish and p.edge1 == bl.VIEWER_BLEND.edge1
