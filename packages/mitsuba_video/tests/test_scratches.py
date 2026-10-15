from __future__ import annotations

import numpy as np
from mitsuba_video import blend as bl
from mitsuba_video.maps import MapSet


def _maps(n: int, rgb: float, rough: float) -> MapSet:
    normal = np.zeros((n, n, 3), np.float32)
    normal[..., 2] = 1.0
    return MapSet(albedo=np.full((n, n, 3), rgb, np.float32), roughness=np.full((n, n), rough, np.float32),
                  normal=normal, height=None)  # fmt: skip


def _along_vs_across(s: np.ndarray) -> tuple[float, float]:
    """Mean correlation of a field with itself shifted 4 texels along x and along y."""
    a = s - s.mean()
    along = float((a * np.roll(a, 4, axis=1)).mean())
    across = float((a * np.roll(a, 4, axis=0)).mean())
    return along, across


def test_scratches_run_along_the_slide_direction():
    n = 96
    region = np.ones((n, n), bool)
    for d, axis in (((1.0, 0.0), "x"), ((0.0, 1.0), "y")):
        direction = np.zeros((n, n, 2), np.float32)
        direction[...] = d
        s = bl.scratch_field(direction, region, 10.0, seed=3)
        assert 0.0 <= s.min() and s.max() <= 1.0 and s.std() > 0.01
        along_x, along_y = _along_vs_across(s)
        if axis == "x":
            assert along_x > 3.0 * abs(along_y)  # streaks are long along x
        else:
            assert along_y > 3.0 * abs(along_x)


def test_scratch_field_is_empty_outside_the_region_and_where_nothing_slid():
    n = 64
    region = np.zeros((n, n), bool)
    region[20:40, 10:50] = True
    direction = np.zeros((n, n, 2), np.float32)
    direction[20:40, 10:50] = (1.0, 0.0)
    s = bl.scratch_field(direction, region, 8.0)
    assert not s[~region].any()
    assert not bl.scratch_field(np.zeros((n, n, 2), np.float32), region, 8.0).any()


def test_scratches_and_burnish_touch_only_worn_texels():
    n = 80
    base, worn = _maps(n, 0.2, 0.6), _maps(n, 0.7, 0.4)
    wear = np.zeros((n, n), np.float32)
    wear[30:50, 10:70] = 0.8
    direction = np.zeros((n, n, 2), np.float32)
    direction[30:50, 10:70] = (1.0, 0.0)
    plain = bl.blend_maps(wear, base, worn)
    p = bl.BlendParams(scratches=1.0, burnish=0.5)
    out = bl.blend_stack(wear, bl.VariantStack.pair(base, worn), p, direction=direction)
    far = np.zeros((n, n), bool)
    far[:20], far[60:] = True, True
    assert np.allclose(out.albedo[far], base.albedo[far])  # burnish spreads ~1.5 texels, not 10
    worn_band = np.zeros((n, n), bool)
    worn_band[34:46, 20:60] = True
    assert not np.allclose(out.albedo[worn_band], plain.albedo[worn_band])  # the striations show
    assert np.allclose(np.linalg.norm(out.normal, axis=-1), 1.0, atol=1e-4)
    assert p.wide and not bl.BlendParams().wide


def test_lower_scratch_density_draws_fewer_lines():
    n = 128
    region = np.ones((n, n), bool)
    direction = np.zeros((n, n, 2), np.float32)
    direction[...] = (1.0, 0.0)
    dense = bl.scratch_field(direction, region, 10.0, seed=5)
    sparse = bl.scratch_field(direction, region, 10.0, seed=5, density=0.005)
    assert (sparse > 0.2).mean() < 0.5 * (dense > 0.2).mean()  # coating shows between the scratches
    assert sparse.max() > 0.9  # the scratches that remain keep their full contrast
    assert not bl.scratch_field(direction, region, 10.0, density=0.0).any()


def test_render_worn_look_overrides():
    import typer
    from mitsuba_video.cli import _blend_overrides

    base = bl.BlendParams()
    assert _blend_overrides(base, 0.0, 0.0, base.scratch_density, None, None) is base
    p = _blend_overrides(base, 0.6, 0.0, 0.01, "0.35,0.95", 0.9)
    assert (p.scratches, p.burnish, p.scratch_density) == (0.6, 0.0, 0.01)
    assert (p.edge0, p.edge1, p.height_strength) == (0.35, 0.95, 0.9)
    for bad in ("0.5", "0.9,0.2", "a,b"):
        try:
            _blend_overrides(base, 0.0, 0.0, base.scratch_density, bad, None)
        except typer.BadParameter:
            continue
        raise AssertionError(bad)


def test_edge_wear_moves_worn_rims_ahead_and_leaves_unworn_texels():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    hole = (np.hypot(yy - 32, xx - 32) < 10).astype(np.float32)  # a pit: its rim is the steep relief
    base, worn = _maps(n, 0.2, 0.6), _maps(n, 0.8, 0.3)
    base.height, worn.height = 0.7 - 0.5 * hole, 0.7 - 0.5 * hole
    edge = bl.edge_field(base.height)
    assert edge[32, 22] > 0.5 and edge[32, 32] < 0.05 and edge[5, 5] < 0.05  # rim steep, pit floor and plateau flat
    wear = np.zeros((n, n), np.float32)
    wear[:, 16:48] = 0.3
    stack = bl.VariantStack.pair(base, worn)
    plain = bl.blend_stack(wear, stack, bl.BlendParams(height_strength=0.0))
    edged = bl.blend_stack(wear, stack, bl.BlendParams(height_strength=0.0, edge_wear=2.0))
    assert edged.albedo[32, 22, 0] > plain.albedo[32, 22, 0] + 0.05  # the rim reads more worn
    assert np.allclose(edged.albedo[5, 30], plain.albedo[5, 30], atol=1e-4)  # flat worn texel: unchanged
    assert np.allclose(edged.albedo[:, :8], base.albedo[:, :8])  # unworn texels stay base
    assert bl.BlendParams(edge_wear=1.0).wide
