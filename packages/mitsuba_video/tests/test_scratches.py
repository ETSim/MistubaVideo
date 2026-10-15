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
