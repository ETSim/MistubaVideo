"""Geometric displacement (mitsuba_video.displace): subdivision, seam welding, UV orientation, inward offsets."""

import numpy as np

from mitsuba_video.displace import DisplacedMesh, sample_height, subdivide, subdivision_levels, weld_groups


def _split_cube():
    """A unit cube with every face its own 4 vertices (hard normals) and its own [0,1] UV square."""
    verts, norms, uvs, faces = [], [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            u, v = np.roll(np.eye(3), 1, axis=0)[axis], np.roll(np.eye(3), 2, axis=0)[axis]
            base = len(verts)
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                verts.append(0.5 * n + (a - 0.5) * u + (b - 0.5) * v)
                norms.append(n)
                uvs.append((a, b))
            faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(verts), np.array(norms), np.array(uvs, float), np.array(faces, np.uint32)


def test_subdivide_shares_edge_midpoints():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    n = np.tile([0.0, 0.0, 1.0], (4, 1))
    uv = v[:, :2].copy()
    f = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    v2, n2, uv2, f2 = subdivide(v, n, uv, f)
    assert len(f2) == 8
    assert len(v2) == 4 + 5  # 5 unique edges, the diagonal shared
    np.testing.assert_allclose(uv2, v2[:, :2])
    np.testing.assert_allclose(np.linalg.norm(n2, axis=1), 1.0)


def test_levels_respect_budget_and_target():
    v, _, _, f = _split_cube()
    # median of the first edges (sides 1, diagonals 1.41) is 1.21 -> 0.60 -> 0.30
    assert subdivision_levels(v, f, target_edge=0.35, max_triangles=10_000) == 2
    assert subdivision_levels(v, f, target_edge=1e-6, max_triangles=12 * 4**3) == 3
    assert subdivision_levels(v, f, target_edge=0.0) == 0


def test_sample_height_uses_image_rows_top_is_v1():
    h = np.zeros((8, 8), np.float32)
    h[0, :] = 1.0  # row 0 = V = 1
    uv = np.array([[0.5, 1.0 - 0.5 / 8], [0.5, 0.5 / 8]])
    np.testing.assert_allclose(sample_height(h, uv), [1.0, 0.0], atol=1e-6)


def test_cube_stays_closed_and_moves_inward():
    v, n, uv, f = _split_cube()
    rng = np.random.default_rng(0)
    height = rng.random((32, 32)).astype(np.float32)
    mesh = DisplacedMesh.build(v, n, uv, f, amplitude=0.01, texture_size=(32, 32), max_triangles=5_000)
    assert mesh.levels >= 2
    out = mesh.displaced(height)
    # Every copy of a position (cube edges, corners) lands on the same displaced point: no cracks.
    for g in np.unique(mesh.group):
        members = out[mesh.group == g]
        np.testing.assert_allclose(members, np.broadcast_to(members[0], members.shape), atol=1e-12)
    # Inward only, and never deeper than the amplitude.
    radial = np.einsum("ij,ij->i", out - mesh.vertices, mesh.group_normal[mesh.group])
    assert radial.max() <= 1e-12
    assert radial.min() >= -0.01 - 1e-12
    assert mesh.top is not None and 0.9 < mesh.top <= 1.0


def test_flat_height_leaves_the_mesh_in_place_and_amplitude_is_capped():
    v, n, uv, f = _split_cube()
    mesh = DisplacedMesh.build(v, n, uv, f, amplitude=10.0, texture_size=(4, 4))
    assert mesh.amplitude <= 0.02 * np.sqrt(3.0) + 1e-9  # 2 % of the diagonal
    np.testing.assert_allclose(mesh.displaced(np.full((4, 4), 0.5, np.float32)), mesh.vertices)
    np.testing.assert_allclose(mesh.displaced(None), mesh.vertices)


def test_weld_groups_cancelling_normals_keep_one():
    v = np.zeros((2, 3))
    n = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    group, normal, count = weld_groups(v, n)
    assert group[0] == group[1] and count[0] == 2
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0)
