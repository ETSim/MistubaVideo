"""Small, static height-map displacement for existing mesh vertices."""

from dataclasses import replace

import numpy as np

from .maps import NormalOptions, fit_size


def displace_body(body, scale: float, max_texture=2048, log=None):
    """Offset vertices by ``scale * (height - 0.5)`` in world units along their authored normals.

    Uses the base height map once, before posing. Mesh density controls silhouette detail; this does not subdivide.
    """
    if scale == 0:
        return body
    if not np.isfinite(scale) or body.scale <= 0:
        raise ValueError("displacement scale must be finite and body scale must be positive")
    size = fit_size(body.material.preferred_size(), max_texture)
    base, _ = body.material.build(size, NormalOptions(), log)
    if base.height is None:
        if log:
            log(f"displace  {body.name}: no base height map; geometry unchanged")
        return body
    height = np.asarray(base.height)
    h, w = height.shape
    uv = np.clip(body.uvs, 0, 1)
    # Pixel centres use image convention, opposite to OBJ's V axis.
    x, y = uv[:, 0] * (w - 1), (1 - uv[:, 1]) * (h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = x - x0, y - y0
    samples = ((1 - ty) * ((1 - tx) * height[y0, x0] + tx * height[y0, x1]) +
               ty * ((1 - tx) * height[y1, x0] + tx * height[y1, x1]))
    normals = body.normals / np.maximum(np.linalg.norm(body.normals, axis=1, keepdims=True), 1e-12)
    vertices = body.vertices + normals * ((samples - 0.5) * scale / body.scale)[:, None]
    tri = vertices[body.faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, body.faces[:, corner], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), body.normals)
    return replace(body, vertices=np.float32(vertices), normals=np.float32(normals))
