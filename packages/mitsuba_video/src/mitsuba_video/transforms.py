"""Rigid transforms: (w, x, y, z) quaternions and ``T(x) * R(q) * S(scale)`` model matrices."""

from __future__ import annotations

import numpy as np


def quaternion_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix for a (w, x, y, z) quaternion (Eigen storage order)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def body_world_transform(position: np.ndarray, q_wxyz: np.ndarray, scale: float) -> np.ndarray:
    """4x4 model matrix ``T(x) * R(q) * S(scale)``."""
    m = np.eye(4)
    m[:3, :3] = quaternion_to_matrix(q_wxyz) * scale
    m[:3, 3] = position
    return m
