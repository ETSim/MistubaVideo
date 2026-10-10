"""Camera paths: fixed framing, tracking a body, or orbiting the scene.

All framing is derived from the recorded trajectories, so a scene at any physical scale (a 0.0025-scaled tray or a
4 m plane) is framed without hand-tuned distances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import Body
from .source import Source
from .transforms import quaternion_to_matrix

UP_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class CameraPose:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray


@dataclass
class CameraSettings:
    mode: str = "fixed"  # fixed | track | orbit
    follow: int | None = None  # body index for track mode (default: first dynamic body)
    azimuth: float = -55.0  # degrees around the up axis
    elevation: float = 28.0  # degrees above the horizon
    fov: float = 35.0  # horizontal field of view, degrees
    zoom: float = 1.0  # > 1 moves closer
    orbit_degrees: float = 120.0  # total sweep over the video in orbit mode
    smoothing: int = 9  # moving-average window (frames) for the track target
    up: str = "y"
    frame_all: bool = False  # frame every body instead of only dynamic ones


def up_basis(axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, horizontal e1, horizontal e2) as a right-handed basis."""
    up = np.zeros(3)
    up[UP_AXES[axis]] = 1.0
    e1 = np.roll(up, 1)  # y-up -> z, z-up -> x
    e2 = np.cross(up, e1)
    return up, e1, e2


def world_corners(body: Body, position: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    lo, hi = body.local_bounds
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return corners @ quaternion_to_matrix(q_wxyz).T + position


def trajectory_bounds(rec: Source, frames, bodies: list[Body]) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for frame in frames:
        pos, quat = rec.pose(frame)
        for body in bodies:
            if body.index < len(pos):
                pts.append(world_corners(body, pos[body.index], quat[body.index]))
    allpts = np.concatenate(pts) if pts else np.zeros((1, 3))
    return allpts.min(axis=0), allpts.max(axis=0)


def scene_bounds(rec: Source, frames) -> tuple[np.ndarray, np.ndarray]:
    return trajectory_bounds(rec, frames, rec.bodies)


def _fit_distance(radius: float, fov_deg: float, aspect: float) -> float:
    half_h = math.radians(fov_deg) * 0.5
    half_v = math.atan(math.tan(half_h) / max(aspect, 1e-6))
    return radius / math.sin(min(half_h, half_v))


def _eye(target: np.ndarray, distance: float, az_deg: float, el_deg: float, axis: str) -> np.ndarray:
    up, e1, e2 = up_basis(axis)
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = math.cos(el) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(el) * up
    return target + distance * d


def _smooth(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) < 3:
        return points
    window = min(window, len(points))
    pad = window // 2
    padded = np.pad(points, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)], axis=1)


def camera_path(rec: Source, frames, settings: CameraSettings, aspect: float) -> list[CameraPose]:
    """One pose per frame in ``frames``. Framing uses the whole recording, so a preview of a few frames is shot
    exactly like the full render."""
    up = up_basis(settings.up)[0]
    dynamic = [b for b in rec.bodies if not b.fixed] or rec.bodies
    framed = rec.bodies if settings.frame_all else dynamic
    lo, hi = trajectory_bounds(rec, rec.frames, framed)
    center = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6)

    if settings.mode == "track":
        follow = next((b for b in rec.bodies if b.index == settings.follow), None) or dynamic[0]
        centers = []
        for frame in frames:
            pos, quat = rec.pose(frame)
            c = world_corners(follow, pos[follow.index], quat[follow.index])
            centers.append(0.5 * (c.min(axis=0) + c.max(axis=0)))
        targets = _smooth(np.asarray(centers), settings.smoothing)
        lo_b, hi_b = follow.local_bounds
        track_radius = 3.5 * 0.5 * float(np.linalg.norm(hi_b - lo_b))
        distance = _fit_distance(track_radius, settings.fov, aspect) / settings.zoom
        return [
            CameraPose(_eye(t, distance, settings.azimuth, settings.elevation, settings.up), t, up) for t in targets
        ]

    distance = 1.08 * _fit_distance(radius, settings.fov, aspect) / settings.zoom
    poses = []
    n = max(len(frames) - 1, 1)
    for i in range(len(frames)):
        az = settings.azimuth + (settings.orbit_degrees * i / n if settings.mode == "orbit" else 0.0)
        poses.append(CameraPose(_eye(center, distance, az, settings.elevation, settings.up), center.copy(), up))
    return poses
