"""Composite-time annotations: labels on the bodies in the image, wear-panel titles, the field's physical full scale and
the header wording.

They are read from a JSON file (``--annotate``) when a frame is composited, so they never change the path-traced
pixels or the resume signature: an existing render is re-annotated with ``--resume --recompose``. Example::

    {"body_labels": {"0": "H 100", "1": "H 200"},
     "panel_titles": {"0": "H 100 enamel"},
     "field_label": "wear depth", "field_max": "5 mm",
     "mover_label": ""}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .camera import CameraPose, world_corners
from .model import Body


@dataclass
class Annotations:
    body_labels: dict[int, str] = field(default_factory=dict)  # tag drawn on the body in the image
    panel_titles: dict[int, str] = field(default_factory=dict)  # wear-panel title (default: the body label)
    field_label: str = ""  # colour bar caption, e.g. "wear depth"
    field_max: str = ""  # colour bar top, e.g. "5 mm"
    # The header's travel line names the moving body with this label; "" leaves the line out (a bare "body 4" means
    # nothing to a viewer).
    mover_label: str = ""
    # False: no header lines on the picture (tf video draws its own title, clock and playback speed there).
    header: bool = True
    # Physical size of field 1.0 (mm for wear: WEAR_DEPTH_MAX_M x wear.rangeScale). > 0 adds each labelled body's
    # measured depth to its tag, from that frame's field (the recording's own data, not a preset number).
    field_scale: float = 0.0
    field_unit: str = "mm"
    # Show the depth only on the last N composited frames (0 = always): the number is read when it is final.
    depth_tail_frames: int = 0
    # Motion arrows on the moving bodies, from the recorded poses (left out when more than max_arrows bodies move:
    # twenty arrows would hide the scene).
    arrows: bool = False
    max_arrows: int = 8
    # Each contact cluster's world AABB (from the recording's per-frame cluster boxes) drawn as a projected wireframe:
    # where the simulation is resolving contact this frame.
    cluster_boxes: bool = False

    @classmethod
    def load(cls, path: Path | str | None) -> Annotations | None:
        if not path:
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {"body_labels", "panel_titles", "field_label", "field_max", "mover_label", "header", "field_scale",
                 "field_unit", "arrows", "max_arrows", "depth_tail_frames", "cluster_boxes"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown annotation keys {sorted(unknown)}; known: {sorted(known)}")
        return cls(
            body_labels={int(k): str(v) for k, v in data.get("body_labels", {}).items()},
            panel_titles={int(k): str(v) for k, v in data.get("panel_titles", {}).items()},
            field_label=str(data.get("field_label", "")),
            field_max=str(data.get("field_max", "")),
            mover_label=str(data.get("mover_label", "")),
            header=bool(data.get("header", True)),
            field_scale=float(data.get("field_scale", 0.0) or 0.0),
            field_unit=str(data.get("field_unit", "mm")),
            arrows=bool(data.get("arrows", False)),
            depth_tail_frames=int(data.get("depth_tail_frames", 0) or 0),
            max_arrows=int(data.get("max_arrows", 8)),
            cluster_boxes=bool(data.get("cluster_boxes", False)),
        )

    def panel_title(self, index: int, field_caption: str) -> str | None:
        """The wear panel's title for body ``index``, or None to keep the renderer's default."""
        if index in self.panel_titles:
            return self.panel_titles[index]
        if index in self.body_labels:  # the colour bar names the field; two-column panels leave ~25 characters
            return self.body_labels[index]
        return None


def project(points: np.ndarray, pose: CameraPose, fov_x_deg: float, width: int, height: int) -> np.ndarray:
    """Pixel coordinates (N, 3): x, y and camera depth of world ``points`` (N, 3) for Mitsuba's perspective sensor
    with ``to_world = look_at(eye, target, up)`` and a horizontal field of view (``fov_axis = "x"``). Mitsuba's
    look_at puts camera +x on the image's left, so image x grows along -left."""
    forward = np.asarray(pose.target, float) - np.asarray(pose.eye, float)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray(pose.up, float), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rel = np.asarray(points, float) - np.asarray(pose.eye, float)
    depth = rel @ forward
    safe = np.where(depth > 1e-9, depth, 1e-9)
    tan_x = math.tan(math.radians(fov_x_deg) / 2.0)
    tan_y = tan_x * height / width
    px = width / 2.0 - (rel @ left) / safe / tan_x * (width / 2.0)
    py = height / 2.0 - (rel @ up) / safe / tan_y * (height / 2.0)
    return np.stack([px, py, depth], axis=1)


@dataclass
class Callout:
    text: str
    box: tuple[float, float, float, float]  # projected bounding box of the body (x0, y0, x1, y1)
    fixed: bool  # fixed bodies (ground tiles) get the tag on the tile, moving ones above them
    value: str = ""  # a measured number under the label (the body's track depth this frame)
    anchor: tuple[float, float] | None = None  # fixed bodies: where the tag's top-left goes (on the tile)


def format_depth(mm: float, full_scale_mm: float = 0.0) -> str:
    """A depth for a tag, in one unit for the whole video: nanometres when the atlas's full scale is under 0.3 um,
    micrometres when it is under 2 mm (or, without one, the value is), else millimetres (the hardness sweep's H 50 tile read "0.14 mm" beside "35 µm",
    2026-11-06); an em dash at the atlas clamp (a clamped value is a recording defect, never printed as a
    measurement)."""
    if full_scale_mm > 0 and mm >= 0.98 * full_scale_mm:
        return "—"
    scale = full_scale_mm if full_scale_mm > 0 else mm
    if scale < 3e-4:  # under 0.3 um full scale: nanometres ("0.00025 um" tags were unreadable, review 2026-11-11)
        if mm < 1e-7:  # under 0.1 nm: less than an atom, not a depth
            return "< 0.1 nm"
        return f"{mm * 1e6:.2g} nm" if mm < 1e-5 else f"{mm * 1e6:.0f} nm"
    micro = scale < 2.0
    if micro:
        return f"{mm * 1000:.2g} µm" if mm < 0.01 else f"{mm * 1000:.0f} µm"
    return f"{mm:.2f} mm" if mm < 1 else f"{mm:.1f} mm"


def track_depth(field: np.ndarray | None, scale: float, threshold: float = 0.5 / 255.0) -> float | None:
    """The depth of a body's worn track this frame: the 95th percentile of its worn texels (a robust "deepest part of
    the track", not one stray texel) times the field's physical scale. None before anything wears."""
    if field is None or scale <= 0:
        return None
    worn = field[field > threshold]
    if worn.size < 16:
        return None
    return float(np.percentile(worn, 95)) * scale


def callouts(bodies: list[Body], positions: np.ndarray | None, orientations: np.ndarray | None, pose: CameraPose,
             fov_x_deg: float, size: tuple[int, int], labels: dict[int, str],
             values: dict[int, str] | None = None,
             settled: tuple[np.ndarray, np.ndarray] | None = None) -> list[Callout]:  # fmt: skip
    """One callout per labelled body that is in front of the camera and at least partly in the image.

    ``settled`` (positions, orientations at the clip's last frame): a fixed body's tag also avoids where the moving
    bodies end up, so it is placed once and does not jump when a body arrives at its corner."""
    if positions is None or orientations is None or not labels:
        return []
    width, height = size
    # Where the moving bodies are on screen: a fixed body's tag goes to a spot none of them covers (the four-materials
    # overhead put the plate's tag on the steel cube, which had stopped at the plate's far edge, 2026-11-12).
    movers = []
    for pos, quat in [(positions, orientations)] + ([settled] if settled is not None else []):
        for body in bodies:
            if body.fixed or body.index >= len(pos):
                continue
            pts = project(world_corners(body, pos[body.index], quat[body.index]), pose, fov_x_deg, width, height)
            if (pts[:, 2] > 0).all():
                movers.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    margin = 0.02 * height
    char_w, line_h = 0.62 * max(14, height // 34), 1.3 * max(14, height // 34)  # the tag font (video.compose_frame)

    def covered(x: float, y: float, text: str) -> bool:
        """Whether a tag whose top-left corner is (x, y) would lie over a moving body (two lines: name and depth)."""
        tx1, ty1 = x + char_w * len(text) + line_h, y + 2.2 * line_h
        return any(x < bx1 + margin and tx1 > bx0 - margin and y < by1 + margin and ty1 > by0 - margin
                   for bx0, by0, bx1, by1 in movers)  # fmt: skip

    out = []
    for body in bodies:  # pose arrays are indexed by body index, as scene.set_poses reads them
        text = labels.get(body.index)
        if not text or body.index >= len(positions):
            continue
        corners = world_corners(body, positions[body.index], orientations[body.index])
        pts = project(corners, pose, fov_x_deg, width, height)
        if (pts[:, 2] <= 0).any():
            continue
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        # the body's centre must be in frame: a macro on one tile otherwise collects its neighbours' tags, clamped
        # into the corner (2026-11-05)
        if not (0 <= cx <= width and 0 <= cy <= height):
            continue
        anchor = None
        if body.fixed:
            # the topmost projected corner, pulled a quarter of the way to the centre: on the tile in every view
            # (the bounding box's top-left corner fell off a tile seen in perspective, 2026-11-05)
            # Then the bottom, left and right corners, the first one no moving body covers.
            picks = [np.argmin(pts[:, 1] * 1e6 + pts[:, 0]), np.argmax(pts[:, 1]), np.argmin(pts[:, 0]),
                     np.argmax(pts[:, 0])]  # fmt: skip
            spots = [(float(pts[k, 0] + 0.25 * (cx - pts[k, 0])), float(pts[k, 1] + 0.25 * (cy - pts[k, 1])))
                     for k in picks]  # fmt: skip
            inside = [p for p in spots if 0 <= p[0] <= width and 0 <= p[1] <= height]
            anchor = next((p for p in inside if not covered(*p, text)), spots[0])
        out.append(Callout(text, (x0, y0, x1, y1), body.fixed, (values or {}).get(body.index, ""), anchor))
    return out


def draw_callouts(image: Image.Image, items: list[Callout], font, *, reserved_top: int = 0) -> None:
    """Tags on a translucent dark plate. ``reserved_top`` keeps tags below the header lines."""
    if not items:
        return
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = max(4, font.size // 3)
    width, height = image.size
    placed: list[tuple[float, float, float, float]] = []
    from .video import font as face_of

    value_font = face_of(max(10, int(font.size * 0.8)), True)
    for item in items:
        left, top, right, bottom = draw.textbbox((0, 0), item.text, font=font)
        w, h = right - left + 2 * pad, bottom - top + 2 * pad
        line_h = h - 2 * pad
        if item.value:
            v_left, v_top, v_right, v_bottom = draw.textbbox((0, 0), item.value, font=value_font)
            w = max(w, v_right - v_left + 2 * pad)
            h += v_bottom - v_top + pad // 2
        x0, y0, x1, _y1 = item.box
        if item.fixed and item.anchor is not None:  # on the tile, toward its far edge
            x, y = item.anchor
        elif item.fixed:
            x, y = max(x0, 0) + pad * 2, max(y0, 0) + pad * 2
        else:  # centred above the body
            x, y = (x0 + x1) / 2 - w / 2, y0 - h - pad
        x = min(max(x, pad), width - w - pad)
        y = min(max(y, reserved_top + pad), height - h - pad)
        for px0, py0, px1, py1 in placed:  # nudge down past an earlier tag it would cover
            if x < px1 and x + w > px0 and y < py1 and y + h > py0:
                y = py1 + pad
        placed.append((x, y, x + w, y + h))
        # plain type with a white halo, no box (advisor, 2026-11-05): the label sits on the object like a caption
        halo = max(2, font.size // 10)
        draw.text((x + pad - left, y + pad - top), item.text, font=font, fill=(22, 22, 26, 255),
                  stroke_width=halo, stroke_fill=(255, 255, 255, 235))  # fmt: skip
        if item.value:  # the measured number, in the accent green of the video chrome
            draw.text((x + pad - v_left, y + pad + line_h + pad // 2 - v_top), item.value, font=value_font,
                      fill=(28, 132, 72, 255), stroke_width=halo, stroke_fill=(255, 255, 255, 235))  # fmt: skip
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


@dataclass
class Arrow:
    start: tuple[float, float]
    end: tuple[float, float]


def motion_arrows(bodies: list[Body], positions: np.ndarray | None, orientations: np.ndarray | None,
                  previous: np.ndarray | None, dt: float, pose: CameraPose, fov_x_deg: float, size: tuple[int, int], *, min_speed: float = 0.03,
                  max_arrows: int = 8) -> list[Arrow]:  # fmt: skip
    """One arrow per moving body along its velocity (finite difference of the recorded positions over ``dt``), from
    the edge of its projected box outwards, longer for faster bodies. Empty when nothing moves, the poses are missing,
    or more than ``max_arrows`` bodies move."""
    if positions is None or orientations is None or previous is None or dt <= 0:
        return []
    width, height = size
    moving = []
    for body in bodies:
        i = body.index
        if body.fixed or i >= len(positions) or i >= len(previous):
            continue
        velocity = (np.asarray(positions[i], float) - np.asarray(previous[i], float)) / dt
        speed = float(np.linalg.norm(velocity))
        if speed >= min_speed:
            moving.append((body, velocity, speed))
    if not moving or len(moving) > max_arrows:
        return []
    out = []
    for body, velocity, speed in moving:
        centre = np.asarray(positions[body.index], float)
        ahead = centre + velocity / speed * 0.05
        pts = project(np.stack([centre, ahead]), pose, fov_x_deg, width, height)
        if (pts[:, 2] <= 0).any():
            continue
        direction = pts[1, :2] - pts[0, :2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue  # moving straight at the camera: no readable arrow
        direction /= norm
        box = project(world_corners(body, centre, orientations[body.index]), pose, fov_x_deg, width, height)
        half = float(np.max(np.abs((box[:, :2] - pts[0, :2]) @ direction)))
        length = height * min(0.11, 0.035 + 0.05 * speed)
        start = pts[0, :2] + direction * (half + height * 0.012)
        end = start + direction * length
        out.append(Arrow((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))))
    return out


def draw_arrows(image: Image.Image, arrows: list[Arrow]) -> None:
    """Solid green arrows with a thin white edge, the accent of the video chrome."""
    if not arrows:
        return
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    shaft = max(5, image.height // 110)
    for arrow in arrows:
        start, end = np.array(arrow.start), np.array(arrow.end)
        d = end - start
        d /= max(1e-6, float(np.linalg.norm(d)))
        n = np.array([-d[1], d[0]])
        head = shaft * 3.2
        base = end - d * head
        tri = [tuple(end), tuple(base + n * head * 0.6), tuple(base - n * head * 0.6)]
        for colour, grow in (((255, 255, 255, 235), 2), ((28, 132, 72, 255), 0)):
            draw.line([tuple(start), tuple(base)], fill=colour, width=shaft + 2 * grow)
            if grow:
                draw.polygon([tuple(np.array(p) + (np.array(p) - base) * 0.12) for p in tri], fill=colour)
            else:
                draw.polygon(tri, fill=colour)
    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))


#: The 12 edges of a box as pairs of corner indices; corner k has bit i set when it sits at the max along axis i.
_BOX_EDGES = [(a, a | 1 << i) for a in range(8) for i in range(3) if not a & 1 << i]

#: Smallest half-extent a drawn cluster box keeps (m): a face contact's box is 1 mm thick, a point contact's a dot.
MIN_BOX_HALF_EXTENT = 0.004


def cluster_aabbs(boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """World axis-aligned bounds ``(lo [3], hi [3])`` of oriented boxes ``(centre, axes one per row, half-extents)``:
    the box's half-size along world axis j is the sum over its axes of |axis_j| x extent."""
    out = []
    for centre, axes, extents in boxes:
        half = np.abs(np.asarray(axes, float)).T @ np.abs(np.asarray(extents, float))
        half = np.maximum(half, MIN_BOX_HALF_EXTENT)
        out.append((np.asarray(centre, float) - half, np.asarray(centre, float) + half))
    return out


def cluster_box_segments(boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]], pose: CameraPose, fov_x_deg: float,
                         size: tuple[int, int]) -> list[tuple[tuple[float, float], tuple[float, float]]]:  # fmt: skip
    """Image-space line segments of every cluster's world AABB (:func:`cluster_aabbs`), edges behind the camera left
    out."""
    width, height = size
    segments = []
    for lo, hi in cluster_aabbs(boxes):
        corners = np.array([[hi[i] if k & 1 << i else lo[i] for i in range(3)] for k in range(8)])
        pts = project(corners, pose, fov_x_deg, width, height)
        for a, b in _BOX_EDGES:
            if pts[a, 2] > 0 and pts[b, 2] > 0:
                segments.append(((float(pts[a, 0]), float(pts[a, 1])), (float(pts[b, 0]), float(pts[b, 1]))))
    return segments


def draw_cluster_boxes(image: Image.Image, segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> None:
    """Thin amber lines with a dark edge, readable on light and dark grounds."""
    if not segments:
        return
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width = max(1, image.height // 400)
    for colour, grow in (((20, 20, 20, 170), 2), ((255, 176, 0, 255), 0)):
        for a, b in segments:
            draw.line([a, b], fill=colour, width=width + grow)
    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))
