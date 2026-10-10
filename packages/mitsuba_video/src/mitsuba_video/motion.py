"""Motion summaries for the video header: cumulative travel of one body, optionally split into passes."""

from __future__ import annotations

import numpy as np

from .model import Body
from .source import Source


def travel_by_frame(
    source: Source, follow: int | None = None, detect_resets: bool = False
) -> tuple[Body | None, dict[str, tuple[float, int]]]:
    """Cumulative distance and pass number of the tracked (or first dynamic) body, per frame key.

    With ``detect_resets`` (sources whose scenarios teleport a body back to its start), a step far above the median
    step counts as a reset: it starts a new pass instead of adding to the distance.
    """
    mover = next((b for b in source.bodies if b.index == follow), None) or next(
        (b for b in source.bodies if not b.fixed), None
    )
    if mover is None or not source.frames:
        return None, {}
    track = np.array([source.pose(f)[0][mover.index] for f in source.frames])
    steps = np.r_[0.0, np.linalg.norm(np.diff(track, axis=0), axis=1)]
    if detect_resets:
        is_reset = steps > max(10.0 * float(np.median(steps)), 1e-9)
    else:
        is_reset = np.zeros_like(steps, dtype=bool)
    slid = np.cumsum(np.where(is_reset, 0.0, steps))
    passes = 1 + np.cumsum(is_reset)
    return mover, {f.key: (float(slid[i]), int(passes[i])) for i, f in enumerate(source.frames)}
