"""Tone mapping, the side panel (UV wear atlas + colour bar), and ffmpeg encoding."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import colormap, linear_to_srgb

BACKGROUND = (18, 18, 22)
TEXT = (232, 232, 236)
MUTED = (150, 150, 160)


def tonemap(rgb: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Linear HDR -> 8-bit sRGB through the ACES filmic fit (Narkowicz 2015)."""
    x = np.maximum(rgb, 0.0) * (2.0**exposure)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    return (linear_to_srgb(mapped) * 255.0 + 0.5).astype(np.uint8)


def write_exr(path: Path, rgb: np.ndarray) -> None:
    import mitsuba as mi

    mi.Bitmap(np.ascontiguousarray(rgb, dtype=np.float32)).write(str(path))


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@dataclass
class PanelSource:
    body_index: int
    title: str
    roi: tuple[int, int, int, int]  # (x0, y0, x1, y1) in atlas pixels, crop applied to every frame


def wear_roi(
    wear: np.ndarray | None, margin: float = 0.12, threshold: float = 0.5 / 255.0
) -> tuple[int, int, int, int]:
    """Bounding box of written wear (> one 8-bit step), padded and squared; full atlas when nothing is worn."""
    if wear is None:
        return (0, 0, 1, 1)
    h, w = wear.shape
    ys, xs = np.nonzero(wear > threshold)
    if len(xs) == 0:
        return (0, 0, w, h)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    side = int(min(max(side * (1.0 + 2.0 * margin), 32), max(w, h)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = int(np.clip(cx - side // 2, 0, max(w - side, 0)))
    y0 = int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    return (x0, y0, min(x0 + side, w), min(y0 + side, h))


def _max_pool(arr: np.ndarray, size: int) -> np.ndarray:
    """Block-max downsample so a sparse wear footprint stays visible in a small panel (values are not rescaled)."""
    factor = max(1, int(np.ceil(max(arr.shape) / size)))
    if factor == 1:
        return arr
    h, w = arr.shape
    padded = np.zeros((-(-h // factor) * factor, -(-w // factor) * factor), arr.dtype)
    padded[:h, :w] = arr
    return padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor).max(axis=(1, 3))


def wear_panel_image(
    wear: np.ndarray | None,
    roi: tuple[int, int, int, int],
    size: int,
    hi: float = 1.0,
    threshold: float = 0.5 / 255.0,
) -> Image.Image:
    """Wear atlas crop as inferno over dark grey (unworn texels stay grey, so the footprint reads clearly)."""
    if wear is None:
        return Image.new("RGB", (size, size), (40, 40, 46))
    x0, y0, x1, y1 = roi
    crop = _max_pool(wear[y0:y1, x0:x1], size)
    rgb = np.empty(crop.shape + (3,), np.float32)
    rgb[:] = np.float32([40, 40, 46]) / 255.0
    mask = crop > threshold
    rgb[mask] = colormap(crop[mask], hi=hi)
    img = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))
    return img.resize((size, size), Image.Resampling.NEAREST)


def colorbar(width: int, height: int) -> Image.Image:
    ramp = colormap(np.linspace(0.0, 1.0, width, dtype=np.float32))
    bar = np.broadcast_to(ramp[None], (height, width, 3))
    return Image.fromarray((bar * 255.0 + 0.5).astype(np.uint8))


def _fit(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> str:
    if draw.textlength(text, font=fnt) <= width:
        return text
    while text and draw.textlength(text + "...", font=fnt) > width:
        text = text[:-1]
    return text + "..."


MIN_PANEL_WIDTH = 48  # px inside the side column; below this compose_frame omits the panels
SERIES = (57, 135, 229)  # dataviz reference palette, categorical slot 1 (dark mode); validated vs the dark surface
GRID = (44, 44, 52)  # one step off the panel surface


def _nice_step(span: float, target_ticks: int = 4) -> float:
    raw = max(span, 1e-12) / target_ticks
    mag = 10.0 ** np.floor(np.log10(raw))
    return float(next(m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * mag >= raw))


def _fmt(v: float, step: float) -> str:
    decimals = max(0, int(-np.floor(np.log10(step)))) if step < 1 else 0
    return f"{v:.{decimals}f}"


@dataclass
class AreaSeries:
    """Worn area of one body over the rendered frames (one value per frame)."""

    title: str
    times: np.ndarray
    values: np.ndarray
    unit: str


def area_chart(series: AreaSeries, index: int, width: int, height: int) -> Image.Image:
    """Line chart of worn area vs time, drawn up to ``index`` on fixed axes (2x supersampled for clean lines).

    Single series, so no legend: the title names it, the end dot carries a direct value label, gridlines are
    hairline and recessive, and text uses text colours only.
    """
    ss = 2
    w, h = width * ss, height * ss
    img = Image.new("RGB", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(img)
    f_title, f_tick = font(max(11, height // 13) * ss), font(max(9, height // 17) * ss)
    pad = 6 * ss

    t0, t1 = float(series.times[0]), float(series.times[-1])
    vmax = float(np.max(series.values)) if len(series.values) else 0.0
    ystep = _nice_step(vmax if vmax > 0 else 1.0)
    ytop = ystep * max(1, int(np.ceil(vmax / ystep - 1e-9)))
    xstep = _nice_step(max(t1 - t0, 1e-6))

    draw.text((0, 0), series.title, font=f_title, fill=TEXT)
    top = f_title.size + 2 * pad
    ylabels = [_fmt(ystep * k, ystep) for k in range(int(round(ytop / ystep)) + 1)]
    left = int(max(draw.textlength(t, font=f_tick) for t in ylabels)) + 2 * pad
    bottom = h - f_tick.size - 2 * pad
    right = w - pad

    def px(t: float, v: float) -> tuple[float, float]:
        x = left + (t - t0) / max(t1 - t0, 1e-12) * (right - left)
        y = bottom - v / ytop * (bottom - top)
        return x, y

    for k, label in enumerate(ylabels):
        _, y = px(t0, ystep * k)
        draw.line([(left, y), (right, y)], fill=GRID, width=ss)
        draw.text((left - pad - draw.textlength(label, font=f_tick), y - f_tick.size / 2), label, font=f_tick,
                  fill=MUTED)  # fmt: skip
    t = np.ceil(t0 / xstep) * xstep
    while t <= t1 + 1e-9:
        x, _ = px(t, 0.0)
        label = _fmt(t, xstep) + (" s" if t + xstep > t1 + 1e-9 else "")
        draw.text((min(x - draw.textlength(label, font=f_tick) / 2, w - draw.textlength(label, font=f_tick)),
                   bottom + pad), label, font=f_tick, fill=MUTED)  # fmt: skip
        t += xstep

    last = min(index, len(series.values) - 1)
    pts = [px(float(series.times[i]), float(series.values[i])) for i in range(last + 1)]
    if len(pts) > 1:
        draw.line(pts, fill=SERIES, width=2 * ss, joint="curve")
    ex, ey = pts[-1]
    r = 4 * ss
    draw.ellipse([ex - r - 2 * ss, ey - r - 2 * ss, ex + r + 2 * ss, ey + r + 2 * ss], fill=BACKGROUND)  # ring
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=SERIES)
    value = f"{float(series.values[last]):.3g} {series.unit}"
    tw = draw.textlength(value, font=f_tick)
    lx = min(max(ex - tw / 2, left), right - tw)
    ly = ey - r - 3 * ss - f_tick.size
    ly = ly if ly > top else ey + r + 3 * ss
    draw.text((lx, ly), value, font=f_tick, fill=TEXT)
    return img.resize((width, height), Image.Resampling.LANCZOS)


def title_card(size: tuple[int, int], title: str, lines: list[str]) -> Image.Image:
    """Opening card at the video's frame size: title, a thin rule, then secondary lines."""
    w, h = size
    img = Image.new("RGB", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(img)
    f_title, f_line = font(max(24, h // 14)), font(max(14, h // 34))
    x = int(w * 0.08)
    block = f_title.size + h // 30 + len(lines) * int(f_line.size * 1.6)
    y = (h - block) // 2
    draw.text((x, y), title, font=f_title, fill=TEXT)
    y = draw.textbbox((x, y), title, font=f_title)[3] + h // 40  # below descenders
    draw.line([(x, y), (x + int(w * 0.18), y)], fill=SERIES, width=max(2, h // 400))
    y += h // 60
    for line in lines:
        draw.text((x, y), line, font=f_line, fill=MUTED)
        y += int(f_line.size * 1.6)
    return img


def compose_frame(
    beauty: np.ndarray,
    panels: list[tuple[PanelSource, np.ndarray | None]],
    header: str,
    subheader: str,
    chart: tuple[AreaSeries, int] | None = None,
    display_max: float = 1.0,
    field_label: str = "field, normalized",
    field_threshold: float = 0.5 / 255.0,
) -> Image.Image:
    """Beauty render with a header, plus a right-hand column of UV wear-atlas panels when ``panels`` is set.

    ``chart`` = (series, frame index) adds the worn-area line chart under the atlas panels.
    """
    h, w = beauty.shape[:2]
    big, small = font(max(14, h // 34)), font(max(11, h // 52))
    pad = max(8, h // 60)
    panel_w = int(h * 0.42) if panels else 0
    if panel_w - 2 * pad < MIN_PANEL_WIDTH:  # thumbnail-size renders: no room for a legible side column
        panels, panel_w = [], 0
    canvas = Image.new("RGB", (w + panel_w, h), BACKGROUND)
    canvas.paste(Image.fromarray(beauty), (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Header with a soft shadow so it reads over bright and dark renders.
    for dx, dy, color in ((1, 1, (0, 0, 0)), (0, 0, TEXT)):
        draw.text((pad + dx, pad + dy), header, font=big, fill=color)
    draw.text((pad, pad + big.size + 4), subheader, font=small, fill=(0, 0, 0))
    draw.text((pad - 1, pad + big.size + 3), subheader, font=small, fill=(210, 210, 216))

    if not panels:
        return canvas

    x = w + pad
    inner = panel_w - 2 * pad
    bar_h = max(8, h // 70)
    footer = 2 * small.size + bar_h + 3 * pad
    chart_h = int(h * 0.3) if chart else 0
    slot = (h - footer - chart_h - pad) // len(panels)
    side = max(16, min(inner, slot - small.size - 2 * pad))
    y = pad
    for source, wear in panels:
        draw.text((x, y), _fit(draw, source.title, small, inner), font=small, fill=TEXT)
        y += small.size + pad // 2
        canvas.paste(wear_panel_image(wear, source.roi, side, display_max, field_threshold), (x + (inner - side) // 2, y))
        y += side + pad
    if chart:
        series, index = chart
        canvas.paste(area_chart(series, index, inner, chart_h - pad), (x, h - footer - chart_h))

    yb = h - footer + pad
    caption = field_label.split(",")[0]
    label = (field_label if display_max >= 1.0
             else f"{caption}, display range 0-{display_max:g} (preview)")  # fmt: skip
    draw.text((x, yb), _fit(draw, label, small, inner), font=small, fill=MUTED)
    yb += small.size + 4
    canvas.paste(colorbar(inner, bar_h), (x, yb))
    yb += bar_h + 2
    draw.text((x, yb), "0", font=small, fill=MUTED)
    top = f"{display_max:g}"
    draw.text((x + inner - draw.textlength(top, font=small), yb), top, font=small, fill=MUTED)
    return canvas


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def encode_video(
    pattern: Path,
    fps: float,
    out: Path,
    crf: int = 18,
    start_number: int = 0,
    title_image: Path | None = None,
    title_seconds: float = 0.0,
    hold_seconds: float = 0.0,
) -> Path:
    """H.264 mp4 from numbered frames; optional faded title card before and a freeze of the last frame after."""
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH")
    titled = title_image is not None and title_seconds > 0
    inputs: list[str] = []
    if titled:
        inputs += ["-loop", "1", "-framerate", f"{fps}", "-t", f"{title_seconds}", "-i", str(title_image)]
    inputs += ["-framerate", f"{fps}", "-start_number", str(start_number), "-i", str(pattern)]
    main = f"[{1 if titled else 0}:v]setsar=1"
    if hold_seconds > 0:
        main += f",tpad=stop_mode=clone:stop_duration={hold_seconds}"
    if titled:
        fade_out = max(title_seconds - 0.5, 0.0)
        graph = (
            f"[0:v]setsar=1,fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out}:d=0.5[t];"
            f"{main},fade=t=in:st=0:d=0.4[m];[t][m]concat=n=2:v=1:a=0[c];"
        )
    else:
        graph = f"{main}[c];"
    graph += "[c]scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,format=yuv420p[v]"
    cmd = [exe, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[v]",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-movflags", "+faststart", str(out)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out


def encode_gif(pattern: Path, fps: float, out: Path, width: int = 960, start_number: int = 0) -> Path:
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH")
    graph = (
        f"fps={min(fps, 25)},scale={width}:-2:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", f"{fps}", "-start_number", str(start_number),
           "-i", str(pattern), "-filter_complex", graph, str(out)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out
