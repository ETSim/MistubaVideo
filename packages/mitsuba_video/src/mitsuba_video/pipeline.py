"""Render pipeline: source -> per-frame Mitsuba renders -> composited frames -> video.

``run_render(settings)`` is the whole job; CLIs only turn options into a ``RenderSettings``. Output layout:

    <out>/beauty/frame_NNNNNN.png   tone-mapped 3D render
    <out>/frames/frame_NNNNNN.png   beauty + header + UV field panel + covered-area chart (what gets encoded)
    <out>/exr/frame_NNNNNN.exr      linear HDR (keep_exr)
    <out>/<field.area_csv>          covered texels / area per frame (the chart's data)
    <out>/render_config.json        settings signature, Mitsuba variant, timings, outputs
    <out>/<stem>.mp4 [.gif]         encoded video
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import camera as cam
from . import scene, video
from .blend import BlendParams, VariantStack, blend_stack
from .color import FieldDisplay, heatmap_albedo
from .maps import NormalOptions, fit_size, resize_scalar
from .model import Body, FieldSpec, Frame
from .motion import travel_by_frame
from .source import Source, open_source

CONFIG_NAME = "render_config.json"
# Frame PNGs are intermediates for the encoder: zlib level 1 is ~5x faster to write than Pillow's default and still
# lossless.
FRAME_PNG = {"compress_level": 1}


class Look(str, Enum):
    heatmap = "heatmap"  # affected material + labelled field colormap on covered texels
    worn = "worn"  # physically based base -> affected appearance only
    plain = "plain"  # base material, field ignored (reference render)


# name -> (width, height, spp); explicit --res / --spp override the preset.
QUALITY_PRESETS: dict[str, tuple[int, int, int]] = {
    "preview": (960, 540, 16),
    "draft": (1280, 720, 64),
    "paper": (1920, 1080, 256),
}


@dataclass
class VideoSettings:
    fps: float = 30.0
    encode: bool = True
    gif: bool = False
    crf: int = 18
    title: str | None = None
    subtitle: list[str] = field(default_factory=list)
    title_seconds: float = 3.0
    hold: float = 2.0


@dataclass
class RenderSettings:
    source: Path
    source_kind: str = "auto"
    source_options: dict[str, object] = field(default_factory=dict)
    field_name: str | None = None  # default: the source's primary field
    out: Path | None = None
    look: Look = Look.heatmap
    width: int = 1920
    height: int = 1080
    spp: int = 128
    max_depth: int = 8
    frames: str = "::1"
    variant: str = "auto"
    camera: cam.CameraSettings = field(default_factory=cam.CameraSettings)
    lighting: scene.LightingSettings = field(default_factory=scene.LightingSettings)
    exposure: float = 0.0
    max_texture: int = 2048
    normals: NormalOptions = field(default_factory=NormalOptions)
    blend: BlendParams = field(default_factory=BlendParams)
    display: FieldDisplay = field(default_factory=FieldDisplay)
    panel: bool = True
    panel_bodies: list[int] = field(default_factory=list)
    hidden_bodies: list[int] = field(default_factory=list)  # left out of the 3D scene; panels still show them
    chart: bool = True
    denoise: bool = False
    seed: int = 0
    keep_exr: bool = False
    resume: bool = False
    video: VideoSettings = field(default_factory=VideoSettings)

    def stem(self, spec: FieldSpec) -> str:
        return f"{spec.video_prefix}_{self.look.value}_{self.camera.mode}"


def parse_frames(spec: str, count: int) -> range:
    """Python slice syntax over source frames: '::1', '0:300:2', '-60:'."""
    parts = (spec.split(":") + ["", "", ""])[:3]
    start, stop, step = (int(p) if p.strip() else None for p in parts)
    return range(count)[slice(start, stop, step)]


def resolve_field(source: Source, name: str | None) -> FieldSpec:
    key = name or source.primary_field
    if key not in source.fields:
        raise ValueError(f"unknown field {key!r}; {source.kind} provides {', '.join(source.fields)}")
    return source.fields[key]


# --- per-frame helpers ---------------------------------------------------------------------------------------


def material_stack(
    body: Body, size: tuple[int, int], normals: NormalOptions, log: Callable[[str], None] | None
) -> VariantStack:
    """The body's material as a ``VariantStack``: ``build_stack`` when the provider has it, else base/worn."""
    build_stack = getattr(body.material, "build_stack", None)
    if build_stack is not None:
        return build_stack(size, normals, log)
    return VariantStack.pair(*body.material.build(size, normals, log))


def _blended_textures(
    look: Look,
    value: np.ndarray | None,
    stack: VariantStack,
    blend: BlendParams,
    display: FieldDisplay,
    threshold: float,
    texel_scale: float,
    direction: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if look is Look.plain or value is None:
        return scene.encode_maps(stack.base)
    maps = blend_stack(value, stack, blend, texel_scale, direction=direction)
    base_color = heatmap_albedo(maps.albedo, value, display, threshold) if look is Look.heatmap else None
    return scene.encode_maps(maps, base_color)


def body_textures(
    look: Look,
    value: np.ndarray | None,
    stack: VariantStack,
    blend: BlendParams,
    display: FieldDisplay = FieldDisplay(),
    threshold: float = 0.5 / 255.0,
    texel_scale: float = 1.0,
    direction: np.ndarray | None = None,
):
    """Mitsuba textures for one body; ``texel_scale`` is render width / field width (for the wear blur).
    ``direction`` is the slide direction at the same size (image axes), for ``blend.scratches``.

    Unworn bodies (no field, or an all-zero one) give the same textures for every body sharing ``stack``, so those
    are encoded once per stack and look settings and shared: treat the returned arrays as read-only."""
    if not (look is Look.plain or value is None or not value.any()):
        return _blended_textures(look, value, stack, blend, display, threshold, texel_scale, direction)
    key = (id(stack), look, value is None, blend, display, threshold, texel_scale)
    hit = _UNWORN_TEXTURES.get(key)
    if hit is None or hit[0] is not stack:
        hit = _UNWORN_TEXTURES[key] = (stack, _blended_textures(look, value, stack, blend, display, threshold,
                                                                texel_scale))  # fmt: skip
    return hit[1]


def _direction_at(source: Source, frame: Frame, body_index: int, size: tuple[int, int],
                  blend: BlendParams) -> np.ndarray | None:  # fmt: skip
    """The body's slide direction (H, W, 2, image axes) resized to ``size``, when scratches are on and the source
    records one; else None."""
    if blend.scratches <= 0.0 or not hasattr(source, "direction"):
        return None
    d = source.direction(frame, body_index)
    if d is None:
        return None
    return np.stack([resize_scalar(d[..., 0], size), resize_scalar(d[..., 1], size)], axis=-1)


# (id(stack), look settings) -> (stack, textures); the stack is kept so a reused id cannot match a dead stack.
_UNWORN_TEXTURES: dict[tuple, tuple[VariantStack, dict[str, np.ndarray]]] = {}


def _crop_stack(stack: VariantStack, rows: slice, cols: slice) -> VariantStack:
    return VariantStack(albedo=[a[rows, cols] for a in stack.albedo], normal=[n[rows, cols] for n in stack.normal],
                        roughness=[r[rows, cols] for r in stack.roughness],
                        height=[h[rows, cols] for h in stack.height])  # fmt: skip


class BodyTextures:
    """One body's textures across frames, re-blended only where its field changed.

    A texel's textures depend on the field within the wear blur's reach (at most 5 texels), so each changed
    ``TILE``-sized tile is re-blended over the tile grown by that reach, from a crop grown by the reach again. The
    result equals a full ``body_textures`` rebuild; a 2048^2 plate with a few new scratches costs a few tiles.
    """

    TILE = 128

    def __init__(self, look: Look, stack: VariantStack, blend: BlendParams, display: FieldDisplay, threshold: float,
                 texel_scale: float) -> None:  # fmt: skip
        self.args = (look, stack, blend, display, threshold, texel_scale)
        px = blend.wear_blur_px * texel_scale
        self.reach = int(np.floor(min(px, 4.0))) + 1 if blend.wear_blur_px > 0.0 and px > 0.01 else 0
        self.value: np.ndarray | None = None
        self.textures: dict[str, np.ndarray] = {}
        self._owned = False  # False while self.textures may be the shared unworn arrays

    def update(self, value: np.ndarray | None, direction: np.ndarray | None = None) -> dict[str, np.ndarray]:
        look, stack, blend, display, threshold, texel_scale = self.args
        # Burnish and scratches reach further than a tile's margin: rebuild the whole body when they are on.
        if (look is Look.plain or value is None or self.value is None or value.shape != self.value.shape
                or not self.textures or blend.wide):  # fmt: skip
            self.textures = body_textures(look, value, stack, blend, display, threshold, texel_scale, direction)
            self._owned = False
        else:
            changed = value != self.value
            if not changed.any():
                return self.textures
            if not self._owned:
                self.textures = {k: v.copy() for k, v in self.textures.items()}
                self._owned = True
            self._update_tiles(value, changed)
        self.value = None if value is None else value.copy()
        return self.textures

    def _update_tiles(self, value: np.ndarray, changed: np.ndarray) -> None:
        look, stack, blend, display, threshold, texel_scale = self.args
        h, w = value.shape
        t, r = self.TILE, self.reach
        ys, xs = np.arange(0, h, t), np.arange(0, w, t)
        tiles = np.logical_or.reduceat(np.logical_or.reduceat(changed, ys, axis=0), xs, axis=1)
        for iy, ix in zip(*np.nonzero(tiles), strict=True):
            y0, x0 = int(ys[iy]), int(xs[ix])
            uy0, ux0, uy1, ux1 = max(y0 - r, 0), max(x0 - r, 0), min(y0 + t + r, h), min(x0 + t + r, w)
            cy0, cx0, cy1, cx1 = max(uy0 - r, 0), max(ux0 - r, 0), min(uy1 + r, h), min(ux1 + r, w)
            rows, cols = slice(cy0, cy1), slice(cx0, cx1)
            sub = _blended_textures(look, value[rows, cols], _crop_stack(stack, rows, cols), blend, display,
                                    threshold, texel_scale)  # fmt: skip
            for key, arr in self.textures.items():
                arr[uy0:uy1, ux0:ux1] = sub[key][uy0 - cy0:uy1 - cy0, ux0 - cx0:ux1 - cx0]


def covered_area(
    source: Source, frames: list[Frame], body: Body, spec: FieldSpec, progress: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """(covered texel count, covered area in scaled mesh units^2) per frame for ``body``."""
    counts, per_texel = [], None
    desc = f"{spec.area_title} body {body.index}"
    for frame in tqdm(frames, desc=desc, unit="frame", disable=not progress, leave=False):
        value = source.field(frame, body.index, spec.name)
        counts.append(0 if value is None else int((value > spec.threshold).sum()))
        if per_texel is None and value is not None:
            per_texel = body.area_per_texel(value.shape)
    counts_arr = np.asarray(counts, np.int64)
    return counts_arr, counts_arr * (per_texel or 0.0)


def _area_series(
    source: Source, frames: list[Frame], body: Body, spec: FieldSpec, out: Path, progress: bool
) -> video.AreaSeries:
    counts, area = covered_area(source, frames, body, spec, progress)
    word = spec.area_title.split()[0]
    unit = getattr(source, "length_unit", "m")
    with (out / spec.area_csv).open("w", encoding="utf-8") as fh:
        fh.write(f"frame,time_s,body,{word}_texels,{word}_area_{unit}2\n")
        for f, n, a in zip(frames, counts, area, strict=True):
            fh.write(f"{f.index},{f.time:.6f},{body.index},{n},{a:.9g}\n")
    small = unit == "m" and area.max() < 0.01
    shown = "cm" if small else unit
    return video.AreaSeries(
        title=f"{spec.area_title}, body {body.index} ({shown}²)",
        times=np.array([f.time for f in frames]),
        values=area * (1e4 if small else 1.0),
        unit=f"{shown}²",
    )


MAX_PANELS = 3


def _panel_sources(source: Source, last: Frame, s: RenderSettings, spec: FieldSpec) -> list[video.PanelSource]:
    """Bodies in the side panel: explicit ``panel_bodies``, else the (up to three) bodies with the most covered
    texels by ``last``, so twenty barely-touched cubes do not crowd out the plate they scratched."""
    if not s.panel:
        return []
    bodies = {b.index: b for b in source.bodies}
    if s.panel_bodies:
        chosen = [(idx, source.field(last, idx, spec.name)) for idx in s.panel_bodies if idx in bodies]
        chosen = [(idx, value) for idx, value in chosen if value is not None][:MAX_PANELS]
    else:
        covered = []
        for idx in bodies:
            value = source.field(last, idx, spec.name)
            count = 0 if value is None else int((value > spec.threshold).sum())
            if count:
                covered.append((count, idx, value))
        covered.sort(key=lambda item: (-item[0], item[1]))
        chosen = [(idx, value) for _, idx, value in covered[:MAX_PANELS]]
    return [video.PanelSource(idx, f"body {idx}: {bodies[idx].label} (UV atlas)",
                              video.wear_roi(value, threshold=spec.threshold))
            for idx, value in chosen]  # fmt: skip


def _digest(value: np.ndarray | None) -> bytes | None:
    return None if value is None else hashlib.blake2b(value.tobytes(), digest_size=16).digest()


# --- config / resume -----------------------------------------------------------------------------------------


def _signature(s: RenderSettings, source: Source, spec: FieldSpec, frames: list[Frame]) -> dict:
    """Everything that changes the pixels of a frame; a resumed render must match it exactly."""
    return {
        "source": str(source.path.resolve()),
        "source_kind": source.kind,
        "source_options": {k: str(v) for k, v in s.source_options.items()},
        "field": spec.name,
        "look": s.look.value,
        "resolution": [s.width, s.height],
        "spp": s.spp,
        "max_depth": s.max_depth,
        "frames": [f.index for f in frames],
        "camera": asdict(s.camera),
        "lighting": asdict(s.lighting),
        "exposure": s.exposure,
        "max_texture": s.max_texture,
        "normals": asdict(s.normals),
        "blend": asdict(s.blend),
        "display": asdict(s.display),
        "panel": s.panel,
        "panel_bodies": s.panel_bodies,
        "hidden_bodies": s.hidden_bodies,
        "chart": s.chart,
        "denoise": s.denoise,
        "seed": s.seed,
    }


def _check_resume(out: Path, signature: dict) -> None:
    path = out / CONFIG_NAME
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8")).get("signature", {})
    current = json.loads(json.dumps(signature, default=str))  # same JSON round trip the stored copy went through
    diff = sorted(k for k in set(previous) | set(current) if previous.get(k) != current.get(k))
    if diff:
        raise RuntimeError(
            f"--resume: frames in {out} were rendered with different settings ({', '.join(diff)}); "
            "use another --out or drop --resume"
        )


def _write_config(out: Path, payload: dict) -> None:
    (out / CONFIG_NAME).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --- encoding ------------------------------------------------------------------------------------------------


def encode_run(out: Path, v: VideoSettings, stem: str, log: Callable[[str], None] = tqdm.write) -> dict[str, str]:
    """Encode ``out/frames`` into ``out/<stem>.mp4`` (+ gif); reusable without re-rendering (``encode``)."""
    frames_dir = out / "frames"
    first = frames_dir / "frame_000000.png"
    if not first.is_file():
        raise FileNotFoundError(f"no rendered frames in {frames_dir}")
    if video.ffmpeg_path() is None:
        log("ffmpeg not found on PATH; frames kept, video skipped")
        return {}
    title_png = None
    if v.title:
        with Image.open(first) as img:
            card = video.title_card(img.size, v.title, list(v.subtitle))
        title_png = out / "title_card.png"
        card.save(title_png)
    pattern = frames_dir / "frame_%06d.png"
    outputs = {
        "video": str(
            video.encode_video(
                pattern,
                v.fps,
                out / f"{stem}.mp4",
                crf=v.crf,
                title_image=title_png,
                title_seconds=v.title_seconds if v.title else 0.0,
                hold_seconds=v.hold,
            )
        )
    }
    if v.gif:
        outputs["gif"] = str(video.encode_gif(pattern, v.fps, out / f"{stem}.gif"))
    log(f"video: {outputs['video']}")
    return outputs


# --- the job -------------------------------------------------------------------------------------------------


def run_render(s: RenderSettings, log: Callable[[str], None] = tqdm.write, progress: bool = True) -> Path:
    """Render ``s.source`` and return the output directory."""
    t_start = time.time()
    source = open_source(s.source, s.source_kind, **s.source_options)
    try:
        if not source.frames or not source.bodies:
            raise ValueError(f"{source.path} has no frames or no renderable bodies")
        spec = resolve_field(source, s.field_name)
        selected = [source.frames[i] for i in parse_frames(s.frames, len(source.frames))]
        if not selected:
            raise ValueError(f"--frames {s.frames!r} selects nothing from {len(source.frames)} frames")
        out = Path(s.out) if s.out else source.root / f"render_{s.look.value}"
        beauty_dir, comp_dir, exr_dir = out / "beauty", out / "frames", out / "exr"
        for d in (beauty_dir, comp_dir) + ((exr_dir,) if s.keep_exr else ()):
            d.mkdir(parents=True, exist_ok=True)

        signature = _signature(s, source, spec, selected)
        if s.resume:
            _check_resume(out, signature)
        else:
            # A fresh render owns its frame folders: stale frames from an earlier run would otherwise be picked up by
            # a later --resume once this run's signature is on disk.
            stale = [f for d in (beauty_dir, comp_dir, exr_dir) if d.is_dir() for f in d.glob("frame_*")]
            for f in stale:
                f.unlink()
            if stale:
                log(f"output    cleared {len(stale)} frame file(s) from a previous render in {out}")

        for key, value in source.describe():
            log(f"{key:<9} {value}")
        log(f"frames    {len(source.frames)} in source, {len(selected)} selected; field '{spec.name}'")
        for body in source.bodies:
            notes = ", ".join(f"{k} {v}" for k, v in body.notes.items())
            log(f"body {body.index}    {body.name}: {len(body.faces)} tris, {'fixed' if body.fixed else 'dynamic'}"
                + (f", {notes}" if notes else ""))  # fmt: skip

        chosen = scene.select_variant(s.variant)
        log(f"mitsuba   {scene.mitsuba().__version__} ({chosen})")

        # Materials at render resolution, sized from the first selected frame's field atlas.
        first, last = selected[0], selected[-1]
        scene_bodies = [b for b in source.bodies if b.index not in s.hidden_bodies]
        if not scene_bodies:
            raise ValueError("--hide leaves no body to render")
        stacks, sizes, scales, textures, last_hash = {}, {}, {}, {}, {}
        for body in tqdm(scene_bodies, desc="materials", unit="body", disable=not progress, leave=False):
            value = source.field(first, body.index, spec.name)
            native = value.shape if value is not None else body.material.preferred_size()
            sizes[body.index] = fit_size(native, s.max_texture)
            scales[body.index] = sizes[body.index][0] / native[1] if native else 1.0
            stacks[body.index] = material_stack(body, sizes[body.index], s.normals, log)
            log(f"blend     body {body.index}: {stacks[body.index].describe()}")
            value_r = None if value is None else resize_scalar(value, sizes[body.index])
            textures[body.index] = BodyTextures(s.look, stacks[body.index], s.blend, s.display, spec.threshold,
                                                scales[body.index])  # fmt: skip
            textures[body.index].update(value_r, _direction_at(source, first, body.index, sizes[body.index], s.blend))
            last_hash[body.index] = _digest(value)

        poses = cam.camera_path(source, selected, s.camera, s.width / s.height)
        fscene = scene.FieldScene(
            scene_bodies,
            {index: t.textures for index, t in textures.items()},
            {b.index: b.material.metallic for b in scene_bodies},
            s.width,
            s.height,
            s.camera.fov,
            cam.scene_bounds(source, source.frames),
            s.lighting,
            up_axis=s.camera.up,
            max_depth=s.max_depth,
            azimuth=s.camera.azimuth,
            denoise_guides=s.denoise,
        )
        denoiser = scene.Denoiser(s.width, s.height, s.denoise, use_guides=True)
        panels = _panel_sources(source, last, s, spec)
        mover, travel = travel_by_frame(source, s.camera.follow, "reset_detection" in source.capabilities)
        multi_pass = bool(travel) and travel[source.frames[-1].key][1] > 1
        travel_word = getattr(source, "travel_label", "travel")
        unit = getattr(source, "length_unit", "m")
        area = None
        if s.chart and panels:
            body = next(b for b in source.bodies if b.index == panels[0].body_index)
            area = _area_series(source, selected, body, spec, out, progress)

        stem = s.stem(spec)
        log(f"render    {s.width}x{s.height}, {s.spp} spp, look={s.look.value}, camera={s.camera.mode}, "
            f"denoiser={'on' if denoiser.active else 'off'}, panels={[p.body_index for p in panels]} -> {out}")  # fmt: skip
        config = {"signature": signature, "status": "rendering", "mitsuba": chosen, "video_stem": stem,
                  "video": asdict(s.video), "texture_sizes": sizes,
                  "bodies": {b.index: {"name": b.name, "scale": b.scale, **b.notes} for b in source.bodies}}  # fmt: skip
        _write_config(out, config)

        timings: list[float] = []
        last_field: dict[int, np.ndarray | None] = {}  # last field array per body: the same object skips re-hashing
        done = {i for i in range(len(selected)) if s.resume and (comp_dir / f"frame_{i:06d}.png").is_file()}
        if done:
            log(f"resume    reusing {len(done)} already rendered frame(s)")
        # Reused frames start the bar (initial=) so its rate and ETA reflect path tracing only.
        bar = tqdm(total=len(selected), initial=len(done), desc="render", unit="frame", disable=not progress,
                   dynamic_ncols=True)  # fmt: skip
        for i, (frame, pose) in enumerate(zip(selected, poses, strict=True)):
            comp_path = comp_dir / f"frame_{i:06d}.png"
            if i in done:
                continue
            t0 = time.time()
            positions, orientations = source.pose(frame)
            fscene.set_poses(positions, orientations)
            panel_values = {}
            for body in source.bodies:
                value = source.field(frame, body.index, spec.name)
                panel_values[body.index] = value
                if body.index in s.hidden_bodies or (value is not None and value is last_field.get(body.index)):
                    continue  # the source handed back the same array: the field did not change
                last_field[body.index] = value
                digest = _digest(value)
                if digest == last_hash[body.index]:
                    continue  # textures only change when the field does
                last_hash[body.index] = digest
                value_r = None if value is None else resize_scalar(value, sizes[body.index])
                direction = _direction_at(source, frame, body.index, sizes[body.index], s.blend)
                fscene.set_textures(body.index, textures[body.index].update(value_r, direction))
            fscene.set_camera(pose)
            rendered = fscene.render(s.spp, seed=s.seed)
            if isinstance(rendered, tuple):
                rgb, albedo, normals = rendered
                rgb = denoiser(rgb, albedo=albedo, normals=normals, to_sensor=fscene.to_sensor())
            else:
                rgb = denoiser(rendered)
            if s.keep_exr:
                video.write_exr(exr_dir / f"frame_{i:06d}.exr", rgb)
            ldr = video.tonemap(rgb, s.exposure)
            Image.fromarray(ldr).save(beauty_dir / f"frame_{i:06d}.png", **FRAME_PNG)

            header = f"t = {frame.time:.3f} s"
            if mover is not None:
                dist, pass_no = travel[frame.key]
                header += f"    body {mover.index} {travel_word} {dist:.2f} {unit}"
                header += f"  (pass {pass_no})" if multi_pass else ""
            sub = f"frame {frame.index}  |  {s.look.value} look  |  Mitsuba {s.spp} spp"
            composite = video.compose_frame(
                ldr,
                [(p, panel_values.get(p.body_index)) for p in panels],
                header,
                sub,
                chart=(area, i) if area is not None else None,
                display_max=s.display.max,
                field_label=spec.label,
                field_threshold=spec.threshold,
            )
            composite.save(comp_path, **FRAME_PNG)
            timings.append(time.time() - t0)
            bar.set_postfix(frame=frame.index, s_per_frame=f"{timings[-1]:.1f}")
            bar.update()
        bar.close()

        outputs = encode_run(out, s.video, stem, log) if s.video.encode else {}
        config.update(
            status="done",
            outputs=outputs,
            seconds_per_frame=float(np.mean(timings)) if timings else None,
            rendered_frames=len(timings),
            total_seconds=time.time() - t_start,
        )
        _write_config(out, config)
        return out
    finally:
        source.close()
