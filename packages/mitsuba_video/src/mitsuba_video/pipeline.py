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


def body_textures(
    look: Look,
    value: np.ndarray | None,
    stack: VariantStack,
    blend: BlendParams,
    display: FieldDisplay = FieldDisplay(),
    threshold: float = 0.5 / 255.0,
    texel_scale: float = 1.0,
):
    """Mitsuba textures for one body; ``texel_scale`` is render width / field width (for the wear blur)."""
    if look is Look.plain or value is None:
        return scene.encode_maps(stack.base)
    maps = blend_stack(value, stack, blend, texel_scale)
    base_color = heatmap_albedo(maps.albedo, value, display, threshold) if look is Look.heatmap else None
    return scene.encode_maps(maps, base_color)


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


def _panel_sources(source: Source, last: Frame, s: RenderSettings, spec: FieldSpec) -> list[video.PanelSource]:
    """Bodies in the side panel: explicit ``panel_bodies``, else every body whose field is non-empty by ``last``."""
    if not s.panel:
        return []
    bodies = source.bodies
    wanted = s.panel_bodies or [b.index for b in bodies if not b.fixed] + [b.index for b in bodies if b.fixed]
    panels = []
    for idx in wanted[:3]:
        body = next((b for b in bodies if b.index == idx), None)
        if body is None:
            continue
        value = source.field(last, idx, spec.name)
        if value is None or (not s.panel_bodies and not (value > spec.threshold).any()):
            continue
        roi = video.wear_roi(value, threshold=spec.threshold)
        panels.append(video.PanelSource(idx, f"body {idx}: {body.label} (UV atlas)", roi))
    return panels


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
        stacks, sizes, scales, initial, last_hash = {}, {}, {}, {}, {}
        for body in tqdm(source.bodies, desc="materials", unit="body", disable=not progress, leave=False):
            value = source.field(first, body.index, spec.name)
            native = value.shape if value is not None else body.material.preferred_size()
            sizes[body.index] = fit_size(native, s.max_texture)
            scales[body.index] = sizes[body.index][0] / native[1] if native else 1.0
            stacks[body.index] = material_stack(body, sizes[body.index], s.normals, log)
            log(f"blend     body {body.index}: {stacks[body.index].describe()}")
            value_r = None if value is None else resize_scalar(value, sizes[body.index])
            initial[body.index] = body_textures(
                s.look, value_r, stacks[body.index], s.blend, s.display, spec.threshold, scales[body.index]
            )
            last_hash[body.index] = _digest(value)

        poses = cam.camera_path(source, selected, s.camera, s.width / s.height)
        fscene = scene.FieldScene(
            source.bodies,
            initial,
            {b.index: b.material.metallic for b in source.bodies},
            s.width,
            s.height,
            s.camera.fov,
            cam.scene_bounds(source, source.frames),
            s.lighting,
            up_axis=s.camera.up,
            max_depth=s.max_depth,
            azimuth=s.camera.azimuth,
        )
        denoiser = scene.Denoiser(s.width, s.height, s.denoise)
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
                digest = _digest(value)
                if digest == last_hash[body.index]:
                    continue  # textures only change when the field does
                last_hash[body.index] = digest
                value_r = None if value is None else resize_scalar(value, sizes[body.index])
                fscene.set_textures(
                    body.index,
                    body_textures(s.look, value_r, stacks[body.index], s.blend, s.display, spec.threshold,
                                  scales[body.index]),  # fmt: skip
                )
            fscene.set_camera(pose)
            rgb = denoiser(fscene.render(s.spp, seed=s.seed))
            if s.keep_exr:
                video.write_exr(exr_dir / f"frame_{i:06d}.exr", rgb)
            ldr = video.tonemap(rgb, s.exposure)
            Image.fromarray(ldr).save(beauty_dir / f"frame_{i:06d}.png")

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
            composite.save(comp_path)
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
