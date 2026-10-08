"""Render pipeline: recording -> per-frame Mitsuba renders -> composited frames -> video.

``run_render(settings)`` is the whole job; the CLI only turns options into a ``RenderSettings``. Output layout:

    <out>/beauty/frame_NNNNNN.png   tone-mapped 3D render
    <out>/frames/frame_NNNNNN.png   beauty + header + UV wear-atlas panel + worn-area chart (what gets encoded)
    <out>/exr/frame_NNNNNN.exr      linear HDR (keep_exr)
    <out>/worn_area.csv             worn texels / area per frame (the chart's data)
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
from . import materials, scene, video
from .recording import Body, Frame, Recording
from .wear_blend import BlendParams, MapSet, blend_maps, heatmap_albedo

WORN_THRESHOLD = 0.5 / 255.0  # one 8-bit step of the exported wear PNG
CONFIG_NAME = "render_config.json"


class Look(str, Enum):
    heatmap = "heatmap"  # worn material + labelled wear colormap on worn texels
    worn = "worn"  # physically based flat -> worn appearance only
    plain = "plain"  # base material, wear ignored (reference render)


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
    panel: bool = True
    panel_bodies: list[int] = field(default_factory=list)
    chart: bool = True
    scales: dict[int, float] = field(default_factory=dict)
    asset_roots: list[Path] = field(default_factory=list)
    denoise: bool = False
    seed: int = 0
    keep_exr: bool = False
    resume: bool = False
    video: VideoSettings = field(default_factory=VideoSettings)

    @property
    def stem(self) -> str:
        return f"wear_{self.look.value}_{self.camera.mode}"


def parse_frames(spec: str, count: int) -> range:
    """Python slice syntax over recorded frames: '::1', '0:300:2', '-60:'."""
    parts = (spec.split(":") + ["", "", ""])[:3]
    start, stop, step = (int(p) if p.strip() else None for p in parts)
    return range(count)[slice(start, stop, step)]


def parse_scales(values: list[str]) -> dict[int, float]:
    """'1=0.5' or 'body_1=0.5' -> {1: 0.5}."""
    out = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep:
            raise ValueError(f"scale override {value!r} is not BODY=SCALE")
        out[int(key.replace("body_", ""))] = float(val)
    return out


# --- per-frame helpers ---------------------------------------------------------------------------------------


def body_textures(look: Look, wear: np.ndarray | None, base: MapSet, worn: MapSet, blend: BlendParams):
    if look is Look.plain or wear is None:
        return scene.encode_maps(base)
    maps = blend_maps(wear, base, worn, blend)
    base_color = heatmap_albedo(maps.albedo, wear) if look is Look.heatmap else None
    return scene.encode_maps(maps, base_color)


def travel_by_frame(rec: Recording, follow: int | None = None) -> tuple[Body | None, dict[str, tuple[float, int]]]:
    """Cumulative sliding distance and pass number of the tracked (or first dynamic) body, per recorded frame.

    Reset scenarios teleport the body back to its start; a step far above the median step is a reset, which starts
    a new pass instead of adding metres of travel.
    """
    mover = next((b for b in rec.bodies if b.index == follow), None) or next(
        (b for b in rec.bodies if not b.fixed), None
    )
    if mover is None:
        return None, {}
    track = np.array([rec.pose(f)[0][mover.index] for f in rec.frames])
    steps = np.r_[0.0, np.linalg.norm(np.diff(track, axis=0), axis=1)]
    is_reset = steps > max(10.0 * float(np.median(steps)), 1e-9)
    slid = np.cumsum(np.where(is_reset, 0.0, steps))
    passes = 1 + np.cumsum(is_reset)
    return mover, {f.key: (float(slid[i]), int(passes[i])) for i, f in enumerate(rec.frames)}


def worn_area(
    rec: Recording, frames: list[Frame], body: Body, progress: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """(worn texel count, worn area in mesh units^2) per frame for ``body``."""
    counts, per_texel = [], None
    for frame in tqdm(frames, desc=f"worn area body {body.index}", unit="frame", disable=not progress, leave=False):
        wear = rec.texture(frame, body.index)
        counts.append(0 if wear is None else int((wear > WORN_THRESHOLD).sum()))
        if per_texel is None and wear is not None:
            per_texel = body.area_per_texel(wear.shape)
    counts_arr = np.asarray(counts, np.int64)
    return counts_arr, counts_arr * (per_texel or 0.0)


def _area_series(
    rec: Recording, frames: list[Frame], body: Body, out: Path, progress: bool
) -> video.AreaSeries:
    counts, area_m2 = worn_area(rec, frames, body, progress)
    with (out / "worn_area.csv").open("w", encoding="utf-8") as fh:
        fh.write("frame,time_s,body,worn_texels,worn_area_m2\n")
        for f, n, a in zip(frames, counts, area_m2, strict=True):
            fh.write(f"{f.index},{f.time:.6f},{body.index},{n},{a:.9g}\n")
    cm2 = area_m2.max() < 0.01
    return video.AreaSeries(
        title=f"worn area, body {body.index} ({'cm' if cm2 else 'm'}²)",
        times=np.array([f.time for f in frames]),
        values=area_m2 * (1e4 if cm2 else 1.0),
        unit="cm²" if cm2 else "m²",
    )


def _panel_sources(rec: Recording, last: Frame, s: RenderSettings) -> list[video.PanelSource]:
    """Bodies shown in the side panel: explicit ``panel_bodies``, else every body that has wear by ``last``."""
    if not s.panel:
        return []
    wanted = s.panel_bodies or [b.index for b in rec.bodies if not b.fixed] + [b.index for b in rec.bodies if b.fixed]
    sources = []
    for idx in wanted[:3]:
        body = next((b for b in rec.bodies if b.index == idx), None)
        if body is None:
            continue
        wear_last = rec.texture(last, idx)
        if wear_last is None or (not s.panel_bodies and not (wear_last > WORN_THRESHOLD).any()):
            continue
        name = body.name.split("_Material")[0]
        sources.append(video.PanelSource(idx, f"body {idx}: {name} (UV atlas)", video.wear_roi(wear_last)))
    return sources


def _digest(wear: np.ndarray | None) -> bytes | None:
    return None if wear is None else hashlib.blake2b(wear.tobytes(), digest_size=16).digest()


# --- config / resume -----------------------------------------------------------------------------------------


def _signature(s: RenderSettings, rec: Recording, frames: list[Frame]) -> dict:
    """Everything that changes the pixels of a frame; a resumed render must match it exactly."""
    return {
        "source": str(rec.path.resolve()),
        "look": s.look.value,
        "resolution": [s.width, s.height],
        "spp": s.spp,
        "max_depth": s.max_depth,
        "frames": [f.index for f in frames],
        "camera": asdict(s.camera),
        "lighting": asdict(s.lighting),
        "exposure": s.exposure,
        "max_texture": s.max_texture,
        "panel": s.panel,
        "panel_bodies": s.panel_bodies,
        "chart": s.chart,
        "scales": {str(k): v for k, v in s.scales.items()},
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
    """Encode ``out/frames`` into ``out/<stem>.mp4`` (+ gif); reusable without re-rendering (``tf-render encode``)."""
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
    rec = Recording(s.source, scale_overrides=s.scales, asset_roots=s.asset_roots)
    try:
        if not rec.frames or not rec.bodies:
            raise ValueError(f"{rec.path} has no frames or no bodies with embedded meshes")
        selected = [rec.frames[i] for i in parse_frames(s.frames, len(rec.frames))]
        if not selected:
            raise ValueError(f"--frames {s.frames!r} selects nothing from {len(rec.frames)} frames")
        out = Path(s.out) if s.out else rec.root / f"render_{s.look.value}"
        beauty_dir, comp_dir, exr_dir = out / "beauty", out / "frames", out / "exr"
        for d in (beauty_dir, comp_dir) + ((exr_dir,) if s.keep_exr else ()):
            d.mkdir(parents=True, exist_ok=True)

        signature = _signature(s, rec, selected)
        if s.resume:
            _check_resume(out, signature)

        log(f"recording {rec.path}")
        log(f"scenario  {rec.scenario_name or 'unknown'}: {len(rec.frames)} frames recorded, {len(selected)} selected")
        for body in rec.bodies:
            note = "" if body.scale_recorded else "  (scale not recorded: pass --scale if this body was scaled)"
            textures = "found" if body.mtl_dir else "not found (Kd colour)"
            log(f"body {body.index}    {body.name}: {len(body.faces)} tris, scale {body.scale:g}, "
                f"{'fixed' if body.fixed else 'dynamic'}, {len(body.variants)} atlas variant(s), "
                f"mtl textures {textures}{note}")  # fmt: skip

        chosen = scene.select_variant(s.variant)
        log(f"mitsuba   {scene.mitsuba().__version__} ({chosen})")

        # Materials at render resolution, sized from the first selected frame's wear atlas.
        blend = BlendParams()
        first, last = selected[0], selected[-1]
        mapsets, sizes, initial, last_hash = {}, {}, {}, {}
        for body in tqdm(rec.bodies, desc="materials", unit="body", disable=not progress, leave=False):
            wear = rec.texture(first, body.index)
            sizes[body.index] = materials.texture_size(body, None if wear is None else wear.shape, s.max_texture)
            mapsets[body.index] = materials.build_mapsets(body, sizes[body.index])
            wear_r = None if wear is None else materials.resize_scalar(wear, sizes[body.index])
            initial[body.index] = body_textures(s.look, wear_r, *mapsets[body.index], blend)
            last_hash[body.index] = _digest(wear)

        poses = cam.camera_path(rec, selected, s.camera, s.width / s.height)
        wscene = scene.WearScene(
            rec.bodies,
            initial,
            {b.index: materials.metallic(b) for b in rec.bodies},
            s.width,
            s.height,
            s.camera.fov,
            cam.scene_bounds(rec, rec.frames),
            s.lighting,
            up_axis=s.camera.up,
            max_depth=s.max_depth,
            azimuth=s.camera.azimuth,
        )
        denoiser = scene.Denoiser(s.width, s.height, s.denoise)
        panels = _panel_sources(rec, last, s)
        mover, travel = travel_by_frame(rec, s.camera.follow)
        multi_pass = bool(travel) and travel[rec.frames[-1].key][1] > 1
        area = None
        if s.chart and panels:
            area = _area_series(rec, selected, rec.body(panels[0].body_index), out, progress)

        log(f"render    {s.width}x{s.height}, {s.spp} spp, look={s.look.value}, camera={s.camera.mode}, "
            f"denoiser={'on' if denoiser.active else 'off'}, panels={[p.body_index for p in panels]} -> {out}")  # fmt: skip
        config = {"signature": signature, "status": "rendering", "mitsuba": chosen, "video_stem": s.stem,
                  "video": asdict(s.video), "texture_sizes": sizes,
                  "body_scales": {b.index: {"scale": b.scale, "recorded": b.scale_recorded} for b in rec.bodies}}  # fmt: skip
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
            positions, orientations = rec.pose(frame)
            wscene.set_poses(positions, orientations)
            panel_wear = {}
            for body in rec.bodies:
                wear = rec.texture(frame, body.index)
                panel_wear[body.index] = wear
                digest = _digest(wear)
                if digest == last_hash[body.index]:
                    continue  # textures only change when the wear does
                last_hash[body.index] = digest
                wear_r = None if wear is None else materials.resize_scalar(wear, sizes[body.index])
                wscene.set_textures(body.index, body_textures(s.look, wear_r, *mapsets[body.index], blend))
            wscene.set_camera(pose)
            rgb = denoiser(wscene.render(s.spp, seed=s.seed))
            if s.keep_exr:
                video.write_exr(exr_dir / f"frame_{i:06d}.exr", rgb)
            ldr = video.tonemap(rgb, s.exposure)
            Image.fromarray(ldr).save(beauty_dir / f"frame_{i:06d}.png")

            header = f"t = {frame.time:.3f} s"
            if mover is not None:
                slid_m, pass_no = travel[frame.key]
                header += f"    body {mover.index} slid {slid_m:.2f} m" + (f"  (pass {pass_no})" if multi_pass else "")
            sub = f"frame {frame.index}  |  {s.look.value} look  |  Mitsuba {s.spp} spp"
            composite = video.compose_frame(
                ldr,
                [(p, panel_wear.get(p.body_index)) for p in panels],
                header,
                sub,
                chart=(area, i) if area is not None else None,
            )
            composite.save(comp_path)
            timings.append(time.time() - t0)
            bar.set_postfix(frame=frame.index, s_per_frame=f"{timings[-1]:.1f}")
            bar.update()
        bar.close()

        outputs = encode_run(out, s.video, s.stem, log) if s.video.encode else {}
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
        rec.close()
