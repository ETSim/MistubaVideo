"""Render pipeline: source -> per-frame Mitsuba renders -> composited frames -> video.

``run_render(settings)`` renders and encodes a sequence. The CLI builds its ``RenderSettings`` from command-line
options. With ``backend`` cycles or eevee, ``mitsuba_video.blender`` renders the beauty frames and saves the scene
package and ``scene.blend`` under ``<out>/blender/``. The pipeline then composites those frames. Output layout:

    <out>/beauty/frame_NNNNNN.png   tone-mapped 3D render
    <out>/frames/frame_NNNNNN.png   beauty + header + UV field panel + covered-area chart (what gets encoded)
    <out>/exr/frame_NNNNNN.exr      linear HDR (keep_exr)
    <out>/<field.area_csv>          covered texels / area per frame (the chart's data)
    <out>/render_config.json        settings signature, Mitsuba variant, timings, outputs
    <out>/<stem>.mp4 [.gif]         encoded video
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import camera as cam
from . import scene, video
from .annotate import Annotations, callouts, cluster_box_segments, format_depth, motion_arrows, track_depth
from .blend import BlendParams, VariantStack, blend_device, blend_stack, resolve_device
from .color import FieldDisplay, heatmap_albedo, heatmap_albedo_device
from .maps import NormalOptions, fit_size, resize_scalar
from .model import Body, FieldSpec, Frame
from .motion import travel_by_frame
from .source import Source, field_value, open_source

CONFIG_NAME = "render_config.json"
# Frame PNGs are intermediates for the encoder: zlib level 1 is ~5x faster to write than Pillow's default and still
# lossless.
FRAME_PNG = {"compress_level": 1}


class Look(str, Enum):
    material = "material"
    heatmap = "heatmap"  # affected material + labelled field colormap on covered texels
    worn = "worn"  # physically based base -> affected appearance only
    plain = "plain"  # base material, field ignored (reference render)


# name -> (width, height, spp); explicit --res / --spp override the preset.
QUALITY_PRESETS: dict[str, tuple[int, int, int]] = {
    "preview": (960, 540, 16),
    "draft": (1280, 720, 64),
    "paper": (1920, 1080, 256),
}
# Suggested sample budgets for guided OptiX on a box-filtered film. CLI presets still use QUALITY_PRESETS.
# These values come from `tf render quality` against a 1024 spp reference
# (2026-11-03, 1280x720, 3 frames each of four_grounds_dragged overhead and scratch_left_right close-up). Worn-region
# PSNR, denoised, 16/32/64/128 spp: overhead 35.6/36.5/37.4/38.2 dB, close-up 31.0/31.9/32.4/33.0 dB; 64 spp is the
# lowest within 1 dB of 128 on both. Paper (1080p, not measured) keeps 128 for headroom; preview stays a quick look.
QUALITY_SPP_DENOISED: dict[str, int] = {"preview": 16, "draft": 64, "paper": 128}


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
    look: Look = Look.plain
    shading: scene.Shading = scene.Shading.pbr
    face_normals: bool = False
    displacement_scale: float = 0.0
    width: int = 1920
    height: int = 1080
    spp: int = 128
    max_depth: int = 8
    rr_depth: int = 5  # Russian roulette starts at this bounce (Mitsuba's default); lower is faster, a little noisier
    frames: str = "::1"
    variant: str = "auto"
    camera: cam.CameraSettings = field(default_factory=cam.CameraSettings)
    lighting: scene.LightingSettings = field(default_factory=scene.LightingSettings)
    exposure: float = 0.0
    max_texture: int = 2048
    blend_device: str = "auto"  # Field blend: auto | cpu (numpy) | gpu (Dr.Jit CUDA); see blend.resolve_device
    # With resume: rebuild every composited frame (panels, chart, header) from the saved beauty frames instead of
    # path tracing again; a frame without a saved beauty is rendered. Composite-only settings may differ.
    recompose: bool = False
    # Composite-only labels (annotate.Annotations JSON): body tags, panel titles, the colour bar's unit. Not part of the
    # signature, so a finished render is re-labelled with --resume --recompose.
    annotate: Path | None = None
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
    # Who renders the beauty frames: Mitsuba (default), or Blender's "cycles" / "eevee" (mitsuba_video.blender). The
    # bodies, poses, camera path, lights and per-frame wear-blended textures are the same; only the renderer differs.
    backend: str = "mitsuba"
    blender: str | None = None  # Blender executable for the Blender backends [default: BLENDER, PATH, install]
    # Geometric displacement (mitsuba_video.displace): millimetres drawn for the full 0..1 height range; 0 = off.
    displace_mm: float = 0.0

    def stem(self, spec: FieldSpec) -> str:
        return f"{spec.video_prefix}_{self.look.value}_{self.camera.mode}"

    @property
    def renderer_label(self) -> str:
        """The renderer as people read it in the video header: "Mitsuba", "Blender Cycles", "Blender EEVEE"."""
        return {"cycles": "Blender Cycles", "eevee": "Blender EEVEE"}.get(self.backend, "Mitsuba")


def parse_frames(spec: str, count: int) -> range:
    """Python slice syntax over source frames: '::1', '0:300:2', '-60:'."""
    parts = (spec.split(":") + ["", "", ""])[:3]
    start, stop, step = (int(p) if p.strip() else None for p in parts)
    return range(count)[slice(start, stop, step)]


def resolve_field(source: Source, name: str | None) -> FieldSpec:
    key = name or source.primary_field
    if not source.fields and name is None:
        return FieldSpec(name="", label="", video_prefix="render")
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
    device: str = "cpu",
) -> dict:
    if look is Look.plain or value is None:
        return scene.encode_maps(stack.base)
    scratches = blend.scratches > 0.0 and direction is not None and direction.shape[:2] == value.shape
    if device == "gpu" and not scratches:
        # Blend, heatmap overlay and encoding stay on the device: the textures are device tensors that
        # FieldScene.set_textures assigns without a download or re-upload. Only the CPU scratch pass needs the maps
        # on the host (the blend_stack branch below).
        dm = blend_device(value, stack, blend, texel_scale)
        base_color = heatmap_albedo_device(dm.albedo, dm.field, display, threshold) if look is Look.heatmap else None
        return scene.encode_maps_device(dm, base_color)
    maps = blend_stack(value, stack, blend, texel_scale, direction=direction, device=device)
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
    device: str = "cpu",
):
    """Mitsuba textures for one body; ``texel_scale`` is render width / field width (for the wear blur).
    ``direction`` is the slide direction at the same size (image axes), for ``blend.scratches``. ``device`` is the
    resolved blend device (cpu | gpu); on the GPU a worn body's textures are ``drjit.cuda.TensorXf`` (H, W, C) that
    never leave the device, unless scratches need the CPU pass.

    Unworn bodies (no field, or an all-zero one) give the same textures for every body sharing ``stack``, so those
    are encoded once per stack and look settings and shared: treat the returned arrays as read-only."""
    if not (look is Look.plain or value is None or not value.any()):
        return _blended_textures(look, value, stack, blend, display, threshold, texel_scale, direction, device)
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
                        height=[h[rows, cols] for h in stack.height],
                        metallic=[m[rows, cols] if np.ndim(m) else m for m in stack.metallic])  # fmt: skip


class BodyTextures:
    """One body's textures across frames, re-blended only where its field changed.

    A texel's textures depend on the field within the wear blur's reach (at most 5 texels), so each changed
    ``TILE``-sized tile is re-blended over the tile grown by that reach, from a crop grown by the reach again. The
    result equals a full ``body_textures`` rebuild; a 2048^2 plate with a few new scratches costs a few tiles.
    On the GPU (``device`` gpu) the whole body is re-blended: one launch is cheaper than cropping tiles, the stack
    stays on the device, and so do the encoded textures (``scene.encode_maps_device``).
    """

    TILE = 128

    def __init__(self, look: Look, stack: VariantStack, blend: BlendParams, display: FieldDisplay, threshold: float,
                 texel_scale: float, device: str = "cpu") -> None:  # fmt: skip
        self.args = (look, stack, blend, display, threshold, texel_scale)
        self.device = device
        px = blend.wear_blur_px * texel_scale
        self.reach = int(np.floor(min(px, 4.0))) + 1 if blend.wear_blur_px > 0.0 and px > 0.01 else 0
        self.value: np.ndarray | None = None
        self.textures: dict[str, np.ndarray] = {}
        self._owned = False  # False while self.textures may be the shared unworn arrays

    def update(self, value: np.ndarray | None, direction: np.ndarray | None = None) -> dict[str, np.ndarray]:
        look, stack, blend, display, threshold, texel_scale = self.args
        # Burnish and scratches reach further than a tile's margin: rebuild the whole body when they are on.
        if (look is Look.plain or value is None or self.value is None or value.shape != self.value.shape
                or not self.textures or blend.wide or self.device == "gpu"):  # fmt: skip
            self.textures = body_textures(look, value, stack, blend, display, threshold, texel_scale, direction,
                                          self.device)  # fmt: skip
            self._owned = False
        else:
            changed = value != self.value
            if not changed.any():
                return self.textures
            if not self._owned:
                self.textures = {k: v.copy() for k, v in self.textures.items()}
                self._owned = True
            self._update_tiles(value, changed)
        # The tile diff (CPU only) needs a private copy; the GPU path never reads it back.
        self.value = None if value is None or self.device == "gpu" else value.copy()
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
        value = field_value(source, frame, body.index, spec.name)
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
        title=f"{spec.area_title}, body {body.index} ({shown}Â²)",
        times=np.array([f.time for f in frames]),
        values=area * (1e4 if small else 1.0),
        unit=f"{shown}Â²",
    )


MAX_PANELS = 6  # compose_frame lays more than three out in two columns


def _panel_sources(source: Source, last: Frame, s: RenderSettings, spec: FieldSpec,
                   notes: Annotations | None = None) -> list[video.PanelSource]:  # fmt: skip
    """Bodies in the side panel: explicit ``panel_bodies``, else the (up to three) bodies with the most covered
    texels by ``last``, so twenty barely-touched cubes do not crowd out the plate they scratched."""
    if not s.panel or not source.fields:
        return []
    bodies = {b.index: b for b in source.bodies}
    if s.panel_bodies:
        chosen = [(idx, field_value(source, last, idx, spec.name)) for idx in s.panel_bodies if idx in bodies]
        chosen = [(idx, value) for idx, value in chosen if value is not None][:MAX_PANELS]
    else:
        covered = []
        for idx in bodies:
            value = field_value(source, last, idx, spec.name)
            count = 0 if value is None else int((value > spec.threshold).sum())
            if count:
                covered.append((count, idx, value))
        covered.sort(key=lambda item: (-item[0], item[1]))
        chosen = [(idx, value) for _, idx, value in covered[:MAX_PANELS]]
    caption = (notes.field_label if notes and notes.field_label else spec.label.split(",")[0])
    titles = {idx: (notes.panel_title(idx, caption) if notes else None) for idx, _ in chosen}
    return [video.PanelSource(idx, titles[idx] or f"body {idx}: {bodies[idx].label} (UV atlas)",
                              video.wear_roi(value, threshold=spec.threshold))
            for idx, value in chosen]  # fmt: skip


def _same_field(a: np.ndarray | None, b: object) -> bool:
    """True when a body's field did not change. Sources that cache decodes hand back the same
    object for unchanged data, so the identity test settles most bodies; the comparison covers fresh arrays."""
    if a is b:
        return True
    return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape
            and bool(np.array_equal(a, b)))  # fmt: skip


_UNSET = object()


@dataclass
class _FrameInput:
    """What the render loop needs from the source for one frame, read ahead on the reader thread."""

    panel_values: dict[int, np.ndarray | None]
    positions: np.ndarray | None = None
    orientations: np.ndarray | None = None
    # the source frame before this one (motion arrows): its positions and the time between the two
    previous: np.ndarray | None = None
    dt: float = 0.0
    # body -> (field resized to the body's texture size, slide direction or None) for bodies whose field changed
    changed: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = field(default_factory=dict)
    # the frame's contact clusters as world oriented boxes (annotate cluster_boxes), when asked for
    clusters: list = field(default_factory=list)


class _FieldReader:
    """Reads frames ahead on one background thread: poses, fields, change detection, resizing and slide directions.
    One thread, because an h5py file must not be read from two threads at once; it is the only reader while the loop
    runs, and frames are read in the order they are submitted, so ``last`` needs no lock."""

    def __init__(self, source: Source, spec: FieldSpec, s: RenderSettings, sizes: dict[int, tuple[int, int]],
                 panel_bodies: list[int], last: dict[int, np.ndarray | None],
                 read_clusters: bool = False) -> None:  # fmt: skip
        self.source, self.spec, self.settings, self.sizes = source, spec, s, sizes
        # Only sources that record contact clusters have them.
        self.read_clusters = read_clusters and hasattr(source, "cluster_boxes")
        self.scene_bodies = {b.index for b in source.bodies if b.index not in s.hidden_bodies}
        self.panel_bodies = list(panel_bodies)
        self.read_bodies = sorted(self.scene_bodies | set(self.panel_bodies))
        self.last = dict(last)  # body -> field its textures were last built from
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="field-reader")

    def submit(self, frame: Frame, recompose: bool) -> Future:
        return self._pool.submit(self._read, frame, recompose)

    def _read(self, frame: Frame, recompose: bool) -> _FrameInput:
        source, name = self.source, self.spec.name
        positions, orientations = source.pose(frame)  # the composite's body callouts need them too
        previous, dt = None, 0.0
        if frame.position > 0:  # the source frame before (not the previous rendered one): velocity for the arrows
            before = source.frames[frame.position - 1]
            previous, dt = source.pose(before)[0], frame.time - before.time
        clusters = source.cluster_boxes(frame) if self.read_clusters else []
        if recompose:
            return _FrameInput({idx: field_value(source, frame, idx, name) for idx in self.panel_bodies},
                               positions, orientations, previous, dt, clusters=clusters)  # fmt: skip
        out = _FrameInput({}, positions, orientations, previous, dt, clusters=clusters)
        for idx in self.read_bodies:
            value = field_value(source, frame, idx, name)
            out.panel_values[idx] = value
            if idx not in self.scene_bodies or _same_field(value, self.last.get(idx, _UNSET)):
                continue  # textures only change when the field does
            self.last[idx] = value
            value_r = None if value is None else resize_scalar(value, self.sizes[idx])
            out.changed[idx] = (value_r, _direction_at(source, frame, idx, self.sizes[idx], self.settings.blend))
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class _FrameWriter:
    """Tone mapping, the beauty/EXR saves and the composite on one background thread, at most ``depth`` frames in
    flight: ``submit`` blocks while the queue is full and re-raises a failed write."""

    def __init__(self, depth: int = 2) -> None:
        self.depth = depth
        self.busy = 0.0  # seconds the writer thread spent writing (only that thread adds to it)
        self._pending: deque[Future] = deque()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-writer")

    def _run(self, job: Callable[[], None]) -> None:
        t0 = time.time()
        job()
        self.busy += time.time() - t0

    def submit(self, job: Callable[[], None]) -> None:
        while len(self._pending) >= self.depth:
            self._pending.popleft().result()
        self._pending.append(self._pool.submit(self._run, job))

    def drain(self) -> None:
        while self._pending:
            self._pending.popleft().result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def _sync_device(device: str) -> None:
    """Wait for queued Dr.Jit kernels, so the blend step's time includes them (they would otherwise land in render)."""
    if device == "gpu":
        import drjit as dr

        dr.sync_thread()


def _write_frame(rgb: np.ndarray | None, ldr: np.ndarray | None, exposure: float, beauty_path: Path,
                 exr_path: Path | None, comp_path: Path,
                 compose: Callable[[np.ndarray], Image.Image]) -> None:  # fmt: skip
    """The writer thread's job for one frame: EXR, tone map and beauty PNG (when ``rgb`` is given; ``ldr`` is an
    already saved beauty for --recompose), then the composite."""
    if rgb is not None:
        if exr_path is not None:
            video.write_exr(exr_path, rgb)
        ldr = video.tonemap(rgb, exposure)
        Image.fromarray(ldr).save(beauty_path, **FRAME_PNG)
    compose(ldr).save(comp_path, **FRAME_PNG)


# --- config / resume -----------------------------------------------------------------------------------------


def _signature(s: RenderSettings, source: Source, spec: FieldSpec, frames: list[Frame]) -> dict:
    """Everything that changes the pixels of a frame; a resumed render must match it exactly."""
    return {
        "source": str(source.path.resolve()),
        "source_kind": source.kind,
        "source_options": {k: str(v) for k, v in s.source_options.items()},
        "field": spec.name,
        "look": s.look.value,
        "shading": s.shading.value,
        "face_normals": s.face_normals,
        "displacement_scale": s.displacement_scale,
        "resolution": [s.width, s.height],
        "spp": s.spp,
        "max_depth": s.max_depth,
        # only when set away from Mitsuba's default, so renders made before the option still resume
        **({"rr_depth": s.rr_depth} if s.rr_depth != 5 else {}),
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
        # only for the Blender backends, so Mitsuba renders made before the option still resume
        **({"backend": s.backend} if s.backend != "mitsuba" else {}),
        # only when on, so renders made before the option still resume
        **({"displace_mm": s.displace_mm} if s.displace_mm > 0 else {}),
    }


# Signature keys that only change the composite around the beauty frame (side panels, chart, field colours).
COMPOSITE_KEYS = frozenset({"panel", "panel_bodies", "chart", "display"})


def _check_resume(out: Path, signature: dict, ignore: frozenset[str] = frozenset()) -> None:
    path = out / CONFIG_NAME
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8")).get("signature", {})
    current = json.loads(json.dumps(signature, default=str))  # same JSON round trip the stored copy went through
    for key, default in (("shading", "pbr"), ("face_normals", False), ("displacement_scale", 0.0)):
        if key in current:
            previous.setdefault(key, default)
    if "normals" in current and isinstance(previous.get("normals"), dict):
        previous["normals"].setdefault("cavity_ao", 0.0)
    for sig in (previous, current):  # the same HDR under a moved folder (resources/environement -> environment,
        light = sig.get("lighting")  # 2026-11-05) lights the same frames: compare the envmap by file name
        if isinstance(light, dict) and light.get("envmap"):
            light["envmap"] = Path(str(light["envmap"]).replace("\\", "/")).name
    blend = previous.get("blend")  # renders from before BlendParams.wear_gamma (2026-11-05) were linear
    if isinstance(blend, dict):
        blend.setdefault("wear_gamma", 1.0)
    diff = sorted(k for k in set(previous) | set(current) if k not in ignore and previous.get(k) != current.get(k))
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


@dataclass
class SceneSetup:
    """A built scene for ``selected``: the per-body textures (blended for the first frame), the Mitsuba scene, the
    denoiser and the camera pose of every selected frame."""

    sizes: dict[int, tuple[int, int]]
    textures: dict[int, BodyTextures]
    first_fields: dict[int, np.ndarray | None]  # the fields the textures were built from
    fscene: scene.FieldScene
    denoiser: scene.Denoiser
    poses: list[cam.CameraPose]


def scene_bodies_of(s: RenderSettings, source: Source) -> list[Body]:
    """The bodies in the 3D scene (all but ``--hide``)."""
    bodies = [b for b in source.bodies if b.index not in s.hidden_bodies]
    if not bodies:
        raise ValueError("--hide leaves no body to render")
    return bodies


def texture_sizes(s: RenderSettings, source: Source, spec: FieldSpec, first: Frame
                  ) -> tuple[dict[int, tuple[int, int]], dict[int, float]]:  # fmt: skip
    """Per scene body: the render texture size (width, height), from the first selected frame's field atlas (or the
    material's own size) capped at ``max_texture``, and render width / field width (the wear blur's texel scale)."""
    sizes, scales = {}, {}
    for body in scene_bodies_of(s, source):
        value = field_value(source, first, body.index, spec.name)
        native = value.shape if value is not None else body.material.preferred_size()
        sizes[body.index] = fit_size(native, s.max_texture)
        scales[body.index] = sizes[body.index][0] / native[1] if native else 1.0
    return sizes, scales


def build_scene(s: RenderSettings, source: Source, spec: FieldSpec, selected: list[Frame], device: str,
                log: Callable[[str], None] = tqdm.write, progress: bool = True) -> SceneSetup:  # fmt: skip
    """Materials at render resolution (sized from the first selected frame's field atlas), blended for that frame,
    and the scene around them. Needs ``scene.select_variant`` first; ``device`` is the resolved blend device."""
    first = selected[0]
    scene_bodies = scene_bodies_of(s, source)
    stacks, textures, first_fields = {}, {}, {}
    sizes, scales = texture_sizes(s, source, spec, first)
    for body in tqdm(scene_bodies, desc="materials", unit="body", disable=not progress, leave=False):
        value = field_value(source, first, body.index, spec.name)
        stacks[body.index] = material_stack(body, sizes[body.index], s.normals, log)
        log(f"blend     body {body.index}: {stacks[body.index].describe()}")
        value_r = None if value is None else resize_scalar(value, sizes[body.index])
        textures[body.index] = BodyTextures(s.look, stacks[body.index], s.blend, s.display, spec.threshold,
                                            scales[body.index], device)  # fmt: skip
        textures[body.index].update(value_r, _direction_at(source, first, body.index, sizes[body.index], s.blend))
        first_fields[body.index] = value

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
        rr_depth=s.rr_depth,
        azimuth=s.camera.azimuth,
        denoise_guides=s.denoise,
        shading=s.shading,
        face_normals=s.face_normals,
        displace=s.displace_mm * {"m": 1e-3, "mm": 1.0}.get(getattr(source, "length_unit", "m"), 1e-3),
        log=log,
    )
    denoiser = scene.Denoiser(s.width, s.height, s.denoise, use_guides=True)
    return SceneSetup(sizes, textures, first_fields, fscene, denoiser, poses)


# --- the job -------------------------------------------------------------------------------------------------


def run_render(s: RenderSettings, log: Callable[[str], None] = tqdm.write, progress: bool = True) -> Path:
    """Render ``s.source`` and return the output directory."""
    if min(s.width, s.height, s.spp, s.max_texture) <= 0:
        raise ValueError("resolution, spp and max_texture must be positive")
    t_start = time.time()
    source = open_source(s.source, s.source_kind, **s.source_options)
    try:
        if not source.frames or not source.bodies:
            raise ValueError(f"{source.path} has no frames or no renderable bodies")
        if s.displacement_scale != 0:
            from .geometry import displace_body
            source.bodies = [displace_body(b, s.displacement_scale, s.max_texture, log) for b in source.bodies]
        if s.backend != "mitsuba" and (s.shading != scene.Shading.pbr or s.face_normals):
            raise ValueError("--shading and --face-normals currently require --backend mitsuba")
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
            _check_resume(out, signature, COMPOSITE_KEYS if s.recompose else frozenset())
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

        on_blender = s.backend != "mitsuba"
        if on_blender:
            from . import blender  # Mitsuba-free path

            if s.backend not in blender.ENGINES:
                raise ValueError(f"unknown backend {s.backend!r}; use mitsuba, {', '.join(blender.ENGINES)}")
            blender_exe = blender.find_blender(s.blender)
            chosen = blender.blender_version(blender_exe)
            log(f"blender   {chosen} ({blender_exe}), engine {s.backend}")
        else:
            chosen = scene.select_variant(s.variant)
            log(f"mitsuba   {scene.mitsuba().__version__} ({chosen})")
        device, why = resolve_device(s.blend_device, s.blend)
        log(f"blend     device {device} ({why}; requested {s.blend_device})")

        last = selected[-1]
        if on_blender:
            sizes, _ = texture_sizes(s, source, spec, selected[0])
            poses = cam.camera_path(source, selected, s.camera, s.width / s.height)
            textures, fscene, denoiser, first_fields = {}, None, None, {}
        else:
            setup = build_scene(s, source, spec, selected, device, log, progress)
            sizes, textures, poses = setup.sizes, setup.textures, setup.poses
            fscene, denoiser, first_fields = setup.fscene, setup.denoiser, setup.first_fields
        notes = Annotations.load(s.annotate)
        panels = _panel_sources(source, last, s, spec, notes)
        mover, travel = travel_by_frame(source, s.camera.follow, "reset_detection" in source.capabilities)
        multi_pass = bool(travel) and travel[source.frames[-1].key][1] > 1
        travel_word = getattr(source, "travel_label", "travel")
        unit = getattr(source, "length_unit", "m")
        area = None
        if s.chart and panels:
            body = next(b for b in source.bodies if b.index == panels[0].body_index)
            area = _area_series(source, selected, body, spec, out, progress)

        stem = s.stem(spec)
        denoise_on = (s.denoise and s.backend == "cycles") if on_blender else denoiser.active
        log(f"render    {s.width}x{s.height}, {s.spp} spp, look={s.look.value}, camera={s.camera.mode}, "
            f"denoiser={'on' if denoise_on else 'off'}, panels={[p.body_index for p in panels]} -> {out}")  # fmt: skip
        config = {"signature": signature, "status": "rendering", "blender" if on_blender else "mitsuba": chosen,
                  "backend": s.backend, "video_stem": stem,
                  "blend_device": device,
                  "video": asdict(s.video), "texture_sizes": sizes,
                  "bodies": {b.index: {"name": b.name, "scale": b.scale, **b.notes} for b in source.bodies}}  # fmt: skip
        _write_config(out, config)

        timings: list[float] = []
        # Seconds per step, summed over rendered frames: where a slow render spends its time. Each step is time on the
        # render loop's thread: "fields" is the wait for the reader thread (plus posing the meshes), "write" the wait
        # for a free slot in the writer's queue (and the final drain).
        steps = dict.fromkeys(("fields", "blend", "upload", "render", "denoise", "write"), 0.0)
        done = {i for i in range(len(selected)) if s.resume and not s.recompose
                and (comp_dir / f"frame_{i:06d}.png").is_file()}  # fmt: skip
        if done:
            log(f"resume    reusing {len(done)} already rendered frame(s)")
        todo = [i for i in range(len(selected)) if i not in done]
        recompose = {i for i in todo if s.recompose and (beauty_dir / f"frame_{i:06d}.png").is_file()}
        if on_blender:
            # Blender renders every beauty frame up front (a resumed render keeps the ones on disk); the loop below
            # then only composites, exactly like --recompose.
            need = [i for i in todo if not (s.resume and (beauty_dir / f"frame_{i:06d}.png").is_file())]
            if need:
                t_b = time.time()
                info = blender.render_beauty(s, source, spec, selected, need, sizes, poses, device, out,
                                             blender_exe, log, progress)  # fmt: skip
                config["blender_run"] = {**info, "seconds_per_frame": (time.time() - t_b) / len(need)}
                _write_config(out, config)
            recompose = set(todo)

        def header_lines(frame: Frame) -> tuple[str, str]:
            if notes is not None and not notes.header:
                return "", ""
            if notes is not None:  # plain words for viewers who have not seen the code
                header = f"simulated time {frame.time:.2f} s"
                if mover is not None and notes.mover_label:
                    dist, pass_no = travel[frame.key]
                    header += f"    {notes.mover_label} {travel_word} {dist:.2f} {unit}"
                kind = "real-time render" if s.backend == "eevee" else "offline path-traced render"
                return header, f"{kind} of the recorded simulation ({s.renderer_label}, {s.spp} spp)"
            header = f"t = {frame.time:.3f} s"
            if mover is not None:
                dist, pass_no = travel[frame.key]
                header += f"    body {mover.index} {travel_word} {dist:.2f} {unit}"
                header += f"  (pass {pass_no})" if multi_pass else ""
            return header, f"frame {frame.index}  |  {s.look.value} look  |  {s.renderer_label} {s.spp} spp"

        # Where the bodies end up: fixed bodies' tags avoid it from the first frame on (annotate.callouts).
        settled = None
        if notes and notes.body_labels:
            end_pos, end_quat = source.pose(last)
            settled = (np.asarray(end_pos), np.asarray(end_quat))

        def composer(i: int, frame: Frame, data: _FrameInput) -> Callable[[np.ndarray], Image.Image]:
            header, sub = header_lines(frame)
            items = [(p, data.panel_values.get(p.body_index)) for p in panels]
            values = {}
            show_depth = (notes is not None and notes.field_scale > 0
                          and (notes.depth_tail_frames <= 0 or i >= len(selected) - notes.depth_tail_frames))  # fmt: skip
            if show_depth:
                for idx in notes.body_labels:
                    depth = track_depth(data.panel_values.get(idx), notes.field_scale, spec.threshold)
                    if depth is not None:
                        text = (format_depth(depth, notes.field_scale) if notes.field_unit == "mm"
                                else f"{depth:.3g} {notes.field_unit}")  # fmt: skip
                        if text != "â€”":  # a clamped depth is not a number: the tag stays a label
                            values[idx] = text
            tags = (callouts([b for b in source.bodies if b.index not in s.hidden_bodies], data.positions, data.orientations, poses[i], s.camera.fov,
                             (s.width, s.height), notes.body_labels, values, settled) if notes else [])  # fmt: skip
            shown = [b for b in source.bodies if b.index not in s.hidden_bodies]
            arrows = (motion_arrows(shown, data.positions, data.orientations, data.previous, data.dt, poses[i], s.camera.fov,
                                    (s.width, s.height), max_arrows=notes.max_arrows)
                      if notes and notes.arrows else [])  # fmt: skip
            boxes = (cluster_box_segments(data.clusters, poses[i], s.camera.fov, (s.width, s.height))
                     if notes and notes.cluster_boxes else [])  # fmt: skip
            label = notes.field_label if notes and notes.field_label else spec.label
            top = notes.field_max if notes else ""
            return lambda ldr: video.compose_frame(ldr, items, header, sub,
                                                   chart=(area, i) if area is not None else None,
                                                   display_max=s.display.max, field_label=label,
                                                   field_threshold=spec.threshold, field_max_text=top,
                                                   callouts=tags, arrows=arrows, cluster_boxes=boxes)  # fmt: skip

        # Reused frames start the bar (initial=) so its rate and ETA reflect path tracing only.
        bar = tqdm(total=len(selected), initial=len(done), desc="render", unit="frame", disable=not progress,
                   dynamic_ncols=True)  # fmt: skip
        # The next frame's fields are read on one reader thread while this one blends and traces, and one writer
        # thread tone-maps, saves and composites the previous frames (at most two in flight).
        # Labelled bodies are read too: their tags carry the measured track depth (composite only).
        read_for = [p.body_index for p in panels]
        if notes and notes.field_scale > 0:
            read_for += [i for i in notes.body_labels if i not in read_for and i in {b.index for b in source.bodies}]
        reader = _FieldReader(source, spec, s, sizes, read_for, first_fields,
                              read_clusters=bool(notes and notes.cluster_boxes))
        writer = _FrameWriter(depth=2)
        t_loop = time.time()
        try:
            pending = reader.submit(selected[todo[0]], todo[0] in recompose) if todo else None
            for n, i in enumerate(todo):
                frame, pose = selected[i], poses[i]
                comp_path = comp_dir / f"frame_{i:06d}.png"
                beauty_path = beauty_dir / f"frame_{i:06d}.png"
                t0 = time.time()
                data = pending.result()
                pending = reader.submit(selected[todo[n + 1]], todo[n + 1] in recompose) if n + 1 < len(todo) else None
                if i in recompose:
                    # Composite only: the path-traced frame is on disk.
                    with Image.open(beauty_path) as im:
                        ldr = np.asarray(im.convert("RGB"))
                    t_w = time.time()
                    steps["fields"] += t_w - t0
                    writer.submit(lambda ldr=ldr, b=beauty_path, c=comp_path, f=composer(i, frame, data):
                                  _write_frame(None, ldr, s.exposure, b, None, c, f))  # fmt: skip
                    steps["write"] += time.time() - t_w
                    timings.append(time.time() - t0)
                    bar.update()
                    continue
                fscene.set_poses(data.positions, data.orientations)
                fscene.set_camera(pose)
                t_blend = t_upload = 0.0
                for idx, (value_r, direction) in data.changed.items():
                    tb = time.time()
                    maps = textures[idx].update(value_r, direction)
                    tu = time.time()
                    fscene.set_textures(idx, maps)
                    t_blend += tu - tb
                    t_upload += time.time() - tu
                if data.changed:
                    tb = time.time()
                    _sync_device(device)
                    t_blend += time.time() - tb
                t_r = time.time()
                steps["fields"] += t_r - t0 - t_blend - t_upload
                steps["blend"] += t_blend
                steps["upload"] += t_upload
                rendered = fscene.render(s.spp, seed=s.seed)
                t_d = time.time()
                steps["render"] += t_d - t_r
                if isinstance(rendered, tuple):
                    rgb, albedo, normals = rendered
                    rgb = denoiser(rgb, albedo=albedo, normals=normals, to_sensor=fscene.to_sensor())
                else:
                    rgb = denoiser(rendered)
                t_w = time.time()
                steps["denoise"] += t_w - t_d
                exr_path = exr_dir / f"frame_{i:06d}.exr" if s.keep_exr else None
                compose = composer(i, frame, data)
                writer.submit(lambda rgb=rgb, b=beauty_path, e=exr_path, c=comp_path, f=compose:
                              _write_frame(rgb, None, s.exposure, b, e, c, f))  # fmt: skip
                steps["write"] += time.time() - t_w
                timings.append(time.time() - t0)
                bar.set_postfix(frame=frame.index, s_per_frame=f"{timings[-1]:.1f}", blend=f"{t_blend:.2f}",
                                render=f"{t_d - t_r:.1f}")  # fmt: skip
                bar.update()
            t_w = time.time()
            writer.drain()
            steps["write"] += time.time() - t_w
        finally:
            reader.close()
            writer.close()
        loop_seconds = time.time() - t_loop
        bar.close()
        if timings:
            per = {k: v / len(timings) for k, v in steps.items()}
            log(f"timing    s/frame over {len(timings)} frames: " + ", ".join(f"{k} {v:.2f}" for k, v in per.items())
                + f" (blend on {device}); wall {loop_seconds / len(timings):.2f}, "
                f"writer thread {writer.busy / len(timings):.2f} in the background")  # fmt: skip
            config["timing_s_per_frame"] = per
            config["writer_thread_s_per_frame"] = writer.busy / len(timings)

        outputs = encode_run(out, s.video, stem, log) if s.video.encode else {}
        config.update(
            status="done",
            outputs=outputs,
            seconds_per_frame=loop_seconds / len(timings) if timings else None,
            rendered_frames=len(timings),
            total_seconds=time.time() - t_start,
        )
        _write_config(out, config)
        return out
    finally:
        source.close()
