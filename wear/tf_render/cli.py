"""``tf-render``: Mitsuba 3 wear videos from TextureFriction ``--record`` runs.

    tf-render render  RUN  [--quality paper] [--camera orbit] [--title ...]   full render + video
    tf-render preview RUN                                                     fast low-res check of framing/look
    tf-render inspect RUN                                                     what the recording contains
    tf-render encode  OUT  [--fps 24] [--title ...]                           re-encode frames, no re-render
    tf-render fixture FILE.h5                                                 tiny synthetic recording

``RUN`` is a ``simulation_<ts>/`` directory or its ``FrictionTexture_*.h5``. ``tf-render RUN ...`` (no command) is
``tf-render render RUN ...``. Without installing: ``python tools/render.py ...`` from the TextureWear repo.
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import typer

from . import __version__
from .camera import CameraSettings
from .materials import NormalOptions
from .pipeline import (
    CONFIG_NAME,
    QUALITY_PRESETS,
    Look,
    RenderSettings,
    VideoSettings,
    encode_run,
    parse_scales,
    run_render,
    travel_by_frame,
    worn_area,
)
from .recording import Recording
from .scene import LightingSettings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Render TextureFriction wear recordings (HDF5) with Mitsuba 3 and encode videos.",
)

SOURCE_HELP = "simulation_<ts>/ directory or FrictionTexture_*.h5"
P_OUT, P_QUALITY, P_CAMERA, P_LIGHT, P_WEAR, P_VIDEO, P_ADV = (
    "Output",
    "Quality",
    "Camera",
    "Lighting",
    "Wear panel",
    "Video",
    "Advanced",
)


class CameraMode(str, Enum):
    fixed = "fixed"  # frames the dynamic bodies' whole path
    track = "track"  # follows one body
    orbit = "orbit"  # sweeps around the path's centre


Quality = Enum("Quality", {name: name for name in QUALITY_PRESETS}, type=str)


def _resolution(res: str | None, quality: Quality, spp: int | None) -> tuple[int, int, int]:
    width, height, preset_spp = QUALITY_PRESETS[quality.value]
    if res:
        try:
            width, height = (int(v) for v in res.lower().split("x"))
        except ValueError as exc:
            raise typer.BadParameter(f"--res {res!r} is not WIDTHxHEIGHT") from exc
    return width, height, spp or preset_spp


def _run(settings: RenderSettings) -> None:
    try:
        run_render(settings)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command()
def render(
    source: Path = typer.Argument(..., exists=True, help=SOURCE_HELP),
    out: Path = typer.Option(None, "--out", "-o", help="Output directory [default: <run>/render_<look>]",
                             rich_help_panel=P_OUT),
    look: Look = typer.Option(Look.heatmap, help="heatmap: worn material + wear colormap; worn: appearance only; "
                              "plain: wear ignored", rich_help_panel=P_OUT),
    frames: str = typer.Option("::1", help="Python slice over recorded frames, e.g. 0:300:2", rich_help_panel=P_OUT),
    resume: bool = typer.Option(False, help="Reuse frames already in --out (settings must match)",
                                rich_help_panel=P_OUT),
    quality: Quality = typer.Option("draft", "--quality", "-q", help="preview 960x540/16 spp, draft 1280x720/64, "
                                    "paper 1920x1080/256", rich_help_panel=P_QUALITY),
    res: str = typer.Option(None, help="WIDTHxHEIGHT, overrides --quality", rich_help_panel=P_QUALITY),
    spp: int = typer.Option(None, help="Samples per pixel, overrides --quality", rich_help_panel=P_QUALITY),
    max_depth: int = typer.Option(8, help="Maximum path depth", rich_help_panel=P_QUALITY),
    max_texture: int = typer.Option(2048, help="Cap on per-body texture resolution", rich_help_panel=P_QUALITY),
    normal_strength: float = typer.Option(1.0, help="Scale normal-map detail (1 = as authored, 0 = flat)",
                                          rich_help_panel=P_QUALITY),
    normal_recenter: float = typer.Option(5.0, help="Recentre normal maps whose mean tilt exceeds this many "
                                          "degrees (mis-encoded maps render black at grazing angles; 0 = off)",
                                          rich_help_panel=P_QUALITY),
    camera: CameraMode = typer.Option(CameraMode.fixed, help="Camera path", rich_help_panel=P_CAMERA),
    follow: int = typer.Option(None, help="Body to track [default: first dynamic body]", rich_help_panel=P_CAMERA),
    azimuth: float = typer.Option(-55.0, help="Degrees around the up axis", rich_help_panel=P_CAMERA),
    elevation: float = typer.Option(28.0, help="Degrees above the horizon", rich_help_panel=P_CAMERA),
    fov: float = typer.Option(35.0, help="Horizontal field of view (degrees)", rich_help_panel=P_CAMERA),
    zoom: float = typer.Option(1.0, help="> 1 moves the camera closer", rich_help_panel=P_CAMERA),
    orbit_degrees: float = typer.Option(120.0, help="Total sweep in --camera orbit", rich_help_panel=P_CAMERA),
    frame_all: bool = typer.Option(False, help="Frame every body, not just the dynamic bodies' paths",
                                   rich_help_panel=P_CAMERA),
    up: str = typer.Option("y", help="World up axis x | y | z (the simulator is y-up)", rich_help_panel=P_CAMERA),
    exposure: float = typer.Option(0.0, help="Exposure in stops before tone mapping", rich_help_panel=P_LIGHT),
    key_light: float = typer.Option(1.1, help="Key light irradiance at the scene centre", rich_help_panel=P_LIGHT),
    fill: float = typer.Option(0.18, help="Constant environment fill", rich_help_panel=P_LIGHT),
    rim_light: float = typer.Option(0.45, help="Rim light strength (0 disables)", rich_help_panel=P_LIGHT),
    envmap: Path = typer.Option(None, exists=True, help="HDR/EXR environment map (replaces the fill)",
                                rich_help_panel=P_LIGHT),
    ground: bool = typer.Option(True, help="Matte floor under the scene", rich_help_panel=P_LIGHT),
    panel: bool = typer.Option(True, help="UV wear-atlas side panel", rich_help_panel=P_WEAR),
    panel_body: list[int] = typer.Option([], help="Body shown in the panel (repeatable) [default: worn bodies]",
                                         rich_help_panel=P_WEAR),
    chart: bool = typer.Option(True, help="Worn-area vs time chart (data also in worn_area.csv)",
                               rich_help_panel=P_WEAR),
    fps: float = typer.Option(30.0, help="Video frame rate", rich_help_panel=P_VIDEO),
    title: str = typer.Option(None, help="Opening title card", rich_help_panel=P_VIDEO),
    subtitle: list[str] = typer.Option([], help="Title-card line (repeatable)", rich_help_panel=P_VIDEO),
    title_seconds: float = typer.Option(3.0, help="Title card duration", rich_help_panel=P_VIDEO),
    hold: float = typer.Option(2.0, help="Freeze the last frame (seconds)", rich_help_panel=P_VIDEO),
    gif: bool = typer.Option(False, help="Also write a GIF", rich_help_panel=P_VIDEO),
    crf: int = typer.Option(18, help="x264 quality (lower is better)", rich_help_panel=P_VIDEO),
    encode: bool = typer.Option(True, help="Encode the video after rendering", rich_help_panel=P_VIDEO),
    variant: str = typer.Option("auto", help="auto | cuda | llvm | cpu | scalar | <mitsuba variant>",
                                rich_help_panel=P_ADV),
    scale: list[str] = typer.Option([], help="Body scale override BODY=SCALE for recordings made before scale "
                                    "was exported", rich_help_panel=P_ADV),
    assets: list[Path] = typer.Option([], help="Extra root containing resources/ for .mtl textures (repeatable; "
                                      "also $TF_ASSET_ROOT)", rich_help_panel=P_ADV),
    denoise: bool = typer.Option(False, help="OptiX denoiser (CUDA only; check it helps on your driver)",
                                 rich_help_panel=P_ADV),
    seed: int = typer.Option(0, help="Sampler seed, fixed across frames for stable noise", rich_help_panel=P_ADV),
    keep_exr: bool = typer.Option(False, help="Also write linear EXR frames", rich_help_panel=P_ADV),
) -> None:
    """Path trace every selected frame, composite the wear panel and chart, and encode the video."""
    width, height, spp_ = _resolution(res, quality, spp)
    try:
        scales = parse_scales(scale)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _run(
        RenderSettings(
            source=source,
            out=out,
            look=look,
            width=width,
            height=height,
            spp=spp_,
            max_depth=max_depth,
            frames=frames,
            variant=variant,
            camera=CameraSettings(mode=camera.value, follow=follow, azimuth=azimuth, elevation=elevation, fov=fov,
                                  zoom=zoom, orbit_degrees=orbit_degrees, up=up, frame_all=frame_all),  # fmt: skip
            lighting=LightingSettings(key_strength=key_light, fill=fill, rim_strength=rim_light,
                                      envmap=str(envmap) if envmap else None, ground=ground),  # fmt: skip
            exposure=exposure,
            max_texture=max_texture,
            normals=NormalOptions(recenter_above_deg=normal_recenter, strength=normal_strength),
            panel=panel,
            panel_bodies=list(panel_body),
            chart=chart,
            scales=scales,
            asset_roots=list(assets),
            denoise=denoise,
            seed=seed,
            keep_exr=keep_exr,
            resume=resume,
            video=VideoSettings(fps=fps, encode=encode, gif=gif, crf=crf, title=title, subtitle=list(subtitle),
                                title_seconds=title_seconds, hold=hold),  # fmt: skip
        )
    )


@app.command()
def preview(
    source: Path = typer.Argument(..., exists=True, help=SOURCE_HELP),
    out: Path = typer.Option(None, "--out", "-o", help="Output directory [default: <run>/render_preview]"),
    look: Look = typer.Option(Look.heatmap, help="heatmap | worn | plain"),
    camera: CameraMode = typer.Option(CameraMode.fixed, help="Camera path"),
    frames: str = typer.Option("::8", help="Python slice over recorded frames"),
    variant: str = typer.Option("auto", help="Mitsuba variant preference"),
) -> None:
    """960x540 at 16 spp on every 8th frame: check framing and look in a minute before a long render."""
    width, height, spp = QUALITY_PRESETS["preview"]
    run_source = Path(source)
    default_out = (run_source if run_source.is_dir() else run_source.parent) / "render_preview"
    _run(
        RenderSettings(
            source=source,
            out=out or default_out,
            look=look,
            width=width,
            height=height,
            spp=spp,
            frames=frames,
            variant=variant,
            camera=CameraSettings(mode=camera.value),
            video=VideoSettings(fps=8.0, hold=1.0),
        )
    )


@app.command()
def inspect(
    source: Path = typer.Argument(..., exists=True, help=SOURCE_HELP),
    assets: list[Path] = typer.Option([], help="Extra root containing resources/ for .mtl textures"),
) -> None:
    """Summarize a recording: bodies, scale, materials, frames, resets and wear coverage."""
    with Recording(source, asset_roots=list(assets)) as rec:
        frames = rec.frames
        typer.echo(f"recording  {rec.path}")
        typer.echo(f"scenario   {rec.scenario_name or 'unknown'}")
        if frames:
            dt = (frames[-1].time - frames[0].time) / max(len(frames) - 1, 1)
            typer.echo(f"frames     {len(frames)}  (t = {frames[0].time:.3f} .. {frames[-1].time:.3f} s, "
                       f"dt ~ {dt:.4f} s)")  # fmt: skip
        mover, travel = travel_by_frame(rec)
        if mover is not None and frames:
            slid, passes = travel[frames[-1].key]
            typer.echo(f"motion     body {mover.index} slid {slid:.3f} m over {passes} pass(es)")
        typer.echo("")
        typer.echo(f"{'body':<5} {'name':<28} {'tris':>6} {'scale':>8} {'kind':<8} {'atlas':>5} {'mtl tex':<8} "
                   f"{'wear texels (first -> last)':>28} {'worn area':>11}")  # fmt: skip
        for body in rec.bodies:
            scale = f"{body.scale:g}" + ("" if body.scale_recorded else "?")
            first = last = 0
            area = 0.0
            if frames:
                counts, areas = worn_area(rec, [frames[0], frames[-1]], body, progress=False)
                first, last, area = int(counts[0]), int(counts[-1]), float(areas[-1])
            typer.echo(f"{body.index:<5} {body.name[:28]:<28} {len(body.faces):>6} {scale:>8} "
                       f"{'fixed' if body.fixed else 'dynamic':<8} {len(body.variants):>5} "
                       f"{'yes' if body.mtl_dir else 'no':<8} {f'{first} -> {last}':>28} {area:>9.4g} m2")  # fmt: skip
        if any(not b.scale_recorded for b in rec.bodies):
            typer.echo("\n? scale not recorded (pre-scale-export recording): pass --scale BODY=SCALE if scaled")


@app.command()
def encode(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="A render output directory (with frames/)"),
    fps: float = typer.Option(None, help="Frame rate [default: as rendered]"),
    title: str = typer.Option(None, help="Opening title card [default: as rendered]"),
    subtitle: list[str] = typer.Option([], help="Title-card line (repeatable)"),
    title_seconds: float = typer.Option(None, help="Title card duration"),
    hold: float = typer.Option(None, help="Freeze the last frame (seconds)"),
    gif: bool = typer.Option(False, help="Also write a GIF"),
    crf: int = typer.Option(None, help="x264 quality (lower is better)"),
    name: str = typer.Option(None, help="Output file stem [default: as rendered]"),
) -> None:
    """Re-encode already rendered frames (new fps, title card, hold) without path tracing again."""
    config_path = run_dir / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    v = VideoSettings(**config.get("video", {}))
    v.fps = fps or v.fps
    v.title = title if title is not None else v.title
    v.subtitle = list(subtitle) or v.subtitle
    v.title_seconds = title_seconds if title_seconds is not None else v.title_seconds
    v.hold = hold if hold is not None else v.hold
    v.crf = crf if crf is not None else v.crf
    v.gif = gif
    try:
        encode_run(run_dir, v, name or config.get("video_stem", "wear"), typer.echo)
    except FileNotFoundError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command()
def fixture(
    path: Path = typer.Argument(Path("FrictionTexture_synthetic.h5"), help="Output .h5 path"),
    frames: int = typer.Option(6, help="Number of frames"),
) -> None:
    """Write a tiny synthetic recording (plane + sliding box) for tests and smoke renders."""
    from .synthetic import write_synthetic

    typer.echo(write_synthetic(path, frames=frames))


@app.command()
def version() -> None:
    """Print tf-render and Mitsuba versions."""
    typer.echo(f"tf-render {__version__}")
    try:
        import mitsuba as mi

        typer.echo(f"mitsuba {mi.__version__}: {', '.join(v for v in mi.variants() if v.endswith('_rgb'))}")
    except ImportError:
        typer.echo("mitsuba not installed")
    typer.echo(f"numpy {np.__version__}")


COMMANDS = {"render", "preview", "inspect", "encode", "fixture", "version"}


def main(argv: list[str] | None = None) -> None:
    """Console entry point. ``tf-render RUN ...`` is shorthand for ``tf-render render RUN ...``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in COMMANDS and not args[0].startswith("-"):
        args.insert(0, "render")
    try:
        app(args=args, prog_name="tf-render")
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception:  # noqa: BLE001 - report, then hard-exit below
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Mitsuba/Dr.Jit can crash during interpreter teardown (seen with mitsuba 3.6.4 on Windows Python 3.13);
    # output is flushed, so skip teardown to keep the exit code meaningful.
    os._exit(code)
