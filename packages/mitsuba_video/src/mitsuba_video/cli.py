"""Render 3D frame sequences and videos with optional scalar fields.

    mitsuba-video render  SOURCE [-q paper] [--camera orbit] [--title ...]   full render + video
    mitsuba-video preview SOURCE                                             fast low-res check of framing/look
    mitsuba-video inspect SOURCE                                             what the source contains
    mitsuba-video encode  OUT [--fps 24] [--title ...]                       re-encode frames, no re-render
    mitsuba-video fixture PATH --kind cube-throw|manifest               tiny synthetic input
    mitsuba-video validate MANIFEST | export manifest SOURCE OUT | sources | version

``SOURCE`` can be a manifest, USD stage or input provided by an installed source plugin.
``mitsuba-video SOURCE ...`` is shorthand for ``mitsuba-video render SOURCE ...``.
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from pathlib import Path

import typer

from . import __version__
from .camera import CameraSettings
from .color import FieldDisplay
from .maps import NormalOptions
from .pipeline import (
    CONFIG_NAME,
    QUALITY_PRESETS,
    Look,
    RenderSettings,
    VideoSettings,
    covered_area,
    encode_run,
    resolve_field,
    run_render,
)
from .scene import LightingSettings, Shading
from .source import available_sources, open_source, parse_options

P_OUT, P_QUALITY, P_CAMERA, P_LIGHT, P_FIELD, P_VIDEO, P_ADV = (
    "Output", "Quality", "Camera", "Lighting", "Field panel", "Video", "Advanced",
)  # fmt: skip
COMMANDS = {"render", "preview", "inspect", "encode", "fixture", "validate", "export", "sources", "version", "doctor", "tui"}


class Backend(str, Enum):
    mitsuba = "mitsuba"  # Mitsuba 3 path tracer (default)
    cycles = "cycles"  # Blender Cycles path tracer (mitsuba_video.blender)
    eevee = "eevee"  # Blender EEVEE rasteriser: fast, approximate



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
    samples = preset_spp if spp is None else spp
    if min(width, height, samples) <= 0:
        raise typer.BadParameter("resolution and samples per pixel must be positive")
    return width, height, samples


def _source_options(option: list[str]) -> dict[str, object]:
    try:
        opts: dict[str, object] = dict(parse_options(option))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return opts


def _fail(exc: Exception) -> typer.Exit:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def build_app(default_source: str = "auto", prog: str = "mitsuba-video",
              default_look: Look = Look.plain, fixture_kind: str = "cube-throw") -> typer.Typer:  # noqa: C901
    """Build a general renderer CLI; source plugins can choose their own defaults."""
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        pretty_exceptions_enable=False,
        help="Render 3D frame sequences and videos with Mitsuba 3. Scalar fields and material transitions are optional.",
    )
    source_help = "Input: a manifest, USD stage, or any installed source plugin"

    def run(settings: RenderSettings) -> None:
        try:
            run_render(settings)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise _fail(exc) from exc

    @app.command()
    def render(
        source: Path = typer.Argument(..., exists=True, help=source_help),
        rr_depth: int = typer.Option(5, help="Bounce at which Russian roulette starts (lower = faster)",
                                     rich_help_panel=P_QUALITY),
        recompose: bool = typer.Option(False, help="With --resume: rebuild the composited frames (panels, chart, "
                                       "header) from the saved beauty frames, path tracing only frames without one"),
        annotate: Path = typer.Option(None, exists=True, dir_okay=False, help="JSON of composite-only labels (body "
                                      "tags, panel titles, colour bar unit; see mitsuba_video.annotate); not part "
                                      "of the resume signature, so --resume --recompose re-labels a finished render",
                                      rich_help_panel=P_FIELD),
        backend: Backend = typer.Option(Backend.mitsuba, help="Renderer: mitsuba, or Blender cycles / eevee (same "
                                        "scene, poses, camera, lights and field textures; also saves "
                                        "<out>/blender/scene.blend)", rich_help_panel=P_ADV),
        blender: str = typer.Option(None, help="Blender executable for --backend cycles/eevee [default: BLENDER, "
                                    "blender on PATH, newest install]", rich_help_panel=P_ADV),
        displace: float = typer.Option(0.0, help="Displace the meshes by each frame's blended height map: millimetres "
                                       "for the full 0..1 height range, inward from the asperity tops (0 = off; "
                                       "subdivides each body up to 1.5M triangles, 6M per scene)", rich_help_panel=P_QUALITY),
        source_kind: str = typer.Option(default_source, "--source", help="Source type (auto-detected by default); "
                                        "see `sources`", rich_help_panel=P_OUT),
        option: list[str] = typer.Option([], "-O", "--option", help="Source option KEY=VALUE (repeatable)",
                                         rich_help_panel=P_OUT),
        field: str = typer.Option(None, help="Field to show [default: the source's primary field]",
                                  rich_help_panel=P_OUT),
        out: Path = typer.Option(None, "--out", "-o", help="Output directory [default: <source>/render_<look>]",
                                 rich_help_panel=P_OUT),
        look: Look = typer.Option(default_look, help="plain: original material; material: field-driven appearance; heatmap: field overlay", rich_help_panel=P_OUT),
        shading: Shading = typer.Option(Shading.pbr, help="pbr | diffuse | metallic | ao", rich_help_panel=P_OUT),
        face_normals: bool = typer.Option(False, help="Use geometric face normals for a faceted mesh",
                                         rich_help_panel=P_QUALITY),
        displacement_scale: float = typer.Option(0.0, help="Static base-height displacement in world units; 0 disables. "
                                                "Offsets existing vertices by scale*(height-0.5); no subdivision",
                                                rich_help_panel=P_QUALITY),
        frames: str = typer.Option("::1", help="Python slice over source frames, e.g. 0:300:2", rich_help_panel=P_OUT),
        resume: bool = typer.Option(False, help="Reuse frames already in --out (settings must match)",
                                    rich_help_panel=P_OUT),
        quality: Quality = typer.Option("draft", "--quality", "-q", help="preview 960x540/16 spp, draft 1280x720/64, "
                                        "paper 1920x1080/256", rich_help_panel=P_QUALITY),
        res: str = typer.Option(None, help="WIDTHxHEIGHT, overrides --quality", rich_help_panel=P_QUALITY),
        spp: int = typer.Option(None, help="Samples per pixel, overrides --quality", rich_help_panel=P_QUALITY),
        max_depth: int = typer.Option(8, help="Maximum path depth", rich_help_panel=P_QUALITY),
        max_texture: int = typer.Option(2048, help="Cap on per-body texture resolution", rich_help_panel=P_QUALITY),
        blend_device: str = typer.Option("auto", help="Field material blend on cpu (numpy), gpu (Dr.Jit CUDA) or auto (gpu when "
                                         "available and the blend has no CPU-only term)", rich_help_panel=P_QUALITY),
        cavity_ao: float = typer.Option(0.0, help="Optional height-derived cavity occlusion strength, 0..1", rich_help_panel=P_QUALITY),
        normal_strength: float = typer.Option(1.0, help="Scale normal-map detail (1 = as authored, 0 = flat)",
                                              rich_help_panel=P_QUALITY),
        normal_recenter: float = typer.Option(5.0, help="Recentre normal maps whose mean tilt exceeds this many "
                                              "degrees (0 = off)", rich_help_panel=P_QUALITY),
        camera: CameraMode = typer.Option(CameraMode.fixed, help="Camera path", rich_help_panel=P_CAMERA),
        follow: int = typer.Option(None, help="Body to track [default: first dynamic body]", rich_help_panel=P_CAMERA),
        azimuth: float = typer.Option(-55.0, help="Degrees around the up axis", rich_help_panel=P_CAMERA),
        elevation: float = typer.Option(28.0, help="Degrees above the horizon", rich_help_panel=P_CAMERA),
        fov: float = typer.Option(35.0, help="Horizontal field of view (degrees)", rich_help_panel=P_CAMERA),
        zoom: float = typer.Option(1.0, help="> 1 moves the camera closer", rich_help_panel=P_CAMERA),
        orbit_degrees: float = typer.Option(120.0, help="Total sweep in --camera orbit", rich_help_panel=P_CAMERA),
        frame_all: bool = typer.Option(False, help="Frame every body, not just the dynamic bodies' paths",
                                       rich_help_panel=P_CAMERA),
        up: str = typer.Option(None, help="World up axis x | y | z [default: the source's]", rich_help_panel=P_CAMERA),
        exposure: float = typer.Option(0.0, help="Exposure in stops before tone mapping", rich_help_panel=P_LIGHT),
        key_light: float = typer.Option(1.1, help="Key light irradiance at the scene centre (0 disables)",
                                        rich_help_panel=P_LIGHT),
        fill: float = typer.Option(0.18, help="Constant environment fill", rich_help_panel=P_LIGHT),
        rim_light: float = typer.Option(0.45, help="Rim light strength (0 disables)", rich_help_panel=P_LIGHT),
        envmap: Path = typer.Option(None, exists=True, help="HDR/EXR environment map (replaces the fill)",
                                    rich_help_panel=P_LIGHT),
        ground: bool = typer.Option(True, help="Matte floor under the scene", rich_help_panel=P_LIGHT),
        panel: bool = typer.Option(True, help="UV field-atlas side panel", rich_help_panel=P_FIELD),
        panel_body: list[int] = typer.Option([], help="Body shown in the panel (repeatable) [default: bodies with a "
                                             "non-empty field]", rich_help_panel=P_FIELD),
        hide: list[int] = typer.Option([], help="Leave this body out of the 3D scene (repeatable); its panel "
                                       "stays, e.g. to film a worn underside from below", rich_help_panel=P_FIELD),
        chart: bool = typer.Option(True, help="Covered-area vs time chart (data also written as CSV)",
                                   rich_help_panel=P_FIELD),
        display_max: float = typer.Option(1.0, "--display-max", help="Top of the colormap "
                                          "(display only); below 1 the colour bar reads 'display range 0-max "
                                          "(preview)'", rich_help_panel=P_FIELD),
        ramp: float = typer.Option(0.0, "--ramp", help="Overlay opacity ramps 0 -> opacity over "
                                   "[0, ramp] (display only; 0 = flat)", rich_help_panel=P_FIELD),
        opacity: float = typer.Option(0.9, "--opacity", help="Heatmap overlay opacity",
                                      rich_help_panel=P_FIELD),
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
        denoise: bool = typer.Option(False, help="OptiX denoiser (CUDA only; check it helps on your driver)",
                                     rich_help_panel=P_ADV),
        seed: int = typer.Option(0, help="Sampler seed, fixed across frames for stable noise", rich_help_panel=P_ADV),
        keep_exr: bool = typer.Option(False, help="Also write linear EXR frames", rich_help_panel=P_ADV),
    ) -> None:
        """Path trace every selected frame, composite the field panel and chart, and encode the video."""
        width, height, spp_ = _resolution(res, quality, spp)
        options = _source_options(option)
        source_up, source_blend = _source_defaults(source, source_kind, options)
        up_axis = up or source_up
        run(
            RenderSettings(
                source=source,
                rr_depth=rr_depth, recompose=recompose, annotate=annotate,
                backend=backend.value, blender=blender, displace_mm=displace,
                source_kind=source_kind,
                source_options=options,
                field_name=field,
                out=out,
                look=look,
                shading=shading,
                face_normals=face_normals,
                displacement_scale=displacement_scale,
                width=width,
                height=height,
                spp=spp_,
                max_depth=max_depth,
                frames=frames,
                variant=variant,
                camera=CameraSettings(mode=camera.value, follow=follow, azimuth=azimuth, elevation=elevation, fov=fov,
                                      zoom=zoom, orbit_degrees=orbit_degrees, up=up_axis, frame_all=frame_all),
                lighting=LightingSettings(key_strength=key_light, fill=fill, rim_strength=rim_light,
                                          envmap=str(envmap) if envmap else None, ground=ground),
                exposure=exposure,
                max_texture=max_texture,
                blend_device=blend_device,
                normals=NormalOptions(recenter_above_deg=normal_recenter, strength=normal_strength, cavity_ao=cavity_ao),
                blend=source_blend,
                display=FieldDisplay(max=display_max, ramp=ramp, opacity=opacity),
                panel=panel,
                panel_bodies=list(panel_body),
                hidden_bodies=list(hide),
                chart=chart,
                denoise=denoise,
                seed=seed,
                keep_exr=keep_exr,
                resume=resume,
                video=VideoSettings(fps=fps, encode=encode, gif=gif, crf=crf, title=title, subtitle=list(subtitle),
                                    title_seconds=title_seconds, hold=hold),
            )
        )  # fmt: skip

    @app.command()
    def preview(
        source: Path = typer.Argument(..., exists=True, help=source_help),
        source_kind: str = typer.Option(default_source, "--source", help="Source type"),
        option: list[str] = typer.Option([], "-O", "--option", help="Source option KEY=VALUE (repeatable)"),
        out: Path = typer.Option(None, "--out", "-o", help="Output directory [default: <source>/render_preview]"),
        look: Look = typer.Option(default_look, help="plain | material | heatmap"),
        camera: CameraMode = typer.Option(CameraMode.fixed, help="Camera path"),
        frames: str = typer.Option("::8", help="Python slice over source frames"),
        variant: str = typer.Option("auto", help="Mitsuba variant preference"),
    ) -> None:
        """960x540 at 16 spp on every 8th frame: check framing and look in a minute before a long render."""
        width, height, spp = QUALITY_PRESETS["preview"]
        options = _source_options(option)
        base = Path(source) if Path(source).is_dir() else Path(source).parent
        source_up, source_blend = _source_defaults(source, source_kind, options)
        run(
            RenderSettings(
                source=source,
                source_kind=source_kind,
                source_options=options,
                out=out or base / "render_preview",
                look=look,
                width=width,
                height=height,
                spp=spp,
                frames=frames,
                variant=variant,
                camera=CameraSettings(mode=camera.value, up=source_up),
                blend=source_blend,
                video=VideoSettings(fps=8.0, hold=1.0),
            )
        )

    @app.command()
    def inspect(
        source: Path = typer.Argument(..., exists=True, help=source_help),
        source_kind: str = typer.Option(default_source, "--source", help="Source type"),
        option: list[str] = typer.Option([], "-O", "--option", help="Source option KEY=VALUE (repeatable)"),
        field: str = typer.Option(None, help="Field to summarise [default: primary]"),
    ) -> None:
        """Summarise a source: bodies, materials, frames, motion and field coverage."""
        from .motion import travel_by_frame

        try:
            src = open_source(source, source_kind, **_source_options(option))
        except (ValueError, FileNotFoundError) as exc:
            raise _fail(exc) from exc
        try:
            spec = resolve_field(src, field)
            for key, value in src.describe():
                typer.echo(f"{key:<10} {value}")
            typer.echo(f"{'source':<10} {src.kind}; fields {', '.join(src.fields) or 'none'} (selected: {spec.name or 'none'})")
            frames = src.frames
            if frames:
                dt = (frames[-1].time - frames[0].time) / max(len(frames) - 1, 1)
                typer.echo(f"{'frames':<10} {len(frames)}  (t = {frames[0].time:.3f} .. {frames[-1].time:.3f} s, "
                           f"dt ~ {dt:.4f} s)")  # fmt: skip
                mover, travel = travel_by_frame(src, None, "reset_detection" in src.capabilities)
                if mover is not None:
                    dist, passes = travel[frames[-1].key]
                    typer.echo(f"{'motion':<10} body {mover.index} {getattr(src, 'travel_label', 'travel')} "
                               f"{dist:.3f} {getattr(src, 'length_unit', 'm')} over {passes} pass(es)")  # fmt: skip
            typer.echo("")
            note_keys = sorted({k for b in src.bodies for k in b.notes})
            header = f"{'body':<5} {'name':<28} {'tris':>6} {'kind':<8} " + " ".join(f"{k:>8}" for k in note_keys)
            typer.echo(header + f" {'texels (first -> last)':>24} {'area':>11}")
            for body in src.bodies:
                first = last = 0
                area = 0.0
                if frames:
                    counts, areas = covered_area(src, [frames[0], frames[-1]], body, spec, progress=False)
                    first, last, area = int(counts[0]), int(counts[-1]), float(areas[-1])
                notes = " ".join(f"{body.notes.get(k, ''):>8}" for k in note_keys)
                typer.echo(f"{body.index:<5} {body.name[:28]:<28} {len(body.faces):>6} "
                           f"{'fixed' if body.fixed else 'dynamic':<8} {notes} {f'{first} -> {last}':>24} "
                           f"{area:>9.4g} {getattr(src, 'length_unit', 'm')}2")  # fmt: skip
        finally:
            src.close()

    @app.command()
    def encode(
        run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="A render output directory"),
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
            encode_run(run_dir, v, name or config.get("video_stem", "field"), typer.echo)
        except FileNotFoundError as exc:
            raise _fail(exc) from exc

    @app.command()
    def fixture(
        path: Path = typer.Argument(..., help="Output location for the chosen fixture"),
        kind: str = typer.Option(fixture_kind, help="Fixture name (built-ins: cube-throw, manifest; plugins may add more)"),
        frames: int = typer.Option(6, min=1, help="Number of frames"),
    ) -> None:
        """Generate a plain animation or a scalar-field example."""
        from .source import available_fixtures

        factories = available_fixtures()
        if kind not in factories:
            raise typer.BadParameter(f"unknown fixture {kind!r}; available: {', '.join(sorted(factories))}")
        typer.echo(factories[kind]()(path, frames=frames))

    @app.command()
    def validate(manifest: Path = typer.Argument(..., exists=True, help="manifest.json or its folder")) -> None:
        """Check a manifest against the v1 JSON schema and that every referenced file exists."""
        from .sources.manifest import ManifestSource

        try:
            src = ManifestSource.open(manifest)
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise _fail(exc) from exc
        missing = sum(1 for f in src.frames for b in src.bodies if src.field(f, b.index) is None)
        typer.echo(f"ok: {len(src.bodies)} bodies, {len(src.frames)} frames, fields {', '.join(src.fields)}; "
                   f"{missing} body-frames without a primary-field image")  # fmt: skip

    export_app = typer.Typer(help="Convert a source to another format.", no_args_is_help=True)

    @export_app.command("manifest")
    def export_manifest_cmd(
        source: Path = typer.Argument(..., exists=True, help=source_help),
        out: Path = typer.Argument(..., help="Output folder"),
        source_kind: str = typer.Option(default_source, "--source", help="Source type"),
        option: list[str] = typer.Option([], "-O", "--option", help="Source option KEY=VALUE (repeatable)"),
        material_size: int = typer.Option(1024, help="Baked material texture size"),
    ) -> None:
        """Write SOURCE as a manifest (OBJ meshes, baked materials, PNG fields, poses.npz)."""
        from .sources.manifest.export import export_manifest

        try:
            src = open_source(source, source_kind, **_source_options(option))
        except (ValueError, FileNotFoundError) as exc:
            raise _fail(exc) from exc
        try:
            typer.echo(export_manifest(src, out, material_size=material_size, log=typer.echo))
        finally:
            src.close()

    app.add_typer(export_app, name="export")

    @app.command()
    def sources() -> None:
        """List registered sources (entry-point group mitsuba_video.sources)."""
        for name, loader in sorted(available_sources().items()):
            try:
                cls = loader()
                doc = (cls.__module__ and sys.modules[cls.__module__].__doc__ or "").strip().splitlines()[0]
                typer.echo(f"{name:<16} {doc}")
            except ImportError as exc:
                typer.echo(f"{name:<16} unavailable ({exc})")

    @app.command()
    def doctor(variant: str = typer.Option("auto", help="Backend to test with a tiny real render")) -> None:
        """Check Mitsuba, GPU/CPU rendering, optional inputs and FFmpeg."""
        from .terminal import doctor_report

        if not doctor_report(variant):
            raise typer.Exit(1)

    @app.command()
    def tui() -> None:
        """Open an interactive terminal setup for a render."""
        from .terminal import render_wizard

        try:
            settings = render_wizard(default_source)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise _fail(exc) from exc
        if settings is not None:
            run(settings)

    @app.command()
    def version() -> None:
        """Print mitsuba-video and Mitsuba versions."""
        typer.echo(f"{prog} {__version__}")
        try:
            import mitsuba as mi

            typer.echo(f"mitsuba {mi.__version__}: {', '.join(v for v in mi.variants() if v.endswith('_rgb'))}")
        except ImportError:
            typer.echo("mitsuba not installed")

    return app


def _source_defaults(source: Path, kind: str, options: dict[str, object]):
    """Source-level defaults (up axis, blend parameters) from one brief open of the source."""
    from .blend import BlendParams

    try:
        src = open_source(source, kind, **options)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError):
        return "y", BlendParams()
    try:
        return src.up_axis, getattr(src, "blend", None) or BlendParams()
    finally:
        src.close()




app = build_app()


def run_main(application: typer.Typer, prog: str, argv: list[str] | None = None) -> None:
    """Console entry: ``prog SOURCE ...`` is shorthand for ``prog render SOURCE ...``; hard-exits after flushing."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in COMMANDS and not args[0].startswith("-"):
        args.insert(0, "render")
    try:
        application(args=args, prog_name=prog)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception:  # noqa: BLE001 - report, then hard-exit below
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Mitsuba/Dr.Jit can crash during interpreter teardown (mitsuba 3.6.4 and 3.9.1 on Windows); output is flushed,
    # so skip teardown to keep the exit code meaningful. On Windows os._exit still runs DLL detach, where the crash
    # happens (the shell then sees 127), so terminate the process outright there.
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code & 0xFFFFFFFF)
    os._exit(code)


def main(argv: list[str] | None = None) -> None:
    run_main(app, "mitsuba-video", argv)
