"""``mitsuba-video``: path-traced videos of rigid bodies whose UV atlases carry a per-texel scalar field.

    mitsuba-video render  SOURCE [-q paper] [--camera orbit] [--title ...]   full render + video
    mitsuba-video preview SOURCE                                             fast low-res check of framing/look
    mitsuba-video inspect SOURCE                                             what the source contains
    mitsuba-video encode  OUT [--fps 24] [--title ...]                       re-encode frames, no re-render
    mitsuba-video fixture PATH --kind manifest|texturefriction               tiny synthetic input
    mitsuba-video validate MANIFEST | export manifest SOURCE OUT | sources | version

``SOURCE`` is anything a registered source accepts: a ``manifest.json`` (or its folder), a TextureFriction
``simulation_<ts>/`` folder or ``.h5``... ``mitsuba-video SOURCE ...`` (no command) means ``render``.
``tf-render`` is the same CLI with ``--source texturefriction`` as default.
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
from .scene import LightingSettings
from .source import available_sources, open_source, parse_options

P_OUT, P_QUALITY, P_CAMERA, P_LIGHT, P_FIELD, P_VIDEO, P_ADV = (
    "Output", "Quality", "Camera", "Lighting", "Field panel", "Video", "Advanced",
)  # fmt: skip
COMMANDS = {"render", "preview", "inspect", "encode", "fixture", "validate", "export", "sources", "version"}


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


def _source_options(option: list[str], scale: list[str], assets: list[Path]) -> dict[str, object]:
    try:
        opts: dict[str, object] = dict(parse_options(option))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if scale:
        opts["scale"] = ",".join(scale)
    if assets:
        opts["assets"] = os.pathsep.join(str(a) for a in assets)
    return opts


def _fail(exc: Exception) -> typer.Exit:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def build_app(default_source: str = "auto", prog: str = "mitsuba-video") -> typer.Typer:  # noqa: C901
    """The CLI, with ``default_source`` as the ``--source`` default (``tf-render`` uses "texturefriction")."""
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        pretty_exceptions_enable=False,
        help="Path-traced videos of rigid bodies with per-texel fields (wear, damage, temperature...) "
        "using Mitsuba 3.",
    )
    source_help = "Input: manifest.json / folder, TextureFriction simulation_<ts>/ or .h5, ..."

    def run(settings: RenderSettings) -> None:
        try:
            run_render(settings)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise _fail(exc) from exc

    @app.command()
    def render(
        source: Path = typer.Argument(..., exists=True, help=source_help),
        source_kind: str = typer.Option(default_source, "--source", help="Source type (auto-detected by default); "
                                        "see `sources`", rich_help_panel=P_OUT),
        option: list[str] = typer.Option([], "-O", "--option", help="Source option KEY=VALUE (repeatable)",
                                         rich_help_panel=P_OUT),
        field: str = typer.Option(None, help="Field to show [default: the source's primary field]",
                                  rich_help_panel=P_OUT),
        out: Path = typer.Option(None, "--out", "-o", help="Output directory [default: <source>/render_<look>]",
                                 rich_help_panel=P_OUT),
        look: Look = typer.Option(Look.heatmap, help="heatmap: affected material + field colormap; worn: appearance "
                                  "only; plain: field ignored", rich_help_panel=P_OUT),
        frames: str = typer.Option("::1", help="Python slice over source frames, e.g. 0:300:2", rich_help_panel=P_OUT),
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
        key_light: float = typer.Option(1.1, help="Key light irradiance at the scene centre", rich_help_panel=P_LIGHT),
        fill: float = typer.Option(0.18, help="Constant environment fill", rich_help_panel=P_LIGHT),
        rim_light: float = typer.Option(0.45, help="Rim light strength (0 disables)", rich_help_panel=P_LIGHT),
        envmap: Path = typer.Option(None, exists=True, help="HDR/EXR environment map (replaces the fill)",
                                    rich_help_panel=P_LIGHT),
        ground: bool = typer.Option(True, help="Matte floor under the scene", rich_help_panel=P_LIGHT),
        panel: bool = typer.Option(True, help="UV field-atlas side panel", rich_help_panel=P_FIELD),
        panel_body: list[int] = typer.Option([], help="Body shown in the panel (repeatable) [default: bodies with a "
                                             "non-empty field]", rich_help_panel=P_FIELD),
        chart: bool = typer.Option(True, help="Covered-area vs time chart (data also written as CSV)",
                                   rich_help_panel=P_FIELD),
        display_max: float = typer.Option(1.0, "--display-max", "--wear-display-max", help="Top of the colormap "
                                          "(display only); below 1 the colour bar reads 'display range 0-max "
                                          "(preview)'", rich_help_panel=P_FIELD),
        ramp: float = typer.Option(0.0, "--ramp", "--wear-ramp", help="Overlay opacity ramps 0 -> opacity over "
                                   "[0, ramp] (display only; 0 = flat)", rich_help_panel=P_FIELD),
        opacity: float = typer.Option(0.9, "--opacity", "--wear-opacity", help="Heatmap overlay opacity",
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
        scale: list[str] = typer.Option([], help="TextureFriction: body scale override BODY=SCALE (= -O scale=...)",
                                        rich_help_panel=P_ADV),
        assets: list[Path] = typer.Option([], help="TextureFriction: extra root holding resources/ "
                                          "(= -O assets=...)", rich_help_panel=P_ADV),
        denoise: bool = typer.Option(False, help="OptiX denoiser (CUDA only; check it helps on your driver)",
                                     rich_help_panel=P_ADV),
        seed: int = typer.Option(0, help="Sampler seed, fixed across frames for stable noise", rich_help_panel=P_ADV),
        keep_exr: bool = typer.Option(False, help="Also write linear EXR frames", rich_help_panel=P_ADV),
    ) -> None:
        """Path trace every selected frame, composite the field panel and chart, and encode the video."""
        width, height, spp_ = _resolution(res, quality, spp)
        options = _source_options(option, scale, assets)
        up_axis = up or _source_up(source, source_kind, options)
        run(
            RenderSettings(
                source=source,
                source_kind=source_kind,
                source_options=options,
                field_name=field,
                out=out,
                look=look,
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
                normals=NormalOptions(recenter_above_deg=normal_recenter, strength=normal_strength),
                blend=_source_blend(source, source_kind, options),
                display=FieldDisplay(max=display_max, ramp=ramp, opacity=opacity),
                panel=panel,
                panel_bodies=list(panel_body),
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
        look: Look = typer.Option(Look.heatmap, help="heatmap | worn | plain"),
        camera: CameraMode = typer.Option(CameraMode.fixed, help="Camera path"),
        frames: str = typer.Option("::8", help="Python slice over source frames"),
        variant: str = typer.Option("auto", help="Mitsuba variant preference"),
    ) -> None:
        """960x540 at 16 spp on every 8th frame: check framing and look in a minute before a long render."""
        width, height, spp = QUALITY_PRESETS["preview"]
        options = _source_options(option, [], [])
        base = Path(source) if Path(source).is_dir() else Path(source).parent
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
                camera=CameraSettings(mode=camera.value, up=_source_up(source, source_kind, options)),
                blend=_source_blend(source, source_kind, options),
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
            src = open_source(source, source_kind, **_source_options(option, [], []))
        except (ValueError, FileNotFoundError) as exc:
            raise _fail(exc) from exc
        try:
            spec = resolve_field(src, field)
            for key, value in src.describe():
                typer.echo(f"{key:<10} {value}")
            typer.echo(f"{'source':<10} {src.kind}; fields {', '.join(src.fields)} (showing '{spec.name}')")
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
        path: Path = typer.Argument(..., help="Output: a folder (manifest) or an .h5 path (texturefriction)"),
        kind: str = typer.Option("manifest", help="manifest | texturefriction"),
        frames: int = typer.Option(6, help="Number of frames"),
    ) -> None:
        """Write a tiny synthetic input (plane + sliding box) for tests and smoke renders."""
        if kind == "manifest":
            from .sources.manifest.fixture import write_synthetic_manifest

            typer.echo(write_synthetic_manifest(path, frames=frames))
        elif kind == "texturefriction":
            from .sources.texturefriction.synthetic import write_synthetic

            typer.echo(write_synthetic(path, frames=frames))
        else:
            raise typer.BadParameter("--kind is manifest or texturefriction")

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
            src = open_source(source, source_kind, **_source_options(option, [], []))
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
    def version() -> None:
        """Print mitsuba-video and Mitsuba versions."""
        typer.echo(f"{prog} {__version__}")
        try:
            import mitsuba as mi

            typer.echo(f"mitsuba {mi.__version__}: {', '.join(v for v in mi.variants() if v.endswith('_rgb'))}")
        except ImportError:
            typer.echo("mitsuba not installed")

    return app


def _peek(source: Path, kind: str, options: dict[str, object]):
    """Open a source briefly to read source-level defaults (up axis, blend parameters)."""
    try:
        return open_source(source, kind, **options)
    except (ValueError, FileNotFoundError, ImportError):
        return None


def _source_up(source: Path, kind: str, options: dict[str, object]) -> str:
    src = _peek(source, kind, options)
    if src is None:
        return "y"
    try:
        return src.up_axis
    finally:
        src.close()


def _source_blend(source: Path, kind: str, options: dict[str, object]):
    from .blend import BlendParams

    src = _peek(source, kind, options)
    if src is None:
        return BlendParams()
    try:
        return getattr(src, "blend", None) or BlendParams()
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


def tf_main(argv: list[str] | None = None) -> None:
    """``tf-render`` console entry: checks the optional h5py dependency before importing the TextureFriction source."""
    import importlib.util

    if importlib.util.find_spec("h5py") is None:
        print('tf-render reads TextureFriction HDF5 recordings and needs h5py: '
              'pip install "mitsuba-video[texturefriction]"', file=sys.stderr)  # fmt: skip
        sys.exit(2)
    from .sources.texturefriction.cli import main as texturefriction_main

    texturefriction_main(argv)
