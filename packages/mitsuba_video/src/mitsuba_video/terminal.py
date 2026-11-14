"""Terminal diagnostics and a guided render setup."""

import importlib.util
import shlex
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from . import scene
from .pipeline import QUALITY_PRESETS, Look, RenderSettings


def doctor_report(variant="auto", console=None):
    console = console or Console()
    table = Table(title="MitsubaVideo diagnostics")
    table.add_column("Check")
    table.add_column("Result")
    ok = True
    try:
        chosen = scene.select_variant(variant)
        table.add_row("Renderer", f"Mitsuba {scene.mitsuba().__version__} / {chosen} (render tested)")
    except (ImportError, RuntimeError) as exc:
        table.add_row("Renderer", str(exc))
        ok = False
    table.add_row("FFmpeg", shutil.which("ffmpeg") or "Missing: install FFmpeg to encode videos")
    for module, extra in (("pxr", "usd"),):
        table.add_row(extra, Text("Available" if importlib.util.find_spec(module) else f"Optional: install [{extra}]"))
    console.print(table)
    return ok


def render_wizard(default_source="auto", console=None):
    console = console or Console()
    console.print(Panel("Choose an input, quality and appearance. Frames stay on disk for inspection and resume.",
                        title="MitsubaVideo"))
    source = Path(Prompt.ask("Manifest, USD stage or plugin input", console=console)).expanduser()
    if not source.exists():
        raise ValueError(f"input does not exist: {source}")
    quality = Prompt.ask("Quality", choices=list(QUALITY_PRESETS), default="preview", console=console)
    look = Prompt.ask("Appearance", choices=[v.value for v in Look], default="plain", console=console)
    shading = Prompt.ask("Surface shading", choices=[v.value for v in scene.Shading], default="pbr", console=console)
    variant = Prompt.ask("Backend", choices=["auto", "cuda", "metal", "cpu", "scalar"], default="auto", console=console)
    out = Path(Prompt.ask("Output folder", default="output/render", console=console)).expanduser()
    width, height, spp = QUALITY_PRESETS[quality]
    console.print(f"{width}x{height}, {spp} samples/pixel; {look}, {shading}; backend {variant}")
    command = ["mitsuba-video", "render", str(source), "--out", str(out), "-q", quality,
               "--look", look, "--shading", shading, "--variant", variant, "--source", default_source]
    console.print("Command: " + shlex.join(command), markup=False)
    if not Confirm.ask("Start rendering?", default=True, console=console):
        return None
    from .cli import _source_defaults

    up, blend = _source_defaults(source, default_source, {})
    settings = RenderSettings(source=source, source_kind=default_source, out=out, width=width, height=height,
                              spp=spp, look=Look(look), shading=scene.Shading(shading), variant=variant, blend=blend)
    settings.camera.up = up
    return settings
