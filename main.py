#!/usr/bin/env python3
"""
Mitsuba Rendering Application

This script provides a command-line interface for rendering 3D scenes with Mitsuba.
"""

import typer
import os
import sys
from typing import Optional, List

from src.utils.logger import RichLogger
from src.utils.timing import timeit
from src.utils.environment import get_app_root_dir
from src.config.constants import RenderDevice
from src.cli.parameters import (
    process_render_parameters, 
    run_multi_obj_rendering,
    process_config_file,
    validate_folder
)

# Configure logger
logger = RichLogger.get_logger("mitsuba_app.main")

# Create Typer app
app = typer.Typer(help="Command-line interface for Mitsuba 3 renderer")

@app.command("multi")
@timeit(log_level="debug")
def render_multi_obj_scene(
    folder: str = typer.Option(..., "--folder", "-f", help="Folder containing OBJ files", callback=validate_folder),
    regex: str = typer.Option(r"\d+\.obj", "--regex", "-r", help="Regex pattern for OBJ filenames"),
    scene_name: str = typer.Option("multi_obj", "--name", "-n", help="Base name for output files"),
    threaded: bool = typer.Option(False, "--threaded/--no-threaded", help="Enable multi-threading"),
    max_workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Max worker threads (if threaded)"),
    enable_video: bool = typer.Option(None, "--video/--no-video", help="Enable video creation"),
    enable_gif: bool = typer.Option(None, "--gif/--no-gif", help="Enable GIF creation"),
    device: str = typer.Option(
        RenderDevice.CPU.value, 
        "--device", "-d", 
        help=f"Rendering device ({RenderDevice.CPU.value}/{RenderDevice.CUDA.value})"
    ),
    quality: List[str] = typer.Option(
        None, "--quality", "-q",
        help="Quality presets (can specify multiple times, e.g., -q low -q medium)"
    ),
    multi_quality: List[str] = typer.Option(
        None, "--multi-quality", "-mq",
        help="Alternative way to specify multiple quality presets"
    ),
    spp: int = typer.Option(None, "--spp", "-s", help="Samples per pixel (overrides quality preset)"),
    width: int = typer.Option(1920, "--width", help="Output image width"),
    height: int = typer.Option(1080, "--height", help="Output image height"),
    output: str = typer.Option("output", "--output", "-o", help="Output directory"),
    max_depth: int = typer.Option(None, "--max-depth", help="Maximum ray bounces (overrides quality preset)"),
    stop_on_error: bool = typer.Option(True, "--stop-on-error/--continue-on-error", help="Stop rendering on errors"),
    views: List[str] = typer.Option(
        None, "--view", "-v", 
        help="Camera views to render (can be specified multiple times). Options: front, right, left, back, top, bottom, perspective"
    ),
    all_views: bool = typer.Option(
        False, "--all-views", help="Render all standard camera views"
    ),
    view_preset: str = typer.Option(
        None, "--view-preset", 
        help="Use a view preset (all, orthographic, technical, 360)"
    ),
    multi_view: bool = typer.Option(
        False, "--multi-view/--single-view", help="Enable rendering each frame from multiple camera views"
    ),
    camera_distance: float = typer.Option(
        4.0, "--camera-distance", help="Camera distance from object center for all views"
    ),
    integrator_type: Optional[str] = typer.Option(
        None, "--integrator", "-i",
        help="Integrator type (path, volpath, direct, etc.)"
    ),
    enable_cache: bool = typer.Option(None, "--cache/--no-cache", help="Enable rendering cache to skip re-rendering identical frames"),
    
    # Logging settings
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with verbose output"),
    log_level: str = typer.Option("info", "--log-level", help="Set logging level (debug, info, warning, error)"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Save logs to a specific file"),
    detailed_logs: bool = typer.Option(False, "--detailed-logs/--simple-logs", help="Enable detailed logs for each frame, quality and view"),
    timing_reports: bool = typer.Option(False, "--timing-reports", help="Generate separate timing reports for each quality and view"),
    
    # Config file option with callback to process immediately
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to custom config.json file", callback=process_config_file),

    # Denoising options
    enable_denoising: Optional[bool] = typer.Option(
        None, "--denoise/--no-denoise",
        help="Enable/disable denoising of rendered images"
    ),
    denoising_type: str = typer.Option(
        "optix", "--denoiser",
        help="Denoiser to use (optix, oidn)"
    ),
    denoising_strength: float = typer.Option(
        1.0, "--denoise-strength",
        help="Denoising strength (0.0-1.0)"
    ),
    denoising_guide_buffers: bool = typer.Option(
        True, "--guide-buffers/--no-guide-buffers",
        help="Generate and use albedo/normal buffers for better denoising"
    ),
    fallback_if_unavailable: bool = typer.Option(
        True, "--denoiser-fallback/--no-denoiser-fallback",
        help="Continue rendering if denoiser is unavailable"
    ),

    # AOV options
    enable_aovs: bool = typer.Option(
        False, "--aovs/--no-aovs",
        help="Enable generation of AOVs (Arbitrary Output Variables) like depth, normals, etc."
    ),
    aov_types: List[str] = typer.Option(
        ["albedo", "depth", "sh_normal"], "--aov-type", "-a",
        help="AOV types to generate (can be specified multiple times): albedo, depth, position, uv, geo_normal, sh_normal"
    ),
    aov_separate_files: bool = typer.Option(
        True, "--aov-separate-files/--aov-combined-file",
        help="Output AOVs as separate files or as channels in a single EXR file"
    ),
):
    """
    Render multiple scenes from OBJ files in a folder.
    
    This command renders a sequence of frames from OBJ files and can create video/GIF outputs.
    Multiple camera views can be rendered for each frame using the --view option.
    Enable --multi-view to render each frame from multiple camera angles.
    Use --multi-quality (-mq) to render the same scene with multiple quality presets.
    Enable --cache to skip re-rendering identical frames.
    You can provide a custom config.json file with the --config option.
    Note: When specifying multiple quality presets, use -mq for each preset (e.g., -mq low -mq medium).
    Note: Both --quality/-q and --multi-quality/-mq can be specified multiple times for rendering with
    different quality settings (e.g., -q low -q medium or -mq low -mq medium).
    Enable --denoise to apply denoising to rendered images, which can significantly
    reduce noise with fewer samples. When using denoising with --guide-buffers,
    additional information like albedo and normals are used for higher quality results.
    If the requested denoiser is not available, the system will automatically
    attempt to use an alternative denoiser. If no denoisers are available,
    rendering will continue without denoising.
    
    To generate AOVs (Arbitrary Output Variables) like depth maps, normal maps, etc.,
    use the --aovs flag and specify which AOV types to output with --aov-type.
    """
    # Process all parameters
    params = process_render_parameters(
        folder=folder,
        regex=regex,
        scene_name=scene_name,
        threaded=threaded,
        max_workers=max_workers,
        enable_video=enable_video,
        enable_gif=enable_gif,
        device=device,
        quality=quality,
        multi_quality=multi_quality,
        spp=spp,
        width=width,
        height=height,
        output=output,
        max_depth=max_depth,
        stop_on_error=stop_on_error,
        views=views,
        all_views=all_views,
        view_preset=view_preset,
        multi_view=multi_view,
        camera_distance=camera_distance,
        integrator_type=integrator_type,
        enable_cache=enable_cache,
        debug=debug,
        log_level=log_level,
        log_file=log_file,
        detailed_logs=detailed_logs,
        timing_reports=timing_reports,
        enable_denoising=enable_denoising,
        denoising_type=denoising_type,
        denoising_strength=denoising_strength,
        denoising_guide_buffers=denoising_guide_buffers,
        fallback_if_unavailable=fallback_if_unavailable,
        enable_aovs=enable_aovs,
        aov_types=aov_types,
        aov_separate_files=aov_separate_files
    )
    
    # Run the rendering process
    success, elapsed = run_multi_obj_rendering(params)
    
    # Report results to the user
    if success:
        typer.echo(f"Rendering completed successfully in {elapsed}")
    else:
        typer.echo(f"Rendering completed with errors in {elapsed}", err=True)
        raise typer.Exit(code=1)

# ...remaining commands (debug, quality) stay the same...
@app.command("debug")
def show_info():
    """
    Display information about the Mitsuba configuration and available variants.
    """
    from src.config.config import display_config
    from rich.console import Console
    from rich.table import Table

    console = Console()
    
    # Show Mitsuba variants
    try:
        variants = MitsubaUtils.get_available_variants()
        table = Table(title="Available Mitsuba Variants")
        table.add_column("Variant", style="green")
        
        for variant in variants:
            table.add_row(variant)
            
        console.print(table)
    except Exception as e:
        logger.error(f"Error getting Mitsuba variants: {e}")
        console.print(f"[red]Error getting Mitsuba variants: {e}[/red]")
    
    # Show configuration
    display_config()

@app.command("quality")
def show_quality_presets():
    """
    Display information about available quality presets.
    """
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Show quality presets
    quality_table = Table(title="Available Quality Presets")
    quality_table.add_column("Preset", style="cyan")
    quality_table.add_column("SPP", style="green")
    quality_table.add_column("Max Depth", style="magenta")
    
    for preset in ["low", "medium", "high", "ultra", "custom"]:
        settings = QualityManager.get_preset_settings(preset)
        quality_table.add_row(
            preset.upper(),
            str(settings.get("spp", "N/A")),
            str(settings.get("maxDepth", settings.get("max_depth", "N/A")))
        )
    
    console.print(quality_table)
    
    # Show current quality settings
    current_preset = QualityManager.get_default_preset()
    console.print(f"\nCurrent quality preset: [bold cyan]{current_preset.upper()}[/bold cyan]")

@app.command("denoise")
@timeit(log_level="info")
def denoise_images(
    input_path: str = typer.Argument(
        ..., help="Input file or directory to denoise"
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output file or directory (defaults to input_path + '_denoised')"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r",
        help="Process directories recursively"
    ),
    denoiser: str = typer.Option(
        "optix", "--denoiser", "-d",
        help="Denoiser to use (optix, oidn)"
    ),
    strength: float = typer.Option(
        1.0, "--strength", "-s",
        help="Denoising strength (0.0-1.0)"
    ),
    file_pattern: str = typer.Option(
        "*.exr", "--pattern", "-p",
        help="File pattern when processing directories"
    ),
):
    """
    Denoise rendered images.
    
    This command applies denoising to existing rendered images. It can denoise
    a single file or all matching files in a directory.
    
    Examples:
        mitsuba denoise render.exr
        mitsuba denoise output/exr/ --recursive --pattern "*.exr"
    """
    from src.mitsuba.denoising import DenoiserManager
    
    # Check if it's a file or directory
    if os.path.isfile(input_path):
        # Denoise a single file
        result_path = DenoiserManager.denoise_exr_file(
            input_path=input_path,
            output_path=output_path,
            denoiser_type=denoiser,
            denoiser_strength=strength
        )
        typer.echo(f"Denoised image saved to: {result_path}")
    
    elif os.path.isdir(input_path):
        # Batch denoise all files in directory
        count = DenoiserManager.batch_denoise_folder(
            folder_path=input_path,
            pattern=file_pattern,
            output_folder=output_path,
            recursive=recursive,
            denoiser_type=denoiser,
            denoiser_strength=strength
        )
        typer.echo(f"Successfully denoised {count} images")
    
    else:
        typer.echo(f"Error: Input path not found: {input_path}", err=True)
        raise typer.Exit(code=1)

@app.command("aov")
def extract_aovs(
    input_path: str = typer.Argument(
        ..., help="Path to input EXR file containing AOVs"
    ),
    output_folder: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output folder for extracted AOVs (defaults to same directory as input)"
    ),
    aov_list: Optional[List[str]] = typer.Option(
        None, "--aov", "-a",
        help="Specific AOVs to extract (default: all available in the file)"
    ),
    format: str = typer.Option(
        "exr", "--format", "-f",
        help="Output format for extracted AOVs (exr, png)"
    ),
):
    """
    Extract AOVs from a multichannel EXR file into separate image files.
    
    This command takes a multichannel EXR file containing AOVs (Arbitrary Output Variables)
    like depth, normals, etc., and extracts each AOV into a separate image file.
    
    Example:
        mitsuba aov rendered_with_aovs.exr --output aovs/ --format png
    """
    from src.processing.aov import AOVProcessor
    
    try:
        # Default output folder to same directory as input if not specified
        if not output_folder:
            output_folder = os.path.dirname(os.path.abspath(input_path))
            
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract AOVs
        extracted_files = AOVProcessor.extract_aovs_from_exr(
            input_path=input_path,
            output_folder=output_folder,
            aov_list=aov_list,
            output_format=format
        )
        
        # Report results
        if extracted_files:
            typer.echo(f"Successfully extracted {len(extracted_files)} AOVs:")
            for aov_name, file_path in extracted_files.items():
                typer.echo(f"  {aov_name}: {os.path.basename(file_path)}")
        else:
            typer.echo("No AOVs found or extracted.")
            
    except Exception as e:
        typer.echo(f"Error extracting AOVs: {e}", err=True)
        raise typer.Exit(code=1)

@app.callback()
def main():
    """
    Mitsuba 3D Renderer CLI
    
    A command-line interface for rendering 3D scenes with the Mitsuba renderer.
    """
    # Log application info
    app_root = get_app_root_dir()
    logger.debug(f"Mitsuba CLI started")
    logger.debug(f"Application root: {app_root}")
    logger.debug(f"Python executable: {sys.executable}")
    logger.debug(f"Working directory: {os.getcwd()}")
    
    # Print basic system info for debugging
    cores = os.cpu_count() or 0
    logger.debug(f"Available CPU cores: {cores}")

if __name__ == "__main__":
    app()
