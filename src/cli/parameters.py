"""
CLI parameter handling for Mitsuba renderer.

This module defines and processes parameters for the command-line interface,
including defaults, validation, and conversion to render settings.
"""

import os
import importlib
import difflib
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing
from datetime import datetime, timedelta
import typer
from pathlib import Path
from tqdm import tqdm

from src.config.config import config_class as config
from src.config.constants import RenderDevice
from src.utils.logger import RichLogger
from src.mitsuba.quality import QualityManager
from src.mitsuba.camera import CameraViewManager
from src.utils.cache import CacheManager
from src.renderers import MultiObjSceneRenderer

# Initialize logger only once
logger = RichLogger.get_logger("mitsuba_app.cli.parameters")

def define_render_parameters(app: typer.Typer) -> None:
    """
    Define all CLI parameters for the render command.
    
    This function exists for documentation purposes - the actual parameters
    are defined directly in the main module using typer.Option.
    """
    pass

def process_config_file(config_file: Optional[str]) -> bool:
    """
    Process a custom config file if specified.
    
    Args:
        config_file: Path to config file or None
        
    Returns:
        True if config was loaded successfully, False otherwise
        
    Raises:
        typer.Exit: If the config file cannot be loaded
    """
    if not config_file:
        return True
        
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        typer.echo(f"Config file not found: {config_file}", err=True)
        raise typer.Exit(code=1)

    try:
        # Get the config instance
        config_instance = config
        
        # Replace the global config with the new one
        success = config_instance.replace_global_instance(config_file)
        
        if success:
            logger.info(f"Loaded custom configuration from {config_file}")
            return True
        else:
            logger.error(f"Failed to load custom configuration from {config_file}")
            typer.echo(f"Error loading config file: {config_file}", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error loading custom config file: {e}")
        typer.echo(f"Error loading config file: {e}", err=True)
        raise typer.Exit(code=1)

def setup_logging(debug: bool, log_level: str, log_file: Optional[str], 
                  detailed_logs: bool, timing_reports: bool = False) -> None:
    """
    Configure logging based on command line parameters and config.
    
    Args:
        debug: Whether debug mode is enabled
        log_level: Logging level (debug, info, warning, error)
        log_file: Path to log file if specified
        detailed_logs: Whether to enable detailed logs
        timing_reports: Whether to enable timing reports
    """
    try:
        # Use RichLogger's configure_from_settings for centralized configuration
        from src.utils.logger import RichLogger
        
        # Build logging settings dictionary
        logging_settings = {
            "file": log_file,
            "level": "DEBUG" if debug else (log_level.upper() if log_level else config.LOG_LEVEL),
            "console": True,
            "debug": {
                "enabled": debug,
                "verbose": debug,
                "detailed": detailed_logs,
                "show_locals": True,
                "log_config": debug
            },
            "reports": {
                "enabled": timing_reports,
                "timing": timing_reports,
                "environment": False,
                "format": config.REPORT_FORMAT if hasattr(config, "REPORT_FORMAT") else "json",
                "folder": config.REPORTS_FOLDER if hasattr(config, "REPORTS_FOLDER") else "reports"
            }
        }
        
        # Apply all settings at once through the helper function
        RichLogger.configure_from_settings(logging_settings)
        
        # Update the config object to match our settings
        if hasattr(config, "logging"):
            if debug:
                config.debug["verboseLogging"] = True
                config.debug["enabled"] = True
            
            if detailed_logs:
                if "debug" not in config.logging:
                    config.logging["debug"] = {}
                config.logging["debug"]["detailed"] = True
                logger.info("Detailed logging enabled")
            
            # Enable timing reports in config if requested
            if timing_reports:
                if "reports" not in config.logging:
                    config.logging["reports"] = {}
                config.logging["reports"]["enabled"] = True
                config.logging["reports"]["timing"] = True
                logger.info("Timing reports enabled")
    except Exception as e:
        # Fallback in case of error
        print(f"Error configuring logging: {e}")
        # Configure log file if specified
        if log_file:
            log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "logs"
            os.makedirs(log_dir, exist_ok=True)
            RichLogger.add_file_handler(log_file)
            logger.info(f"Logging to file: {os.path.abspath(log_file)}")
        
        # Set log level based on parameters
        if debug:
            RichLogger.set_level("DEBUG")
            logger.debug("Debug mode enabled with verbose logging")
        elif log_level:
            RichLogger.set_level(log_level.upper())
            logger.info(f"Log level set to {log_level.upper()}")
        else:
            # Use log level from config
            RichLogger.set_level(getattr(config, "LOG_LEVEL", "INFO"))

def process_quality_parameters(quality: Optional[List[str]], multi_quality: List[str]) -> List[str]:
    """
    Process quality-related parameters.
    
    Args:
        quality: List of quality preset names from -q option
        multi_quality: List of quality presets from -mq option
        
    Returns:
        List of quality presets to render
    """
    qualities_to_render = []
    
    # Combine both quality options, with multi_quality taking precedence if both are used
    if multi_quality and len(multi_quality) > 0:
        logger.info(f"Rendering with multiple quality presets: {', '.join(multi_quality)}")
        qualities_to_render = multi_quality
    elif quality and len(quality) > 0:
        logger.info(f"Rendering with quality presets: {', '.join(quality)}")
        qualities_to_render = quality
    else:
        # No quality specified, use default
        default_quality = QualityManager.get_default_preset()
        logger.info(f"Using default quality preset: {default_quality}")
        qualities_to_render = [default_quality]
        
    return qualities_to_render

def process_camera_parameters(
    views: List[str], 
    all_views: bool, 
    view_preset: str, 
    multi_view: bool,
    camera_distance: float
) -> Tuple[List[str], bool]:
    """
    Process camera-related parameters.
    
    Args:
        views: List of camera view names
        all_views: Whether to render all views
        view_preset: Camera view preset name
        multi_view: Whether multi-view mode is enabled
        camera_distance: Camera distance from object
        
    Returns:
        Tuple of (camera_views, is_multi_view)
    """
    try:
        is_multi_view = multi_view
        
        # Force multi-view mode if we have multiple camera views
        if multi_view or all_views or view_preset or (views and len(views) > 1):
            if not multi_view:
                is_multi_view = True
                logger.info("Multiple camera views specified, enabling multi-view rendering mode")
            
            # Get camera views for multi-view rendering
            camera_views = CameraViewManager.get_camera_views(all_views, view_preset, views, is_multi_view)
            
            if camera_views:
                logger.debug(f"Using {len(camera_views)} camera views for multi-view rendering: {', '.join(camera_views)}")
                # Enable multi-view in config (helps with proper path generation)
                config.features["enable"]["multiView"] = True
                
                # Configure camera distance from config or command-line
                if camera_distance:
                    if not hasattr(config.rendering, "camera"):
                        config.rendering["camera"] = {}
                    config.rendering["camera"]["distance"] = camera_distance
                    logger.debug(f"Setting camera distance to {camera_distance}")
        else:
            # Single view mode - use default or specified view
            single_view = views[0] if views and len(views) == 1 else "perspective"
            camera_views = [single_view]
            logger.debug(f"Using single camera view: {single_view}")
            config.features["enable"]["multiView"] = False
            
        return camera_views, is_multi_view
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(error_msg)
        typer.echo(error_msg, err=True)
        raise typer.Exit(code=1)

def validate_folder(folder: str) -> str:
    """
    Validate that the specified folder exists and resolve path.
    
    Args:
        folder: Path to folder
    
    Returns:
        Normalized path to folder
        
    Raises:
        typer.Exit: If the folder cannot be found
    """
    # Normalize folder path
    folder_path = os.path.normpath(folder)
    
    # Check if folder exists, try to find similar folders if it doesn't
    if os.path.isdir(folder_path):
        return folder_path
        
    # Try as absolute path
    if os.path.isdir(os.path.abspath(folder_path)):
        return os.path.abspath(folder_path)
    
    # Try relative to current directory
    if os.path.isdir(os.path.join(os.getcwd(), folder_path)):
        return os.path.join(os.getcwd(), folder_path)
    
    # Suggest similar folder names
    all_dirs = [d for d in os.listdir() if os.path.isdir(d)]
    similar_dirs = difflib.get_close_matches(folder, all_dirs, n=3, cutoff=0.5)
    
    if similar_dirs:
        suggestions = ", ".join([f"'{d}'" for d in similar_dirs])
        error_msg = f"Folder not found: '{folder}'. Did you mean: {suggestions}?"
    else:
        error_msg = f"Folder not found: '{folder}'"
    
    logger.error(error_msg)
    typer.echo(error_msg, err=True)
    raise typer.Exit(code=1)

def prepare_render_params(
    spp: Optional[int], 
    width: int, 
    height: int, 
    max_depth: Optional[int],
    integrator_type: Optional[str]
) -> Dict[str, Any]:
    """
    Prepare render parameters from CLI options.
    
    Args:
        spp: Samples per pixel
        width: Image width
        height: Image height
        max_depth: Maximum ray depth
        integrator_type: Integrator type
        
    Returns:
        Dictionary of render parameters
    """
    # Base parameters
    render_params = {
        "width": width,
        "height": height,
        "integrator_type": integrator_type or config.rendering.get("integrator", {}).get("type", "path"),
    }
    
    # Apply safe SPP calculation if explicitly provided
    if spp is not None:
        from src.mitsuba.quality import QualityManager
        safe_spp = QualityManager.safe_sample_count(spp, width, height)
        if safe_spp != spp:
            logger.warning(f"Reducing requested SPP from {spp} to {safe_spp} to stay under the 2^32 sample limit")
        render_params["spp"] = safe_spp
    
    # Override individual settings if explicitly provided
    if max_depth is not None:
        render_params["max_depth"] = max_depth
        
    return render_params

def update_config_from_cli(
    output: Optional[str], 
    enable_video: Optional[bool], 
    enable_gif: Optional[bool],
    threaded: Optional[bool],
    timing_reports: bool
) -> None:
    """
    Update configuration settings from CLI parameters.
    
    Args:
        output: Output directory
        enable_video: Whether to enable video creation
        enable_gif: Whether to enable gif creation
        threaded: Whether to use multi-threading
        timing_reports: Whether to enable timing reports
    """
    # Override config settings only if explicitly provided
    if output is not None:
        config.OUTPUT_FOLDER = output
    if enable_video is not None:
        config.ENABLE_VIDEO = enable_video
    if enable_gif is not None:
        config.ENABLE_GIF = enable_gif
    if threaded is not None:
        config.MULTI_THREADED = threaded
    
    # Enable timing reports in config if requested
    if timing_reports:
        if not hasattr(config.features, "reports"):
            config.features["reports"] = {}
        config.features["reports"]["enabled"] = True
        config.features["reports"]["timing"] = True
        logger.info("Timing reports enabled")

def setup_cache(enable_cache: Optional[bool], output_folder: str, camera_views: List[str], quality: str, render_params: Dict[str, Any]) -> Optional[CacheManager]:
    """
    Setup caching if enabled.
    
    Args:
        enable_cache: Whether to enable caching (None = use config setting)
        output_folder: Output directory
        camera_views: List of camera views
        quality: Quality preset
        render_params: Render parameters
        
    Returns:
        CacheManager instance if caching is enabled, None otherwise
    """
    # Update cache settings if specified
    if enable_cache is not None:
        config.features["cache"]["enabled"] = enable_cache
        logger.info(f"Frame caching {'enabled' if enable_cache else 'disabled'}")
    
    # Check if caching is enabled
    cache_enabled = config.features.get("cache", {}).get("enabled", False)
    if cache_enabled:
        logger.info("Cache enabled - will skip rendering identical frames")
        return CacheManager(
            cache_dir=os.path.join(output_folder, config.paths["output"]["folders"]["cache"]),
            quality=quality,
            camera_views=camera_views,
            render_params=render_params
        )
    
    return None

def get_worker_count(threaded: bool, max_workers: Optional[int]) -> Optional[int]:
    """
    Determine the number of worker threads to use.
    
    Args:
        threaded: Whether threading is enabled
        max_workers: Maximum worker threads (if specified)
        
    Returns:
        Number of worker threads or None if threading is disabled
    """
    if not threaded:
        return None
        
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        logger.debug(f"Using {max_workers} worker threads")
        
    return max_workers

def prepare_integrator_override(integrator_type: Optional[str]) -> Dict[str, Any]:
    """
    Prepare integrator override parameters.
    
    Args:
        integrator_type: Integrator type name
        
    Returns:
        Dictionary with integrator override parameters
    """
    integrator_override = {}
    if integrator_type:
        integrator_override["type"] = integrator_type
        
    return integrator_override

def set_mitsuba_variant(device: str) -> Tuple[bool, str]:
    """
    Set the Mitsuba variant based on the selected device.
    
    Args:
        device: The rendering device to use
        
    Returns:
        Tuple of (success status, variant name)
    """
    from src.mitsuba.mitsuba import MitsubaUtils
    success, variant, message = MitsubaUtils.set_variant(device)
    logger.debug(message)
    return success, variant

def process_render_parameters(
    # Input folder and file options
    folder: str,
    regex: str,
    scene_name: str,
    
    # Performance options
    threaded: bool,
    max_workers: Optional[int],
    stop_on_error: bool,
    
    # Output options
    enable_video: Optional[bool],
    enable_gif: Optional[bool],
    output: Optional[str],
    
    # Rendering device
    device: str,
    
    # Quality settings
    quality: Optional[List[str]],
    multi_quality: List[str],
    spp: Optional[int],
    width: int,
    height: int,
    max_depth: Optional[int],
    
    # Camera settings
    views: List[str],
    all_views: bool,
    view_preset: Optional[str],
    multi_view: bool,
    camera_distance: float,
    
    # Other rendering settings
    integrator_type: Optional[str],
    enable_cache: Optional[bool],
    
    # Denoising options
    enable_denoising: Optional[bool],
    denoising_type: str,
    denoising_strength: float,
    denoising_guide_buffers: bool,
    
    # AOV options
    enable_aovs: bool,
    aov_types: List[str],
    aov_separate_files: bool,
    
    # Logging settings
    debug: bool,
    log_level: str,
    log_file: Optional[str],
    detailed_logs: bool,
    timing_reports: bool,
    
    # Moved this parameter to the end since it has a default value
    fallback_if_unavailable: bool = True,
) -> Dict[str, Any]:
    """
    Process all render parameters from CLI and prepare everything for rendering.
    
    Returns:
        Dictionary containing all processed parameters needed for rendering
    """
    # Set up logging first to ensure proper diagnostics
    setup_logging(debug, log_level, log_file, detailed_logs, timing_reports)
    
    # Update cache settings in config
    if enable_cache is not None:
        config.features["cache"]["enabled"] = enable_cache
        logger.info(f"Frame caching {'enabled' if enable_cache else 'disabled'}")
    
    cache_enabled = config.features.get("cache", {}).get("enabled", False)
    if cache_enabled:
        logger.info("Cache enabled - will skip rendering identical frames")
    
    # Generate timestamp for output organization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle quality settings (we've changed quality type to be List[str])
    qualities_to_render = process_quality_parameters(quality, multi_quality)
    
    # Process camera views
    camera_views, is_multi_view = process_camera_parameters(
        views, all_views, view_preset, multi_view, camera_distance
    )
    
    # Process render parameters 
    base_render_params = prepare_render_params(spp, width, height, max_depth, integrator_type)
    
    # Update config with output settings
    update_config_from_cli(output, enable_video, enable_gif, threaded, timing_reports)
    
    # Handle integrator overrides
    integrator_override = prepare_integrator_override(integrator_type)
    
    # Set up threading and worker count
    if threaded is None:
        threaded = config.MULTI_THREADED
    if enable_video is None:
        enable_video = config.ENABLE_VIDEO
    if enable_gif is None:
        enable_gif = config.ENABLE_GIF
    
    worker_count = get_worker_count(threaded, max_workers)
    
    # Set Mitsuba variant
    set_success, variant = set_mitsuba_variant(device)
    logger.debug(f"Using Mitsuba variant: {variant}")
    
    # Process denoising settings
    if enable_denoising is not None:
        config.DENOISING_ENABLED = enable_denoising
        config.features["denoising"]["enabled"] = enable_denoising
        logger.info(f"Image denoising {'enabled' if enable_denoising else 'disabled'}")
    
    denoising_enabled = config.DENOISING_ENABLED
    
    if denoising_enabled:
        # Check first if any denoiser is available to warn the user early
        try:
            from src.mitsuba.denoising import OPTIX_AVAILABLE
            if not OPTIX_AVAILABLE:
                logger.warning("Denoising is enabled but OptiX denoiser is not available")
                if fallback_if_unavailable:
                    logger.info("Will continue rendering without denoising")
                else:
                    logger.error("Cannot continue without an OptiX denoiser when --no-denoiser-fallback is set")
                    raise typer.Exit(code=1)
        except ImportError as e:
            logger.warning(f"Failed to check denoiser availability: {e}")
            
        # Update denoising configuration
        if denoising_type:
            config.DENOISING_TYPE = denoising_type
            config.features["denoising"]["type"] = denoising_type
        
        config.DENOISING_STRENGTH = denoising_strength
        config.features["denoising"]["strength"] = denoising_strength
        
        config.DENOISING_USE_GUIDE_BUFFERS = denoising_guide_buffers
        config.features["denoising"]["useGuideBuffers"] = denoising_guide_buffers
        
        config.DENOISING_FALLBACK = fallback_if_unavailable
        config.features["denoising"]["fallbackIfUnavailable"] = fallback_if_unavailable
        
        logger.debug(f"Denoising configuration: type={config.DENOISING_TYPE}, "
                   f"strength={config.DENOISING_STRENGTH}, "
                   f"guide buffers={'enabled' if denoising_guide_buffers else 'disabled'}")
    
    # Process AOV settings
    if enable_aovs:
        config.AOV_ENABLED = enable_aovs
        config.features["aov"]["enabled"] = enable_aovs
        
        # Validate AOV types
        valid_aov_types = ["albedo", "depth", "position", "uv", "geo_normal", "sh_normal", "prim_index", "shape_index"]
        validated_aov_types = []
        for aov_type in aov_types:
            if aov_type in valid_aov_types:
                validated_aov_types.append(aov_type)
            else:
                logger.warning(f"Unknown AOV type '{aov_type}', ignoring")
                
        if not validated_aov_types:
            logger.warning("No valid AOV types specified, using defaults")
            validated_aov_types = ["albedo", "depth", "sh_normal"]
        
        config.AOV_TYPES = validated_aov_types
        config.features["aov"]["types"] = validated_aov_types
        config.features["aov"]["output_separate_files"] = aov_separate_files
        
        # Update AOV string in config
        config._setup_aov_attributes()
        
        # Enable guide buffers for denoising if configured
        if config.features["aov"]["guide_buffers_for_denoising"] and enable_denoising:
            logger.info("Using AOVs as guide buffers for denoising")
            config.DENOISING_USE_GUIDE_BUFFERS = True
            config.features["denoising"]["useGuideBuffers"] = True
            
        logger.info(f"AOVs enabled with types: {', '.join(validated_aov_types)}")
    
    # Return all processed parameters for the render function
    return {
        "folder_path": folder,
        "regex": regex,
        "scene_name": scene_name,
        "threaded": threaded,
        "max_workers": worker_count,
        "stop_on_error": stop_on_error,
        "qualities_to_render": qualities_to_render,
        "camera_views": camera_views,
        "base_render_params": base_render_params,
        "integrator_override": integrator_override,
        "cache_enabled": cache_enabled,
        "timestamp": timestamp,
        "denoising_enabled": denoising_enabled,
        "denoising_config": config.features.get("denoising", {}),
        "fallback_if_unavailable": fallback_if_unavailable,
    }

def run_multi_obj_rendering(params: Dict[str, Any]) -> Tuple[bool, timedelta]:
    """
    Execute the multi-object rendering process with the processed parameters.
    
    Args:
        params: Dictionary of processed parameters
        
    Returns:
        Tuple of (success, elapsed_time)
    """
    from src.utils.environment import ensure_environment
    from src.renderers import MultiObjSceneRenderer
    
    # Make sure the environment is properly configured
    ensure_environment()
    
    # Log the current configuration
    logger.info(f"Output folder: {config.OUTPUT_FOLDER}")
    logger.debug(f"Video output: {config.ENABLE_VIDEO}")
    logger.debug(f"GIF output: {config.ENABLE_GIF}")
    logger.debug(f"Multi-threaded: {params['threaded']}")
    logger.debug(f"Using template XML file: {config.TEMPLATE_XML}")
    
    start_time = datetime.now()
    
    try:
        # Main progress bar for multiple quality presets
        with tqdm(total=len(params['qualities_to_render']), 
                  desc="Quality presets", 
                  position=0, 
                  leave=True, 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as quality_pbar:
                  
            # Loop through each quality preset and render
            overall_success = True
            for q_index, current_quality in enumerate(params['qualities_to_render']):
                quality_pbar.set_description(f"Quality preset: {current_quality.upper()} ({q_index+1}/{len(params['qualities_to_render'])})")
                logger.info(f"Rendering with quality preset: {current_quality}")
                
                # Create a copy of base params and apply quality settings
                render_params = params['base_render_params'].copy()
                render_params = QualityManager.apply_preset_to_params(current_quality, render_params)
                
                # Add quality and timestamp to scene name for better organization
                current_scene_name = f"{params['scene_name']}_{current_quality}"
                if len(params['qualities_to_render']) > 1:
                    current_scene_name = f"{params['scene_name']}_{current_quality}_{params['timestamp']}"
                
                # Initialize cache manager if caching is enabled
                cache_manager = None
                if params['cache_enabled']:
                    cache_manager = CacheManager(
                        cache_dir=os.path.join(config.OUTPUT_FOLDER, config.paths["output"]["folders"]["cache"]),
                        quality=current_quality,
                        camera_views=params['camera_views'],
                        render_params=render_params
                    )
                
                # Render the scenes with this quality preset
                success = MultiObjSceneRenderer.render_multi_obj_scene_from_folder(
                    folder=params['folder_path'],
                    regex=params['regex'],
                    scene_name=current_scene_name,
                    threaded=params['threaded'],
                    max_workers=params['max_workers'],
                    render_params=render_params,
                    stop_on_error=params['stop_on_error'],
                    camera_views=params['camera_views'],
                    integrator_override=params['integrator_override'],
                    cache_manager=cache_manager
                )
                
                overall_success = overall_success and success
                quality_pbar.update(1)
                
                if not success and params['stop_on_error']:
                    logger.error(f"Rendering stopped due to errors in quality preset: {current_quality}")
                    break
            
            # Calculate and log total time
            end_time = datetime.now()
            elapsed = end_time - start_time
            logger.info(f"Rendering completed in {elapsed}")
            
            return overall_success, elapsed
            
    except Exception as e:
        logger.error(f"Error during rendering: {e}", exc_info=True)
        end_time = datetime.now()
        elapsed = end_time - start_time
        return False, elapsed