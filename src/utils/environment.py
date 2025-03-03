"""
Utilities for environment setup and management.

This module provides functions to ensure the rendering environment
is properly configured before running any rendering jobs.
"""

import os
import datetime
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.utils.logger import RichLogger
from src.config.config import config_class as config
from src.utils.common import (
    ensure_directory, 
    copy_file,
    get_system_info as _get_system_info,
    check_package_availability
)
from src.config.constants import (
    REPORTS_FOLDER, OUTPUT_FOLDER, SCENE_FOLDER, MESH_FOLDER,
    LOGS_FOLDER, CACHE_FOLDER, DEFAULT_MITSUBA_VARIANT,
    AVAILABLE_VARIANTS, ENV_VARS, FFMPEG_PATH
)
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.environment")

# Make sure these are explicitly exported for backward compatibility
create_directory = ensure_directory

@timeit(log_level="debug")
def ensure_environment() -> Dict[str, Any]:
    """
    Ensure the environment is properly set up for rendering.
    
    Returns:
        Dictionary containing environment setup information.
    """
    env_info = {
        "setup_time": datetime.datetime.now().isoformat(),
        "directories_created": [],
        "template_copied": False,
        "mitsuba_variant": get_mitsuba_variant(),
        "ffmpeg_available": is_ffmpeg_available(),
        "system_info": get_system_info(),
    }
    
    # List of directories to create
    dirs = [
        os.path.join(config.OUTPUT_FOLDER, config.EXR_FOLDER),
        os.path.join(config.OUTPUT_FOLDER, config.PNG_FOLDER), 
        os.path.join(config.OUTPUT_FOLDER, config.VIDEO_FOLDER),
        os.path.join(config.OUTPUT_FOLDER, config.GIF_FOLDER),
        os.path.join(config.OUTPUT_FOLDER, config.SCENE_FOLDER),
        os.path.join(config.OUTPUT_FOLDER, config.MESH_FOLDER),
        os.path.join(config.OUTPUT_FOLDER, REPORTS_FOLDER),
        os.path.join(config.OUTPUT_FOLDER, LOGS_FOLDER),
    ]
    
    # Only create cache directory if caching is enabled
    if hasattr(config, 'ENABLE_CACHE') and config.ENABLE_CACHE:
        dirs.append(os.path.join(config.OUTPUT_FOLDER, CACHE_FOLDER))
    
    for d in dirs:
        dir_path = create_directory(d)
        env_info["directories_created"].append(str(dir_path))
    
    # Copy the default template to the output directory if necessary
    try:
        template_path = ensure_default_template()
        env_info["template_copied"] = True
        env_info["template_path"] = template_path
    except FileNotFoundError as e:
        logger.warning(f"Could not copy template: {e}")
        env_info["template_copied"] = False
    
    # Write environment information to JSON report only if enabled
    if hasattr(config, 'ENABLE_REPORTS') and config.ENABLE_REPORTS and \
       hasattr(config, 'ENABLE_ENVIRONMENT_REPORT') and config.ENABLE_ENVIRONMENT_REPORT:
        report_path = os.path.join(config.OUTPUT_FOLDER, REPORTS_FOLDER, "environment_setup.json")
        create_directory(os.path.dirname(report_path))
        with open(report_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        logger.debug(f"Environment report saved to {report_path}")
    
    logger.debug("Environment setup completed")
    
    # If in debug mode, print system information
    if config.DEBUG.get("enabled", False):
        print_system_info()
    
    return env_info

@timeit(log_level="debug")
def get_app_root_dir() -> Path:
    """
    Get the application's root directory.
    
    Returns:
        Path to the app root directory.
    """
    # The app root is typically where main.py is located
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

@timeit(log_level="debug")
def ensure_default_template() -> str:
    """
    Ensure the default template XML file exists and is copied to the output directory.
    
    Returns:
        The destination path of the default template.
        
    Raises:
        FileNotFoundError: If no template file is found in any expected location.
    """
    app_root = get_app_root_dir()
    configured_path = config.paths["templates"]["scene"]
    
    possible_locations = [
        str(app_root / "src/assets/templates/scene_template.xml"),
        str(app_root / "src/templates/scene_template.xml"),
        configured_path,
        "src/assets/templates/scene_template.xml",
    ]
    
    template_path = None
    for loc in possible_locations:
        if os.path.isfile(loc):
            template_path = loc
            break

    if not template_path:
        logger.error("Default template XML file not found in any expected location.")
        raise FileNotFoundError("Default template XML file not found. Please ensure it exists in src/assets/templates/")
    
    # Define the destination path in the output directory
    dest_path = os.path.join(config.OUTPUT_FOLDER, config.SCENE_FOLDER, "default.xml")
    copy_file(template_path, dest_path)
    logger.debug(f"Copied template from {template_path} to {dest_path}")
    
    return dest_path

@timeit(log_level="debug")
def get_temp_dir() -> Path:
    """
    Get a temporary directory for intermediate files.
    
    Returns:
        Path to a temporary directory.
    """
    temp_dir = Path(tempfile.gettempdir()) / "mitsuba_tmp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

@timeit(log_level="debug")
def cleanup_environment(temp_files: Optional[List[str]] = None) -> None:
    """
    Clean up temporary files created during rendering.
    
    Args:
        temp_files: List of temporary files to remove.
    """
    if not temp_files:
        return
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
    logger.debug("Environment cleanup completed")

def get_mitsuba_variant() -> str:
    """
    Get the current Mitsuba variant being used.
    
    Returns:
        String representing the Mitsuba variant.
    """
    # First try to get from environment variable
    variant = os.environ.get(ENV_VARS["VARIANT"])
    
    if not variant:
        # Try to get from loaded config
        try:
            variant = getattr(config, "MITSUBA_VARIANT", DEFAULT_MITSUBA_VARIANT)
        except:
            variant = DEFAULT_MITSUBA_VARIANT
    
    # Validate that it's a known variant
    if variant not in AVAILABLE_VARIANTS:
        logger.warning(f"Unknown Mitsuba variant '{variant}'. Using default '{DEFAULT_MITSUBA_VARIANT}'.")
        variant = DEFAULT_MITSUBA_VARIANT
        
    return variant

def print_system_info() -> None:
    """Print system information for debugging."""
    system_info = get_system_info()
    logger.debug("System Information:")
    for key, value in system_info.items():
        if isinstance(value, dict):
            logger.debug(f"  {key}: {json.dumps(value, indent=2)}")
        else:
            logger.debug(f"  {key}: {value}")

def get_system_info() -> Dict[str, Any]:
    """Get detailed system information"""
    return _get_system_info()

def is_mitsuba_available() -> bool:
    """Check if Mitsuba is available"""
    # Use the single package check function, not the list version
    return check_package_availability("mitsuba")

def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available"""
    try:
        if not os.path.exists(config.FFMPEG_PATH) and not os.path.exists(FFMPEG_PATH):
            return False
        # Use the run_subprocess from common.py
        from src.utils.common import run_subprocess
        return_code, _, _ = run_subprocess([config.FFMPEG_PATH or FFMPEG_PATH, "-version"])
        return return_code == 0
    except Exception:
        return False

if __name__ == "__main__":
    # When run directly, print environment information
    ensure_environment()
    print_system_info()

# Export all important functions and aliases
__all__ = [
    'ensure_environment',
    'create_directory',
    'ensure_directory',
    'get_app_root_dir',
    'ensure_default_template',
    'get_mitsuba_variant',
    'is_ffmpeg_available',
    'is_mitsuba_available',
    'get_system_info'
]