"""
Application-wide constants for the Mitsuba renderer.

This module defines constants used throughout the application,
including folder names, file extensions, and default settings.
"""

import os
import shutil
import platform
from enum import Enum, auto
from typing import Dict, Any

# Output directories and asset paths
OUTPUT_FOLDER = "output"
EXR_FOLDER = "exr"
PNG_FOLDER = "png"
VIDEO_FOLDER = "video"
GIF_FOLDER = "gif"
SCENE_FOLDER = "scenes"
MESH_FOLDER = "meshes"
REPORTS_FOLDER = "reports"
LOGS_FOLDER = "logs"
CACHE_FOLDER = "cache"

# File extensions and files
EXR_EXT = ".exr"
PNG_EXT = ".png"
XML_EXT = ".xml"
OBJ_EXT = ".obj"
PLY_EXT = ".ply"
VIDEO_EXT = ".mp4"
GIF_EXT = ".gif"
VIDEO_FILE = ".mp4"
GIF_FILE = ".gif"
LOG_FILE = "mitsuba.log"

# Rendering configuration
FRAMERATE = 24
QUALITY_PRESETS = {
    "low": {"spp": 64, "max_depth": 3},
    "medium": {"spp": 512, "max_depth": 8},
    "high": {"spp": 2048, "max_depth": 16},
    "ultra": {"spp": 8192, "max_depth": 24}
}

# FFmpeg executable path 
FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

# Configuration file names
CONFIG_JSON_FILENAME = "config.json"
CONFIG_SCHEMA_FILENAME = os.path.join("src", "assets", "schema", "config_schema.json")
ENV_FILE = ".env"

# Default settings
MULTI_THREADED_DEFAULT = True
MAX_THREADS = os.cpu_count() or 4

# Template settings
TEMPLATE_XML = os.path.join("src", "assets", "templates", "scene_template.xml")
DEFAULT_TEMPLATE_FILENAME = "scene_template.xml"
DEFAULT_TEMPLATE_DESTINATION = "default.xml"

# Output flags (enable/disable)
ENABLE_GIF = True
ENABLE_VIDEO = True
ENABLE_EXR = True
ENABLE_PNG = True

# Rendering settings defaults
DEFAULT_SPP = 128
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_MAX_DEPTH = 8
DEFAULT_OBJ_REGEX = r"\d+\.obj"
DEFAULT_SCENE_PREFIX = "multi_obj"

# Performance settings
DEFAULT_MAX_WORKERS = None  # Will be set based on CPU count
DEFAULT_STOP_ON_ERROR = True

# Logging and debug settings (consolidated)
DEFAULT_LOG_SETTINGS = {
    "file": None, 
    "level": "INFO",
    "console": True,
    "reports": {
        "enabled": False,
        "environment": False,
        "timing": False,
        "format": "json",
        "folder": REPORTS_FOLDER
    },
    "debug": {
        "enabled": False,
        "verbose": False,
        "detailed": False,
        "show_locals": True,
        "log_config": False
    }
}

# Temporary directory constant
TEMP_DIR_NAME = "mitsuba_tmp"

# Additional asset paths
ASSETS_DIR = os.path.join("src", "assets")
TEMPLATES_DIR = os.path.join(ASSETS_DIR, "templates")
SCHEMA_DIR = os.path.join(ASSETS_DIR, "schema")

# System information
SYSTEM_INFO = {
    "platform": platform.system(),
    "platform_version": platform.version(),
    "python_version": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "architecture": platform.architecture()[0]
}

# Mitsuba specific settings
DEFAULT_MITSUBA_VARIANT = "scalar_rgb"
AVAILABLE_VARIANTS = [
    "scalar_rgb",
    "scalar_spectral",
    "cuda_rgb",
    "cuda_spectral",
    "cuda_ad_rgb",
    "cuda_ad_spectral",
    "llvm_rgb",
    "llvm_spectral",
]

# Environment variable names
ENV_VAR_PREFIX = "MITSUBA_"
ENV_VARS = {
    "VARIANT": f"{ENV_VAR_PREFIX}VARIANT",
    "OUTPUT_DIR": f"{ENV_VAR_PREFIX}OUTPUT_DIR",
    "LOG_LEVEL": f"{ENV_VAR_PREFIX}LOG_LEVEL",
    "THREADS": f"{ENV_VAR_PREFIX}THREADS",
    "FFMPEG": f"{ENV_VAR_PREFIX}FFMPEG"
}

# Frame patterns for FFmpeg
FRAME_PATTERN = "frame_%d"
FRAME_VIEW_PATTERN = "frame_%d_%s"  # Pattern with view: frame_0_perspective.png

# Denoising configuration defaults
DENOISING_ENABLED = False
DENOISING_TYPE = "optix"  # 'optix' or 'oidn'
DENOISING_STRENGTH = 1.0  # Range 0.0-1.0
DENOISING_ALWAYS_USE_GUIDE_BUFFERS = False  # Whether to always generate guide buffers
DENOISING_AUTO_DENOISE = True  # Whether to automatically denoise after rendering

class RenderDevice(Enum):
    """Enum for rendering device types."""
    CPU = "cpu"
    CUDA = "cuda"
    OPTIX = "optix"
    LLVM = "llvm"
    
    @classmethod
    def get_default(cls) -> str:
        """Get the default rendering device based on system capabilities."""
        # Simple logic for default device - can be expanded
        if platform.system() == "Windows" and shutil.which("nvcc") is not None:
            return cls.CUDA.value
        return cls.CPU.value

class LogLevel(Enum):
    """Enum for log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @classmethod
    def get_level(cls, name: str) -> str:
        """Get the log level from a string name, case-insensitive."""
        try:
            return getattr(cls, name.upper()).value
        except (AttributeError, TypeError):
            return cls.INFO.value

class OutputFormat(Enum):
    """Enum for output file formats."""
    EXR = "exr"
    PNG = "png"
    JPG = "jpg"
    TIFF = "tiff"
    
    @classmethod
    def get_extension(cls, format_type: str) -> str:
        """Get file extension from format type."""
        if format_type.lower() == cls.EXR.value:
            return EXR_EXT
        elif format_type.lower() == cls.PNG.value:
            return PNG_EXT
        elif format_type.lower() == cls.JPG.value:
            return ".jpg"
        elif format_type.lower() == cls.TIFF.value:
            return ".tiff"
        return ""

def get_default_values() -> Dict[str, Any]:
    """Get a dictionary of default values for configuration."""
    return {
        "output_folder": OUTPUT_FOLDER,
        "framerate": FRAMERATE,
        "spp": DEFAULT_SPP,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "max_depth": DEFAULT_MAX_DEPTH,
        "device": RenderDevice.get_default(),
        "variant": DEFAULT_MITSUBA_VARIANT,
        "multi_threaded": MULTI_THREADED_DEFAULT,
        "max_threads": MAX_THREADS,
        "enable_gif": ENABLE_GIF,
        "enable_video": ENABLE_VIDEO,
        "enable_exr": ENABLE_EXR,
        "enable_png": ENABLE_PNG,
        "log_level": DEFAULT_LOG_SETTINGS["level"]
    }
