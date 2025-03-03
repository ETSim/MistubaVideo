"""
Configuration management for the Mitsuba application.

This module loads and manages application configuration from various sources,
including JSON files, environment variables, and .env files.
"""

import sys
import os
import json
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List, Union, Tuple
import importlib

from src.config.constants import (
    OUTPUT_FOLDER,
    EXR_FOLDER,
    PNG_FOLDER,
    VIDEO_FOLDER,
    GIF_FOLDER,
    SCENE_FOLDER,
    MESH_FOLDER,
    REPORTS_FOLDER,
    LOGS_FOLDER,
    CACHE_FOLDER,
    TEMPLATE_XML,
    FFMPEG_PATH,
    EXR_EXT,
    PNG_EXT,
    VIDEO_FILE,
    GIF_FILE,
    MULTI_THREADED_DEFAULT,
    FRAMERATE,
    ENABLE_GIF,
    ENABLE_VIDEO,
    ENABLE_EXR,
    ENABLE_PNG,
    CONFIG_JSON_FILENAME,
    CONFIG_SCHEMA_FILENAME,
    ENV_FILE,
    DEFAULT_LOG_SETTINGS,  # Add this missing import
)
from rich.console import Console
from rich.table import Table
from src.utils.logger import RichLogger

logger = RichLogger.get_logger("mitsuba_app.config")

# Default denoising configuration
DEFAULT_DENOISING_CONFIG = {
    "enabled": True,
    "type": "optix",
    "strength": 1.0,
    "useGuideBuffers": False,
    "useTemporal": False,
    "fallbackIfUnavailable": True,
    "applyToQuality": {
        "low": True,
        "medium": True,
        "high": False,
        "ultra": False
    },
    "reconstruction_filter": "box"
}

# Default AOV configuration
DEFAULT_AOV_CONFIG = {
    "enabled": False,
    "types": ["albedo", "depth", "sh_normal"],
    "guide_buffers_for_denoising": True,
    "output_separate_files": True
}

class Config:
    """
    Configuration manager that loads settings from various sources
    with a priority order: environment variables > config.json > .env > defaults.

    Implements the singleton pattern to ensure only one instance exists.
    """
    _instance = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        """Set up the configuration by loading from various sources."""
        # Set default configuration values using constants and complete schema defaults
        self.defaults: Dict[str, Any] = {
            "paths": {
                "output": {
                    "root": OUTPUT_FOLDER,
                    "folders": {
                        "exr": EXR_FOLDER,
                        "png": PNG_FOLDER,
                        "video": VIDEO_FOLDER,
                        "gif": GIF_FOLDER,
                        "scenes": SCENE_FOLDER,
                        "meshes": MESH_FOLDER,
                        "reports": REPORTS_FOLDER,
                        "logs": LOGS_FOLDER,
                        "cache": CACHE_FOLDER,
                    },
                },
                "templates": {
                    "scene": TEMPLATE_XML,
                },
                "ffmpeg": FFMPEG_PATH,
            },
            "files": {
                "extensions": {
                    "exr": EXR_EXT,
                    "png": PNG_EXT,
                },
                "defaults": {
                    "video": VIDEO_FILE,
                    "gif": GIF_FILE,
                },
            },
            "rendering": {
                "multiThreaded": MULTI_THREADED_DEFAULT,
                "framerate": 24,  # Default framerate
                "timing": {
                    "video": {
                        "frameInterval": 1.0,  # Default 1 second between frames
                        "duration": None,      # No default duration (use frameInterval)
                    },
                    "gif": {
                        "frameInterval": 1.0,  # Default 1 second between frames
                        "duration": None,      # No default duration (use frameInterval)
                    }
                },
                "defaults": {
                    "spp": 2048,
                    "resolution": {
                        "width": 1920,
                        "height": 1080,
                    },
                    "maxDepth": 16,
                    "device": "cpu",
                    "stopOnError": True,
                },
                "threading": {
                    "enabled": False,
                    "maxWorkers": None,
                },
                "denoising": {
                    "defaults": {
                        "enabled": True,
                        "quality": {
                            "low": True,
                            "medium": True,
                            "high": False,
                            "ultra": False
                        },
                        "reconstruction_filter": "box"
                    },
                    "guide_buffers": {
                        "enabled": False,
                        "spp": 16
                    }
                },
                "aov": {
                    "enabled": False,
                    "types": {
                        "albedo": True,
                        "depth": True,
                        "position": False,
                        "uv": False,
                        "geo_normal": False,
                        "sh_normal": True,
                        "prim_index": False,
                        "shape_index": False
                    },
                    "output_separate_files": True,
                    "guide_buffers_for_denoising": True
                }
            },
            "features": {
                "enable": {
                    "gif": ENABLE_GIF,
                    "video": ENABLE_VIDEO,
                    "exr": ENABLE_EXR,
                    "png": ENABLE_PNG,
                },
                "reports": {
                    "enabled": False,  # Disable reports by default
                    "environment": False,
                    "timing": False,
                },
                "cache": {
                    "enabled": False,  # Disable caching by default
                    "meshes": False,
                    "scenes": False,
                },
                "denoising": DEFAULT_DENOISING_CONFIG,
                "aov": DEFAULT_AOV_CONFIG,
            },
            "debug": {
                "enabled": False,
                "verboseLogging": False,
            },
            "patterns": {
                "objFiles": r"\d+\.obj",
                "scenePrefix": "multi_obj",
            },
            "logging": DEFAULT_LOG_SETTINGS
        }

        # Begin with the defaults
        config_data: Dict[str, Any] = self.defaults.copy()

        # Load configuration from config.json if it exists
        self._load_from_json(config_data)

        # If no config.json is found, try to load from .env file
        if not os.path.exists(os.path.join(os.getcwd(), CONFIG_JSON_FILENAME)):
            self._load_from_env(config_data)

        # Finally, override with system environment variables
        self._override_from_env(config_data)

        # Set each configuration key as an attribute of the instance
        for key, value in config_data.items():
            setattr(self, key, value)

        # Add convenience attributes for quick access
        self.MULTI_THREADED = self.rendering.get("multiThreaded")
        self.FRAMERATE = self.rendering.get("framerate")
        self.FRAME_INTERVAL = self.rendering.get("frameInterval", 0.1)
        self.RENDERING_DEFAULTS = self.rendering.get("defaults", {})
        self.RENDERING_THREADING = self.rendering.get("threading", {})
        self.DEBUG = self.debug
        self.PATTERNS = self.patterns

        # Flatten feature flags into top-level attributes
        feature_enables = self.features.get("enable", {})
        self.ENABLE_GIF = feature_enables.get("gif", False)
        self.ENABLE_VIDEO = feature_enables.get("video", False)
        self.ENABLE_EXR = feature_enables.get("exr", False)
        self.ENABLE_PNG = feature_enables.get("png", False)

        # Add report settings as top-level attributes
        report_settings = self.features.get("reports", {})
        self.ENABLE_REPORTS = report_settings.get("enabled", False)
        self.ENABLE_ENVIRONMENT_REPORT = report_settings.get("environment", False)
        self.ENABLE_TIMING_REPORT = report_settings.get("timing", False)
        
        # Add cache settings as top-level attributes
        cache_settings = self.features.get("cache", {})
        self.ENABLE_CACHE = cache_settings.get("enabled", False)
        self.ENABLE_MESH_CACHE = cache_settings.get("meshes", False)
        self.ENABLE_SCENE_CACHE = cache_settings.get("scenes", False)

        # Convenience attributes for output paths
        output_paths = self.paths.get("output", {})
        self.OUTPUT_FOLDER = output_paths.get("root", "output")
        folders = output_paths.get("folders", {})
        self.EXR_FOLDER = folders.get("exr", "exr")
        self.PNG_FOLDER = folders.get("png", "png")
        self.VIDEO_FOLDER = folders.get("video", "video")
        self.GIF_FOLDER = folders.get("gif", "gif")
        self.SCENE_FOLDER = folders.get("scenes", "scenes")
        self.MESH_FOLDER = folders.get("meshes", "meshes")
        self.REPORTS_FOLDER = folders.get("reports", "reports")
        self.LOGS_FOLDER = folders.get("logs", "logs")
        self.CACHE_FOLDER = folders.get("cache", "cache")
        self.FFMPEG_PATH = self.paths.get("ffmpeg", "ffmpeg")

        # Flatten file extensions and defaults for easy access
        file_extensions = self.files.get("extensions", {})
        self.EXR_EXT = file_extensions.get("exr", ".exr")
        self.PNG_EXT = file_extensions.get("png", ".png")
        
        file_defaults = self.files.get("defaults", {})
        self.VIDEO_FILE = file_defaults.get("video", "video.mp4")
        self.GIF_FILE = file_defaults.get("gif", "animation.gif")

        # Flatten the template path for easy access
        self.TEMPLATE_XML = self.paths.get("templates", {}).get("scene", TEMPLATE_XML)

        # Update timing-related convenience attributes
        rendering_timing = self.rendering.get("timing", {})
        video_timing = rendering_timing.get("video", {})
        gif_timing = rendering_timing.get("gif", {})

        self.VIDEO_FRAME_INTERVAL = video_timing.get("frameInterval", 1.0)
        self.VIDEO_DURATION = video_timing.get("duration", None)
        self.GIF_FRAME_INTERVAL = gif_timing.get("frameInterval", 1.0)
        self.GIF_DURATION = gif_timing.get("duration", None)

        # Add logging configuration as top-level attributes for easy access
        logging_config = getattr(self, "logging", {})
        self.LOG_FILE = logging_config.get("file", None)
        self.LOG_LEVEL = logging_config.get("level", "INFO").upper()
        self.DETAILED_LOGS = logging_config.get("detailed", False)
        
        # Add report configuration as top-level attributes
        reports_config = logging_config.get("reports", {})
        self.REPORT_FORMAT = reports_config.get("format", "json")
        self.REPORT_FOLDER = reports_config.get("folder", "reports")
        
        # Apply settings to RichLogger
        if self.LOG_FILE:
            from src.utils.logger import RichLogger
            RichLogger.add_file_handler(self.LOG_FILE)

        # Add denoising-specific convenience attributes
        self._setup_denoising_attributes()

        # Add AOV-specific convenience attributes
        self._setup_aov_attributes()

        # Initialize logging configuration
        self._setup_logging()
        
    def _setup_denoising_attributes(self):
        """Set up convenience attributes for denoising configuration."""
        # Get the denoising config
        denoising_config = self.features.get("denoising", DEFAULT_DENOISING_CONFIG)
        
        # Add top-level denoising attributes
        self.DENOISING_ENABLED = denoising_config.get("enabled", True)
        self.DENOISING_TYPE = denoising_config.get("type", "optix")
        self.DENOISING_STRENGTH = denoising_config.get("strength", 1.0)
        self.DENOISING_USE_GUIDE_BUFFERS = denoising_config.get("useGuideBuffers", False)
        self.DENOISING_USE_TEMPORAL = denoising_config.get("useTemporal", False)
        self.DENOISING_FALLBACK = denoising_config.get("fallbackIfUnavailable", True)
        
        # Add quality-specific denoising settings
        quality_settings = denoising_config.get("applyToQuality", {})
        self.DENOISING_QUALITY_LOW = quality_settings.get("low", True)
        self.DENOISING_QUALITY_MEDIUM = quality_settings.get("medium", True)
        self.DENOISING_QUALITY_HIGH = quality_settings.get("high", False)
        self.DENOISING_QUALITY_ULTRA = quality_settings.get("ultra", False)

    def _setup_aov_attributes(self):
        """Set up convenience attributes for AOV configuration."""
        # Get the AOV config
        aov_config = self.features.get("aov", DEFAULT_AOV_CONFIG)
        
        # Add top-level AOV attributes
        self.AOV_ENABLED = aov_config.get("enabled", False)
        self.AOV_TYPES = aov_config.get("types", ["albedo", "depth", "sh_normal"])
        self.AOV_SEPARATE_FILES = aov_config.get("output_separate_files", True)
        self.AOV_GUIDE_BUFFERS_FOR_DENOISING = aov_config.get("guide_buffers_for_denoising", True)
        
        # Create a formatted AOVs string for the integrator
        if self.AOV_ENABLED and self.AOV_TYPES:
            aov_list = []
            for aov_type in self.AOV_TYPES:
                if aov_type == "albedo":
                    aov_list.append("albedo:albedo")
                elif aov_type == "depth":
                    aov_list.append("dd.y:depth")
                elif aov_type == "position":
                    aov_list.append("p:position")
                elif aov_type == "uv":
                    aov_list.append("uv:uv")
                elif aov_type == "geo_normal":
                    aov_list.append("ng:geo_normal")
                elif aov_type == "sh_normal":
                    aov_list.append("nn:sh_normal")
                elif aov_type == "prim_index":
                    aov_list.append("pi:prim_index")
                elif aov_type == "shape_index":
                    aov_list.append("si:shape_index")
            
            self.AOV_STRING = ",".join(aov_list)
        else:
            self.AOV_STRING = ""

    def _setup_logging(self):
        """Configure logging based on current settings."""
        try:
            from src.utils.logger import RichLogger
            RichLogger.configure_from_settings(self.logging)
        except ImportError:
            # Logger might not be available yet during initial import
            pass

    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a specific JSON file.
        
        Args:
            config_file: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # Update the configuration with values from the file
            for key, value in config_data.items():
                if hasattr(self, key):
                    # If the attribute exists, update it
                    setattr(self, key, value)
                else:
                    # Otherwise, add it as a new attribute
                    setattr(self, key, value)
                    
            # Update the convenience attributes to reflect changes
            self._update_convenience_attributes()
                
            logger.info(f"Configuration loaded from {config_file}")
            
            # If we're in debug mode, log all the loaded settings
            if self.debug.get("verboseLogging", False):
                self._log_config_values()
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            raise

    def find_config_file(self, name: Optional[str] = None) -> Optional[str]:
        """
        Find a configuration file using the configured search paths.
        
        Args:
            name: Name of the configuration file (defaults to config.json)
            
        Returns:
            Path to the configuration file if found, None otherwise
        """
        # Get search paths from config
        search_paths = self.features.get('config', {}).get('searchPaths', ['.',])
        name = name or self.features.get('config', {}).get('defaultPath', 'config.json')
        
        # Search for the config file
        for path in search_paths:
            config_path = os.path.join(path, name)
            if os.path.exists(config_path):
                return config_path
                
        return None

    def replace_global_instance(self, new_config_file: str) -> bool:
        """
        Replace this config instance with settings from a new config file.
        
        Args:
            new_config_file: Path to the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the config from the file
            self.load_from_file(new_config_file)
            
            # Re-setup logging with new settings
            self._setup_logging()
            
            # Update module references to ensure all parts of the app see the changes
            import src.config
            reload_result = importlib.reload(src.config)
            
            logger.info(f"Global configuration replaced with values from {new_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to replace configuration: {e}")
            return False

    def _update_convenience_attributes(self):
        """Update convenience attributes to reflect the current configuration."""
        # Add convenience attributes for quick access
        self.MULTI_THREADED = self.rendering.get("multiThreaded")
        self.FRAMERATE = self.rendering.get("framerate")
        self.FRAME_INTERVAL = self.rendering.get("frameInterval", 0.1)
        self.RENDERING_DEFAULTS = self.rendering.get("defaults", {})
        self.RENDERING_THREADING = self.rendering.get("threading", {})
        self.DEBUG = self.debug

        # Flatten feature flags into top-level attributes
        feature_enables = self.features.get("enable", {})
        self.ENABLE_GIF = feature_enables.get("gif", False)
        self.ENABLE_VIDEO = feature_enables.get("video", False)
        self.ENABLE_EXR = feature_enables.get("exr", False)
        self.ENABLE_PNG = feature_enables.get("png", False)

        # Add report settings as top-level attributes
        report_settings = self.features.get("reports", {})
        self.ENABLE_REPORTS = report_settings.get("enabled", False)
        self.ENABLE_ENVIRONMENT_REPORT = report_settings.get("environment", False)
        self.ENABLE_TIMING_REPORT = report_settings.get("timing", False)
        
        # Add cache settings as top-level attributes
        cache_settings = self.features.get("cache", {})
        self.ENABLE_CACHE = cache_settings.get("enabled", False)
        self.ENABLE_MESH_CACHE = cache_settings.get("meshes", False)
        self.ENABLE_SCENE_CACHE = cache_settings.get("scenes", False)

        # Convenience attributes for output paths
        output_paths = self.paths.get("output", {})
        self.OUTPUT_FOLDER = output_paths.get("root", "output")
        folders = output_paths.get("folders", {})
        self.EXR_FOLDER = folders.get("exr", "exr")
        self.PNG_FOLDER = folders.get("png", "png")
        self.VIDEO_FOLDER = folders.get("video", "video")
        self.GIF_FOLDER = folders.get("gif", "gif")
        self.SCENE_FOLDER = folders.get("scenes", "scenes")
        self.MESH_FOLDER = folders.get("meshes", "meshes")
        self.REPORTS_FOLDER = folders.get("reports", "reports")
        self.LOGS_FOLDER = folders.get("logs", "logs")
        self.CACHE_FOLDER = folders.get("cache", "cache")
        self.FFMPEG_PATH = self.paths.get("ffmpeg", "ffmpeg")

        # Flatten file extensions and defaults for easy access
        file_extensions = self.files.get("extensions", {})
        self.EXR_EXT = file_extensions.get("exr", ".exr")
        self.PNG_EXT = file_extensions.get("png", ".png")
        
        file_defaults = self.files.get("defaults", {})
        self.VIDEO_FILE = file_defaults.get("video", "video.mp4")
        self.GIF_FILE = file_defaults.get("gif", "animation.gif")

        # Flatten the template path for easy access
        self.TEMPLATE_XML = self.paths.get("templates", {}).get("scene", TEMPLATE_XML)

        # Update timing-related convenience attributes
        rendering_timing = self.rendering.get("timing", {})
        video_timing = rendering_timing.get("video", {})
        gif_timing = rendering_timing.get("gif", {})

        self.VIDEO_FRAME_INTERVAL = video_timing.get("frameInterval", 1.0)
        self.VIDEO_DURATION = video_timing.get("duration", None)
        self.GIF_FRAME_INTERVAL = gif_timing.get("frameInterval", 1.0)
        self.GIF_DURATION = gif_timing.get("duration", None)
        
        # Add logging configuration as top-level attributes for easy access
        logging_config = getattr(self, "logging", {})
        self.LOG_FILE = logging_config.get("file", None)
        self.LOG_LEVEL = logging_config.get("level", "INFO").upper()
        self.DETAILED_LOGS = logging_config.get("detailed", False)
        
        # Add report configuration as top-level attributes
        reports_config = logging_config.get("reports", {})
        self.REPORT_FORMAT = reports_config.get("format", "json")
        self.REPORT_FOLDER = reports_config.get("folder", "reports")
        
        # Apply settings to logger
        if hasattr(self, 'LOG_LEVEL'):
            from src.utils.logger import RichLogger
            RichLogger.set_level(self.LOG_LEVEL)
        
        # Add denoising-specific convenience attributes
        self._setup_denoising_attributes()

        # Update AOV attributes
        self._setup_aov_attributes()
    
    def _log_config_values(self):
        """Log all configuration values for debugging."""
        logger.debug("Current configuration values:")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if isinstance(value, dict):
                    logger.debug(f"  {key}: {json.dumps(value, indent=2)}")
                else:
                    logger.debug(f"  {key}: {value}")

    def _load_from_json(self, config_data: Dict[str, Any]) -> None:
        """Load configuration from JSON files using separate schemas."""
        json_path = os.path.join(os.getcwd(), CONFIG_JSON_FILENAME)
        if (os.path.exists(json_path)):
            try:
                try:
                    with open(json_path, "r", encoding='utf-8') as f:  # Explicitly specify UTF-8
                        json_config = json.load(f)
                except json.JSONDecodeError as je:
                    logger.error(f"JSON syntax error in {json_path}: {str(je)}")
                    logger.error(f"Error at line {je.lineno}, column {je.colno}")
                    return  # Exit early on JSON syntax errors
                
                # Load and validate with individual schemas
                schema_dir = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "assets", "schema", "schemas"
                )
                
                for schema_name in ["paths", "rendering", "features"]:
                    schema_path = os.path.join(schema_dir, f"{schema_name}_schema.json")
                    if os.path.exists(schema_path):
                        try:
                            with open(schema_path, "r", encoding='utf-8') as f:
                                schema = json.load(f)
                                # Validate only the relevant section
                                if schema_name in json_config:
                                    validate(
                                        instance={schema_name: json_config[schema_name]},
                                        schema=schema
                                    )
                        except json.JSONDecodeError as je:
                            logger.error(f"JSON syntax error in schema {schema_path}: {str(je)}")
                            continue
                        except ValidationError as ve:
                            logger.error(f"Validation error for {schema_name}: {str(ve)}")
                            continue

                # Merge JSON values into config_data
                config_data.update(json_config)
                
            except Exception as e:
                logger.error(f"Unexpected error loading config: {str(e)}")
                logger.debug("Using default configuration")

    def _load_from_env(self, config_data: Dict[str, Any]) -> None:
        """Load configuration from a .env file."""
        env_path = os.path.join(os.getcwd(), ENV_FILE)
        if (os.path.exists(env_path)):
            load_dotenv(dotenv_path=env_path)
            for key in self.defaults.keys():
                value = os.getenv(key)
                if value is not None:
                    # Force type conversion if necessary
                    if key.lower() == "framerate":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    config_data[key] = value

    def _override_from_env(self, config_data: Dict[str, Any]) -> None:
        """Override configuration with system environment variables."""
        for key in self.defaults.keys():
            env_val = os.getenv(key)
            if env_val is not None:
                config_data[key] = env_val

    def get_denoising_config(self, quality: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the full denoising configuration or for a specific quality level.
        
        Args:
            quality: Optional quality level (low, medium, high, ultra)
            
        Returns:
            Dictionary with denoising configuration
        """
        # Get the full denoising configuration
        denoising_config = self.features.get("denoising", DEFAULT_DENOISING_CONFIG)
        
        # If no quality specified, return the full config
        if quality is None:
            return denoising_config
            
        # Check if denoising should be applied for this quality
        quality = quality.lower()
        apply_to_quality = denoising_config.get("applyToQuality", {})
        
        # If this quality is specifically set to False, return disabled config
        if quality in apply_to_quality and not apply_to_quality.get(quality, True):
            return {**denoising_config, "enabled": False}
            
        return denoising_config

    def should_denoise_for_quality(self, quality: str) -> bool:
        """
        Determine if denoising should be applied for a specific quality level.
        
        Args:
            quality: Quality level (low, medium, high, ultra)
            
        Returns:
            True if denoising should be applied, False otherwise
        """
        # First check if denoising is globally enabled
        if not self.features.get("denoising", {}).get("enabled", False):
            return False
            
        # Check quality-specific settings
        quality_settings = self.features.get("denoising", {}).get("applyToQuality", {})
        
        # If specific setting exists, use it
        if quality.lower() in quality_settings:
            return quality_settings[quality.lower()]
            
        # Default recommendations based on quality
        if quality.lower() in ["low", "medium"]:
            return True
        elif quality.lower() in ["high", "ultra"]:
            return False
        
        # Default fallback
        return True
    
    def get_reconstruction_filter_for_denoising(self) -> str:
        """
        Get the recommended reconstruction filter type for denoising.
        
        Returns:
            Reconstruction filter type string (e.g., 'box', 'gaussian')
        """
        # Get from rendering.denoising settings first
        render_settings = self.rendering.get("denoising", {}).get("defaults", {})
        filter_type = render_settings.get("reconstruction_filter", "box")
        
        # If not set there, check features.denoising
        if not filter_type:
            filter_type = self.features.get("denoising", {}).get("reconstruction_filter", "box")
            
        return filter_type

    def prepare_scene_for_denoising(self, scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a scene dictionary for denoising by setting the appropriate reconstruction filter.
        
        Args:
            scene_dict: The Mitsuba scene dictionary
            
        Returns:
            Modified scene dictionary optimized for denoising
        """
        import copy
        scene_dict = copy.deepcopy(scene_dict)
        
        # Set box reconstruction filter which is optimal for the OptiX denoiser
        filter_type = self.get_reconstruction_filter_for_denoising()
        
        if 'sensor' in scene_dict and 'film' in scene_dict['sensor']:
            if 'rfilter' not in scene_dict['sensor']['film']:
                scene_dict['sensor']['film']['rfilter'] = {}
            
            scene_dict['sensor']['film']['rfilter']['type'] = filter_type
            logger.debug(f"Set {filter_type} reconstruction filter for optimal denoising")
            
        return scene_dict

    def get_aov_list_string(self) -> str:
        """
        Get a formatted AOV list string for use with the Mitsuba AOV integrator.
        
        Returns:
            String with format "name:type,name:type"
        """
        if not self.AOV_ENABLED:
            return ""
            
        aov_list = []
        for aov_type in self.AOV_TYPES:
            if aov_type == "albedo":
                aov_list.append("albedo:albedo")
            elif aov_type == "depth":
                aov_list.append("dd.y:depth")
            elif aov_type == "position":
                aov_list.append("p:position")
            elif aov_type == "uv":
                aov_list.append("uv:uv")
            elif aov_type == "geo_normal":
                aov_list.append("ng:geo_normal")
            elif aov_type == "sh_normal":
                aov_list.append("nn:sh_normal")
            elif aov_type == "prim_index":
                aov_list.append("pi:prim_index")
            elif aov_type == "shape_index":
                aov_list.append("si:shape_index")
                
        return ",".join(aov_list)

    def prepare_scene_dict_for_aovs(self, scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a scene dictionary for AOVs by adding the AOV integrator if enabled.
        
        Args:
            scene_dict: The Mitsuba scene dictionary
            
        Returns:
            Modified scene dictionary with AOV integrator if enabled
        """
        import copy
        scene_dict = copy.deepcopy(scene_dict)
        
        if not self.AOV_ENABLED:
            return scene_dict
            
        # Get the original integrator
        original_integrator = scene_dict.get("integrator", {"type": "path"})
        
        # Create the AOV integrator wrapper
        scene_dict["integrator"] = {
            "type": "aov",
            "aovs": self.get_aov_list_string(),
            "integrator": original_integrator
        }
        
        logger.debug(f"Added AOV integrator with types: {self.AOV_TYPES}")
        return scene_dict

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration as a dictionary.

    Returns:
        Dictionary with default configuration values.
    """
    mitsuba_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        "OUTPUT_FOLDER": os.path.join(mitsuba_root, "output"),
        "TEMPLATE_XML": os.path.join(mitsuba_root, "src/assets/templates/scene_template.xml"),
        "MITSUBA_VARIANT": "cuda_ad_rgb",  # Default variant
    }


# Instantiate the config singleton
config_class = Config()


def display_config() -> None:
    """Display the current configuration in a formatted table."""
    console = Console()
    
    # Basic configuration
    table = Table(title="Mitsuba Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Add basic configuration
    keys = [
        "OUTPUT_FOLDER", "EXR_FOLDER", "PNG_FOLDER", "VIDEO_FOLDER",
        "GIF_FOLDER", "SCENE_FOLDER", "MESH_FOLDER", "FRAMERATE",
        "FFMPEG_PATH", "ENABLE_GIF", "ENABLE_VIDEO", "ENABLE_EXR", "ENABLE_PNG",
        "TEMPLATE_XML"
    ]

    for key in keys:
        value = getattr(config_class, key, "N/A")
        table.add_row(key, str(value))

    console.print(table)
    
    # Denoising configuration
    denoising_table = Table(title="Denoising Configuration")
    denoising_table.add_column("Setting", style="cyan", no_wrap=True)
    denoising_table.add_column("Value", style="green")
    
    denoising_keys = [
        "DENOISING_ENABLED", "DENOISING_TYPE", "DENOISING_STRENGTH", 
        "DENOISING_USE_GUIDE_BUFFERS", "DENOISING_USE_TEMPORAL",
        "DENOISING_FALLBACK"
    ]
    
    for key in denoising_keys:
        value = getattr(config_class, key, "N/A")
        denoising_table.add_row(key, str(value))
    
    console.print(denoising_table)


if __name__ == "__main__":
    display_config()
