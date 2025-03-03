"""
Utilities for managing default rendering settings from configuration.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import mitsuba as mi

from src.utils.logger import RichLogger
from src.config.config import config_class as config

logger = RichLogger.get_logger("mitsuba_app.utils.render_settings")

class RenderSettings:
    """
    Utility class for creating rendering components from configuration.
    """
    
    @staticmethod
    def create_default_integrator(override_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create an integrator based on configuration settings.
        
        Args:
            override_params: Optional dictionary of parameters to override config defaults
            
        Returns:
            Configured integrator instance
        """
        # Get default settings from config
        try:
            integrator_config = config.rendering["integrator"]
            integrator_type = integrator_config.get("type", "path")
            params = {
                "max_depth": integrator_config.get("max_depth", 8),
                "rr_depth": integrator_config.get("rr_depth", 5),
                "hide_emitters": integrator_config.get("hide_emitters", False)
            }
            
            # Override with any provided params
            if override_params:
                params.update(override_params)
                
            # Create the integrator
            logger.debug(f"Creating integrator of type '{integrator_type}' with params: {params}")
            return mi.load_dict({
                "type": integrator_type,
                **params
            })
        
        except (AttributeError, KeyError) as e:
            logger.warning(f"Failed to create integrator from config: {e}. Using default path tracer.")
            return mi.load_dict({
                "type": "path",
                "max_depth": 8
            })
    
    @staticmethod
    def create_default_film(width: Optional[int] = None, 
                            height: Optional[int] = None,
                            override_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a film based on configuration settings.
        
        Args:
            width: Optional width to override config value
            height: Optional height to override config value
            override_params: Optional dictionary of parameters to override config defaults
            
        Returns:
            Configured film instance
        """
        # Get default settings from config
        try:
            film_config = config.rendering["film"]
            film_type = film_config.get("type", "hdrfilm")
            
            # Get filter settings
            rfilter_config = film_config.get("rfilter", {})
            rfilter_type = rfilter_config.get("type", "gaussian")
            rfilter_params = {
                "radius": rfilter_config.get("radius", 2.0)
            }
            
            # Build film params
            params = {
                "width": width or film_config.get("width", 1920),
                "height": height or film_config.get("height", 1080),
                "file_format": film_config.get("file_format", "openexr"),
                "pixel_format": film_config.get("pixel_format", "rgb"),
                "component_format": film_config.get("component_format", "float16"),
                "rfilter": {
                    "type": rfilter_type,
                    **rfilter_params
                }
            }
            
            # Override with any provided params
            if override_params:
                # Handle nested rfilter params
                if "rfilter" in override_params:
                    params["rfilter"].update(override_params.pop("rfilter", {}))
                params.update(override_params)
                
            # Create the film
            logger.debug(f"Creating film of type '{film_type}' with params: {params}")
            return mi.load_dict({
                "type": film_type,
                **params
            })
            
        except (AttributeError, KeyError) as e:
            logger.warning(f"Failed to create film from config: {e}. Using default hdrfilm.")
            return mi.load_dict({
                "type": "hdrfilm",
                "width": width or 1920,
                "height": height or 1080,
                "rfilter": {
                    "type": "gaussian"
                }
            })
    
    @staticmethod
    def create_default_sensor(film: Optional[Any] = None,
                             sampler: Optional[Any] = None,
                             override_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a sensor based on configuration settings.
        
        Args:
            film: Optional film to attach to the sensor
            sampler: Optional sampler to attach to the sensor
            override_params: Optional dictionary of parameters to override config defaults
            
        Returns:
            Configured sensor instance
        """
        # Get default settings from config
        try:
            sensor_config = config.rendering["sensor"]
            sensor_type = sensor_config.get("type", "perspective")
            
            # Build sensor params
            params = {}
            
            # Add type-specific parameters
            if sensor_type in ["perspective", "orthographic"]:
                params.update({
                    "near_clip": sensor_config.get("near_clip", 0.01),
                    "far_clip": sensor_config.get("far_clip", 10000.0),
                })
            
            if sensor_type in ["perspective", "thinlens"]:
                # Only include one of fov or focal_length
                if "fov" in sensor_config:
                    params["fov"] = sensor_config.get("fov", 45.0)
                    params["fov_axis"] = sensor_config.get("fov_axis", "x")
                else:
                    params["focal_length"] = sensor_config.get("focal_length", "50mm")
                    
            if sensor_type == "thinlens":
                params.update({
                    "aperture_radius": sensor_config.get("aperture_radius", 0.1),
                    "focus_distance": sensor_config.get("focus_distance", 5.0),
                })
            
            # Override with any provided params
            if override_params:
                params.update(override_params)
                
            # Add film and sampler if provided
            if film:
                params["film"] = film
            if sampler:
                params["sampler"] = sampler
            
            # Create the sensor
            logger.debug(f"Creating sensor of type '{sensor_type}' with params: {params}")
            return mi.load_dict({
                "type": sensor_type,
                **params
            })
            
        except (AttributeError, KeyError) as e:
            logger.warning(f"Failed to create sensor from config: {e}. Using default perspective camera.")
            
            # Build a minimal dictionary
            sensor_dict = {
                "type": "perspective",
                "fov": 45.0
            }
            
            # Add film and sampler if provided
            if film:
                sensor_dict["film"] = film
            if sampler:
                sensor_dict["sampler"] = sampler
                
            return mi.load_dict(sensor_dict)
