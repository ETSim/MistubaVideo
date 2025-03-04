"""
Camera positioning utilities for Mitsuba scenes.
"""

import numpy as np
import difflib
from typing import Dict, Tuple, List, Any, Optional
from enum import Enum

class CameraView(Enum):
    """Enumeration of standard camera views."""
    FRONT = "front"
    RIGHT = "right"
    LEFT = "left"
    BACK = "back"
    TOP = "top"
    BOTTOM = "bottom"
    PERSPECTIVE = "perspective"  # Default 3/4 view


class CameraUtils:
    """Utility class for camera positioning in Mitsuba scenes."""
    
    @staticmethod
    def get_camera_transform(view: str, distance: float = 4.0) -> Dict[str, Any]:
        """
        Generate camera transform parameters for a given view.
        
        Args:
            view: The camera view (front, right, left, back, top, bottom, perspective)
            distance: Distance from origin
            
        Returns:
            Dictionary with camera transform parameters
        """
        try:
            # Convert string to enum if it's a string
            if isinstance(view, str):
                view = CameraView(view.lower())
        except ValueError:
            # Fall back to perspective view if the string doesn't match any enum value
            view = CameraView.PERSPECTIVE
            
        if view == CameraView.FRONT:
            origin = [0, 0, distance]
            target = [0, 0, 0]
            up = [0, 1, 0]
        elif view == CameraView.RIGHT:
            origin = [distance, 0, 0]
            target = [0, 0, 0]
            up = [0, 1, 0]
        elif view == CameraView.LEFT:
            origin = [-distance, 0, 0]
            target = [0, 0, 0]
            up = [0, 1, 0]
        elif view == CameraView.BACK:
            origin = [0, 0, -distance]
            target = [0, 0, 0]
            up = [0, 1, 0]
        elif view == CameraView.TOP:
            origin = [0, distance, 0]
            target = [0, 0, 0]
            up = [0, 0, -1]
        elif view == CameraView.BOTTOM:
            origin = [0, -distance, 0]
            target = [0, 0, 0]
            up = [0, 0, 1]
        else:  # Default to perspective 3/4 view
            origin = [distance, distance, distance]
            target = [0, 0, 0]
            up = [0, 1, 0]
            
        return {
            "origin": origin,
            "target": target,
            "up": up
        }
    
    @staticmethod
    def format_transform_for_xml(transform: Dict[str, Any]) -> str:
        """
        Format a camera transform as XML for Mitsuba scene.
        
        Args:
            transform: Camera transform dictionary
            
        Returns:
            XML string for the transform
        """
        origin_str = ", ".join(map(str, transform["origin"]))
        target_str = ", ".join(map(str, transform["target"]))
        up_str = ", ".join(map(str, transform["up"]))
        
        return f'<lookat origin="{origin_str}" target="{target_str}" up="{up_str}"/>'
    
    @staticmethod
    def get_all_standard_views(distance: float = 4.0) -> Dict[str, Dict[str, Any]]:
        """
        Get camera transforms for all standard views.
        
        Args:
            distance: Camera distance from origin
            
        Returns:
            Dictionary of view name to camera transform
        """
        return {
            view.value: CameraUtils.get_camera_transform(view, distance)
            for view in CameraView
        }
    
    @staticmethod
    def get_views_from_preset(preset_name: str) -> List[str]:
        """
        Get a list of camera views based on a named preset.
        
        Args:
            preset_name: The name of the preset ("all", "orthographic", "technical", etc.)
            
        Returns:
            List of camera view names
        """
        try:
            from src.config.config import config_class as config
            
            # Try to get from configuration
            presets = config.features["views"]["presets"]
            if preset_name in presets:
                return presets[preset_name]
        except (AttributeError, KeyError, ImportError):
            pass
        
        # Fallback presets if not in configuration
        presets = {
            "all": [view.value for view in CameraView],
            "orthographic": ["front", "right", "top"],
            "technical": ["front", "right", "top", "bottom"],
            "360": ["front", "right", "back", "left"]
        }
        
        return presets.get(preset_name, ["perspective"])
    
    @staticmethod
    def get_default_views() -> List[str]:
        """
        Get the default camera views from configuration.
        
        Returns:
            List of default camera view names
        """
        try:
            from src.config.config import config_class as config
            
            # Check if multi-view is enabled by default
            if config.features["enable"].get("multiView", False):
                # Return configured default views
                return config.features["views"].get("defaultViews", ["perspective"])
            else:
                # Multi-view disabled, return just the default view
                return ["perspective"]
        except (AttributeError, KeyError, ImportError):
            # Fallback to perspective view
            return ["perspective"]
    
    @staticmethod
    def get_camera_settings_for_view(view_name: str, distance: float = 4.0) -> Dict[str, str]:
        """
        Get camera settings for a named view.
        
        Args:
            view_name: Name of the view (perspective, top, etc.)
            distance: Camera distance from target
            
        Returns:
            Dictionary with camera settings for XML
        """
        camera = CameraViewManager.get_view(view_name)
        
        # Scale distance if needed
        if distance != 4.0:
            camera = CameraViewManager.scale_camera_distance(camera, distance)
            
        return camera


class CameraViewManager:
    """Manager class for handling camera view selection and validation."""
    
    @staticmethod
    def get_view(view_name: str) -> Dict[str, str]:
        """
        Get camera settings for a specific named view.
        
        Args:
            view_name: Name of the camera view (e.g., 'front', 'top', 'perspective')
            
        Returns:
            Dictionary with camera transform settings for XML
        """
        from src.utils.logger import RichLogger
        logger = RichLogger.get_logger("mitsuba_app.camera.viewmanager")
        
        # Convert to lowercase for case-insensitive matching
        view_name = view_name.lower()
        logger.debug(f"Getting camera settings for view: '{view_name}'")
        
        # Validate the view name
        valid_views = [v.value for v in CameraView]
        if view_name not in valid_views:
            logger.warning(f"Invalid view name '{view_name}', falling back to perspective view")
            logger.debug(f"Valid view names: {', '.join(valid_views)}")
            view_name = "perspective"
        
        # Get camera transform from CameraUtils
        transform = CameraUtils.get_camera_transform(view_name)
        logger.debug(f"Retrieved camera transform for '{view_name}': {transform}")
        
        # Convert to the format needed for XML
        result = {
            "origin": ", ".join(map(str, transform["origin"])),
            "target": ", ".join(map(str, transform["target"])),
            "up": ", ".join(map(str, transform["up"]))
        }
        logger.debug(f"Formatted for XML: {result}")
        
        return result
    
    @staticmethod
    def scale_camera_distance(camera_settings: Dict[str, str], distance_scale: float) -> Dict[str, str]:
        """
        Scale the distance of camera position from target.
        
        Args:
            camera_settings: Camera settings dictionary with origin, target, up
            distance_scale: Scale factor for distance
            
        Returns:
            Updated camera settings with scaled distance
        """
        # The original settings have strings like "0, 0, 4"
        # Parse the origin coordinates
        origin_coords = [float(x.strip()) for x in camera_settings["origin"].split(",")]
        target_coords = [float(x.strip()) for x in camera_settings["target"].split(",")]
        
        # Calculate direction vector
        direction = [o - t for o, t in zip(origin_coords, target_coords)]
        
        # Calculate distance
        current_distance = np.sqrt(sum(d*d for d in direction))
        
        if current_distance > 0:
            # Normalize direction
            direction = [d / current_distance for d in direction]
            
            # Scale to new distance
            new_distance = distance_scale
            scaled_direction = [d * new_distance for d in direction]
            
            # Calculate new origin
            new_origin = [t + d for t, d in zip(target_coords, scaled_direction)]
            
            # Return updated settings
            return {
                "origin": ", ".join(map(str, new_origin)),
                "target": camera_settings["target"],
                "up": camera_settings["up"]
            }
        
        # If we can't calculate distance (e.g., origin = target), return unmodified
        return camera_settings
        
    @staticmethod
    def get_camera_views(all_views: bool = False, 
                        view_preset: Optional[str] = None,
                        views: Optional[List[str]] = None,
                        multi_view: bool = False) -> List[str]:
        """
        Resolve camera views based on priorities:
        1. all_views flag
        2. view_preset
        3. specific views
        4. default (perspective only)
        
        Args:
            all_views: If True, return all standard views
            view_preset: Name of view preset to use
            views: List of specific view names
            multi_view: If True, ensure multiple views are returned
            
        Returns:
            List of camera view names
        """
        # Check for --all-views flag
        if all_views:
            camera_views = [view.value for view in CameraView]
            return camera_views
            
        # Check for view preset
        if view_preset:
            camera_views = CameraUtils.get_views_from_preset(view_preset)
            return camera_views
            
        # Check for specific views
        if views and len(views) > 0:
            CameraViewManager.validate_camera_views(views)
            return views
        
        # For multi-view mode, return a good default set if no views specified
        if multi_view:
            return ["front", "right", "perspective", "top"]
        
        # Default to perspective view only
        return ["perspective"]
    
    @staticmethod
    def validate_camera_views(views: List[str]) -> None:
        """
        Validate that all camera views are valid, raising ValueError with
        helpful suggestions if not.
        
        Args:
            views: List of camera view names to validate
            
        Raises:
            ValueError: If any view name is invalid
        """
        valid_views = [view.value for view in CameraView]
        
        for view in views:
            if view not in valid_views:
                closest_match = difflib.get_close_matches(view, valid_views, n=1)
                suggestion = f" Did you mean '{closest_match[0]}'?" if closest_match else ""
                error_msg = f"Invalid camera view: '{view}'.{suggestion} Valid options are: {', '.join(valid_views)}"
                raise ValueError(error_msg)
