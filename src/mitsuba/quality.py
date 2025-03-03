"""
Quality management for Mitsuba renderer.

This module defines quality presets and manages quality settings conversion.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
import math
from enum import Enum

from src.config.config import config_class as config
from src.utils.logger import RichLogger

logger = RichLogger.get_logger("mitsuba_app.quality")

# Maximum number of samples for most Mitsuba variants (2^32)
MAX_SAMPLE_COUNT = 4294967296  # 2^32

class QualityPreset(Enum):
    """Enumeration of rendering quality presets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CUSTOM = "custom"


class QualityManager:
    """Manager for quality presets and settings."""
    
    @staticmethod
    def get_default_preset() -> str:
        """
        Get default quality preset from config.
        
        Returns:
            Default quality preset name
        """
        return config.rendering.get("quality", {}).get("preset", "medium")
    
    @staticmethod
    def get_preset_settings(preset: str) -> Dict[str, Any]:
        """
        Get settings for a specific quality preset.
        
        Args:
            preset: Quality preset name (low, medium, high, ultra)
            
        Returns:
            Dictionary containing quality settings for the preset
        """
        if preset.lower() not in ["low", "medium", "high", "ultra"]:
            logger.warning(f"Unknown quality preset '{preset}', using default")
            preset = QualityManager.get_default_preset()
        
        # Get preset from config
        presets = config.rendering.get("quality", {}).get("presets", {})
        if preset.lower() in presets:
            return presets[preset.lower()]
        
        # Default presets if not in config
        default_presets = {
            "low": {"spp": 64, "maxDepth": 3},
            "medium": {"spp": 512, "maxDepth": 8},
            "high": {"spp": 2048, "maxDepth": 16},
            "ultra": {"spp": 8192, "maxDepth": 32}
        }
        
        return default_presets.get(preset.lower(), default_presets["medium"])
    
    @staticmethod
    def safe_sample_count(spp: int, width: int, height: int, max_samples: int = MAX_SAMPLE_COUNT) -> int:
        """
        Ensure the total sample count doesn't exceed the maximum limit.
        
        Args:
            spp: Samples per pixel
            width: Image width
            height: Image height
            max_samples: Maximum total samples allowed
            
        Returns:
            Adjusted SPP value that keeps total samples under the limit
        """
        # Calculate total samples
        total_samples = spp * width * height
        
        # If within limit, return original SPP
        if total_samples <= max_samples:
            return spp
        
        # Calculate maximum safe SPP
        max_spp = int(max_samples / (width * height))
        
        # Apply a safety margin of 10% to avoid edge cases
        safe_spp = int(max_spp * 0.9)
        
        # Ensure we have at least 1 sample per pixel
        safe_spp = max(1, safe_spp)
        
        # Warn about the adjustment
        logger.warning(f"Reducing SPP from {spp} to {safe_spp} to stay under the 2^32 sample limit")
        logger.info(f"Original would use {total_samples:,} samples, adjusted will use {safe_spp * width * height:,} samples")
        
        return safe_spp
    
    @staticmethod
    def apply_preset_to_params(preset: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply quality preset settings to render parameters.
        
        Args:
            preset: Quality preset name
            params: Existing render parameters
            
        Returns:
            Updated render parameters with quality settings applied
        """
        settings = QualityManager.get_preset_settings(preset)
        
        # Copy existing params
        result = params.copy()
        
        # Apply SPP if not already set
        if "spp" not in result:
            result["spp"] = settings.get("spp", 512)
            
        # Apply max depth if not already set
        if "max_depth" not in result:
            result["max_depth"] = settings.get("maxDepth", settings.get("max_depth", 8))
        
        # Check if width and height are set
        if "width" in result and "height" in result:
            # Ensure SPP doesn't exceed maximum sample count
            original_spp = result["spp"]
            result["spp"] = QualityManager.safe_sample_count(
                result["spp"], result["width"], result["height"]
            )
            
            # Log if SPP was adjusted
            if result["spp"] != original_spp:
                logger.info(f"Quality '{preset}': SPP adjusted from {original_spp} to {result['spp']} to stay within sample limits")
        
        return result

    @staticmethod
    def get_quality_settings(preset: str, width: int, height: int) -> Dict[str, Any]:
        """
        Get quality settings with safe SPP adjustments based on image dimensions.
        
        Args:
            preset: Quality preset name
            width: Image width
            height: Image height
            
        Returns:
            Dictionary with quality settings
        """
        settings = QualityManager.get_preset_settings(preset)
        
        # Copy settings to avoid modifying the original
        result = settings.copy()
        
        # Ensure SPP doesn't exceed maximum sample count
        original_spp = result.get("spp", 512)
        result["spp"] = QualityManager.safe_sample_count(original_spp, width, height)
        
        # Log if SPP was adjusted
        if result["spp"] != original_spp:
            logger.info(f"Quality '{preset}': SPP adjusted from {original_spp} to {result['spp']} to stay within sample limits")
        
        return result
    
    @staticmethod
    def update_custom_preset(spp: Optional[int] = None, max_depth: Optional[int] = None) -> None:
        """
        Update the custom quality preset with new values.
        
        Args:
            spp: Samples per pixel
            max_depth: Maximum ray depth
        """
        if not hasattr(config.rendering, "quality"):
            config.rendering["quality"] = {}
            
        if "presets" not in config.rendering["quality"]:
            config.rendering["quality"]["presets"] = {}
            
        if "custom" not in config.rendering["quality"]["presets"]:
            config.rendering["quality"]["presets"]["custom"] = {}
            
        custom_preset = config.rendering["quality"]["presets"]["custom"]
        
        if spp is not None:
            custom_preset["spp"] = spp
            
        if max_depth is not None:
            custom_preset["maxDepth"] = max_depth
    
    @staticmethod
    def get_all_quality_presets() -> List[str]:
        """
        Get a list of all available quality presets.
        
        Returns:
            List of quality preset names
        """
        return [preset.value for preset in QualityPreset]
