"""
Utilities for processing AOVs (Arbitrary Output Variables) in Mitsuba renders.

This module provides functionality to extract and manipulate AOVs from multichannel EXR images.
"""

import os
from typing import Dict, List, Optional, Union, Any
import mitsuba as mi
import numpy as np

from src.utils.logger import RichLogger
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.processing.aov")

class AOVProcessor:
    """
    Class for processing AOV (Arbitrary Output Variables) from rendered images.
    """
    
    # Known AOV names and their descriptions
    AOV_TYPES = {
        "albedo": "Surface diffuse reflectance",
        "depth": "Distance from camera",
        "position": "World space coordinates",
        "uv": "Surface UV coordinates",
        "geo_normal": "Geometric normal",
        "sh_normal": "Shading normal",
        "prim_index": "Primitive index",
        "shape_index": "Shape index"
    }
    
    @classmethod
    @timeit(log_level="debug")
    def extract_aovs_from_exr(cls, 
                              input_path: str, 
                              output_folder: str,
                              aov_list: Optional[List[str]] = None,
                              output_format: str = "exr") -> Dict[str, str]:
        """
        Extract AOV layers from a multichannel EXR file to separate files.
        
        Args:
            input_path: Path to the input EXR file with AOVs
            output_folder: Directory to save extracted AOVs
            aov_list: List of AOVs to extract (None = extract all found)
            output_format: Format for output files (exr or png)
            
        Returns:
            Dictionary mapping AOV names to output file paths
        """
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return {}
            
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            # Load the multichannel EXR
            bitmap = mi.Bitmap(input_path)
            logger.debug(f"Loaded EXR with {bitmap.channel_count()} channels")
            
            # Extract AOVs from the loaded bitmap
            return cls.extract_aovs_from_bitmap(
                bitmap=bitmap,
                output_folder=output_folder,
                base_name=os.path.splitext(os.path.basename(input_path))[0],
                aov_list=aov_list,
                output_format=output_format
            )
            
        except Exception as e:
            logger.error(f"Failed to extract AOVs from {input_path}: {e}")
            return {}
    
    @classmethod
    def extract_aovs_from_bitmap(cls, 
                                bitmap: mi.Bitmap, 
                                output_folder: str,
                                base_name: str, 
                                aov_list: Optional[List[str]] = None,
                                output_format: str = "exr") -> Dict[str, str]:
        """
        Extract AOV layers from a multichannel bitmap to separate files.
        
        Args:
            bitmap: Source multichannel bitmap with AOVs
            output_folder: Directory to save extracted AOVs
            base_name: Base name for output files
            aov_list: List of AOVs to extract (None = extract all found)
            output_format: Format for output files (exr or png)
            
        Returns:
            Dictionary mapping AOV names to output file paths
        """
        result = {}
        
        try:
            # Split the bitmap into channels
            channels = dict(bitmap.split())
            logger.debug(f"Found channels: {list(channels.keys())}")
            
            # Identify AOV channels
            aov_channels = {}
            for channel_name, channel_bitmap in channels.items():
                # Skip the root channel
                if channel_name == "<root>":
                    continue
                    
                # Map standard AOV channel names to more readable names
                if channel_name == "albedo":
                    aov_name = "albedo"
                elif channel_name == "dd.y" or channel_name == "depth":
                    aov_name = "depth"
                elif channel_name == "p" or channel_name == "position":
                    aov_name = "position"
                elif channel_name == "uv":
                    aov_name = "uv"
                elif channel_name == "ng" or channel_name == "geo_normal":
                    aov_name = "geo_normal"
                elif channel_name == "nn" or channel_name == "sh_normal":
                    aov_name = "sh_normal"
                elif channel_name == "pi" or channel_name == "prim_index":
                    aov_name = "prim_index"
                elif channel_name == "si" or channel_name == "shape_index":
                    aov_name = "shape_index"
                else:
                    # Keep original name for unknown channels
                    aov_name = channel_name
                
                # If specific AOVs requested, filter to only those
                if aov_list is None or aov_name in aov_list:
                    aov_channels[aov_name] = channel_bitmap
            
            # Save each AOV to a file
            for aov_name, aov_bitmap in aov_channels.items():
                # Create file path
                if output_format.lower() == "exr":
                    file_path = os.path.join(output_folder, f"{base_name}_{aov_name}.exr")
                    aov_bitmap.write(file_path)
                elif output_format.lower() == "png":
                    file_path = os.path.join(output_folder, f"{base_name}_{aov_name}.png")
                    
                    # For depth maps, apply normalization/scaling for better visualization
                    if aov_name == "depth":
                        # Convert to numpy for normalization
                        depth_array = np.array(aov_bitmap)
                        # Normalize to 0-1 range (handle case where all values are the same)
                        depth_min = np.min(depth_array)
                        depth_max = np.max(depth_array)
                        if depth_min == depth_max:
                            normalized = np.zeros_like(depth_array)
                        else:
                            normalized = (depth_array - depth_min) / (depth_max - depth_min)
                        # Create new bitmap from normalized data
                        normalized_bitmap = mi.Bitmap(normalized)
                        normalized_bitmap.write(file_path)
                    # For normal maps, ensure they're in a good range for visualization
                    elif aov_name in ["sh_normal", "geo_normal"]:
                        # Convert from [-1,1] to [0,1] range for better PNG display
                        normal_array = np.array(aov_bitmap)
                        # Map from [-1,1] to [0,1]
                        normalized = (normal_array + 1.0) * 0.5
                        # Create new bitmap from mapped data
                        normal_bitmap = mi.Bitmap(normalized)
                        normal_bitmap.write(file_path)
                    else:
                        # For other AOV types, just write directly
                        aov_bitmap.write(file_path)
                
                # Record the output file path
                result[aov_name] = file_path
                logger.debug(f"Saved {aov_name} AOV to {file_path}")
                
            # Return dictionary of extracted AOVs
            logger.info(f"Extracted {len(result)} AOVs: {', '.join(result.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract AOVs from bitmap: {e}")
            return result
            
    @classmethod
    def get_available_aov_types(cls) -> List[str]:
        """
        Get a list of all available AOV types.
        
        Returns:
            List of AOV type names
        """
        return list(cls.AOV_TYPES.keys())

    @classmethod
    def is_valid_aov_type(cls, aov_type: str) -> bool:
        """
        Check if an AOV type is valid.
        
        Args:
            aov_type: AOV type name to check
            
        Returns:
            True if valid, False otherwise
        """
        return aov_type in cls.AOV_TYPES

    @classmethod
    def batch_extract_aovs(cls,
                          input_folder: str,
                          output_folder: str,
                          pattern: str = "*.exr",
                          recursive: bool = False,
                          aov_list: Optional[List[str]] = None,
                          output_format: str = "exr") -> int:
        """
        Batch extract AOVs from multiple EXR files.
        
        Args:
            input_folder: Folder containing EXR files
            output_folder: Output folder for extracted AOVs
            pattern: File pattern to match (default: "*.exr")
            recursive: Whether to search subdirectories
            aov_list: List of AOV types to extract
            output_format: Output format (exr or png)
            
        Returns:
            Number of files processed
        """
        import glob
        
        # Find matching files
        if recursive:
            files = []
            for root, _, _ in os.walk(input_folder):
                matches = glob.glob(os.path.join(root, pattern))
                files.extend(matches)
        else:
            files = glob.glob(os.path.join(input_folder, pattern))
        
        # Process each file
        count = 0
        for file_path in files:
            # Create output subfolder based on file name
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_output_folder = os.path.join(output_folder, file_name)
            
            # Extract AOVs
            result = cls.extract_aovs_from_exr(
                input_path=file_path,
                output_folder=file_output_folder,
                aov_list=aov_list,
                output_format=output_format
            )
            
            if result:
                count += 1
        
        return count