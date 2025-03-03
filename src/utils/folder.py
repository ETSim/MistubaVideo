"""
Folder and file management utilities for the Mitsuba application.
"""

import os
import json
from typing import Any, Tuple
from src.utils.logger import RichLogger
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.utils.folder")

class FolderUtils:
    """
    Utility class for folder operations.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def ensure_folder(base: str, subfolder: str) -> str:
        """
        Ensure that the subfolder exists within the base folder.
        
        Args:
            base: Base directory path
            subfolder: Subfolder name to create
            
        Returns:
            Full path to the created/existing folder
        """
        folder = os.path.join(base, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.debug(f"Created folder: {folder}")
        else:
            logger.debug(f"Folder already exists: {folder}")
        return folder
    
    @staticmethod
    @timeit(log_level="debug")
    def setup_directories(base_folder: str, config) -> Tuple[str, str, str, str, str]:
        """
        Create all required output directories.
        
        Args:
            base_folder: Base folder for all outputs
            config: Configuration object with folder settings
            
        Returns:
            Tuple containing paths to (exr_folder, png_folder, video_folder, gif_folder, scene_folder)
        """
        exr_folder = os.path.join(base_folder, config.EXR_FOLDER)
        png_folder = os.path.join(base_folder, config.PNG_FOLDER)
        video_folder = os.path.join(base_folder, config.VIDEO_FOLDER)
        gif_folder = os.path.join(base_folder, config.GIF_FOLDER)
        scene_folder = os.path.join(base_folder, config.SCENE_FOLDER)
        
        folders = [exr_folder, png_folder, video_folder, gif_folder, scene_folder]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            
        logger.debug(f"All output directories ensured in {base_folder}")
        return exr_folder, png_folder, video_folder, gif_folder, scene_folder
    
    @staticmethod
    @timeit(log_level="debug")
    def serialize_value(val: Any) -> Any:
        """
        Recursively convert a value into a JSON-serializable format.
        
        Args:
            val: The value to serialize
            
        Returns:
            JSON-serializable form of the input
        """
        return json.loads(json.dumps(val))
