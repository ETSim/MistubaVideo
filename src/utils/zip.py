"""
ZIP file utilities for the Mitsuba application.
"""

import os
import zipfile
from typing import List, Optional
from src.utils.logger import RichLogger
from src.utils.folder import FolderUtils
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.utils.zip")

class ZipDecoder:
    """
    Utility class for extracting ZIP files.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def decode_zip_file(zip_path: str, dest_folder: str, subfolders: Optional[List[str]] = None) -> None:
        """
        Extracts the zip file into the destination folder.
        
        Args:
            zip_path: Path to the ZIP archive
            dest_folder: Directory to extract files into
            subfolders: Optional list of required subfolder names to be created inside dest_folder.
                If not provided, defaults to ['env', 'scenes', 'frames', 'meshes']
        
        Raises:
            FileNotFoundError: If the zip file is not found
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        # Ensure destination folder exists.
        os.makedirs(dest_folder, exist_ok=True)
        logger.debug(f"Destination folder ensured: {dest_folder}")
        
        # Set default subfolders if none provided.
        if subfolders is None:
            subfolders = ["env", "scenes", "frames", "meshes"]
        
        # Create required subfolders.
        for sub in subfolders:
            FolderUtils.ensure_folder(dest_folder, sub)
        
        # Extract the zip file.
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            logger.debug(f"Extracting {total_files} files from '{zip_path}'")
            zip_ref.extractall(dest_folder)
            
        logger.debug(f"Successfully extracted '{zip_path}' into '{dest_folder}'")
