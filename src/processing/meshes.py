import os
import re
import shutil
from pathlib import Path
from typing import List, Optional
from src.utils.logger import RichLogger

logger = RichLogger.get_logger("mitsuba_app.processing.obj")

class ObjProcessor:
    """
    Utility class for processing OBJ files.
    """
    
    @staticmethod
    def get_matching_obj_files(folder: str, regex: str) -> List[str]:
        """
        Find OBJ files in a folder that match a regex pattern.
        
        Args:
            folder: Path to folder containing OBJ files
            regex: Regular expression pattern to match filenames
            
        Returns:
            List of matching OBJ file paths
        """
        # Convert to Path object to handle both Windows and POSIX paths
        folder_path = Path(folder)
        
        if not folder_path.is_dir():
            # Try with current working directory if it's a relative path
            cwd_folder_path = Path.cwd() / folder
            if cwd_folder_path.is_dir():
                folder_path = cwd_folder_path
                logger.debug(f"Using resolved path: {folder_path}")
            else:
                # Check for similar paths for better error messages
                try:
                    parent_dir = Path.cwd()
                    similar_folders = [d.name for d in parent_dir.iterdir() if d.is_dir()]
                    if similar_folders:
                        logger.debug(f"Available folders in current directory: {', '.join(similar_folders)}")
                    
                    raise FileNotFoundError(f"Folder not found: {folder}")
                except Exception as e:
                    raise FileNotFoundError(f"Folder not found: {folder}")
        
        # Convert back to string for compatibility with existing code
        folder = str(folder_path)
        
        pattern = re.compile(regex)
        # Use fullmatch to ensure the entire filename fits the pattern.
        matched_files = [os.path.join(folder, f) for f in os.listdir(folder) if pattern.fullmatch(f)]
        
        if not matched_files:
            logger.warning(f"No files match regex '{regex}' in folder '{folder}'")
            available_files = os.listdir(folder)[:10]  # Show first 10 files
            logger.debug(f"Available files in folder (first 10): {available_files}")
            raise ValueError(f"No files match regex '{regex}' in folder '{folder}'")
        
        # Sort files numerically by extracting the number from the filename.
        matched_files = sorted(matched_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        return matched_files
    
    @staticmethod
    def copy_to_mesh_folder(obj_file: str, mesh_folder: str) -> str:
        """
        Copy an OBJ file to the mesh folder if it doesn't already exist there.
        
        Args:
            obj_file: Path to OBJ file
            mesh_folder: Target folder for meshes
            
        Returns:
            Path to the copied OBJ file
        """
        base_name = os.path.basename(obj_file)
        dest_path = os.path.join(mesh_folder, base_name)
        
        if not os.path.exists(dest_path):
            shutil.copy(obj_file, dest_path)
            logger.debug(f"Copied OBJ '{obj_file}' to mesh folder: {dest_path}")
        
        return dest_path
