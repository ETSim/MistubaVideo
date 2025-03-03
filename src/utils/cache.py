"""
Cache management utilities for Mitsuba renderer.
"""

import os
import json
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.logger import RichLogger

logger = RichLogger.get_logger("mitsuba_app.utils.cache")

class CacheManager:
    """
    Manager for caching and retrieving rendered frames to avoid redundant rendering.
    Uses hash-based cache keys to identify identical rendering tasks.
    """
    
    def __init__(self, cache_dir: str, quality: str = "medium", 
                camera_views: Optional[List[str]] = None,
                render_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for storing cache files
            quality: Quality preset being used
            camera_views: List of camera views being rendered
            render_params: Render parameters for hashing
        """
        self.cache_dir = cache_dir
        self.quality = quality
        self.camera_views = camera_views or ["perspective"]
        self.render_params = render_params or {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create metadata subdirectory for cache info
        self.metadata_dir = os.path.join(self.cache_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Maps file hashes to cache information
        self.cache_map = self._load_cache_map()
        
        logger.debug(f"Initialized cache manager with {len(self.cache_map)} cached items")
    
    def _load_cache_map(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache map from disk."""
        cache_map_path = os.path.join(self.metadata_dir, "cache_map.json")
        
        if os.path.exists(cache_map_path):
            try:
                with open(cache_map_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache map: {e}")
        
        return {}
    
    def _save_cache_map(self) -> None:
        """Save the cache map to disk."""
        cache_map_path = os.path.join(self.metadata_dir, "cache_map.json")
        
        try:
            with open(cache_map_path, 'w') as f:
                json.dump(self.cache_map, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache map: {e}")
    
    def _compute_hash(self, obj_file: str, view: str) -> str:
        """
        Compute a hash key for a rendering task.
        
        Args:
            obj_file: Path to the OBJ file
            view: Camera view name
            
        Returns:
            Hash string that uniquely identifies this rendering task
        """
        # Get file modification time and size
        try:
            stats = os.stat(obj_file)
            file_mtime = stats.st_mtime
            file_size = stats.st_size
        except Exception:
            # If stat fails, use current time and zero size
            file_mtime = time.time()
            file_size = 0
        
        # Combine all parameters that affect rendering outcome
        hash_components = [
            os.path.basename(obj_file),
            str(file_mtime),
            str(file_size),
            view,
            self.quality,
            str(self.render_params.get("spp", 0)),
            str(self.render_params.get("max_depth", 0)),
            str(self.render_params.get("width", 0)),
            str(self.render_params.get("height", 0))
        ]
        
        # Create hash
        hash_input = "|".join(hash_components)
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        
        return hash_value
    
    def get_cached_path(self, obj_file: str, view: str) -> Optional[str]:
        """
        Check if a rendered image is available in cache.
        
        Args:
            obj_file: Path to OBJ file
            view: Camera view name
            
        Returns:
            Path to cached image if available, None otherwise
        """
        hash_key = self._compute_hash(obj_file, view)
        
        if hash_key in self.cache_map:
            cached_info = self.cache_map[hash_key]
            cached_path = cached_info.get("path")
            
            # Check if the cache file actually exists
            if cached_path and os.path.exists(cached_path):
                logger.debug(f"Cache hit for {os.path.basename(obj_file)}, view '{view}'")
                return cached_path
            else:
                # Remove invalid cache entry
                logger.debug(f"Removing invalid cache entry for {os.path.basename(obj_file)}, view '{view}'")
                del self.cache_map[hash_key]
                self._save_cache_map()
        
        logger.debug(f"Cache miss for {os.path.basename(obj_file)}, view '{view}'")
        return None
    
    def add_to_cache(self, obj_file: str, view: str, rendered_path: str) -> None:
        """
        Add a rendered image to the cache.
        
        Args:
            obj_file: Path to OBJ file
            view: Camera view name
            rendered_path: Path to the rendered image file
        """
        try:
            # Compute hash key
            hash_key = self._compute_hash(obj_file, view)
            
            # Create cache path
            obj_basename = os.path.basename(obj_file)
            view_safe = view.replace(" ", "_")
            cache_filename = f"cache_{self.quality}_{hash_key[:8]}_{obj_basename}_{view_safe}{Path(rendered_path).suffix}"
            cache_path = os.path.join(self.cache_dir, cache_filename)
            
            # Copy the file to cache
            shutil.copyfile(rendered_path, cache_path)
            
            # Update cache map
            self.cache_map[hash_key] = {
                "path": cache_path,
                "obj_file": obj_file,
                "view": view,
                "quality": self.quality,
                "params": self.render_params,
                "added": time.time()
            }
            
            # Save the updated cache map
            self._save_cache_map()
            
            logger.debug(f"Added {os.path.basename(obj_file)}, view '{view}' to cache")
            
        except Exception as e:
            logger.warning(f"Failed to add to cache: {e}")
    
    def clear_cache(self) -> int:
        """
        Clear all cached files.
        
        Returns:
            Number of files removed
        """
        count = 0
        
        # Remove all cache files
        for hash_key, cache_info in self.cache_map.items():
            path = cache_info.get("path")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {path}: {e}")
        
        # Clear cache map
        self.cache_map = {}
        self._save_cache_map()
        
        logger.info(f"Cleared {count} files from cache")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        oldest_time = float('inf')
        newest_time = 0
        quality_counts = {}
        valid_entries = 0
        
        for hash_key, cache_info in self.cache_map.items():
            path = cache_info.get("path")
            if path and os.path.exists(path):
                valid_entries += 1
                
                # Get file size
                try:
                    size = os.path.getsize(path)
                    total_size += size
                except:
                    pass
                
                # Track timestamps
                added_time = cache_info.get("added", 0)
                oldest_time = min(oldest_time, added_time)
                newest_time = max(newest_time, added_time)
                
                # Track quality distribution
                quality = cache_info.get("quality", "unknown")
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        # Format timestamps
        if oldest_time == float('inf'):
            oldest_time = 0
        
        return {
            "total_entries": len(self.cache_map),
            "valid_entries": valid_entries,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_entry": datetime.datetime.fromtimestamp(oldest_time).isoformat() if oldest_time > 0 else "N/A",
            "newest_entry": datetime.datetime.fromtimestamp(newest_time).isoformat() if newest_time > 0 else "N/A",
            "quality_distribution": quality_counts
        }
