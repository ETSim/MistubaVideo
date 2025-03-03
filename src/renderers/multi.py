import os
import re
import time
import glob
import concurrent.futures
import threading
import signal
from tqdm import tqdm
from datetime import timedelta, datetime
from pathlib import Path
import shutil
from typing import Dict, Any, List, Tuple, Optional
import json
import drjit as dr
import numpy as np

from .base import BaseRenderer
from src.processing import ObjProcessor, SceneProcessor
from src.config.config import config_class as config
from src.utils.logger import RichLogger
from src.utils.timing import timeit
from src.config.constants import OUTPUT_FOLDER
from src.mitsuba.camera import CameraUtils, CameraView, CameraViewManager
from src.mitsuba.render_settings import RenderSettings
from src.utils.cache import CacheManager
from src.mitsuba.denoising import DenoiserManager, OPTIX_AVAILABLE

logger = RichLogger.get_logger("mitsuba_app.renderers.multi")

class MultiObjSceneRenderer(BaseRenderer):
    """
    Renderer for multi-object scenes from OBJ files.
    """
    
    def __init__(self, obj_files: List[str], scene_prefix: str = "multi_obj", 
                 base_folder: str = OUTPUT_FOLDER, threaded: bool = False,
                 camera_views: Optional[List[str]] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        Initialize the multi-object scene renderer.
        
        Args:
            obj_files: List of OBJ files to render
            scene_prefix: Prefix for output files
            base_folder: Base directory for all output files
            threaded: Whether to use threading for rendering
            camera_views: List of camera views to render (if None, uses only default view)
            cache_manager: Optional cache manager for frame caching
        """
        super().__init__(scene_prefix, base_folder)
        self.obj_files = obj_files
        self.threaded = threaded
        self.scene_xml_paths = []  # Store prepared XML paths
        self.render_timings = []  # Track render time for each frame
        self.camera_views = camera_views or ["perspective"]  # Default to perspective view if not specified
        self.cache_manager = cache_manager  # Store the cache manager
        
        # Track cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def prepare_scene_xmls(self):
        """
        Prepare all scene XML files before rendering.
        
        Returns:
            List of prepared XML file paths
        """
        logger.debug(f"Preparing {len(self.obj_files)} scene XML files with {len(self.camera_views)} camera views...")
        
        scene_xml_paths = []
        for i, obj_file in enumerate(tqdm(self.obj_files, desc="Preparing scene XMLs")):
            try:
                scene_xml_dict = self._prepare_scene_xml(obj_file, i)
                scene_xml_paths.append(scene_xml_dict)
                
                # Copy OBJ file to mesh folder
                try:
                    ObjProcessor.copy_to_mesh_folder(obj_file, self.mesh_folder)
                except Exception as e:
                    logger.error(f"Error copying OBJ '{obj_file}' to mesh folder: {e}")
                
            except Exception as e:
                logger.error(f"Error preparing scene XML for frame {i}: {e}")
                scene_xml_paths.append(None)  # Add None to maintain index consistency
        
        self.scene_xml_paths = scene_xml_paths
        logger.debug(f"Prepared {len([p for p in scene_xml_paths if p is not None])} scene XML files")
        return scene_xml_paths

    @timeit(log_level="debug")
    def render_frame(self, index: int, scene_xml_paths: Dict[str, str]) -> bool:
        """
        Render a single frame from prepared XML files with timing and nested progress display.
        
        Args:
            index: Frame index
            scene_xml_paths: Dictionary of view name to scene XML file path
            
        Returns:
            True if the rendering was successful, False otherwise
        """
        if scene_xml_paths is None or not scene_xml_paths:
            logger.error(f"No scene XML prepared for frame {index}")
            return False
            
        frame_start = time.time()
        obj_file = self.obj_files[index] if index < len(self.obj_files) else f"{index}.obj"
        obj_name = os.path.basename(obj_file)
        
        total_views = len(scene_xml_paths)
        logger.debug(f"Rendering frame {index} with {total_views} camera views")
        
        # Track detailed timing data for this frame if detailed logging is enabled
        detailed_timing = config.debug.get("detailedLogs", False)
        frame_timing_data = {
            "frame_index": index,
            "obj_file": obj_name,
            "start_time": datetime.now().isoformat(),
            "camera_views": list(scene_xml_paths.keys()),
            "views_count": total_views,
            "views_data": {}
        }
        
        success = True
        
        # Create a tqdm progress bar for the views within this frame
        with tqdm(total=total_views, desc=f"Frame {index} views", 
                  leave=False, position=1, 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as view_pbar:
            
            for i, (view, scene_xml_path) in enumerate(scene_xml_paths.items()):
                view_start = time.time()
                output_name = f"frame_{index}_{view}"
                
                try:
                    view_pbar.set_description(f"Frame {index}: '{view}' view")
                    logger.info(f"Rendering frame {index}, view {i+1}/{total_views}: '{view}'")
                    
                    # Initialize view timing data if detailed logging is enabled
                    view_timing_data = {
                        "view": view,
                        "start_time": datetime.now().isoformat(),
                        "cache_hit": False
                    }
                    
                    # Check cache first if enabled
                    cached_path = None
                    if self.cache_manager:
                        cached_path = self.cache_manager.get_cached_path(obj_file, view)
                    
                    if cached_path:
                        # Cache hit - copy from cache to output
                        self.cache_hits += 1
                        output_exr = self.get_output_path(self.exr_folder, output_name, ".exr")
                        os.makedirs(os.path.dirname(output_exr), exist_ok=True)
                        shutil.copyfile(cached_path, output_exr)
                        logger.info(f"Using cached version for frame {index}, view '{view}'")
                        
                        if detailed_timing:
                            view_timing_data["cache_hit"] = True
                            view_timing_data["cached_path"] = cached_path
                        
                        # Also create PNG if needed
                        if hasattr(config, "ENABLE_PNG") and config.ENABLE_PNG:
                            output_png = self.get_output_path(self.png_folder, output_name, ".png")
                            # Convert cached EXR to PNG
                            from src.processing.image import ImageConverter
                            ImageConverter.convert_exr_to_png(cached_path, output_png)
                    else:
                        # Cache miss - render normally
                        self.cache_misses += 1
                        
                        # Create a thread monitoring timer to detect rendering hangs
                        render_complete = threading.Event()
                        timer = threading.Timer(300, lambda: logger.error(f"Rendering timeout for frame {index}, view '{view}'") 
                                   if not render_complete.is_set() else None)
                        timer.daemon = True
                        timer.start()
                        
                        try:
                            # Render the scene
                            output_exr = self.render_scene(scene_xml_path, output_name)
                            
                            # Add to cache if enabled
                            if self.cache_manager and output_exr and os.path.exists(output_exr):
                                self.cache_manager.add_to_cache(obj_file, view, output_exr)
                            
                            # Cleanup
                            render_complete.set()
                            timer.cancel()
                            
                        except Exception as e:
                            render_complete.set()
                            timer.cancel()
                            raise e
                        
                        if detailed_timing:
                            view_timing_data["cache_hit"] = False
                    
                    # Update timing info
                    view_time = time.time() - view_start
                    view_pbar.set_postfix(time=f"{view_time:.2f}s")
                    logger.debug(f"  View '{view}' rendered in {timedelta(seconds=view_time)} ({view_time:.2f}s)")
                    
                    # Record detailed view timing if enabled
                    if detailed_timing:
                        view_timing_data["elapsed_seconds"] = view_time
                        view_timing_data["end_time"] = datetime.now().isoformat()
                        view_timing_data["success"] = True
                        frame_timing_data["views_data"][view] = view_timing_data
                        
                        # Log to structured timing logs if timing reports are enabled
                        if config.features.get("reports", {}).get("timing", False):
                            quality = self.scene_prefix.split("_")[-1] if "_" in self.scene_prefix else "unknown"
                            RichLogger.log_timing_data(
                                frame=index,
                                view=view,
                                quality=quality,
                                elapsed=view_time,
                                output_dir=self.base_folder,
                                additional_data={
                                    "obj_file": obj_name,
                                    "cache_hit": view_timing_data.get("cache_hit", False)
                                }
                            )
                        
                except Exception as e:
                    success = False
                    logger.error(f"Error rendering frame {index}, view '{view}': {e}")
                    
                    # Record error in view timing data if detailed logging is enabled
                    if detailed_timing:
                        view_timing_data["success"] = False
                        view_timing_data["error"] = str(e)
                        view_timing_data["elapsed_seconds"] = time.time() - view_start
                        view_timing_data["end_time"] = datetime.now().isoformat()
                        frame_timing_data["views_data"][view] = view_timing_data
                
                # Update the views progress bar regardless of success/failure
                view_pbar.update(1)
                
        # Record timing for the complete frame (all views)
        elapsed = time.time() - frame_start
        self.render_timings.append((index, obj_name, elapsed))
        logger.debug(f"Frame {index} with {total_views} views completed in {timedelta(seconds=elapsed)} ({elapsed:.2f}s)")
        
        # Save frame timing report if detailed logging is enabled
        if detailed_timing:
            frame_timing_data["elapsed_seconds"] = elapsed
            frame_timing_data["end_time"] = datetime.now().isoformat()
            frame_timing_data["success"] = success
            
            # Create logs directory in output folder if it doesn't exist
            logs_dir = os.path.join(self.base_folder, "logs", "frames")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save detailed timing report for this frame
            report_path = os.path.join(logs_dir, f"frame_{index}_timing.json")
            with open(report_path, 'w') as f:
                json.dump(frame_timing_data, f, indent=2)
        
        return success

    def _prepare_scene_xml(self, obj_file: str, index: int) -> Dict[str, str]:
        """
        Prepare scene XML for rendering with multiple views.
        
        Args:
            obj_file: Path to OBJ file
            index: Frame index
            
        Returns:
            Dictionary mapping view names to XML file paths
        """
        result = {}
        
        # If multiple camera views are requested
        if len(self.camera_views) > 1:
            # Base path for XML files
            base_path = os.path.join(self.scene_folder, f"{self.scene_prefix}_{index}")
            
            # Create XMLs for all camera views
            result = SceneProcessor.create_scene_xmls_with_multiple_views(
                obj_file=obj_file,
                output_base_path=base_path,
                views=self.camera_views,
                obj_path_in_xml=f"../meshes/{os.path.basename(obj_file)}"
            )
        else:
            # Single camera view (default behavior)
            view = self.camera_views[0]
            scene_xml_path = os.path.join(self.scene_folder, f"{self.scene_prefix}_{index}.xml")
            xml_path = SceneProcessor.create_scene_xml_for_obj(
                obj_file=obj_file,
                output_xml_path=scene_xml_path,
                obj_path_in_xml=f"../meshes/{os.path.basename(obj_file)}",
                camera_view=view
            )
            result[view] = xml_path
            
        return result

    def render_frames(self, max_workers: Optional[int] = None) -> bool:
        """
        Prepare all scene XMLs and then render all frames, either sequentially or with threading.
        
        Args:
            max_workers: Maximum number of worker threads (if threaded)
            
        Returns:
            True if all frames were rendered successfully, False otherwise
        """
        total_frames = len(self.obj_files)
        all_timings = []
        
        # First phase: prepare all scene XMLs
        self.prepare_scene_xmls()
        
        # Second phase: render all frames
        if self.threaded:
            return self._render_frames_threaded(max_workers, all_timings)
        else:
            return self._render_frames_sequential(all_timings)

    def _render_frames_threaded(self, max_workers: Optional[int], all_timings: List) -> bool:
        """
        Render frames using multithreading
        
        Returns:
            True if all frames were rendered successfully, False otherwise
        """
        total_frames = len(self.scene_xml_paths)
        success = True
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs with start time tracking
            futures = {}
            for i, scene_xml_path in enumerate(self.scene_xml_paths):
                if scene_xml_path is not None:  # Only submit jobs for successfully prepared XMLs
                    future = executor.submit(self.render_frame, i, scene_xml_path)
                    futures[future] = (i, scene_xml_path, time.time())
            
            # Process completed futures with timing
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc="Rendering frames (threaded)"):
                i, scene_xml_path, start_time = futures[future]
                try:
                    result = future.result()
                    if not result:
                        success = False
                    end_time = time.time()
                    elapsed = end_time - start_time
                    all_timings.append((i, f"{i}.obj", elapsed))
                except Exception as e:
                    success = False
                    logger.error(f"Error in threaded rendering frame {i}: {e}")
        
        # Log timing statistics
        self._log_timing_statistics(all_timings)
        logger.debug("Finished rendering all frames concurrently.")
        return success

    def _render_frames_sequential(self, all_timings: List) -> bool:
        """
pare with         Render frames sequentially with nested progress tracking.
        
        Returns:
            True if all frames were rendered successfully, False otherwise
        """
        total_frames = len(self.scene_xml_paths)
        success = True
        
        try:
            # Main progress bar for frames
            with tqdm(total=total_frames, desc="Rendering frames", position=0) as pbar:
                for i, scene_xml_path in enumerate(self.scene_xml_paths):
                    if scene_xml_path is None:
                        logger.warning(f"Skipping frame {i} - no scene XML available")
                        success = False
                        pbar.update(1)
                        continue
                        
                    frame_start = time.time()
                    try:
                        pbar.set_description(f"Rendering frame {i}/{total_frames}")
                        
                        frame_success = self.render_frame(i, scene_xml_path)
                        if not frame_success:
                            success = False
                        
                        frame_time = time.time() - frame_start
                        all_timings.append((i, f"{i}.obj", frame_time))
                        
                        # Update main progress bar with current frame timing
                        pbar.set_postfix(time=f"{frame_time:.2f}s", success=frame_success)
                        
                    except Exception as e:
                        success = False
                        logger.error(f"Error rendering frame {i}: {e}")
                        # Continue with next frame
                    
                    # Always update the progress bar
                    pbar.update(1)
                    
        except KeyboardInterrupt:
            logger.warning("Rendering interrupted by user")
            return False
        
        # Log timing statistics
        self._log_timing_statistics(all_timings)
        logger.debug("Finished rendering all frames sequentially.")
        return success

    def create_video_and_gif(self) -> None:
        """Create video and GIF from rendered frames"""
        
        # Check if we're using multiple camera views
        if len(self.camera_views) > 1:
            # Create separate video/GIF for each camera view
            for view in self.camera_views:
                self._create_video_and_gif_for_view(view)
        else:
            # Single view - use the first/only view in the list
            view = self.camera_views[0]
            self._create_video_and_gif_for_view(view)
            
    def _create_video_and_gif_for_view(self, view: str) -> None:
        """Create video and GIF for a specific camera view."""
        
        # Pattern for frame files with the specific view
        pattern = f"{self.scene_prefix}_frame_*_{view}{config.PNG_EXT}"
        png_files = list(glob.glob(os.path.join(self.png_folder, pattern)))
        
        if not png_files:
            logger.warning(f"No PNG files found matching '{pattern}'. Skipping video and GIF creation for view '{view}'.")
            return
            
        logger.debug(f"Found {len(png_files)} PNG files for view '{view}' video/GIF creation")
        
        # Pattern for FFmpeg using frame number
        ffmpeg_pattern = f"{self.scene_prefix}_frame_%d_{view}{config.PNG_EXT}"
        
        # Create video if enabled
        if config.ENABLE_VIDEO:
            try:
                # Use view in filename to keep videos separate
                video_filename = f"{self.scene_prefix}_{view}_video{config.VIDEO_FILE}"
                video_path = self.create_video(video_filename, ffmpeg_pattern)
                logger.debug(f"Video created for view '{view}' at {video_path}")
            except Exception as e:
                logger.error(f"Failed to create video for view '{view}': {e}")
                
        # Create GIF if enabled
        if config.ENABLE_GIF:
            try:
                # Use view in filename to keep GIFs separate
                gif_filename = f"{self.scene_prefix}_{view}_animation{config.GIF_FILE}"
                gif_path = self.create_gif(gif_filename, ffmpeg_pattern)
                logger.debug(f"GIF created for view '{view}' at {gif_path}")
            except Exception as e:
                logger.error(f"Failed to create GIF for view '{view}': {e}")

    def render_scene(self, scene_xml_path: str, output_name: str) -> Optional[str]:
        """
        Render a scene from an XML file and save outputs.
        
        Args:
            scene_xml_path: Path to the scene XML file
            output_name: Base name for output files
            
        Returns:
            Path to the EXR output file or None if rendering failed
        """
        import mitsuba as mi
        
        # Make output paths
        output_exr = self.get_output_path(self.exr_folder, output_name, ".exr")
        output_png = self.get_output_path(self.png_folder, output_name, ".png") if config.ENABLE_PNG else None
        
        # Create necessary directories
        os.makedirs(os.path.dirname(output_exr), exist_ok=True)
        if output_png:
            os.makedirs(os.path.dirname(output_png), exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Load scene using Mitsuba API
            from src.mitsuba.mitsuba import MitsubaUtils
            
            # If AOVs are enabled, modify the scene parameters
            scene_params = {}
            if config.AOV_ENABLED:
                scene_params["use_aovs"] = "true"
                scene_params["aovs"] = config.AOV_STRING
                logger.info(f"Rendering with AOVs: {config.AOV_STRING}")
                
            # Load the scene with parameters
            scene = MitsubaUtils.load_scene_from_file(scene_xml_path, params=scene_params)
            
            # Render the scene
            original_image = MitsubaUtils.render_scene(scene)
            
            # Save output as EXR
            original_image.write(output_exr)
            logger.debug(f"EXR rendered to {output_exr}")
            
            # If AOVs were enabled and we should extract them to separate files
            if config.AOV_ENABLED and config.AOV_SEPARATE_FILES:
                # Create AOV directory
                aov_folder = os.path.join(self.exr_folder, "aovs", output_name)
                os.makedirs(aov_folder, exist_ok=True)
                
                # Extract AOVs to separate files
                from src.processing.aov import AOVProcessor
                aov_files = AOVProcessor.extract_aovs_from_bitmap(
                    bitmap=original_image,
                    output_folder=aov_folder,
                    base_name=output_name
                )
                logger.debug(f"Extracted {len(aov_files)} AOV layers to {aov_folder}")
                
            # Check if denoising is enabled
            denoising_enabled = config.DENOISING_ENABLED
            
            # First verify that denoisers are available if denoising is requested
            from src.mitsuba.denoising import OPTIX_AVAILABLE
            
            if denoising_enabled:
                if not OPTIX_AVAILABLE:
                    logger.warning("Denoising is enabled but OptiX denoiser is not available. Continuing with non-denoised image.")
                    denoising_enabled = False
            
            # Proceed with denoising if still enabled and denoisers are available
            if denoising_enabled and OPTIX_AVAILABLE:
                try:
                    # Get denoising configuration
                    denoiser_strength = config.DENOISING_STRENGTH
                    use_guide_buffers = config.DENOISING_USE_GUIDE_BUFFERS
                    
                    logger.debug(f"Denoising with OptiX denoiser (strength={denoiser_strength})")
                    
                    # Use DenoiserManager for the actual denoising
                    from src.mitsuba.denoising import DenoiserManager
                    
                    # If using guide buffers and we have AOVs, use those directly
                    if use_guide_buffers and config.AOV_ENABLED:
                        # If we have a multichannel image with AOVs, extract guide buffers
                        channels = dict(original_image.split())
                        albedo = channels.get('albedo')
                        normals = channels.get('sh_normal') or channels.get('normals')
                        
                        if albedo and normals:
                            logger.debug("Using AOV guide buffers for denoising")
                            denoised_image = DenoiserManager.denoise_image_safe(
                                image=original_image,
                                albedo=albedo,
                                normals=normals,
                                denoiser_strength=denoiser_strength
                            )
                        else:
                            # Fall back to basic denoising
                            logger.debug("Using basic denoising (no guide buffers found in AOVs)")
                            denoised_image = DenoiserManager.denoise_image_safe(
                                image=original_image,
                                denoiser_strength=denoiser_strength
                            )
                    # Option 1: Denoise the already rendered image without guide buffers
                    elif not use_guide_buffers:
                        denoised_image = DenoiserManager.denoise_image_safe(
                            image=original_image,
                            denoiser_strength=denoiser_strength
                        )
                    # Option 2: Render with guide buffers for denoising
                    else:
                        try:
                            # Render albedo (diffuse colors) buffer - reduce sample count for speed
                            albedo_params = {"aovs": "albedo", "spp": 16}
                            albedo = MitsubaUtils.render_scene(scene, albedo_params)
                            
                            # Render normal buffer - reduce sample count for speed
                            normal_params = {"aovs": "nn:sh_normal", "spp": 16}
                            normals = MitsubaUtils.render_scene(scene, normal_params)
                            
                            # Denoise with guide buffers
                            denoised_image = DenoiserManager.denoise_image_safe(
                                image=original_image,
                                albedo=albedo,
                                normals=normals,
                                denoiser_strength=denoiser_strength
                            )
                        except Exception as e:
                            logger.warning(f"Failed to render guide buffers: {e}. Trying without guide buffers.")
                            denoised_image = DenoiserManager.denoise_image_safe(
                                image=original_image,
                                denoiser_strength=denoiser_strength
                            )
                    
                    # If we actually got a denoised image back, save it
                    if denoised_image is not original_image:  # Reference comparison intentional
                        # Save denoised EXR
                        denoised_exr = output_exr.replace(".exr", "_denoised.exr")
                        denoised_image.write(denoised_exr)
                        logger.debug(f"Denoised EXR saved to {denoised_exr}")
                        
                        # For PNG output, use the denoised image
                        if output_png:
                            from src.processing.image import ImageConverter
                            ImageConverter.convert_exr_to_png(denoised_exr, output_png)
                            logger.debug(f"Denoised PNG saved to {output_png}")
                    else:
                        # Denoising failed or returned original image, use standard conversion
                        if output_png:
                            from src.processing.image import ImageConverter
                            ImageConverter.convert_exr_to_png(output_exr, output_png)
                            logger.debug(f"PNG saved to {output_png}")
                    
                except Exception as e:
                    logger.warning(f"Denoising failed: {e}, continuing with non-denoised image")
                    # If denoising fails, fall back to non-denoised PNG conversion
                    if output_png:
                        from src.processing.image import ImageConverter
                        ImageConverter.convert_exr_to_png(output_exr, output_png)
            else:
                # No denoising - standard PNG conversion if enabled
                if output_png:
                    from src.processing.image import ImageConverter
                    ImageConverter.convert_exr_to_png(output_exr, output_png)
                    logger.debug(f"PNG saved to {output_png}")
            
            # Log rendering time
            elapsed = time.time() - start_time
            logger.debug(f"Scene rendered in {elapsed:.2f}s")
            
            return output_exr
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error rendering scene {scene_xml_path} (after {elapsed:.2f}s): {e}")
            return None

    @classmethod
    def render_multi_obj_scene_from_folder(cls, folder: str, regex: str, scene_name: str, 
                                         threaded: bool = False, max_workers: Optional[int] = None, 
                                         render_params: Optional[Dict[str, Any]] = None,
                                         stop_on_error: bool = True,
                                         camera_views: Optional[List[str]] = None,
                                         integrator_override: Optional[Dict[str, Any]] = None,
                                         cache_manager: Optional[CacheManager] = None,
                                         denoising_enabled: bool = False,
                                         denoising_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Render scenes for all matching OBJ files in a folder.
        
        Args:
            folder: Path to folder containing OBJ files
            regex: Regular expression pattern to match filenames
            scene_name: Prefix for output files
            threaded: Whether to use multithreading
            max_workers: Maximum number of worker threads (if threaded)
            render_params: Dictionary with rendering parameters (spp, width, height)
            stop_on_error: Whether to stop execution on first rendering error
            camera_views: List of camera views to render
            integrator_override: Override parameters for the integrator
            cache_manager: Optional cache manager for frame caching
            denoising_enabled: Whether to enable denoising
            denoising_config: Configuration for denoising
            
        Returns:
            True if all frames were rendered successfully, False otherwise
        """
        logger.debug(f"Rendering multi-object scenes from folder: {folder}")
        
        if render_params:
            logger.debug(f"Rendering with custom parameters: {render_params}")
            
        if camera_views:
            logger.debug(f"Rendering with camera views: {', '.join(camera_views)}")
        
        try:
            # Get matching OBJ files
            matched_files = ObjProcessor.get_matching_obj_files(folder, regex)
            logger.debug(f"Found {len(matched_files)} files matching regex '{regex}'")
            
            if not matched_files:
                logger.error(f"No OBJ files found matching the regex '{regex}' in folder '{folder}'")
                return False
                
            # If no camera views specified, use defaults from config
            if not camera_views:
                camera_views = CameraUtils.get_default_views()
                
            logger.info(f"Rendering with camera views: {', '.join(camera_views)}")
            
            # Create and configure renderer
            renderer = cls(matched_files, scene_name, threaded=threaded, 
                        camera_views=camera_views, cache_manager=cache_manager)
            
            # Create integrator parameters by combining config defaults with any overrides
            if integrator_override:
                integrator_params = {}
                
                # Get integrator type from override or from render_params
                integrator_type = integrator_override.get("type", 
                                  render_params.get("integrator_type", "path"))
                
                # Get standard parameters that apply to most integrators
                if "max_depth" in render_params:
                    integrator_params["max_depth"] = render_params["max_depth"]
                    
                # Add any additional override parameters
                integrator_params.update(integrator_override)
                integrator_params["type"] = integrator_type
                
                logger.debug(f"Using integrator settings: {integrator_params}")
                # TODO: Pass integrator_params to scene creation
            
            # Apply rendering parameters to template if specified
            if render_params:
                SceneProcessor.update_template_with_params(
                    config.TEMPLATE_XML,
                    render_params.get("spp", 256),
                    render_params.get("width", 1920),
                    render_params.get("height", 1080),
                    render_params.get("max_depth", 8)
                )
            
            # Configure denoising if enabled
            if denoising_enabled:
                logger.info("Denoising is enabled for this rendering")
                if denoising_config:
                    # Check if OptiX is available before setting up denoising
                    if not OPTIX_AVAILABLE:  # Use OPTIX_AVAILABLE directly
                        logger.warning("OptiX denoiser is not available, disabling denoising")
                        denoising_enabled = False
                
                # Configure denoising in global config for the renderer to use
                config.features["denoising"] = {
                    "enabled": denoising_enabled,
                    **(denoising_config or {})
                }
            
            # Render all frames
            success = renderer.render_frames(max_workers)
            
            if not success and stop_on_error:
                logger.error("Rendering stopped due to errors")
                return False
            
            # Create video and GIF only if rendering was successful
            if success:
                renderer.create_video_and_gif()
            else:
                logger.warning("Skipping video/GIF creation because of rendering errors")
            
            # Log cache statistics if cache was used
            if cache_manager and renderer.cache_hits + renderer.cache_misses > 0:
                total = renderer.cache_hits + renderer.cache_misses
                hit_rate = (renderer.cache_hits / total) * 100 if total > 0 else 0
                logger.info(f"Cache statistics: {renderer.cache_hits} hits, {renderer.cache_misses} misses ({hit_rate:.1f}% hit rate)")
                
                # If debug is enabled, show more detailed cache stats
                if config.debug.get("verboseLogging", False):
                    stats = cache_manager.get_cache_stats()
                    logger.debug(f"Cache size: {stats['total_size_mb']:.2f} MB, Entries: {stats['valid_entries']}")
            
            logger.debug("Rendering completed.")
            return success
            
        except Exception as e:
            logger.error(f"Error during multi-object rendering: {e}", exc_info=True)
            return False
