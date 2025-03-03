import os
import time
import signal
import threading
from pathlib import Path
from typing import Optional, Any, Dict

import mitsuba as mi

from src.config.config import config_class as config
from src.utils.logger import RichLogger
from src.utils.timing import TimingManager
from src.processing.image import ImageConverter
from src.processing.ffmpeg import FFmpegProcessor
from src.utils.common import ensure_directory

logger = RichLogger.get_logger("mitsuba_app.renderers.base")


class BaseRenderer:
    """
    Base class for scene rendering that handles common functionality.
    """

    def __init__(self, scene_prefix: str, base_folder: Optional[str] = None) -> None:
        """
        Initialize the base renderer with output directories and settings.

        Args:
            scene_prefix: Prefix for output filenames.
            base_folder: Base directory for all output files. If not provided,
                         defaults to the output root defined in configuration.
        """
        self.scene_prefix = scene_prefix
        # Use provided base_folder or fall back to the configuration value.
        self.base_folder = base_folder or config.paths["output"]["root"]

        # Define output directories based on configuration.
        self.exr_folder = os.path.join(self.base_folder, config.paths["output"]["folders"]["exr"])
        self.png_folder = os.path.join(self.base_folder, config.paths["output"]["folders"]["png"])
        self.video_folder = os.path.join(self.base_folder, config.paths["output"]["folders"]["video"])
        self.gif_folder = os.path.join(self.base_folder, config.paths["output"]["folders"]["gif"])
        self.scene_folder = os.path.join(self.base_folder, config.paths["output"]["folders"]["scenes"])
        self.reports_folder = os.path.join(self.base_folder, config.REPORTS_FOLDER)
        self.mesh_folder = os.path.join(self.base_folder, config.MESH_FOLDER)

        # Create all necessary directories.
        for folder in [
            self.exr_folder,
            self.png_folder,
            self.video_folder,
            self.gif_folder,
            self.scene_folder,
            self.reports_folder,
            self.mesh_folder,
        ]:
            ensure_directory(folder)

        # List to track rendering timings.
        self.render_timings: list[float] = []

    def get_output_path(self, folder: str, name: str, ext: str) -> str:
        """
        Generate an output file path with proper naming.

        Args:
            folder: Target folder for the file.
            name: Name part of the file (without prefix/extension).
            ext: File extension.

        Returns:
            Full path to the output file.
        """
        return os.path.join(folder, f"{self.scene_prefix}_{name}{ext}")

    def render_scene(self, scene_xml_path: str, output_name: str, timeout: int = 300) -> str:
        """
        Render a scene from an XML file and save the output.

        Args:
            scene_xml_path: Path to the scene XML file.
            output_name: Base name for the output files (without prefix/extension).
            timeout: Maximum time in seconds to allow for rendering.

        Returns:
            Path to the rendered EXR file.

        Raises:
            Exception: Propagates exceptions encountered during rendering.
        """
        # Ensure output directories exist.
        ensure_directory(self.exr_folder)
        ensure_directory(self.png_folder)

        output_exr = self.get_output_path(self.exr_folder, output_name, config.EXR_EXT)
        os.makedirs(os.path.dirname(output_exr), exist_ok=True)

        try:
            logger.debug(f"Loading scene from XML: {scene_xml_path}")
            scene = mi.load_file(scene_xml_path)
            logger.debug("Scene loaded successfully, starting render...")

            # Render the scene and obtain a bitmap with timeout monitoring
            render_start = time.time()
            img = mi.render(scene)
            render_time = time.time() - render_start
            logger.debug(f"Scene rendered in {render_time:.2f} seconds")
            
            bmp = mi.Bitmap(img)
            logger.debug(f"Bitmap created, saving to {output_exr}")

            # Save the rendered image.
            bmp.write(output_exr)
            logger.debug(f"Rendered scene saved as {output_exr}")

            # Convert to PNG if both PNG and EXR outputs are enabled.
            if config.ENABLE_PNG and config.ENABLE_EXR:
                output_png = self.get_output_path(self.png_folder, output_name, config.PNG_EXT)
                ImageConverter.convert_exr_to_png(output_exr, output_png)
                logger.debug(f"Converted rendered EXR to PNG: {output_png}")

            return output_exr

        except Exception as e:
            logger.error(f"Error rendering scene {scene_xml_path}: {e}")
            raise

    def create_video(
        self,
        output_name: Optional[str] = None,
        pattern: Optional[str] = None,
        framerate: Optional[int] = None,
    ) -> str:
        """
        Create a video from rendered frames.

        Args:
            output_name: Name for the video file. Defaults to the configured default.
            pattern: Filename pattern for input frames. If None, uses default pattern.
            framerate: Frames per second for the video.

        Returns:
            Path to the created video file.
        """
        output_name = output_name or config.files["defaults"]["video"]
        # Use provided pattern or fall back to default
        pattern = pattern or f"{self.scene_prefix}_{config.patterns.get('framePattern', 'frame_%d')}{config.files['extensions']['png']}"
        framerate = framerate or config.rendering["framerate"]

        output_video = os.path.join(self.video_folder, output_name)
        FFmpegProcessor.create_video(self.png_folder, output_video, pattern, framerate)
        return output_video

    def create_gif(
        self,
        output_name: Optional[str] = None,
        pattern: Optional[str] = None,
        framerate: Optional[int] = None,
    ) -> str:
        """
        Create a GIF from rendered frames.

        Args:
            output_name: Name for the GIF file. Defaults to the configured default.
            pattern: Filename pattern for input frames. If None, uses default pattern.
            framerate: Frames per second for the GIF.

        Returns:
            Path to the created GIF file.
        """
        output_name = output_name or config.files["defaults"]["gif"]
        # Use provided pattern or fall back to default
        pattern = pattern or f"{self.scene_prefix}_{config.patterns.get('framePattern', 'frame_%d')}{config.files['extensions']['png']}"
        framerate = framerate or config.rendering["framerate"]

        output_gif = os.path.join(self.gif_folder, output_name)
        FFmpegProcessor.create_gif(self.png_folder, output_gif, pattern, framerate)
        return output_gif

    def _log_timing_statistics(self, timings: list[float]) -> None:
        """Log statistics about frame rendering times."""
        if not timings:
            return
        TimingManager.log_timing_statistics(
            timings=timings,
            prefix=self.scene_prefix,
            output_folder=self.reports_folder,
        )
