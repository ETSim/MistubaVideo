"""
FFmpeg processing utilities for video and GIF creation.
"""

import os
import re
import time
import glob
import subprocess
from typing import List
from tqdm import tqdm

from src.utils.logger import RichLogger
from src.config.config import config_class as config
from src.utils.timing import timeit
from src.utils.common import ensure_directory, run_subprocess

logger = RichLogger.get_logger("mitsuba_app.processing.ffmpeg")

class FFmpegHelper:
    @staticmethod
    @timeit(log_level="debug")
    def check_ffmpeg() -> str:
        """
        Ensure FFmpeg is available; return its path from the config.
        
        Returns:
            Path to FFmpeg executable
        
        Raises:
            RuntimeError: If FFmpeg is not found
        """
        if not config.paths["ffmpeg"]:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        return config.paths["ffmpeg"]

    @staticmethod
    @timeit(log_level="debug")
    def run_command(command: List[str], description: str) -> None:
        """
        Run a command (such as FFmpeg) and show progress via tqdm with enhanced display.
        This implementation tries to parse FFmpeg output to show actual progress percentages.
        
        Args:
            command: Command list to execute
            description: Description to show in progress bar
            
        Raises:
            RuntimeError: If the command returns a non-zero exit status
        """
        logger.debug(f"Executing command: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Try to determine total duration or frames for better progress display
        progress_pattern = re.compile(r'frame=\s*(\d+)')
        time_pattern = re.compile(r'time=(\d+:\d+:\d+\.\d+)')
        duration_seconds = None
        
        # Initialize tqdm with unknown total but better formatting
        with tqdm(total=100, desc=description, unit="%", 
                 bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]") as pbar:
            last_percent = 0
            start_time = time.time()
            
            while True:
                line = process.stderr.readline()
                if line == "" and process.poll() is not None:
                    break
                
                if line:
                    # Look for frame information to update progress
                    frame_match = progress_pattern.search(line)
                    time_match = time_pattern.search(line)
                    
                    # If we found time info, try to calculate percentage
                    if time_match:
                        time_str = time_match.group(1)
                        h, m, s = time_str.split(':')
                        current_seconds = float(h) * 3600 + float(m) * 60 + float(s)
                        
                        # If we don't know the total duration yet, just show elapsed time
                        if duration_seconds is None:
                            elapsed = time.time() - start_time
                            pbar.set_postfix(elapsed=f"{elapsed:.1f}s", current=time_str)
                        else:
                            percent = min(int((current_seconds / duration_seconds) * 100), 100)
                            if percent > last_percent:
                                pbar.update(percent - last_percent)
                                last_percent = percent
                    
                    # If we found frame info, show it
                    elif frame_match:
                        frame = frame_match.group(1)
                        pbar.set_postfix(frame=frame)
                        
                    # Log any errors
                    if "error" in line.lower():
                        pbar.write(line.strip())
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command returned non-zero exit status {process.returncode}")

    @staticmethod
    @timeit(log_level="debug")
    def build_ffmpeg_command_video(input_pattern: str, output_video: str, framerate: int = None) -> List[str]:
        """
        Construct the FFmpeg command for creating an MP4 video.
        Uses framerate if specified, otherwise uses frame interval from config.
        """
        base_cmd = [FFmpegHelper.check_ffmpeg(), "-y"]

        # Use either framerate or frame interval, with framerate taking precedence
        if (framerate is not None and framerate > 0):
            base_cmd.extend(["-framerate", str(framerate)])
        elif config.rendering.get("frameInterval", 0) > 0:
            interval = config.rendering["frameInterval"]
            effective_framerate = 1.0 / interval
            base_cmd.extend(["-framerate", f"{effective_framerate:.2f}"])

        base_cmd.extend([
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ])
        
        return base_cmd

    @staticmethod
    @timeit(log_level="debug")
    def build_ffmpeg_command_gif(input_pattern: str, output_gif: str, framerate: int = None) -> List[str]:
        """
        Construct the FFmpeg command for creating an animated GIF.
        Uses framerate if specified, otherwise uses frame interval from config.
        """
        # Determine effective framerate
        if framerate is not None and framerate > 0:
            effective_framerate = framerate
        elif config.rendering.get("frameInterval", 0) > 0:
            interval = config.rendering["frameInterval"]
            effective_framerate = 1.0 / interval
        else:
            effective_framerate = 30  # default fallback

        return [
            FFmpegHelper.check_ffmpeg(),
            "-y",
            "-framerate", f"{effective_framerate:.2f}",
            "-i", input_pattern,
            "-filter_complex",
            f"[0:v] fps={effective_framerate:.2f},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split [a][b];"
            "[a] palettegen [p];"
            "[b][p] paletteuse",
            output_gif
        ]

class FFmpegProcessor:
    """
    Processor class for creating videos and GIFs using FFmpeg.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def get_ffmpeg_path() -> str:
        """Get FFmpeg executable path from config."""
        return config.FFMPEG_PATH
    
    @staticmethod
    @timeit(log_level="debug")
    def create_video(png_folder: str, output_video: str, pattern: str, framerate: int = None) -> str:
        """
        Create an MP4 video from PNG images using FFmpeg.
        
        Args:
            png_folder: Folder containing PNG images
            output_video: Path to output video file
            pattern: Pattern for input frames (e.g., "frame_%d.png")
            framerate: Frames per second (default: from config)
            
        Returns:
            Path to the created video or empty string on failure
        """
        # Calculate effective framerate based on timing settings
        if config.VIDEO_DURATION:
            # If duration is set, calculate framerate to match desired duration
            num_frames = len(glob.glob(os.path.join(png_folder, pattern.replace("%d", "*"))))
            effective_framerate = num_frames / config.VIDEO_DURATION
        elif config.VIDEO_FRAME_INTERVAL > 0:
            # Use frame interval if specified
            effective_framerate = 1.0 / config.VIDEO_FRAME_INTERVAL
        else:
            # Fall back to default framerate
            effective_framerate = framerate or config.rendering["framerate"]

        # Update FFmpeg command with calculated framerate
        cmd = FFmpegHelper.build_ffmpeg_command_video(
            input_pattern=os.path.join(png_folder, pattern),
            output_video=output_video,
            framerate=effective_framerate
        )
        
        if framerate is None:
            framerate = config.rendering["framerate"]
            
        # First check if there are any matching PNG files
        search_pattern = pattern.replace("%d", "*")
        matching_files = glob.glob(os.path.join(png_folder, search_pattern))
        
        if not matching_files:
            logger.warning(f"No PNG files found matching '{os.path.join(png_folder, search_pattern)}'. Cannot create video.")
            return ""
            
        logger.debug(f"Found {len(matching_files)} PNG files for video creation")
        
        # Ensure the output directory exists
        ensure_directory(os.path.dirname(output_video))
        
        # Build the command - FIX: Avoid using filter_complex as a string with quotes
        cmd = [
            FFmpegProcessor.get_ffmpeg_path(),
            "-y",
            "-framerate", str(framerate),
            "-i", os.path.join(png_folder, pattern),
        ]
        
        # Apply frame interval if configured - FIX: Use proper command argument structure
        frame_interval = config.rendering.get("frameInterval", 0)
        if frame_interval > 0:
            cmd.extend([
                "-filter:v", f"setpts={frame_interval}*PTS"
            ])
            
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ])
        
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run the command
        return_code, stdout, stderr = run_subprocess(cmd, timeout=60)
        
        if return_code == 0:
            logger.debug(f"Video created at {output_video}")
            return output_video
        else:
            logger.error(f"Failed to create video: {stderr}")
            return ""

    @staticmethod
    @timeit(log_level="debug")
    def create_gif(png_folder: str, output_gif: str, pattern: str, framerate: int = None) -> str:
        """
        Create an animated GIF from PNG images using FFmpeg.
        
        Args:
            png_folder: Folder containing PNG images
            output_gif: Path to output GIF file
            pattern: Pattern for input frames (e.g., "frame_%d.png")
            framerate: Frames per second (default: from config)
            
        Returns:
            Path to the created GIF or empty string on failure
        """
        # Calculate effective framerate based on timing settings
        if config.GIF_DURATION:
            # If duration is set, calculate framerate to match desired duration
            num_frames = len(glob.glob(os.path.join(png_folder, pattern.replace("%d", "*"))))
            effective_framerate = num_frames / config.GIF_DURATION
        elif config.GIF_FRAME_INTERVAL > 0:
            # Use frame interval if specified
            effective_framerate = 1.0 / config.GIF_FRAME_INTERVAL
        else:
            # Fall back to default framerate
            effective_framerate = framerate or config.rendering["framerate"]

        # Update FFmpeg command with calculated framerate
        cmd = FFmpegHelper.build_ffmpeg_command_gif(
            input_pattern=os.path.join(png_folder, pattern),
            output_gif=output_gif,
            framerate=effective_framerate
        )
        
        if framerate is None:
            framerate = config.rendering["framerate"]
            
        # First check if there are any matching PNG files
        search_pattern = pattern.replace("%d", "*")
        matching_files = glob.glob(os.path.join(png_folder, search_pattern))
        
        if not matching_files:
            logger.warning(f"No PNG files found matching '{os.path.join(png_folder, search_pattern)}'. Cannot create GIF.")
            return ""
            
        logger.debug(f"Found {len(matching_files)} PNG files for GIF creation")
        
        # Ensure the output directory exists
        ensure_directory(os.path.dirname(output_gif))
        
        # FIX: Properly structure the filter_complex argument without embedded quotes
        filter_complex = f"[0:v] fps={framerate},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split [a][b]; [a] palettegen [p]; [b][p] paletteuse"
        
        # Apply frame interval if configured
        frame_interval = config.rendering.get("frameInterval", 0)
        if frame_interval > 0:
            # Integrate setpts into the filter complex chain correctly
            filter_complex = f"[0:v] fps={framerate},setpts={frame_interval}*PTS,scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split [a][b]; [a] palettegen [p]; [b][p] paletteuse"
        
        # Build the command
        cmd = [
            FFmpegProcessor.get_ffmpeg_path(),
            "-y",
            "-framerate", str(framerate),
            "-i", os.path.join(png_folder, pattern),
            "-filter_complex", filter_complex,
            output_gif
        ]
        
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run the command
        return_code, stdout, stderr = run_subprocess(cmd, timeout=60)
        
        if return_code == 0:
            logger.debug(f"GIF created at {output_gif}")
            return output_gif
        else:
            logger.error(f"Failed to create GIF: {stderr}")
            return ""
