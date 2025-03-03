"""
Image processing utilities for the Mitsuba application.
"""

import mitsuba as mi
from src.utils.logger import RichLogger
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.processing.image")

class ImageConverter:
    """
    Utility class for image format conversion.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def convert_exr_to_png(input_exr: str, output_png: str) -> None:
        """
        Convert an EXR image to an 8-bit sRGB PNG using Mitsuba.
        
        Args:
            input_exr: Path to input EXR file
            output_png: Path to output PNG file
        """
        try:
            bmp = mi.Bitmap(input_exr)
            bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
            bmp.write(output_png)
            logger.debug(f"Converted {input_exr} to {output_png}")
        except Exception as e:
            logger.error(f"Error converting {input_exr} to PNG: {e}")
            raise
    
    @staticmethod
    @timeit(log_level="debug")
    def convert_exr_to_hdr(input_exr: str, output_hdr: str) -> None:
        """
        Convert an EXR image to an HDR format file.
        
        Args:
            input_exr: Path to input EXR file
            output_hdr: Path to output HDR file
        """
        try:
            bmp = mi.Bitmap(input_exr)
            bmp.write(output_hdr)
            logger.debug(f"Converted {input_exr} to {output_hdr}")
        except Exception as e:
            logger.error(f"Error converting {input_exr} to HDR: {e}")
            raise
