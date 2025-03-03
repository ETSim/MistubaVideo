"""
Module for image denoising capabilities in the Mitsuba renderer.

Supports the NVIDIA OptiX AI Denoiser for high-quality denoising of rendered images.
"""

import os
import sys
import glob
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import mitsuba as mi
import numpy as np

from src.utils.logger import RichLogger
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.denoising")

def convert_to_tensor(image: Any) -> Any:
    """
    Convert an image to a TensorXf if available.
    
    Args:
        image: Input image (bitmap, tensor, or numpy array)
    
    Returns:
        Converted tensor or original image if conversion isn't possible
    """
    try:
        import drjit
        if hasattr(drjit, "TensorXf"):
            # If already a tensor, return as is
            if isinstance(image, drjit.TensorXf):
                return image
            
            # If a bitmap, convert to numpy then tensor
            if isinstance(image, mi.Bitmap):
                return drjit.TensorXf(np.array(image))
            
            # If numpy array, convert directly
            if isinstance(image, np.ndarray):
                return drjit.TensorXf(image)
            
    except ImportError:
        pass  # drjit not available
    
    # Return original if no conversion possible
    return image

def convert_from_tensor(tensor: Any, pixel_format: Any = None) -> mi.Bitmap:
    """
    Convert a tensor to a Mitsuba bitmap.
    
    Args:
        tensor: Input tensor or array
        pixel_format: Optional pixel format for the bitmap
    
    Returns:
        Mitsuba bitmap
    """
    # If already a bitmap, return as is
    if isinstance(tensor, mi.Bitmap):
        return tensor
    
    # Convert to numpy array if needed
    if hasattr(tensor, "__array__"):
        array = np.array(tensor)
    else:
        array = tensor
    
    # Create bitmap with pixel format if provided
    if pixel_format is not None:
        return mi.Bitmap(array, pixel_format)
    else:
        return mi.Bitmap(array)

# Check if OptiX denoiser is available - improved detection logic
OPTIX_AVAILABLE = False
try:
    # First check if we have CUDA variants
    variants = mi.variants()
    logger.debug(f"Available Mitsuba variants: {variants}")
    
    # Filter for CUDA variants - ensure we're filtering strings
    cuda_variants = []
    for v in variants:
        if isinstance(v, str) and ("cuda" in v.lower() or "gpu" in v.lower()):
            cuda_variants.append(v)
            
    if not cuda_variants:
        logger.warning("No CUDA variants found in Mitsuba - OptiX denoiser will not be available")
    else:
        # Store current variant to restore it later
        try:
            current_variant = mi.variant()
            logger.debug(f"Current variant: {current_variant}")
            
            # Set a CUDA variant to check for OptiX denoiser
            test_variant = cuda_variants[0] 
            logger.debug(f"Testing OptiX availability with variant: {test_variant}")
            mi.set_variant(test_variant)
            
            # Check if OptixDenoiser is available in this variant
            if hasattr(mi, "OptixDenoiser"):
                # Try to create a small test denoiser to confirm it works
                try:
                    test_denoiser = mi.OptixDenoiser(input_size=(64, 64), albedo=False, normals=False)
                    OPTIX_AVAILABLE = True
                    logger.info("NVIDIA OptiX denoiser is available and working")
                except Exception as e:
                    logger.warning(f"OptiX denoiser constructor failed: {e}")
                    OPTIX_AVAILABLE = False
            else:
                logger.warning("OptixDenoiser class not found in Mitsuba CUDA variant")
                
            # Restore original variant
            if current_variant is not None:
                try:
                    mi.set_variant(current_variant)
                    logger.debug(f"Restored variant to: {current_variant}")
                except Exception as e:
                    logger.warning(f"Error restoring variant: {e}")
        except Exception as e:
            logger.warning(f"Error while testing OptiX denoiser: {e}")
            OPTIX_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error checking for OptiX denoiser: {e}")
    OPTIX_AVAILABLE = False

logger.info(f"OptiX denoiser availability: {'AVAILABLE' if OPTIX_AVAILABLE else 'NOT AVAILABLE'}")

def is_tensorxf_available() -> bool:
    """Check if TensorXf is available in drjit."""
    try:
        import drjit
        return hasattr(drjit, "TensorXf")
    except ImportError:
        return False

class DenoiserManager:
    """
    Class to manage image denoising operations using the OptiX AI Denoiser.
    """
    
    # Cache for denoiser instances to avoid recreating them for same-sized images
    _denoiser_cache = {}
    # Store denoiser configuration along with the instance
    _denoiser_config = {}
    
    @staticmethod
    def _get_cached_denoiser(size: Tuple[int, int], use_albedo: bool = False, use_normals: bool = False, temporal: bool = False):
        """
        Get or create a denoiser with specific parameters, using cache to avoid recreation.
        
        Args:
            size: Image size (width, height)
            use_albedo: Whether to use albedo buffers (default: False)
            use_normals: Whether to use normal buffers (default: False)
            temporal: Whether to use temporal denoising (default: False)
            
        Returns:
            OptixDenoiser instance or None if creation fails
        """
        # Always use basic denoiser without guide buffers for reliability
        use_albedo = False
        use_normals = False
        if temporal:
            logger.info("Disabling temporal denoising for reliability")
            temporal = False
            
        if not OPTIX_AVAILABLE:
            logger.info("OptiX denoiser not available")
            return None
            
        # Set CUDA variant first
        variants = mi.variants()
        cuda_variants = [v for v in variants if isinstance(v, str) and ("cuda" in v.lower() or "gpu" in v.lower())]
        if not cuda_variants:
            logger.warning("No CUDA variants available, OptiX denoiser cannot be used")
            return None
        
        # Ensure we're using a CUDA variant
        if not any(x in mi.variant() for x in ["cuda", "gpu"]):
            # Try to set a CUDA variant
            for variant in ["cuda_ad_rgb", "cuda_rgb", "cuda_spectral"]:
                if variant in variants:
                    try:
                        mi.set_variant(variant)
                        logger.info(f"Switched to {variant} variant for OptiX denoising")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to set variant {variant}: {e}")
        
        # Log current variant
        logger.info(f"Current variant for denoising: {mi.variant()}")
        
        # Check if we have a cached denoiser for these parameters
        cache_key = (size, use_albedo, use_normals, temporal)
        if cache_key not in DenoiserManager._denoiser_cache:
            try:
                # Create the basic denoiser - NO guide buffers
                logger.info(f"Creating new basic denoiser (size={size}, no guide buffers)")
                DenoiserManager._denoiser_cache[cache_key] = mi.OptixDenoiser(
                    input_size=size,
                    albedo=False,
                    normals=False,
                    temporal=False
                )
                # Store the configuration
                DenoiserManager._denoiser_config[cache_key] = {
                    "use_albedo": False,
                    "use_normals": False,
                    "temporal": False,
                    "size": size
                }
                logger.info(f"Successfully created basic OptiX denoiser")
            except Exception as e:
                logger.warning(f"Failed to create OptiX denoiser: {e}")
                return None
        
        return DenoiserManager._denoiser_cache.get(cache_key)
    
    @staticmethod
    def prepare_scene_for_denoising(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a scene dictionary for denoising by setting the appropriate reconstruction filter.
        
        Args:
            scene_dict: The Mitsuba scene dictionary
            
        Returns:
            Modified scene dictionary optimized for denoising
        """
        # Deep copy to avoid modifying the original
        import copy
        scene_dict = copy.deepcopy(scene_dict)
        
        # Set box reconstruction filter for optimal denoising
        if 'sensor' in scene_dict and 'film' in scene_dict['sensor']:
            if 'rfilter' not in scene_dict['sensor']['film']:
                scene_dict['sensor']['film']['rfilter'] = {}
            
            scene_dict['sensor']['film']['rfilter']['type'] = 'box'
            logger.debug("Set box reconstruction filter for optimal denoising")
            
        return scene_dict

    @staticmethod
    def create_aov_integrator_for_denoising(base_integrator: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an AOV integrator to capture albedo and normal buffers for better denoising.
        
        Args:
            base_integrator: The base integrator configuration
            
        Returns:
            AOV integrator configuration
        """
        return {
            'type': 'aov',
            'aovs': 'albedo:albedo,normals:sh_normal',
            'integrator': base_integrator
        }
    
    @staticmethod
    @timeit(log_level="debug")
    def denoise_image(
        image: Union[mi.Bitmap, str],
        albedo: Optional[Union[mi.Bitmap, str]] = None,
        normals: Optional[Union[mi.Bitmap, str]] = None, 
        denoiser_strength: float = 1.0,
        prev_frame: Optional[Union[mi.Bitmap, str]] = None,
        optical_flow: Optional[Union[mi.Bitmap, np.ndarray]] = None,
        to_sensor: Optional[Any] = None
    ) -> mi.Bitmap:
        """
        Denoise an image using the OptiX AI Denoiser.
        
        Args:
            image: Input image as Mitsuba Bitmap or path to EXR file
            albedo: Optional albedo buffer for improved denoising
            normals: Optional surface normals buffer for improved denoising
            denoiser_strength: Strength of denoising (0.0 to 1.0)
            prev_frame: Optional previous frame for temporal denoising
            optical_flow: Optional optical flow for temporal denoising
            to_sensor: Optional transform to convert normals to sensor space
            
        Returns:
            Denoised image as Mitsuba Bitmap
            
        Raises:
            ValueError: If the denoiser is not available or inputs are invalid
        """
        # Check if OptiX is available
        if not OPTIX_AVAILABLE:
            logger.warning("OptiX denoiser not available, returning original image")
            return image if isinstance(image, mi.Bitmap) else mi.Bitmap(image)
            
        # Handle string paths
        if isinstance(image, str):
            image = mi.Bitmap(image)
            
        # Get image shape
        if hasattr(image, 'size'):
            size = image.size()
        elif hasattr(image, 'shape') and len(image.shape) >= 2:
            size = image.shape[:2]
        else:
            raise ValueError(f"Invalid image format, cannot determine size: {type(image)}")
        
        logger.info(f"Denoising image of size {size}")
        
        # Use the simplest, most basic approach for maximum reliability
        logger.info("Using basic denoising without guide buffers")
        
        try:
            # Create a basic denoiser directly - not from cache
            logger.info("Creating basic denoiser with no guide buffers")
            denoiser = mi.OptixDenoiser(
                input_size=size,
                albedo=False,
                normals=False,
                temporal=False
            )
            
            # Apply denoising using the simplest approach
            logger.info("Denoising image with basic OptiX denoiser")
            denoised = denoiser(image)
            logger.info("Denoising completed successfully")
            
            return denoised
                
        except Exception as e:
            logger.error(f"Error in OptiX denoising: {e}", exc_info=True)
            return image  # Return original image on error
    
    @staticmethod
    @timeit(log_level="info")  
    def denoise_exr_file(
        input_path: str,
        output_path: Optional[str] = None,
        denoiser_strength: float = 1.0,
        to_sensor: Optional[Any] = None
    ) -> str:
        """
        Denoise an EXR file and save the result.
        
        Args:
            input_path: Path to input EXR file
            output_path: Path for output denoised EXR (defaults to input_path + '_denoised.exr')
            denoiser_strength: Strength of denoising (0.0 to 1.0)
            to_sensor: Optional transform to convert normals to sensor space
            
        Returns:
            Path to denoised EXR file
            
        Raises:
            FileNotFoundError: If input file not found
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input EXR file not found: {input_path}")
            
        # Default output path if not specified
        if output_path is None:
            input_name = os.path.splitext(input_path)[0]
            output_path = f"{input_name}_denoised.exr"
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Load image
        try:
            image = mi.Bitmap(input_path)
            logger.debug(f"Loaded EXR with {image.channel_count() if hasattr(image, 'channel_count') else 'unknown'} channels")
        except Exception as e:
            logger.error(f"Failed to load EXR file {input_path}: {e}")
            raise
            
        # Try to extract albedo and normal buffers if this is a multichannel EXR
        albedo = None
        normals = None
        
        if hasattr(image, 'channel_count') and image.channel_count() > 3:
            try:
                # Split the multichannel EXR to get individual buffers
                channels = dict(image.split())
                logger.debug(f"Found channels: {list(channels.keys())}")
                
                # Look for albedo buffer
                if 'albedo' in channels:
                    albedo = channels['albedo']
                    logger.debug("Found albedo buffer in EXR")
                
                # Look for normals buffer
                if 'sh_normal' in channels:
                    normals = channels['sh_normal']
                    logger.debug("Found normals buffer in EXR")
                elif 'normals' in channels:
                    normals = channels['normals']
                    logger.debug("Found normals buffer in EXR")
                
                # Use the root channel as the main image
                if '<root>' in channels:
                    image = channels['<root>']
            except Exception as e:
                logger.warning(f"Error splitting multichannel EXR: {e}")
            
        # Denoise image
        denoised = DenoiserManager.denoise_image_safe(
            image=image, 
            albedo=albedo,
            normals=normals,
            denoiser_strength=denoiser_strength,
            to_sensor=to_sensor
        )
        
        # Save denoised image
        denoised.write(output_path)
        logger.info(f"Denoised image saved to {output_path}")
        
        return output_path
    
    @staticmethod
    def denoise_image_safe(
        image: Union[mi.Bitmap, str],
        albedo: Optional[Union[mi.Bitmap, str]] = None,
        normals: Optional[Union[mi.Bitmap, str]] = None, 
        denoiser_strength: float = 1.0,
        prev_frame: Optional[Union[mi.Bitmap, str]] = None,
        optical_flow: Optional[np.ndarray] = None,
        to_sensor: Optional[Any] = None
    ) -> mi.Bitmap:
        """
        Safely denoise an image, falling back to the original if denoising fails.
        
        This simplified version prioritizes reliability over advanced features.
        
        Args:
            image: Input image as Mitsuba Bitmap or path to EXR file
            albedo: Optional albedo buffer for improved denoising (not used in basic mode)
            normals: Optional surface normals buffer for improved denoising (not used in basic mode)
            denoiser_strength: Strength of denoising (0.0 to 1.0)
            prev_frame: Optional previous frame for temporal denoising (not used in basic mode)
            optical_flow: Optional optical flow for temporal denoising (not used in basic mode)
            to_sensor: Optional transform to convert normals to sensor space
            
        Returns:
            Denoised image as Mitsuba Bitmap, or original image if denoising fails
        """
        # Handle string paths
        if isinstance(image, str):
            try:
                image = mi.Bitmap(image)
            except Exception as e:
                logger.error(f"Failed to load image from {image}: {e}")
                return image  # Return the original path
        
        # Check if OptiX is available
        if not OPTIX_AVAILABLE:
            logger.warning("OptiX denoiser not available - skipping denoising")
            return image
        
        # Log the denoising attempt
        logger.info(f"Attempting to denoise image with strength={denoiser_strength}")
        
        # Use simplest approach for maximum reliability
        try:
            # Basic denoising - just the image without guide buffers
            logger.info("Using basic denoising without guide buffers")
            
            # Get image shape first
            if hasattr(image, 'size'):
                size = image.size()
            elif hasattr(image, 'shape') and len(image.shape) >= 2:
                size = image.shape[:2]
            else:
                logger.error(f"Cannot determine image size from {type(image)}")
                return image
            
            # Create simple denoiser directly - not from cache
            try:
                logger.info(f"Creating basic denoiser for image size {size}")
                denoiser = mi.OptixDenoiser(
                    input_size=size,
                    albedo=False,
                    normals=False,
                    temporal=False
                )
                
                # Simple denoise call
                logger.info("Applying basic denoising")
                denoised = denoiser(image)
                logger.info("Basic denoising succeeded")
                return denoised
            except Exception as e:
                logger.error(f"Basic denoising failed: {e}")
                return image
                
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            logger.debug(f"Error details:", exc_info=True)
            return image  # Return the original image if denoising fails

# Make sure to export a simulated AVAILABLE_DENOISERS for backwards compatibility with older code
AVAILABLE_DENOISERS = {"optix": OPTIX_AVAILABLE}

