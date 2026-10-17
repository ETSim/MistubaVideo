"""
Module for image denoising capabilities in the Mitsuba renderer.

Supports the NVIDIA OptiX AI Denoiser for high-quality denoising of rendered images.
When albedo and shading-normal guides are available, they are passed through; without
them OptiX barely reduces Monte Carlo noise on these scenes.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
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
                    test_denoiser = mi.OptixDenoiser(input_size=(64, 64), albedo=True, normals=True)
                    OPTIX_AVAILABLE = True
                    logger.info("NVIDIA OptiX denoiser is available and working (with guide buffers)")
                except Exception:
                    try:
                        test_denoiser = mi.OptixDenoiser(input_size=(64, 64), albedo=False, normals=False)
                        OPTIX_AVAILABLE = True
                        logger.info("NVIDIA OptiX denoiser is available (beauty-only fallback)")
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


def _image_size(image: Any) -> Tuple[int, int]:
    if hasattr(image, "size"):
        size = image.size()
        return (int(size[0]), int(size[1]))
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return (int(image.shape[1]), int(image.shape[0]))  # (width, height) for OptiX
    raise ValueError(f"Invalid image format, cannot determine size: {type(image)}")


def _as_bitmap(image: Union[mi.Bitmap, str, np.ndarray]) -> mi.Bitmap:
    if isinstance(image, mi.Bitmap):
        return image
    if isinstance(image, str):
        return mi.Bitmap(image)
    return mi.Bitmap(np.asarray(image, dtype=np.float32))


class DenoiserManager:
    """
    Class to manage image denoising operations using the OptiX AI Denoiser.
    """
    
    # Cache for denoiser instances to avoid recreating them for same-sized images
    _denoiser_cache = {}
    # Store denoiser configuration along with the instance
    _denoiser_config = {}
    
    @staticmethod
    def _ensure_cuda_variant() -> bool:
        variants = mi.variants()
        cuda_variants = [v for v in variants if isinstance(v, str) and ("cuda" in v.lower() or "gpu" in v.lower())]
        if not cuda_variants:
            logger.warning("No CUDA variants available, OptiX denoiser cannot be used")
            return False
        if not any(x in mi.variant() for x in ["cuda", "gpu"]):
            for variant in ["cuda_ad_rgb", "cuda_rgb", "cuda_spectral"]:
                if variant in variants:
                    try:
                        mi.set_variant(variant)
                        logger.info(f"Switched to {variant} variant for OptiX denoising")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to set variant {variant}: {e}")
        logger.info(f"Current variant for denoising: {mi.variant()}")
        return True

    @staticmethod
    def _get_cached_denoiser(size: Tuple[int, int], use_albedo: bool = False, use_normals: bool = False, temporal: bool = False):
        """
        Get or create a denoiser with specific parameters, using cache to avoid recreation.
        
        Args:
            size: Image size (width, height)
            use_albedo: Whether to use albedo buffers
            use_normals: Whether to use normal buffers
            temporal: Whether to use temporal denoising (kept off for reliability)
            
        Returns:
            OptixDenoiser instance or None if creation fails
        """
        if temporal:
            logger.info("Disabling temporal denoising for reliability")
            temporal = False
            
        if not OPTIX_AVAILABLE:
            logger.info("OptiX denoiser not available")
            return None

        if not DenoiserManager._ensure_cuda_variant():
            return None
        
        cache_key = (size, use_albedo, use_normals, temporal)
        if cache_key not in DenoiserManager._denoiser_cache:
            try:
                logger.info(
                    f"Creating OptiX denoiser (size={size}, albedo={use_albedo}, normals={use_normals})"
                )
                DenoiserManager._denoiser_cache[cache_key] = mi.OptixDenoiser(
                    input_size=size,
                    albedo=use_albedo,
                    normals=use_normals,
                    temporal=False,
                )
                DenoiserManager._denoiser_config[cache_key] = {
                    "use_albedo": use_albedo,
                    "use_normals": use_normals,
                    "temporal": False,
                    "size": size,
                }
            except Exception as e:
                if use_albedo or use_normals:
                    logger.warning(f"Guided OptiX denoiser failed ({e}); retrying beauty-only")
                    return DenoiserManager._get_cached_denoiser(size, False, False, False)
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
    def _apply(
        image: mi.Bitmap,
        albedo: Optional[mi.Bitmap],
        normals: Optional[mi.Bitmap],
        to_sensor: Optional[Any],
    ) -> mi.Bitmap:
        size = _image_size(image)
        use_guides = albedo is not None and normals is not None
        denoiser = DenoiserManager._get_cached_denoiser(size, use_guides, use_guides, False)
        if denoiser is None:
            return image

        # Multichannel Bitmap path: channel names when the beauty EXR already holds AOVs.
        if hasattr(image, "channel_count") and image.channel_count() > 3 and use_guides:
            channels = dict(image.split())
            albedo_ch = "albedo" if "albedo" in channels else ""
            normals_ch = (
                "sh_normal" if "sh_normal" in channels
                else "normals" if "normals" in channels
                else ""
            )
            if albedo_ch and normals_ch:
                kwargs = {"albedo_ch": albedo_ch, "normals_ch": normals_ch}
                if to_sensor is not None:
                    kwargs["to_sensor"] = to_sensor
                return denoiser(image, **kwargs)

        # Separate beauty + guide buffers as tensors.
        noisy = convert_to_tensor(image)
        if use_guides:
            kwargs = {
                "albedo": convert_to_tensor(albedo),
                "normals": convert_to_tensor(normals),
            }
            if to_sensor is not None:
                kwargs["to_sensor"] = to_sensor
            out = denoiser(noisy, **kwargs)
        else:
            out = denoiser(noisy)
        return convert_from_tensor(out)
    
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
            denoiser_strength: Strength of denoising (0.0 to 1.0); reserved for blend
            prev_frame: Optional previous frame for temporal denoising (unused)
            optical_flow: Optional optical flow for temporal denoising (unused)
            to_sensor: Optional transform to convert normals to sensor space
            
        Returns:
            Denoised image as Mitsuba Bitmap
        """
        if not OPTIX_AVAILABLE:
            logger.warning("OptiX denoiser not available, returning original image")
            return image if isinstance(image, mi.Bitmap) else mi.Bitmap(image)

        image = _as_bitmap(image)
        albedo_bm = _as_bitmap(albedo) if albedo is not None else None
        normals_bm = _as_bitmap(normals) if normals is not None else None
        logger.info(
            f"Denoising image of size {_image_size(image)} "
            f"(guides={'on' if albedo_bm is not None and normals_bm is not None else 'off'})"
        )
        try:
            denoised = DenoiserManager._apply(image, albedo_bm, normals_bm, to_sensor)
            if denoiser_strength < 1.0:
                # Soft blend toward the noisy input when strength < 1.
                a = np.asarray(image, dtype=np.float32)[..., :3]
                b = np.asarray(denoised, dtype=np.float32)[..., :3]
                mixed = (1.0 - denoiser_strength) * a + denoiser_strength * b
                return mi.Bitmap(mixed)
            return denoised
        except Exception as e:
            logger.error(f"Error in OptiX denoising: {e}", exc_info=True)
            return image
    
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
        
        Args:
            image: Input image as Mitsuba Bitmap or path to EXR file
            albedo: Optional albedo buffer for improved denoising
            normals: Optional surface normals buffer for improved denoising
            denoiser_strength: Strength of denoising (0.0 to 1.0)
            prev_frame: Optional previous frame for temporal denoising (unused)
            optical_flow: Optional optical flow for temporal denoising (unused)
            to_sensor: Optional transform to convert normals to sensor space
            
        Returns:
            Denoised image as Mitsuba Bitmap, or original image if denoising fails
        """
        try:
            return DenoiserManager.denoise_image(
                image=image,
                albedo=albedo,
                normals=normals,
                denoiser_strength=denoiser_strength,
                prev_frame=prev_frame,
                optical_flow=optical_flow,
                to_sensor=to_sensor,
            )
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            logger.debug("Error details:", exc_info=True)
            if isinstance(image, str):
                try:
                    return mi.Bitmap(image)
                except Exception:
                    return image  # type: ignore[return-value]
            return image

# Make sure to export a simulated AVAILABLE_DENOISERS for backwards compatibility with older code
AVAILABLE_DENOISERS = {"optix": OPTIX_AVAILABLE}
