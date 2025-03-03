import mitsuba as mi
from typing import List, Optional, Tuple, Dict, Any
from src.utils.logger import RichLogger
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.mitsuba_utils")

class MitsubaUtils:
    """Helper class for common Mitsuba operations."""
    
    @staticmethod
    @timeit(log_level="debug")
    def set_variant(device_type: str) -> Tuple[bool, str, str]:
        """
        Set the Mitsuba variant based on the device type.
        
        Args:
            device_type: The device type ('cpu' or 'cuda')
            
        Returns:
            Tuple of (success, variant, message)
        """
        success = True
        message = ""
        variant = "scalar_rgb"  # Default variant
        
        try:
            available_variants = mi.variants()
            logger.debug(f"Available Mitsuba variants: {', '.join(available_variants)}")
            
            if device_type.lower() == "cpu":
                mi.set_variant("scalar_rgb")
                message = f"Using CPU rendering with variant: scalar_rgb"
            elif device_type.lower() == "cuda":
                # Try different CUDA variants in order of preference
                cuda_variants = [
                    "cuda_rgb",            # First choice
                    "cuda_ad_rgb",         # Second choice
                    "cuda_spectral",       # Third choice
                    "cuda_ad_spectral",    # Fourth choice
                    "gpu_rgb",             # Legacy name
                    "gpu_spectral"         # Legacy name
                ]
                
                selected_variant = None
                for cuda_variant in cuda_variants:
                    if cuda_variant in available_variants:
                        selected_variant = cuda_variant
                        break
                        
                if selected_variant:
                    mi.set_variant(selected_variant)
                    variant = selected_variant
                    message = f"Using GPU rendering with variant: {selected_variant}"
                else:
                    message = f"No CUDA variants available. Falling back to CPU rendering."
                    logger.warning(message)
                    logger.debug(f"Available variants: {', '.join(available_variants)}")
                    mi.set_variant("scalar_rgb")
                    success = False
            else:
                message = f"Invalid device type '{device_type}'. Using default CPU rendering."
                logger.warning(message)
                mi.set_variant("scalar_rgb")
                success = False
                
            logger.debug(f"Set Mitsuba variant to: {mi.variant()}")
            return success, variant, message
            
        except ImportError as e:
            message = f"Mitsuba import error: {e}. Falling back to CPU rendering."
            logger.error(message)
            mi.set_variant("scalar_rgb")
            return False, "scalar_rgb", message
        except Exception as e:
            message = f"Error setting Mitsuba variant: {e}. Falling back to CPU rendering."
            logger.error(message)
            mi.set_variant("scalar_rgb")
            return False, "scalar_rgb", message
    
    @staticmethod
    @timeit(log_level="debug")
    def get_available_variants() -> List[str]:
        """Get a list of available Mitsuba variants."""
        return mi.variants()
    
    @staticmethod
    @timeit(log_level="debug")
    def render_scene(scene: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Render a Mitsuba scene with optional custom parameters.
        
        Args:
            scene: The Mitsuba scene to render
            params: Optional parameters dictionary
            
        Returns:
            Rendered image as Mitsuba Bitmap
        """
        import mitsuba as mi
        import drjit as dr
        
        try:
            # Determine rendering parameters
            if params is None:
                # If no parameters provided, render with defaults
                result = mi.render(scene)
            else:
                # Convert dictionary params to SceneParameters object
                scene_params = mi.SceneParameters()
                
                # Handle SPP (samples per pixel)
                if 'spp' in params:
                    scene_params.spp = params['spp']
                    logger.debug(f"Setting SPP to {params['spp']}")
                
                # Handle AOV (Arbitrary Output Variable) for denoising buffers
                if 'aovs' in params:
                    # Configure AOVs
                    if params['aovs'] == 'albedo':
                        scene_params.aovs = ['albedo']
                        logger.debug("Rendering albedo buffer")
                    elif params['aovs'] == 'normal':
                        scene_params.aovs = ['normal']
                        logger.debug("Rendering normal buffer")
                
                # Render with configured parameters
                logger.debug(f"Rendering scene with custom parameters")
                result = mi.render(scene, params=scene_params)
            
            # Check if result is a TensorXf (CUDA variant) and convert to bitmap if needed
            if hasattr(dr, 'TensorXf') and isinstance(result, dr.TensorXf):
                logger.debug("Converting CUDA tensor to Mitsuba bitmap")
                # Convert tensor to NumPy array and then to Mitsuba bitmap
                import numpy as np
                result_np = np.array(result)
                result_bitmap = mi.Bitmap(result_np)
                return result_bitmap
            elif hasattr(result, 'write'):
                # Already a bitmap or has write method, return as is
                return result
            else:
                # Unknown type, try to convert to bitmap
                logger.debug(f"Converting result of type {type(result).__name__} to bitmap")
                return mi.Bitmap(result)
                
        except Exception as e:
            logger.error(f"Error during scene rendering: {e}")
            # Re-raise to handle at higher level
            raise
    
    @staticmethod
    @timeit(log_level="debug")
    def load_scene_from_file(file_path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Load a Mitsuba scene from a file with optional parameters.
        
        Args:
            file_path: Path to scene file
            params: Optional dictionary of parameters to pass to the scene
            
        Returns:
            Loaded Mitsuba scene
        """
        logger.debug(f"Loading scene from: {file_path}")
        
        if params:
            logger.debug(f"Using scene parameters: {params}")
            return mi.load_file(file_path, **params)
        else:
            return mi.load_file(file_path)
    
    @staticmethod
    @timeit(log_level="debug")
    def load_scene_from_dict(scene_dict: Dict[str, Any]) -> Any:
        """Load a Mitsuba scene from a dictionary."""
        return mi.load_dict(scene_dict)
