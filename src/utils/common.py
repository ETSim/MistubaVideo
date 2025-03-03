"""
Common utility functions shared across multiple modules.
"""

import os
import sys
import platform
import subprocess
import importlib.util
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, TypeVar
from pathlib import Path
import shutil

# Import necessary constants
from src.config.constants import FFMPEG_PATH

# Define common type variables
F = TypeVar('F', bound=Callable[..., Any])

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a simple logger - used during early initialization before RichLogger is available.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Simple logger for common module
logger = get_logger("mitsuba_app.common")

def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists at the specified path.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

# Alias for backward compatibility
create_directory = ensure_directory

def run_subprocess(
    command: List[str], 
    timeout: int = 5, 
    capture_output: bool = True,
    check: bool = False
) -> Tuple[int, str, str]:
    """
    Run a subprocess command and return its results safely.
    
    Args:
        command: Command list to execute
        timeout: Maximum time to wait for command to complete
        capture_output: Whether to capture stdout and stderr
        check: Whether to raise an exception on non-zero return code
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    stdout, stderr = "", ""
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            timeout=timeout,
            check=check
        )
        
        return_code = result.returncode
        if capture_output:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            
        return return_code, stdout, stderr
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out after {timeout}s: {' '.join(command)}")
        return 1, stdout, f"Command timed out after {timeout}s"
    except subprocess.SubprocessError as e:
        logger.error(f"Error executing command: {' '.join(command)} - {e}")
        return 1, stdout, str(e)
    except Exception as e:
        logger.error(f"Unexpected error running command: {' '.join(command)} - {e}")
        return 1, stdout, str(e)

def check_package_availability(package_name: str) -> bool:
    """
    Check if a Python package is available.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if the package is available, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None

def check_packages_availability(packages: List[str]) -> Dict[str, bool]:
    """
    Check if required Python packages are available.
    
    Args:
        packages: List of package names to check.
        
    Returns:
        Dictionary mapping package names to availability status.
    """
    availability = {}
    
    for package in packages:
        is_available = importlib.util.find_spec(package) is not None
        availability[package] = is_available
        
        if not is_available:
            logger.warning(f"Required package '{package}' is not installed.")
    
    return availability

def get_system_info() -> Dict[str, Any]:
    """
    Get detailed information about the current system.
    
    Returns:
        Dictionary containing system information
    """
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "python_path": sys.executable,
        "cpu_count": os.cpu_count(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "machine": platform.machine(),
        "gpu_info": get_gpu_info(),
        "ffmpeg": is_ffmpeg_available(),
        "mitsuba": is_mitsuba_available()
    }

def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary containing GPU information if available.
    """
    gpu_info = {"available": False, "devices": []}
    
    if platform.system() == "Windows":
        # Try to get NVIDIA GPU info using nvidia-smi
        return_code, stdout, _ = run_subprocess(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]
        )
        
        if return_code == 0:
            lines = stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header
                gpu_info["available"] = True
                for i, line in enumerate(lines[1:]):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_info["devices"].append({
                            "id": i,
                            "name": parts[0],
                            "memory": parts[1],
                            "driver": parts[2]
                        })
    elif platform.system() == "Linux":
        # Check for NVIDIA GPUs on Linux
        return_code, stdout, _ = run_subprocess(["lspci", "-nn"])
        if return_code == 0 and "NVIDIA" in stdout:
            gpu_info["available"] = True
            # Could parse more detailed info if needed
            
    return gpu_info

def is_mitsuba_available() -> bool:
    """
    Check if Mitsuba is available in the current environment.
    
    Returns:
        True if Mitsuba is available, False otherwise.
    """
    return check_package_availability("mitsuba")

def is_ffmpeg_available() -> bool:
    """
    Check if FFmpeg is available in the system path.
    
    Returns:
        True if FFmpeg is available, False otherwise.
    """
    try:
        # Try to get ffmpeg path from config first
        from src.config.config import config_class as config
        ffmpeg_path = getattr(config, "FFMPEG_PATH", FFMPEG_PATH)
        
        # Fallback to just "ffmpeg" if no path is configured
        if not ffmpeg_path:
            ffmpeg_path = "ffmpeg"
            
        return_code, _, _ = run_subprocess([ffmpeg_path, "-version"])
        return return_code == 0
    except Exception:
        return False

def copy_file(source: str, destination: str, overwrite: bool = True) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if the file was copied successfully, False otherwise
    """
    if not os.path.exists(source):
        logger.error(f"Source file does not exist: {source}")
        return False
        
    if os.path.exists(destination) and not overwrite:
        logger.debug(f"Destination file already exists: {destination}")
        return True
        
    try:
        # Ensure the destination directory exists
        ensure_directory(os.path.dirname(destination))
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error copying file from {source} to {destination}: {e}")
        return False

def check_python_version(min_version: Tuple[int, int, int] = (3, 7, 0)) -> bool:
    """
    Check if the current Python version meets the minimum requirements.
    
    Args:
        min_version: Minimum required Python version as a tuple (major, minor, micro).
        
    Returns:
        True if the Python version is compatible, False otherwise.
    """
    current_version = sys.version_info[:3]
    is_compatible = current_version >= min_version
    
    if not is_compatible:
        logger.warning(f"Python version {'.'.join(map(str, current_version))} "
                      f"is below the minimum required {'.'.join(map(str, min_version))}")
    
    return is_compatible

def check_mitsuba_availability() -> Dict[str, bool]:
    """
    Check if Mitsuba is available and which variants are available.
    
    Returns:
        Dictionary containing Mitsuba availability information.
    """
    result = {
        "available": False,
        "variants": {}
    }
    
    # Check if mitsuba can be imported
    if importlib.util.find_spec("mitsuba") is None:
        logger.warning("Mitsuba package is not installed or not in PYTHONPATH.")
        return result
    
    # Mitsuba is available, check which variants are available
    result["available"] = True
    
    for variant in AVAILABLE_VARIANTS:
        try:
            # Try to dynamically set the variant
            import mitsuba
            mitsuba.set_variant(variant)
            result["variants"][variant] = True
        except Exception:
            result["variants"][variant] = False
    
    return result

def check_ffmpeg_availability() -> Dict[str, bool]:
    """
    Check if FFmpeg is available and functioning.
    
    Returns:
        Dictionary containing FFmpeg availability information.
    """
    result = {
        "available": False,
        "version": None,
        "path": FFMPEG_PATH
    }
    
    try:
        # Run FFmpeg version command
        ffmpeg_process = subprocess.run(
            [FFMPEG_PATH, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        
        if ffmpeg_process.returncode == 0:
            result["available"] = True
            # Extract version information from output
            version_line = ffmpeg_process.stdout.split('\n')[0]
            if "ffmpeg version" in version_line:
                result["version"] = version_line.split("ffmpeg version")[1].strip().split()[0]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    
    return result

def check_cuda_availability() -> Dict[str, bool]:
    """
    Check if CUDA is available.
    
    Returns:
        Dictionary containing CUDA availability information.
    """
    result = {
        "available": False,
        "version": None,
        "nvcc_available": False
    }
    
    # Check if CUDA Python packages are available
    cuda_packages = ["torch.cuda", "pycuda", "cupy"]
    for package in cuda_packages:
        try:
            spec = importlib.util.find_spec(package.split('.')[0])
            if spec is not None:
                if package == "torch.cuda":
                    import torch
                    result["available"] = torch.cuda.is_available()
                    if result["available"]:
                        result["version"] = torch.version.cuda
                        break
                elif package == "pycuda":
                    import pycuda.driver as cuda
                    cuda.init()
                    result["available"] = True
                    result["version"] = ".".join(map(str, cuda.get_version()))
                    break
                elif package == "cupy":
                    import cupy
                    result["available"] = cupy.cuda.is_available()
                    if result["available"]:
                        result["version"] = cupy.cuda.runtime.runtimeGetVersion()
                        break
        except ImportError:
            continue
        except Exception:
            # Other exceptions like CUDA initialization errors
            continue
    
    # Check if nvcc is available
    try:
        nvcc_process = subprocess.run(
            ["nvcc", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=5
        )
        if nvcc_process.returncode == 0:
            result["nvcc_available"] = True
            # Get CUDA version from nvcc output
            if not result["version"]:
                version_line = nvcc_process.stdout.split('\n')[3]
                if "release" in version_line:
                    result["version"] = version_line.split("release")[1].strip().split(",")[0].strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    
    return result

def check_system_compatibility() -> Dict[str, bool]:
    """
    Run all system compatibility checks.
    
    Returns:
        Dictionary containing results of all compatibility checks.
    """
    compatibility = {
        "python": check_python_version(),
        "packages": check_packages_availability([
            "numpy", "rich", "pillow", "tqdm", "jsonschema"
        ]),
        "mitsuba": check_mitsuba_availability(),
        "ffmpeg": check_ffmpeg_availability(),
        "cuda": check_cuda_availability(),
        "system": {
            "platform": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count()
        }
    }
    
    return compatibility

def display_compatibility_report(compatibility: Optional[Dict[str, bool]] = None) -> None:
    """
    Display a formatted report of system compatibility.
    
    Args:
        compatibility: Compatibility check results. If None, checks will be run.
    """
    if compatibility is None:
        compatibility = check_system_compatibility()
    
    console.print(Panel("[bold blue]Mitsuba System Compatibility Report[/bold blue]", 
                        border_style="blue"))
    
    # System Information
    sys_info = compatibility["system"]
    sys_table = Table(title="System Information", show_header=True, header_style="bold magenta")
    sys_table.add_column("Property", style="cyan")
    sys_table.add_column("Value", style="green")
    
    sys_table.add_row("Platform", sys_info["platform"])
    sys_table.add_row("Version", sys_info["version"])
    sys_table.add_row("Machine", sys_info["machine"])
    sys_table.add_row("Processor", sys_info["processor"])
    sys_table.add_row("CPU Count", str(sys_info["cpu_count"]))
    
    console.print(sys_table)
    console.print("")
    
    # Python Information
    py_table = Table(title="Python Environment", show_header=True, header_style="bold magenta")
    py_table.add_column("Component", style="cyan")
    py_table.add_column("Status", style="green")
    py_table.add_column("Version", style="blue")
    
    py_status = "✅" if compatibility["python"] else "❌"
    py_table.add_row("Python", py_status, platform.python_version())
    
    # Packages
    for package, available in compatibility["packages"].items():
        pkg_status = "✅" if available else "❌"
        version = ""
        if available:
            try:
                pkg = __import__(package)
                if hasattr(pkg, "__version__"):
                    version = pkg.__version__
                elif hasattr(pkg, "version"):
                    version = pkg.version
            except (ImportError, AttributeError):
                pass
        py_table.add_row(package, pkg_status, version)
    
    console.print(py_table)
    console.print("")
    
    # Mitsuba
    mitsuba_info = compatibility["mitsuba"]
    mitsuba_table = Table(title="Mitsuba Availability", show_header=True, header_style="bold magenta")
    mitsuba_table.add_column("Variant", style="cyan")
    mitsuba_table.add_column("Available", style="green")
    
    mitsuba_available = mitsuba_info["available"]
    main_status = "✅" if mitsuba_available else "❌"
    mitsuba_table.add_row("Mitsuba (Core)", main_status)
    
    if mitsuba_available and "variants" in mitsuba_info:
        for variant, available in mitsuba_info["variants"].items():
            var_status = "✅" if available else "❌"
            mitsuba_table.add_row(variant, var_status)
    
    console.print(mitsuba_table)
    console.print("")
    
    # CUDA
    cuda_info = compatibility["cuda"]
    cuda_table = Table(title="CUDA Availability", show_header=True, header_style="bold magenta")
    cuda_table.add_column("Component", style="cyan")
    cuda_table.add_column("Status", style="green")
    cuda_table.add_column("Version", style="blue")
    
    cuda_status = "✅" if cuda_info["available"] else "❌"
    cuda_version = cuda_info["version"] if cuda_info["version"] else "N/A"
    cuda_table.add_row("CUDA Runtime", cuda_status, cuda_version)
    
    nvcc_status = "✅" if cuda_info["nvcc_available"] else "❌"
    cuda_table.add_row("NVCC Compiler", nvcc_status, "")
    
    console.print(cuda_table)
    console.print("")
    
    # FFmpeg
    ffmpeg_info = compatibility["ffmpeg"]
    ffmpeg_table = Table(title="FFmpeg Availability", show_header=True, header_style="bold magenta")
    ffmpeg_table.add_column("Component", style="cyan")
    ffmpeg_table.add_column("Status", style="green")
    ffmpeg_table.add_column("Version", style="blue")
    ffmpeg_table.add_column("Path", style="yellow")
    
    ffmpeg_status = "✅" if ffmpeg_info["available"] else "❌"
    ffmpeg_version = ffmpeg_info["version"] if ffmpeg_info["version"] else "N/A"
    ffmpeg_path = ffmpeg_info["path"] if ffmpeg_info["path"] else "N/A"
    ffmpeg_table.add_row("FFmpeg", ffmpeg_status, ffmpeg_version, ffmpeg_path)
    
    console.print(ffmpeg_table)
    
    # Overall assessment
    console.print("")
    overall_compatible = (
        compatibility["python"] and
        all(compatibility["packages"].values()) and
        mitsuba_available and
        ffmpeg_info["available"]
    )