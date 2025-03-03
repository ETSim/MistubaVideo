import time
import functools
import csv
import os
import logging
from datetime import datetime, timedelta
from typing import Callable, Any, TypeVar, cast, List, Optional, Tuple, Union
from pathlib import Path

from src.utils.logger import RichLogger

# Use proper logging level constants
logger = RichLogger.get_logger("mitsuba_app.timing", logging.INFO)

F = TypeVar('F', bound=Callable[..., Any])

def timeit(log_level: str = "debug", with_args: bool = False) -> Callable[[F], F]:
    """
    A decorator that logs the execution time of a function.
    
    Args:
        log_level: The logging level to use ('debug', 'info', 'warning', or 'error')
        with_args: Whether to include function arguments in the log message
        
    Returns:
        The decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            start_datetime = datetime.now()
            
            func_name = func.__qualname__
            
            # Log the function call
            args_str = ""
            if with_args:
                # Format args and kwargs for logging
                args_parts = [f"{arg!r}" for arg in args[1:]]  # Skip self for methods
                kwargs_parts = [f"{k}={v!r}" for k, v in kwargs.items()]
                all_args = args_parts + kwargs_parts
                args_str = f" with args: ({', '.join(all_args)})"
                
            logger.debug(f"Starting {func_name}{args_str}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Calculate the time taken
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed = timedelta(seconds=elapsed_time)
            
            # Choose the appropriate logging level
            if log_level.lower() == "debug":
                logger.debug(f"{func_name} completed in {elapsed} ({elapsed_time:.3f}s)")
            elif log_level.lower() == "info":
                logger.info(f"{func_name} completed in {elapsed} ({elapsed_time:.3f}s)")
            elif log_level.lower() == "warning":
                logger.warning(f"{func_name} completed in {elapsed} ({elapsed_time:.3f}s)")
            else:
                logger.debug(f"{func_name} completed in {elapsed} ({elapsed_time:.3f}s)")
            
            return result
        return cast(F, wrapper)
    return decorator


class TimingManager:
    """
    Manager class for timing-related operations.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def log_timing_statistics(timings: List[Tuple[int, str, float]], 
                             prefix: Optional[str] = None,
                             output_folder: Optional[str] = None) -> None:
        """
        Log statistics about frame rendering times and create a report.
        
        Args:
            timings: List of timing tuples (index, filename, seconds)
            prefix: Prefix for report filename (optional)
            output_folder: Output folder for reports (optional)
        """
        if not timings:
            logger.warning("No timing data available to log")
            return
        
        # Calculate statistics
        times = [t[2] for t in timings]
        total_time = sum(times)
        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Find fastest and slowest frames
        slowest_idx = times.index(max_time)
        fastest_idx = times.index(min_time)
        slowest_frame = timings[slowest_idx][1]
        fastest_frame = timings[fastest_idx][1]
        
        # Log the statistics
        logger.debug(f"Frame timing statistics:")
        logger.debug(f"  Total time: {timedelta(seconds=total_time)} ({total_time:.2f}s)")
        logger.debug(f"  Average time per frame: {timedelta(seconds=avg_time)} ({avg_time:.2f}s)")
        logger.debug(f"  Fastest frame: {fastest_frame} in {timedelta(seconds=min_time)} ({min_time:.2f}s)")
        logger.debug(f"  Slowest frame: {slowest_frame} in {timedelta(seconds=max_time)} ({max_time:.2f}s)")
        
        # Create a timing report file
        if output_folder:
            TimingManager.create_timing_report(timings, prefix, output_folder)
    
    @staticmethod
    @timeit(log_level="debug")
    def create_timing_report(timings: List[Tuple[int, str, float]], 
                            prefix: Optional[str] = None,
                            output_folder: str = "reports") -> str:
        """
        Create a CSV report file with frame timings.
        
        Args:
            timings: List of timing tuples (index, filename, seconds)
            prefix: Prefix for report filename
            output_folder: Output folder for reports
            
        Returns:
            Path to the created report file or empty string if reports are disabled
        """
        # Check if reporting is enabled
        from src.config.config import config_class as config
        if hasattr(config, 'ENABLE_REPORTS') and not config.ENABLE_REPORTS or \
           hasattr(config, 'ENABLE_TIMING_REPORT') and not config.ENABLE_TIMING_REPORT:
            logger.debug("Timing report generation skipped (disabled in config)")
            return ""
        
        try:
            # Use common utility for directory creation
            from src.utils.common import ensure_directory
            ensure_directory(output_folder)
            
            # Generate report filename
            report_name = f"{prefix}_timing_report.csv" if prefix else "timing_report.csv"
            report_file = os.path.join(output_folder, report_name)
            
            # Write CSV report
            with open(report_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Filename", "Seconds", "Formatted"])
                
                for i, filename, seconds in sorted(timings, key=lambda x: x[0]):
                    formatted = str(timedelta(seconds=seconds))
                    writer.writerow([i, filename, f"{seconds:.4f}", formatted])
            
            logger.debug(f"Timing report saved to {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to create timing report: {e}")
            return ""
