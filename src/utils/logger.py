import logging
import os
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json

# Install rich traceback handler for improved exception display
install(show_locals=True)

class RichLogger:
    """
    A logger class that uses Rich for pretty, formatted console output.
    """
    
    # Class variable to store loggers by name
    _loggers = {}
    
    # Default theme configuration
    DEFAULT_THEME = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "success": "green bold",
        "timestamp": "dim",
        "function": "bright_blue",
        "module": "magenta"
    }
    
    # Store file handlers to avoid duplicate handlers
    _file_handlers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_level: int = logging.INFO, log_file: Optional[str] = None, theme: Optional[Dict[str, str]] = None) -> logging.Logger:
        """
        Get a configured logger instance.
        
        Args:
            name: The name for the logger
            log_level: Logging level. Defaults to logging.INFO.
            log_file: Optional path to write logs to a file
            theme: Optional custom theme overrides
            
        Returns:
            logging.Logger: A configured logger instance
        """
        # Return existing logger if already created
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Configure rich console with theme
        final_theme = cls.DEFAULT_THEME.copy()
        if theme:
            final_theme.update(theme)
            
        console = Console(theme=Theme(final_theme))
        
        # Configure handler
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            show_time=True,
            show_path=True
        )
        
        # Format without timestamp since Rich adds it
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        
        # Get and configure the logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # Remove any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
            
        # Add our rich handler
        logger.addHandler(rich_handler)
        
        # Add file handler if requested
        if log_file:
            cls.add_file_handler(log_file, logger)
        
        # Add success level
        logging.SUCCESS = 25  # between INFO and WARNING
        logging.addLevelName(logging.SUCCESS, "SUCCESS")
        
        def success(self, message, *args, **kwargs):
            self.log(logging.SUCCESS, message, *args, **kwargs)
            
        logger.success = success.__get__(logger)
        
        # Store in class dict
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def add_file_handler(cls, log_file: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Add a file handler to the logger or to all existing loggers.
        
        Args:
            log_file: Path to the log file
            logger: Specific logger instance or None to add to all loggers
        """
        # Check if we already have this file handler
        if log_file in cls._file_handlers:
            return
            
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create file handler with detailed formatting
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # Store the handler for later reference
            cls._file_handlers[log_file] = file_handler
            
            # Add to specific logger or all loggers
            if logger:
                logger.addHandler(file_handler)
            else:
                for logger_instance in cls._loggers.values():
                    logger_instance.addHandler(file_handler)
                    
        except Exception as e:
            print(f"Failed to set up file logging to {log_file}: {e}")
    
    @classmethod
    def create_structured_log(cls, 
                            category: str, 
                            data: Dict[str, Any], 
                            log_dir: str = "logs",
                            prefix: Optional[str] = None) -> str:
        """
        Create a structured log file for specific events like rendering timing.
        
        Args:
            category: Category of the log (frame, quality, view)
            data: Data to log
            log_dir: Directory for log files
            prefix: Optional prefix for the log filename
            
        Returns:
            Path to the created log file
        """
        try:
            # Ensure the log directory exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix_str = f"{prefix}_" if prefix else ""
            filename = f"{prefix_str}{category}_{timestamp}.json"
            log_path = os.path.join(log_dir, filename)
            
            # Add metadata
            data["_metadata"] = {
                "timestamp": timestamp,
                "category": category,
                "created_at": datetime.now().isoformat()
            }
            
            # Write the log file
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            return log_path
            
        except Exception as e:
            print(f"Failed to create structured log: {e}")
            return ""
    
    @classmethod
    def log_timing_data(cls, 
                       frame: int, 
                       view: str, 
                       quality: str, 
                       elapsed: float,
                       output_dir: str,
                       additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log detailed timing data for a specific rendering task.
        
        Args:
            frame: Frame number
            view: Camera view name
            quality: Quality preset
            elapsed: Elapsed time in seconds
            output_dir: Output directory for logs
            additional_data: Any additional data to include in the log
        """
        try:
            # Create logs directory in output folder
            logs_dir = os.path.join(output_dir, "logs", "timing")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create the timing data structure
            timing_data = {
                "frame": frame,
                "view": view,
                "quality": quality,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": str(timedelta(seconds=elapsed)),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional data if provided
            if additional_data:
                timing_data.update(additional_data)
            
            # Create log files for different aggregations
            
            # Per-frame log
            frame_log = os.path.join(logs_dir, f"frame_{frame}.json")
            cls._append_to_json_log(frame_log, timing_data)
            
            # Per-quality log
            quality_log = os.path.join(logs_dir, f"quality_{quality}.json")
            cls._append_to_json_log(quality_log, timing_data)
            
            # Per-view log
            view_log = os.path.join(logs_dir, f"view_{view}.json")
            cls._append_to_json_log(view_log, timing_data)
            
            # Combined log
            combined_log = os.path.join(logs_dir, "all_timings.json")
            cls._append_to_json_log(combined_log, timing_data)
            
        except Exception as e:
            print(f"Failed to log timing data: {e}")
    
    @staticmethod
    def _append_to_json_log(log_file: str, data: Dict[str, Any]) -> None:
        """
        Append data to a JSON log file, creating it if it doesn't exist.
        
        Args:
            log_file: Path to the log file
            data: Data to append
        """
        try:
            # Load existing data or create new structure
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    # If it's a dict with 'entries', append to entries list
                    if isinstance(existing_data, dict) and 'entries' in existing_data:
                        existing_data['entries'].append(data)
                        existing_data['count'] = len(existing_data['entries'])
                        existing_data['last_updated'] = datetime.now().isoformat()
                    else:
                        # Convert to new format
                        existing_data = {
                            'count': 1,
                            'created_at': datetime.now().isoformat(),
                            'last_updated': datetime.now().isoformat(),
                            'entries': [data]
                        }
                except:
                    # If file exists but is invalid, create new structure
                    existing_data = {
                        'count': 1,
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat(),
                        'entries': [data]
                    }
            else:
                # Create new log structure
                existing_data = {
                    'count': 1,
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'entries': [data]
                }
            
            # Write updated data back to the file
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to append to JSON log {log_file}: {e}")

    @classmethod
    def set_level(cls, level_name: str) -> None:
        """
        Set the log level for all existing loggers.
        
        Args:
            level_name: The name of the log level to set (debug, info, warning, error)
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        
        level = level_map.get(level_name.lower(), logging.INFO)
        
        for logger in cls._loggers.values():
            logger.setLevel(level)

    @classmethod
    def configure_from_settings(cls, settings: Dict[str, Any]) -> None:
        """
        Configure all logging from a settings dictionary (typically from config).
        
        Args:
            settings: Dictionary with logging settings
        """
        # Extract config values
        log_file = settings.get("file")
        log_level = settings.get("level", "INFO")
        console_enabled = settings.get("console", True)
        debug_settings = settings.get("debug", {})
        
        # Set up logging level
        cls.set_level(log_level)
        
        # Set up file logging if enabled
        if log_file:
            cls.add_file_handler(log_file)
        
        # Configure exception handling
        show_locals = debug_settings.get("show_locals", True)
        from rich.traceback import install as install_traceback
        install_traceback(show_locals=show_locals)
        
        # Log configuration if requested
        if debug_settings.get("log_config", False):
            logger = cls.get_logger("mitsuba_app.config")
            logger.debug("Logging configuration:")
            logger.debug(f"  Level: {log_level}")
            logger.debug(f"  File: {log_file or 'None'}")
            logger.debug(f"  Console: {'Enabled' if console_enabled else 'Disabled'}")
            logger.debug(f"  Debug: {'Enabled' if debug_settings.get('enabled', False) else 'Disabled'}")
            logger.debug(f"  Detailed logs: {'Enabled' if debug_settings.get('detailed', False) else 'Disabled'}")