import logging
import os
import sys
import datetime
from pathlib import Path
from typing import Union

_active_log_file = None

def setup_logging(log_level: Union[str, int] = "INFO") -> str:
    """
    Configures logging to both console and a timestamped file in var/logs.
    Ensures setup only happens once per execution.
    """
    global _active_log_file
    if _active_log_file is not None:
        return _active_log_file

    if isinstance(log_level, int):
        numeric_level = log_level
    else:
        numeric_level = getattr(logging, str(log_level).upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO

    # Resolve absolute path to the project root (3 levels up from src/utils/logging_config.py)
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "var" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"execution_{timestamp}.log"

    # Configure logging with both StreamHandler (console) and FileHandler
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Use force=True in basicConfig to override any existing configuration
    # Note: force=True is available in Python 3.8+
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ],
        force=True
    )

    _active_log_file = str(log_file)
    logging.getLogger(__name__).info("Logging initialized. File: %s", _active_log_file)
    return _active_log_file
