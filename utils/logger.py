import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from utils.config import LOGS_DIR

# Global log file configuration
LOG_FILE_PATH: Path = LOGS_DIR / "cortex_ai.log"
MAX_LOG_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT: int = 5

# Define a standard, clean log format
LOG_FORMAT: str = "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s"

def setup_logger(name: str = "cortex_ai", log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with both console and rotating file handlers.
    
    Args:
        name (str): Name of the logger, typically __name__.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times for the same name
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT)
    
    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # 2. Rotating File Handler
    try:
        file_handler = RotatingFileHandler(
            filename=LOG_FILE_PATH,
            maxBytes=MAX_LOG_FILE_SIZE_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fall back to printing a warning if file logging fails to initialize
        print(f"Warning: Failed to initialize file logger at {LOG_FILE_PATH}: {e}", file=sys.stderr)
        
    return logger
