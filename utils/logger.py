"""
Logging configuration for the Code Explainer Agent.
"""
import logging
import sys
from typing import Optional

def setup_logging(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Name of the logger.
        log_level: Logging level (default: logging.INFO).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger

def set_log_level(level: str) -> None:
    """Set the log level for all loggers.
    
    Args:
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level)
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
