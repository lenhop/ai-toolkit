"""
Logging utilities for AI Toolkit.

This module provides convenient logging setup and utilities.

Functions:
    setup_logger(name, level, log_file, log_format, date_format, console, file_mode, max_bytes, backup_count, rotation): Setup and configure a logger
    get_logger(name): Get an existing logger or create a new one
    log_info(message, logger): Log an info message
    log_error(message, logger, exc_info): Log an error message
    log_warning(message, logger): Log a warning message
    log_debug(message, logger): Log a debug message
    log_critical(message, logger): Log a critical message
    set_log_level(level, logger): Set logging level
    disable_logging(logger): Disable logging
    enable_logging(logger): Enable logging
"""

import logging
import sys
from typing import Optional, Union
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name: str = 'ai_toolkit',
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    console: bool = True,
    file_mode: str = 'a',
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    rotation: Optional[str] = None  # 'size' or 'time'
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format string
        date_format: Custom date format string
        console: Whether to log to console
        file_mode: File mode for log file ('a' for append, 'w' for write)
        max_bytes: Maximum bytes per log file (for rotation)
        backup_count: Number of backup files to keep
        rotation: Rotation type ('size' or 'time')
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if date_format is None:
        date_format = '%Y-%m-%d %H:%M:%S'
    
    formatter = logging.Formatter(log_format, date_format)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if rotation == 'size':
            file_handler = RotatingFileHandler(
                log_file,
                mode=file_mode,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        elif rotation == 'time':
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file, mode=file_mode)
        
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'ai_toolkit') -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_info(message: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an info message.
    
    Args:
        message: Message to log
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    logger.info(message)


def log_error(message: str, logger: Optional[logging.Logger] = None, 
              exc_info: bool = False) -> None:
    """
    Log an error message.
    
    Args:
        message: Message to log
        logger: Logger instance (uses default if None)
        exc_info: Whether to include exception info
    """
    if logger is None:
        logger = get_logger()
    logger.error(message, exc_info=exc_info)


def log_warning(message: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a warning message.
    
    Args:
        message: Message to log
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    logger.warning(message)


def log_debug(message: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a debug message.
    
    Args:
        message: Message to log
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    logger.debug(message)


def log_critical(message: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a critical message.
    
    Args:
        message: Message to log
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    logger.critical(message)


def set_log_level(level: Union[int, str], logger: Optional[logging.Logger] = None) -> None:
    """
    Set logging level.
    
    Args:
        level: Logging level
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging(logger: Optional[logging.Logger] = None) -> None:
    """
    Disable logging.
    
    Args:
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    logger.disabled = True


def enable_logging(logger: Optional[logging.Logger] = None) -> None:
    """
    Enable logging.
    
    Args:
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    logger.disabled = False
