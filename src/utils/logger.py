"""
Module: utils.logger

Description:
    Logging configuration and utilities for the RAG system

Inputs:
    - None (module-level configuration)

Outputs:
    - Configured logger instances

Dependencies:
    - logging
    - sys
    - os

Usage:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the entire application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format string

    Returns:
        None

    Example:
        >>> setup_logging(log_level="INFO", log_file="app.log")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[]
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Example:
        >>> class MyClass(LoggerMixin):
        ...     def my_method(self):
        ...         self.logger.info("Method called")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class instance."""
        return logging.getLogger(self.__class__.__name__)
