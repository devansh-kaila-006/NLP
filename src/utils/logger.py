"""
Logging utilities for RAG pipeline
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.config import LOGGING_CONFIG, LOGS_DIR


def setup_logger(name: str = "rag_pipeline", level: str = "INFO") -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Clear existing handlers
    logger.propagate = False

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (with rotation)
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "rag_pipeline") -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(name)


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    print(f"\n✅ Logger test complete. Check log file: {LOGS_DIR / 'test.log'}")
