"""
Logger Module
=============

Provides centralized logging configuration for the project.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str, level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_file: Optional file path for log output

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class PipelineLogger:
    """Context manager for logging pipeline execution."""

    def __init__(self, name: str, log_file: Optional[str] = "logs/pipeline.log"):
        self.name = name
        self.logger = setup_logger(f"pipeline.{name}", log_file=log_file)
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{'='*50}")
        self.logger.info(f"STARTING: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type:
            self.logger.error(f"FAILED: {self.name} - {exc_val}")
        else:
            self.logger.info(f"COMPLETED: {self.name}")
        self.logger.info(f"Duration: {duration}")
        return False

    def info(self, message: str):
        self.logger.info(message)
