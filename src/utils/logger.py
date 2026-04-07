"""
Structured Logging System with Rotation

Provides centralized logging with:
- Console and file output
- Log rotation
- Structured formatting
- Multiple log levels
- Context managers

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.error("Error occurred", exc_info=True)
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.COLORS:
            levelname_color = self.COLORS[record.levelno] + levelname + LogColors.RESET
            record.levelname = levelname_color

        # Format the message
        result = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    Setup structured logger with rotation.

    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (optional)
        level: Logging level
        max_bytes: Max file size before rotation
        backup_count: Number of backup files
        console: Enable console output
        json_format: Use JSON format for file output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)

        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(
    name: str, log_file: Optional[str] = None, level: Optional[int] = None
) -> logging.Logger:
    """
    Get or create logger with default configuration.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        level: Optional logging level

    Returns:
        Logger instance
    """
    # Try to load settings
    try:
        from src.utils.settings import settings

        if level is None:
            level = getattr(logging, settings.log_level.upper(), logging.INFO)
        if log_file is None and settings.environment == "production":
            log_file = settings.paths.logs / f"{name.replace('.', '_')}.log"
    except:
        level = level or logging.INFO

    return setup_logger(name, log_file=log_file, level=level)


class LoggerContext:
    """Context manager for temporary logging configuration."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(func):
    """Decorator to log function calls."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}", exc_info=True)
            raise

    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test console logging
    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test file logging
    file_logger = setup_logger(
        "test_file", log_file="outputs_pipeline/logs/test.log", level=logging.DEBUG
    )

    file_logger.info("This goes to both console and file")

    # Test JSON logging
    json_logger = setup_logger(
        "test_json",
        log_file="outputs_pipeline/logs/test.json",
        level=logging.INFO,
        json_format=True,
        console=False,
    )

    json_logger.info("This is JSON formatted")

    # Test context manager
    logger.info("Normal logging level")
    with LoggerContext(logger, logging.DEBUG):
        logger.debug("Temporary debug level")
    logger.debug("Back to normal (this won't show if level is INFO)")

    # Test decorator
    @log_function_call
    def test_function(x, y):
        return x + y

    result = test_function(5, 3)
    print(f"Result: {result}")
