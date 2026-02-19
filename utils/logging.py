"""Shared logging utilities for the water meter ML project."""

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, category: str = "data") -> logging.Logger:
    """Create a configured logger with file and console output.

    Args:
        name: Log file prefix (e.g. "00_clean_data", "build_obb").
              The final filename will be {name}_{YYYYMMDD_HHMMSS}.log
        category: Subdirectory inside logs/ ("data" or "tools").

    Returns:
        A configured logging.Logger instance.
    """
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs" / category
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler — DEBUG level, full detail
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler — INFO level, clean output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger
