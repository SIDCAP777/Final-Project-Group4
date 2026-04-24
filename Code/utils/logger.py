# ============================================================
# logger.py
# Sets up a logger that prints messages to the console and
# also saves them to a file in outputs/logs/.
# Use: logger = get_logger("train_classical"); logger.info("...")
# ============================================================

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    # Make sure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Build a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Create the logger object
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format for every log message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler that writes to the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler that prints to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Writing logs to: {log_file}")
    return logger
