import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

# Anchored to the repository root (parent of utils/) so logging works
# regardless of the caller's current working directory, instead of relying
# on relative paths that only resolved correctly when run from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_configured_loggers: set = set()


def setup_logger(name: str = "medical_vlm", config_path: str = "configs/logging.yaml") -> logging.Logger:
    """
    Initializes a production-grade logger writing to both console and a
    rotating log file. Safe to call repeatedly for the same `name` (e.g. on
    module re-import or across DataLoader worker processes) - handlers are
    attached at most once per logger name.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    log_level = logging.INFO
    log_file = "logs/execution.log"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = PROJECT_ROOT / config_path

    if resolved_config_path.exists():
        with open(resolved_config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
            sys_log = cfg.get("system_logging", {})
            level_str = sys_log.get("level", "INFO")
            log_level = getattr(logging, level_str.upper(), logging.INFO)
            log_file = sys_log.get("file", log_file)
            log_format = sys_log.get("format", log_format)

    resolved_log_file = Path(log_file)
    if not resolved_log_file.is_absolute():
        resolved_log_file = PROJECT_ROOT / log_file
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(log_format)

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(log_level)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # Rotating instead of an unbounded FileHandler so logs/execution.log
    # doesn't grow forever across the many EDA / smoke-test runs of a
    # multi-week dissertation project.
    f_handler = RotatingFileHandler(
        resolved_log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    f_handler.setLevel(log_level)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    _configured_loggers.add(name)
    return logger
