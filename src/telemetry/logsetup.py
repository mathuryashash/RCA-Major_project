"""Small rotating file logger for the headless collector."""

import logging
from logging.handlers import RotatingFileHandler

from . import config

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        try:
            config.app_dir().mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                config.log_path(), maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUPS, encoding="utf-8",
            )
        except OSError:
            # Imports must remain safe when a test sandbox cannot write the
            # user's profile. A real collector run will retry setup next time.
            return logger
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        root = logging.getLogger("telemetry")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        _configured = True
    return logger
