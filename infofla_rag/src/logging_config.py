# src/logging_config.py
import logging
import logging.config
import os
from pathlib import Path


def setup_logging() -> None:

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s [%(levelname)s] "
                    "%(name)s - %(message)s"
                ),
            },
            "verbose": {
                "format": (
                    "%(asctime)s [%(levelname)s] "
                    "%(name)s (%(filename)s:%(lineno)d) - %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
            "file_app": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "verbose",
                "filename": str(log_dir / "app.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
                "level": "INFO",
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "verbose",
                "filename": str(log_dir / "error.log"),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
                "level": "ERROR",
            },
        },
        "loggers": {
            # RAG 관련 모듈 (pipeline / api / vllm 등)
            "rag": {
                "handlers": ["console", "file_app", "file_error"],
                "level": "INFO",
                "propagate": False,
            },
            # 필요하면 다른 네임스페이스도 추가 가능
        },
        "root": {
            "handlers": ["console", "file_app", "file_error"],
            "level": "WARNING",
        },
    }

    logging.config.dictConfig(logging_config)
