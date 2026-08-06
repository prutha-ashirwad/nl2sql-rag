"""Logging setup, applied once at start-up by the entry points."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_LOGGER_NAME = "nl2sql"


def configure_logging(level: str = "INFO") -> None:
    """Attach a single stderr handler to the package logger.

    Repeated calls replace the existing handler rather than stacking another one.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the package logger."""
    suffix = name.removeprefix(f"{_ROOT_LOGGER_NAME}.")
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{suffix}")
