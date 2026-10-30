"""Structured logging framework for AUTOGEN AI."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured, single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception trace if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include structured extras if provided
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            log_payload["data"] = record.structured_data

        return json.dumps(log_payload)


class ConsoleFormatter(logging.Formatter):
    """Clean, human-readable console formatter with timestamp and level."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configures the root AUTOGEN logging subsystem.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        json_format: If True, emits JSON logs. If False, emits human-readable console text.
    """
    root_logger = logging.getLogger("autogen")
    root_logger.setLevel(level.upper())

    # Remove existing handlers to avoid duplicates on reconfiguration
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        stream_handler.setFormatter(StructuredJsonFormatter())
    else:
        stream_handler.setFormatter(ConsoleFormatter())

    root_logger.addHandler(stream_handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retrieves a named logger under the 'autogen' namespace hierarchy."""
    if name == "autogen" or name.startswith("autogen."):
        return logging.getLogger(name)
    return logging.getLogger(f"autogen.{name}")
