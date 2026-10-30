"""Unit tests for structured logging."""

from __future__ import annotations

import json
import logging

import pytest

from autogen.core.logging import StructuredJsonFormatter, configure_logging, get_logger


@pytest.mark.unit
def test_structured_json_formatter():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="autogen.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test event occurred",
        args=(),
        exc_info=None,
    )
    record.__dict__["structured_data"] = {"agent_id": "agent_0", "score": 42.5}

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "autogen.test"
    assert parsed["message"] == "Test event occurred"
    assert parsed["data"]["agent_id"] == "agent_0"
    assert parsed["data"]["score"] == 42.5
    assert "timestamp" in parsed


@pytest.mark.unit
def test_get_logger_naming():
    logger1 = get_logger("simulation")
    assert logger1.name == "autogen.simulation"

    logger2 = get_logger("autogen.agents")
    assert logger2.name == "autogen.agents"


@pytest.mark.unit
def test_configure_logging(capsys: pytest.CaptureFixture[str]):
    configure_logging(level="DEBUG", json_format=True)
    logger = get_logger("core_test")
    logger.debug("Debug message test")

    captured = capsys.readouterr()
    assert "Debug message test" in captured.out
    parsed = json.loads(captured.out.strip())
    assert parsed["level"] == "DEBUG"
    assert parsed["logger"] == "autogen.core_test"
