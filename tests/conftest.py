"""Shared pytest fixtures and test environment setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure packages/ is on sys.path for test discovery and running without editable install
packages_dir = Path(__file__).resolve().parent.parent / "packages"
if str(packages_dir) not in sys.path:
    sys.path.insert(0, str(packages_dir))


@pytest.fixture
def sample_valid_yaml(tmp_path: Path) -> Path:
    yaml_content = """
app_name: "Test Platform"
environment: "test"
logging:
  level: "DEBUG"
  format: "json"
seeds:
  master_seed: 999
  train_seeds_count: 3
  eval_seeds_count: 3
world:
  width: 16
  height: 16
  max_steps: 100
"""
    file_path = tmp_path / "valid_test.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_invalid_yaml(tmp_path: Path) -> Path:
    yaml_content = """
app_name: "Invalid Platform"
world:
  width: -5
  height: "invalid_string"
"""
    file_path = tmp_path / "invalid_test.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")
    return file_path
