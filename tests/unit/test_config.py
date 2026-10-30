"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autogen.core.config import LogFormat, load_settings


@pytest.mark.unit
def test_default_settings():
    settings = load_settings()
    assert settings.app_name == "AUTOGEN AI Research Platform"
    assert settings.environment == "development"
    assert settings.world.width == 32
    assert settings.world.height == 32
    assert settings.seeds.master_seed == 42


@pytest.mark.unit
def test_load_valid_yaml(sample_valid_yaml: Path):
    settings = load_settings(sample_valid_yaml)
    assert settings.app_name == "Test Platform"
    assert settings.environment == "test"
    assert settings.logging.level == "DEBUG"
    assert settings.logging.format == LogFormat.JSON
    assert settings.seeds.master_seed == 999
    assert settings.world.width == 16
    assert settings.world.height == 16
    assert settings.world.max_steps == 100


@pytest.mark.unit
def test_reject_invalid_yaml(sample_invalid_yaml: Path):
    with pytest.raises(ValidationError) as exc_info:
        load_settings(sample_invalid_yaml)
    errors = exc_info.value.errors()
    assert len(errors) >= 1


@pytest.mark.unit
def test_reject_extra_keys(tmp_path: Path):
    yaml_with_extra = tmp_path / "extra_keys.yaml"
    yaml_with_extra.write_text("extra_unknown_field: 123\napp_name: 'Test'", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(yaml_with_extra)


@pytest.mark.unit
def test_reject_non_mapping_yaml(tmp_path: Path):
    yaml_list = tmp_path / "list.yaml"
    yaml_list.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(yaml_list)


@pytest.mark.unit
def test_missing_file_raises_not_found(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_settings(missing_file)


@pytest.mark.unit
def test_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTOGEN_APP_NAME", "Overridden App")
    monkeypatch.setenv("AUTOGEN_WORLD__WIDTH", "48")
    settings = load_settings()
    assert settings.app_name == "Overridden App"
    assert settings.world.width == 48
