"""Configuration models and YAML loader with strict Pydantic validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: LogFormat = Field(default=LogFormat.CONSOLE)


class RandomSeedSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    master_seed: int = Field(default=42, ge=0)
    train_seeds_count: int = Field(default=10, ge=1)
    eval_seeds_count: int = Field(default=10, ge=1)


class WorldGridSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=32, ge=4, le=1024)
    height: int = Field(default=32, ge=4, le=1024)
    max_steps: int = Field(default=1000, ge=1)


class SimulationSettings(BaseSettings):
    """Global system configuration for AUTOGEN simulation platform."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOGEN_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
    )

    app_name: str = "AUTOGEN AI Research Platform"
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production|test)$",
    )
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    seeds: RandomSeedSettings = Field(default_factory=RandomSeedSettings)
    world: WorldGridSettings = Field(default_factory=WorldGridSettings)


def load_settings(yaml_path: Path | str | None = None) -> SimulationSettings:
    """Loads and validates simulation settings from an optional YAML file.

    If yaml_path is None, loads default settings merged with environment variables.
    If yaml_path is provided, parses YAML and strictly validates against schema.

    Raises:
        FileNotFoundError: If the specified yaml file does not exist.
        ValueError: If YAML syntax is invalid or root is not a mapping.
        ValidationError: If configuration fails schema validation.
    """
    if yaml_path is None:
        return SimulationSettings()

    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as err:
        raise ValueError(f"Malformed YAML syntax in {path}: {err}") from err

    if data is None:
        return SimulationSettings()

    if not isinstance(data, dict):
        raise ValueError(f"YAML configuration root must be a mapping, got {type(data).__name__}")

    return SimulationSettings.model_validate(data)
