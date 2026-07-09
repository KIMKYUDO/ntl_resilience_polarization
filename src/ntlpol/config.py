from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update nested dictionaries.

    Parameters
    ----------
    base:
        Mutable base dictionary.
    updates:
        Dictionary containing override values.
    """
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return data


@dataclass
class Config:
    """Small convenience wrapper around a nested config dictionary."""

    data: dict[str, Any]
    source_path: Path | None = None

    def get(self, dotted_key: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, default=None)
        if value is None:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return self.data

    def validate_phase1(self) -> None:
        """Validate keys required by the Phase 1 project skeleton."""
        required_keys = [
            "project.name",
            "project.country",
            "spatial.grid_resolution_km",
            "time_window.pre_months",
            "time_window.post_months_for_measurement",
            "ntl.channels",
            "targets.delayed_recovery.slow_quantile",
            "validation.primary_split",
        ]
        for key in required_keys:
            self.require(key)

        channels = self.require("ntl.channels")
        if not isinstance(channels, list) or len(channels) == 0:
            raise ConfigError("ntl.channels must be a non-empty list")

        pre = int(self.require("time_window.pre_months"))
        event_months = int(self.require("time_window.event_months"))
        post = int(self.require("time_window.post_months_for_measurement"))
        expected = pre + event_months + post
        actual = int(self.require("time_window.full_sequence_months"))
        if expected != actual:
            raise ConfigError(
                f"full_sequence_months must equal pre + event + post. "
                f"Expected {expected}, got {actual}."
            )


def load_config(
    path: str | Path = "configs/base.yaml",
    overrides: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Config:
    """Load the project config.

    Examples
    --------
    >>> cfg = load_config("configs/base.yaml")
    >>> cfg.require("project.name")
    'ntl_resilience_polarization'
    """
    path = Path(path)
    data = load_yaml(path)
    if overrides:
        data = _deep_update(data, overrides)
    cfg = Config(data=data, source_path=path)
    if validate:
        cfg.validate_phase1()
    return cfg
