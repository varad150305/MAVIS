"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Container for the validated runtime configuration."""

    data: dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve a nested configuration value."""
        current: Any = self.data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]

        return current

    @property
    def system_name(self) -> str:
        """Return the configured system name."""
        return str(self.get("system", "name", default="unknown"))

    @property
    def features(self) -> dict[str, bool]:
        """Return feature flags."""
        return self.get("features", default={})


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration.

    Returns:
        A validated Config object.

    Raises:
        ConfigError: If the file is missing, malformed, or invalid.
    """

    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigError(
            f"Configuration path is not a file: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML syntax in {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Unable to read configuration file: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a YAML mapping.")

    _validate_config(data)

    return Config(data=data)


def _validate_config(data: dict[str, Any]) -> None:
    """Validate the required configuration structure."""

    required_sections = [
        "system",
        "features",
        "audio",
        "vad",
        "asr",
        "translation",
        "cpu",
        "logging",
        "performance",
        "output",
    ]

    for section in required_sections:
        if section not in data:
            raise ConfigError(
                f"Missing required configuration section: '{section}'"
            )

    required_flags = [
        "use_adaptive_buffer",
        "use_local_agreement",
        "use_stability_controller",
        "use_translation_stability",
    ]

    features = data["features"]

    if not isinstance(features, dict):
        raise ConfigError("'features' must be a mapping.")

    for flag in required_flags:
        if flag not in features:
            raise ConfigError(
                f"Missing required feature flag: '{flag}'"
            )

        if not isinstance(features[flag], bool):
            raise ConfigError(
                f"Feature flag '{flag}' must be true or false."
            )

    audio = data["audio"]

    _require_positive_number(
        audio,
        "sample_rate",
        "audio.sample_rate",
    )

    _require_positive_number(
        audio,
        "channels",
        "audio.channels",
    )

    _require_positive_number(
        audio,
        "chunk_duration_ms",
        "audio.chunk_duration_ms",
    )

    _require_positive_number(
        audio,
        "buffer_size_ms",
        "audio.buffer_size_ms",
    )

    _require_nonnegative_number(
        audio,
        "overlap_ms",
        "audio.overlap_ms",
    )

    if audio["overlap_ms"] >= audio["buffer_size_ms"]:
        raise ConfigError(
            "audio.overlap_ms must be smaller than audio.buffer_size_ms."
        )

    cpu = data["cpu"]

    _require_positive_number(
        cpu,
        "threads",
        "cpu.threads",
    )

    _require_positive_number(
        cpu,
        "intra_op_threads",
        "cpu.intra_op_threads",
    )

    _require_positive_number(
        cpu,
        "inter_op_threads",
        "cpu.inter_op_threads",
    )

    logging_config = data["logging"]

    if logging_config.get("level") not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise ConfigError(
            "logging.level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    fixed_buffer = data.get(
        "fixed_buffer"
    )

    if fixed_buffer is not None:

        if not isinstance(
            fixed_buffer,
            dict,
        ):
            raise ConfigError(
                "'fixed_buffer' must be a mapping."
            )

        if "enabled" not in fixed_buffer:
            raise ConfigError(
                "Missing configuration value: "
                "'fixed_buffer.enabled'"
            )

        if not isinstance(
            fixed_buffer["enabled"],
            bool,
        ):
            raise ConfigError(
                "'fixed_buffer.enabled' must be true or false."
            )

        _require_nonnegative_number(
            fixed_buffer,
            "pre_roll_ms",
            "fixed_buffer.pre_roll_ms",
        )

        _require_positive_number(
            fixed_buffer,
            "max_utterance_duration_ms",
            "fixed_buffer.max_utterance_duration_ms",
        )

        _require_nonnegative_number(
            fixed_buffer,
            "min_utterance_duration_ms",
            "fixed_buffer.min_utterance_duration_ms",
        )

        if (
            fixed_buffer["min_utterance_duration_ms"]
            > fixed_buffer["max_utterance_duration_ms"]
        ):
            raise ConfigError(
                "fixed_buffer.min_utterance_duration_ms "
                "cannot exceed "
                "fixed_buffer.max_utterance_duration_ms."
            )

def _require_positive_number(
    section: dict[str, Any],
    key: str,
    path: str,
) -> None:
    """Validate that a configuration value is positive."""

    if key not in section:
        raise ConfigError(f"Missing configuration value: '{path}'")

    value = section[key]

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"'{path}' must be numeric.")

    if value <= 0:
        raise ConfigError(f"'{path}' must be greater than zero.")


def _require_nonnegative_number(
    section: dict[str, Any],
    key: str,
    path: str,
) -> None:
    """Validate that a configuration value is non-negative."""

    if key not in section:
        raise ConfigError(f"Missing configuration value: '{path}'")

    value = section[key]

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"'{path}' must be numeric.")

    if value < 0:
        raise ConfigError(f"'{path}' cannot be negative.")