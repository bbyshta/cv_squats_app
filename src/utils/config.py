"""Configuration loading helpers."""

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed YAML mapping.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is empty, invalid, or not a mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except ImportError as exc:
        raise ValueError("PyYAML is required to load YAML config files") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {config_path}: {exc}") from exc

    if data is None:
        raise ValueError(f"YAML config is empty: {config_path}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {config_path}")

    return data

