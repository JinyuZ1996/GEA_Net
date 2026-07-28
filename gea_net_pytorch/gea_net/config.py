from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _resolve_paths(value: Any, base_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, base_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, base_dir, key) for v in value]
    if isinstance(value, str) and key in {"root", "cache_dir", "output_dir"}:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return str(path)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = _resolve_paths(config, config_path.parent)
    config["_config_path"] = str(config_path)
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
