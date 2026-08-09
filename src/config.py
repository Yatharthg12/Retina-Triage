from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

class ConfigError(ValueError):
    pass

def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("RETINATRIAGE_CONFIG", ROOT / "configs/default.json"))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot load configuration: {config_path}") from exc
    if "extends" in data:
        parent = config_path.parent / data.pop("extends")
        data = _merge(load_config(parent), data)
    for key in ("paths", "quality", "triage", "uploads"):
        if key not in data:
            raise ConfigError(f"Missing configuration section: {key}")
    overrides = {
        "database": os.getenv("RETINATRIAGE_DATABASE"),
        "model": os.getenv("RETINATRIAGE_MODEL_PATH"),
    }
    for key, value in overrides.items():
        if value:
            data["paths"][key] = value
    for key, value in list(data["paths"].items()):
        p = Path(value)
        data["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    data["_root"] = str(ROOT)
    return data

