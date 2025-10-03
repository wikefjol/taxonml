# src/taxml/core/io.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
import yaml

def read_yaml(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}

def write_json(path: Path, obj: Any, *, atomic: bool = True, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(obj, f, indent=indent)
        os.replace(tmp, path)
    else:
        with path.open("w") as f:
            json.dump(obj, f, indent=indent)
