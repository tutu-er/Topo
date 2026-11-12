"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: dict) -> None:
    """Write JSON with UTF-8 encoding."""

    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame as CSV."""

    frame.to_csv(path, index=True)

