"""Compatibility entry point for the standalone RNJ-wzzT core."""

from __future__ import annotations

import sys
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1] / "rnj_wzzt_core"
sys.path.insert(0, str(CORE_ROOT))

from rnj_wzzt.cli import DEFAULT_CASES, DEFAULT_OUTPUT, main, run

__all__ = ["DEFAULT_CASES", "DEFAULT_OUTPUT", "main", "run"]


if __name__ == "__main__":
    main()
