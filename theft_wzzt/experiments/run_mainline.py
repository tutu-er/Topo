"""Mainline reproduction entry for the standalone theft_wzzt package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from theft_wzzt.cli import DEFAULT_CASES, DEFAULT_OUTPUT, main, run

__all__ = ["DEFAULT_CASES", "DEFAULT_OUTPUT", "main", "run"]


if __name__ == "__main__":
    main()
