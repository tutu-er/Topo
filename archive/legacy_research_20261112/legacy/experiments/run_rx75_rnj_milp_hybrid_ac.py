"""Compatibility alias for the authoritative standalone core pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "rnj_wzzt_core"
sys.path.insert(0, str(CORE_ROOT))
IMPLEMENTATION = importlib.import_module("rnj_wzzt.pipeline")

if __name__ == "__main__":
    IMPLEMENTATION.main()
else:
    sys.modules[__name__] = IMPLEMENTATION
