"""Load an authoritative module from the standalone RNJ-wzzT source tree."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def alias_module(module_name: str, target: str) -> ModuleType:
    core_root = Path(__file__).resolve().parents[1] / "rnj_wzzt_core"
    core_text = str(core_root)
    if core_text not in sys.path:
        sys.path.insert(0, core_text)
    implementation = importlib.import_module(target)
    sys.modules[module_name] = implementation
    return implementation
