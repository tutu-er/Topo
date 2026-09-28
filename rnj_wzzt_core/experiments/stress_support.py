"""Compatibility entry point for research_experiments.rnj.stress_support."""

from importlib import import_module
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_implementation = import_module("research_experiments.rnj.stress_support")
if __name__ == "__main__":
    _main = getattr(_implementation, "main", None)
    if _main is not None:
        _main()
else:
    sys.modules[__name__] = _implementation
