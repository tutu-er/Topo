"""Compatibility alias for the independent matrix-repair experiment."""
import sys
from research_experiments.rnj import matrix_constraints as _implementation

sys.modules[__name__] = _implementation
