"""RNJ aggregation, alternative estimators, and comparative experiments.

Experiments use the standalone core as a dependency; the core never imports
this package. A source checkout can use the sibling core without installation.
"""

import sys
from pathlib import Path

_core_root = Path(__file__).resolve().parents[2] / "rnj_wzzt_core"
if _core_root.is_dir() and str(_core_root) not in sys.path:
    sys.path.insert(0, str(_core_root))
