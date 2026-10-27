"""Research flow using the active checkout's production WZZT/AC core."""
from pathlib import Path
import sys

# CLI execution must use this checkout, not an unrelated editable installation.
_core=Path(__file__).resolve().parents[3]/'rnj_wzzt_core'
if _core.is_dir() and str(_core) not in sys.path:
    sys.path.insert(0,str(_core))
