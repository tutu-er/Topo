"""Run from any working directory: python /path/to/this/folder/run.py demo."""
from pathlib import Path
import sys

# Also support Python isolated mode (-I), which omits the script directory.
# This adds only this copied folder, never the original repository.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
