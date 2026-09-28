"""Run the preserved theft experiments with their original workspace paths."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence


THEFT_WORKSPACE = Path(__file__).resolve().parents[2] / "theft_wzzt"


def main(argv: Sequence[str] | None = None) -> int:
    """Forward arguments and the exit code without importing research code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    completed = subprocess.run(
        [sys.executable, str(THEFT_WORKSPACE / "experiments" / "run_theft.py"), *arguments],
        cwd=THEFT_WORKSPACE,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
