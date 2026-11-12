"""Small JSON and source receipts shared by the CLI and retained DR solver."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

import numpy as np


def plain(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=plain, ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def contract_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + array.tobytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source_receipt():
    root = Path(__file__).resolve().parent
    return {p.relative_to(root).as_posix(): ref(p)["sha256"] for p in sorted(root.rglob("*.py"))
            if "outputs" not in p.relative_to(root).parts}


def environment():
    packages = {}
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, packages=packages)
