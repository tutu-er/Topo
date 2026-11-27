"""Read final four-feeder records and locate private output for new runs."""
import json
import os
from copy import deepcopy
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

PUBLISHED = Path(__file__).resolve().parents[2] / "experiments" / "four_feeders"


def cache_root():
    if configured := os.environ.get("PTID_FOUR_CACHE"):
        return Path(configured).resolve()
    from ptid.runs.four_feeders import REPO, data_root
    try:
        repo = data_root().parent
    except FileNotFoundError:
        repo = REPO  # The synthetic node experiments require no archived inputs.
    return repo / "outputs" / "topology_study_four_feeders"


@lru_cache(maxsize=1)
def _evidence():
    return json.loads((PUBLISHED / "evidence.json").read_text(encoding="utf-8"))


def read_published(filename):
    """Return one frozen job keyed by its original JSON filename."""
    return deepcopy(_evidence()[filename])


def read_record(directory, filename):
    path = directory / filename
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if directory.resolve() == PUBLISHED.resolve():
        return read_published(filename)
    raise FileNotFoundError(path)


def record_names(directory, pattern):
    names = {p.name for p in directory.glob(pattern)}
    if directory.resolve() == PUBLISHED.resolve():
        names.update(name for name in _evidence() if fnmatchcase(name, pattern))
    return sorted(names, key=str.casefold)


def input_records(output, pattern):
    """Replay a local run when present, otherwise use the complete published series."""
    return output if record_names(output, pattern) else PUBLISHED
