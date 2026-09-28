"""Migration boundaries: legacy imports, worker identity, and source manifests."""

import argparse
import importlib
import inspect
from pathlib import Path
import pickle
import runpy
import subprocess
import sys

import pytest

from research_experiments.rnj.stress_support import source_fingerprint


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "rnj_wzzt_core"


@pytest.mark.parametrize("name", [
    "analyze_advantage_stress", "rooted_ablation_support", "rooted_aggregation",
    "rooted_study_baselines", "run_advantage_stress", "run_common_mode_comparison",
    "run_negative_offdiag_logdet", "run_root_information_study", "run_rooted_ablation",
    "run_scenario_benchmark", "stress_support", "temporal_preprocessing",
])
def test_legacy_research_import_is_the_same_module(name):
    if name == "run_negative_offdiag_logdet":
        pytest.importorskip("cvxpy")
    canonical = importlib.import_module(f"research_experiments.rnj.{name}")
    legacy = importlib.import_module(f"rnj_wzzt_core.experiments.{name}")
    assert legacy is canonical


def test_legacy_tuning_import_preserves_monkeypatches(monkeypatch):
    canonical = importlib.import_module("research_experiments.rnj.scripts.run_matched_rnj_tuning")
    legacy = importlib.import_module("rnj_wzzt_core.scripts.run_matched_rnj_tuning")
    assert legacy is canonical
    marker = object()
    monkeypatch.setattr(legacy, "source_fingerprint", marker)
    assert canonical.source_fingerprint is marker


@pytest.mark.parametrize("name,worker", [
    ("run_advantage_stress", "safe_job"),
    ("run_scenario_benchmark", "run_job"),
    ("run_rooted_ablation", "run_job"),
])
def test_legacy_cli_dispatches_picklable_canonical_workers(name, worker, monkeypatch):
    class StopBeforeRunning(RuntimeError):
        pass

    def inspect_arguments(*args, **kwargs):
        namespace = inspect.currentframe().f_back.f_globals
        function = namespace[worker]
        assert function.__module__ == f"research_experiments.rnj.{name}"
        assert pickle.loads(pickle.dumps(function)) is function
        raise StopBeforeRunning

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_arguments)
    with pytest.raises(StopBeforeRunning):
        runpy.run_path(str(CORE / "experiments" / f"{name}.py"), run_name="__main__")


def test_spawn_workers_can_import_relocated_modules(tmp_path):
    # Windows uses spawn by default. Force it everywhere to verify that workers
    # can unpickle the canonical callables after importing their real modules.
    script = tmp_path / "spawn_check.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from multiprocessing import get_context\n"
        "from research_experiments.rnj import run_advantage_stress, run_scenario_benchmark, run_rooted_ablation\n"
        "def identify(function):\n"
        "    return function.__module__, function.__name__\n"
        "if __name__ == '__main__':\n"
        "    functions = [run_advantage_stress.safe_job, run_scenario_benchmark.run_job, run_rooted_ablation.run_job]\n"
        "    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as pool:\n"
        "        actual = list(pool.map(identify, functions))\n"
        "    assert actual == [(f.__module__, f.__name__) for f in functions]\n",
        encoding="utf-8",
    )
    result = subprocess.run([sys.executable, "-B", str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_primary_fingerprint_tracks_implementations_and_excludes_later_tuning():
    manifest = {Path(name).as_posix(): digest for name, digest in source_fingerprint(CORE).items()}
    assert "rnj_wzzt_core/rnj_wzzt/pipeline.py" in manifest
    for name in ("rooted_aggregation", "graph_adapters", "sensitivity_geometry", "temporal_preprocessing"):
        assert f"research_experiments/rnj/{name}.py" in manifest
    assert "research_experiments/rnj/__init__.py" in manifest
    assert "research_experiments/__init__.py" in manifest
    assert not any(name.startswith("rnj_wzzt_core/experiments/") for name in manifest)
    assert not any(name.startswith("research_experiments/rnj/scripts/") for name in manifest)
