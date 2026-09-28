from pathlib import Path
from inspect import signature
import subprocess
import sys

import pytest

import rnj_wzzt.cli as cli
import rnj_wzzt.pipeline as pipeline
from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.reporting import _truth_nontrivial_clades


def test_all_core_modules_import_without_parent_or_research(tmp_path):
    core = Path(__file__).resolve().parents[1]
    script = """
import importlib, pathlib, pkgutil, sys
core = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(core))
import rnj_wzzt
for item in pkgutil.walk_packages(rnj_wzzt.__path__, rnj_wzzt.__name__ + '.'):
    module = importlib.import_module(item.name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(core), item.name
for name in sys.modules:
    assert not name.startswith(('terminal_case33', 'research_experiments', 'theft_wzzt', 'experiments')), name
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(core)],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_exact_four_case_registry() -> None:
    assert tuple(CASE_BUILDERS) == (
        "paper15",
        "soumalas11",
        "flynn16",
        "pengwah18",
    )


def test_truth_clades_are_nonempty_for_every_case() -> None:
    for build in CASE_BUILDERS.values():
        net = build()
        terminals = (
            net.buses.loc[
                net.buses["bus_type"].eq("observed_terminal"),
                "bus_id",
            ]
            .astype(int)
            .tolist()
        )
        assert _truth_nontrivial_clades(net, terminals)


def test_cli_python_compatibility_delegate_forwards_only_explicit_options(monkeypatch) -> None:
    captured = {}

    def fake_run(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    result = cli.run(Path("result"), cases=("paper15",), time_limit=123.0)
    assert result == {"ok": True}
    assert captured == {
        "output_dir": Path("result"), "cases": ("paper15",), "time_limit": 123.0,
    }


@pytest.mark.parametrize("entrypoint", ["main", "advanced_main", "pipeline_main"])
def test_all_entrypoints_inherit_the_python_pipeline_defaults(monkeypatch, entrypoint):
    calls = []
    computation = pipeline.run

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(sys, "argv", ["entrypoint"])
    if entrypoint == "pipeline_main":
        monkeypatch.setattr(pipeline, "run", capture)
        pipeline.main()
    else:
        monkeypatch.setattr(cli, "run_pipeline", capture)
        getattr(cli, entrypoint)()
    assert calls == [((), {})]
    options = signature(computation).bind(*calls[0][0], **calls[0][1])
    options.apply_defaults()
    assert options.arguments["output_dir"] == Path("outputs/reference")
    assert options.arguments["scenario_suite"] == "reference"
    assert options.arguments["selection_only"] is False
    assert options.arguments["run_baseline"] is False
    assert options.arguments["bootstrap_replicates"] == 100
    assert cli.DEFAULT_OUTPUT == pipeline.DEFAULT_OUTPUT


def test_unified_cli_forwards_explicit_settings_without_resetting_them():
    captured = {}
    cli.main([
        "--output", "custom", "--cases", "paper15", "flynn16",
        "--scenario-suite", "legacy", "--scenario-count", "2",
        "--samples-per-scenario", "192", "--training-replicate", "4",
        "--validation-replicate", "5", "--pq-noise-rel", "0.01",
        "--voltage-noise-rel", "0.002", "--root-observation", "exact",
        "--root-sigma", "0.02", "--root-meter-noise-rel", "0.003",
        "--impedance-scale", "1.5", "--tolerance-factor", "0.2",
        "--bootstrap-replicates", "9", "--block-length", "8",
        "--confidence-threshold", "0.8", "--maximum-candidate-count", "3",
        "--time-limit", "42", "--milp-solver", "gurobi",
        "--coefficient-bound", "3", "--run-baseline",
    ], runner=lambda **kwargs: captured.update(kwargs))
    assert captured == {
        "output_dir": Path("custom"), "cases": ("paper15", "flynn16"),
        "scenario_suite": "legacy", "scenario_count": 2,
        "samples_per_scenario": 192, "training_replicate": 4,
        "validation_replicate": 5, "pq_noise_rel": .01,
        "voltage_noise_rel": .002, "root_observation": "exact",
        "root_sigma": .02, "root_meter_noise_rel": .003,
        "impedance_scale": 1.5, "tolerance_factor": .2,
        "bootstrap_replicates": 9, "block_length": 8,
        "confidence_threshold": .8, "maximum_candidate_count": 3,
        "time_limit": 42., "milp_solver": "gurobi",
        "coefficient_bound": 3., "run_baseline": True,
    }


@pytest.mark.parametrize("flags, expected", [
    (["--selection-only"], {"selection_only": True}),
    (["--run-baseline"], {"run_baseline": True}),
    (["--run-milp", "--hybrid-only"], {"selection_only": False, "run_baseline": False}),
])
def test_diagnostic_flags_and_legacy_aliases_are_explicit(flags, expected):
    captured = {}
    cli.main(flags, runner=lambda **kwargs: captured.update(kwargs))
    assert captured == expected


@pytest.mark.parametrize("arguments", [
    ["--selection-only", "--run-milp"],
    ["--run-baseline", "--hybrid-only"],
    ["--samples-per-scenario", "1.5"],
    ["--cases", "unknown"],
])
def test_cli_rejects_conflicting_or_malformed_options(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments, runner=lambda **_: pytest.fail("must not run"))
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--support-search-mode", "rnj"],
        ["--candidate-pool-mode", "rnj"],
        ["--contract-blocks"],
        ["--deembedding-weight", "0.5"],
    ],
)
def test_advanced_cli_rejects_removed_switch(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["advanced", *arguments])
    with pytest.raises(SystemExit) as exc:
        cli.advanced_main(runner=lambda *_args, **_kwargs: pytest.fail("must not run"))
    assert exc.value.code == 2


def test_cli_requires_observed_root(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["advanced", "--root-observation", "unobserved"])
    with pytest.raises(SystemExit) as exc:
        cli.advanced_main(runner=lambda *_args, **_kwargs: pytest.fail("must not run"))
    assert exc.value.code == 2
