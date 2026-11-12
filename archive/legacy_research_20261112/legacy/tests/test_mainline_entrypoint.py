from pathlib import Path
from inspect import signature

import experiments.run_mainline as mainline
import rnj_wzzt.cli as core_cli
import rnj_wzzt.pipeline as pipeline


def test_mainline_forwards_the_reviewed_configuration(monkeypatch) -> None:
    captured = {}

    def fake_run(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(core_cli, "run_pipeline", fake_run)
    result = mainline.run(Path("result"), cases=("paper15",), time_limit=321.0)

    assert result == {"ok": True}
    assert captured["output_dir"] == Path("result")
    assert captured["cases"] == ("paper15",)
    assert captured["time_limit"] == 321.0
    assert set(captured) == {"output_dir", "cases", "time_limit"}
    effective = signature(pipeline.run).bind(**captured)
    effective.apply_defaults()
    assert effective.arguments["scenario_suite"] == "reference"
    assert effective.arguments["scenario_count"] == 3
    assert effective.arguments["samples_per_scenario"] == 96
    assert effective.arguments["bootstrap_replicates"] == 100
    assert effective.arguments["confidence_threshold"] == .75
    assert effective.arguments["maximum_candidate_count"] == 2
    assert effective.arguments["selection_only"] is False
    assert effective.arguments["run_baseline"] is False


def test_root_entrypoint_is_the_standalone_core() -> None:
    assert mainline.run.__module__ == "rnj_wzzt.cli"
