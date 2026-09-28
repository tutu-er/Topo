from pathlib import Path

import experiments.run_mainline as mainline
import rnj_wzzt.cli as core_cli


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
    assert captured["scenario_count"] == 3
    assert captured["samples_per_scenario"] == 96
    assert captured["bootstrap_replicates"] == 100
    assert captured["confidence_threshold"] == 0.75
    assert captured["maximum_candidate_count"] == 2
    assert captured["selection_only"] is False
    assert captured["run_baseline"] is False
    assert captured["contract_blocks"] is True
    assert "support_search_mode" not in captured
    assert "candidate_pool_mode" not in captured
    assert captured["time_limit"] == 321.0


def test_root_entrypoint_is_the_standalone_core() -> None:
    assert mainline.run.__module__ == "rnj_wzzt.cli"
