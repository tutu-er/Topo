from pathlib import Path

import rnj_wzzt.cli as cli
from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.reporting import _truth_nontrivial_clades


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


def test_cli_forwards_reviewed_configuration(monkeypatch) -> None:
    captured = {}

    def fake_run(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    result = cli.run(Path("result"), cases=("paper15",), time_limit=123.0)
    assert result == {"ok": True}
    assert captured["scenario_count"] == 3
    assert captured["samples_per_scenario"] == 96
    assert captured["contract_blocks"] is True
    assert captured["candidate_pool_mode"] == "rnj"
    assert captured["time_limit"] == 123.0
