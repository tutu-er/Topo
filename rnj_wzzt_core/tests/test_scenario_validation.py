import pytest

from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.scenario import validate_terminal_load_only_scenario


def test_terminal_only_validation_api_is_self_contained() -> None:
    for build in CASE_BUILDERS.values():
        net = build()
        summary = net.check_terminal_load_only(strict=True)
        direct_summary = validate_terminal_load_only_scenario(net, strict=True)

        assert summary == direct_summary
        assert summary["violations"] == []
        assert summary["is_connected"] is True
        assert summary["is_acyclic"] is True


def test_terminal_label_requires_observation() -> None:
    net = CASE_BUILDERS["paper15"]()
    net.buses.loc[net.buses["bus_id"].eq(101), "is_observed"] = False

    with pytest.raises(ValueError, match="observed_terminal node is not observed"):
        validate_terminal_load_only_scenario(net, strict=True)
    assert "observed_terminal node is not observed" in validate_terminal_load_only_scenario(net, strict=False)["violations"]
