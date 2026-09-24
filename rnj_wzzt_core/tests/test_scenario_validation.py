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
