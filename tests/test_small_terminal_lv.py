from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.scenario.validate_scenario import validate_terminal_load_only_scenario


def test_small_terminal_lv_is_strict_terminal_load_only():
    net = build_small_terminal_lv_case()
    summary = validate_terminal_load_only_scenario(net, strict=True)
    assert summary["observed_terminal_count"] == 6
    assert summary["hidden_internal_count"] == 3
    assert summary["violations"] == []
    hidden = net.buses[net.buses["bus_type"].eq("hidden_internal")]
    assert hidden["hidden_degree"].min() >= 3
    assert set(net.load_buses()) == {101, 102, 103, 104, 105, 106}
