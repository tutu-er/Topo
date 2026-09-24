from terminal_case33.data.paper_style_terminal_lv import (
    build_paper_style_resource_assignment,
    build_paper_style_terminal_lv_case,
)


def test_paper_style_terminal_lv_is_strict_terminal_only():
    net = build_paper_style_terminal_lv_case()
    summary = net.check_terminal_load_only(strict=True)
    assert summary["observed_terminal_count"] == 15
    assert summary["hidden_internal_count"] == 6
    assert summary["degree_2_hidden_chains"] == []
    assert all(net.buses.loc[net.buses["bus_type"].eq("hidden_internal"), "hidden_degree"] >= 3)


def test_paper_style_resource_assignment_has_mixed_der_nodes():
    net = build_paper_style_terminal_lv_case()
    assignment = build_paper_style_resource_assignment(net)
    component_count = (
        assignment["pv_capacity_kw"].gt(0).astype(int)
        + assignment["wind_capacity_kw"].gt(0).astype(int)
        + assignment["small_generator_capacity_kw"].gt(0).astype(int)
    )
    assert len(assignment) == 15
    assert component_count.gt(1).sum() >= 6
    assert {"base_load", "pv", "wind", "small_generator"}.issubset(set(assignment["profile_class"]))
