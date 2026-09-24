from terminal_case33.data.case33_der_assignment import build_case33_der_assignment
from terminal_case33.data.case33bw_raw import load_raw_case33bw


def test_case33_der_assignment_covers_all_buses():
    raw = load_raw_case33bw(include_tie_lines=True)
    assignment = build_case33_der_assignment(raw)
    assert len(assignment) == 33
    assert assignment["bus_id"].is_unique
    assert assignment.loc[assignment["bus_id"].eq(1), "profile_class"].iloc[0] == "root"
    assert {"pv", "wind", "small_generator", "base_load"}.issubset(set(assignment["profile_class"]))
    assert assignment.loc[assignment["profile_class"].isin(["pv", "wind", "small_generator"]), "der_capacity_kw"].gt(0).all()
