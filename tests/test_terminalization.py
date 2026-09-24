import networkx as nx

from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.scenario.validate_scenario import validate_terminal_load_only_scenario


def test_hybrid_leaf_terminalization_strict():
    raw = load_raw_case33bw()
    net = terminalize_case33(raw, mode="hybrid_leaf", seed=1)
    summary = validate_terminal_load_only_scenario(net, strict=True)
    graph = net.to_networkx_graph()
    leaves = {n for n, d in graph.degree() if d == 1 and n != net.root_bus}
    assert set(net.load_buses()).issubset(leaves)
    hidden = net.buses[net.buses["bus_type"].eq("hidden_internal")]
    assert (hidden["pd_kw"] == 0.0).all()
    assert (hidden["qd_kvar"] == 0.0).all()
    assert abs(net.buses["pd_kw"].sum() - raw.buses["pd_kw"].sum()) < 1e-9
    assert bool(net.buses.loc[net.buses["bus_id"].eq(net.root_bus), "is_observed"].iloc[0])
    raw_nonroot_loads = raw.buses[(raw.buses["bus_id"] != raw.root_bus) & ((raw.buses["pd_kw"] != 0) | (raw.buses["qd_kvar"] != 0))]
    assert summary["observed_terminal_count"] == len(raw_nonroot_loads)
    assert nx.is_tree(graph)

