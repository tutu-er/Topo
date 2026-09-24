import networkx as nx

from terminal_case33.data.case33bw_raw import load_raw_case33bw


def test_raw_case33_counts_and_loads():
    raw = load_raw_case33bw(include_tie_lines=True)
    closed = raw.branches[raw.branches["status"]]
    assert len(raw.buses) == 33
    assert len(closed) == 32
    graph = nx.Graph()
    graph.add_nodes_from(raw.buses["bus_id"])
    graph.add_edges_from(zip(closed["from_bus"], closed["to_bus"]))
    assert nx.is_connected(graph)
    assert nx.is_tree(graph)
    assert abs(raw.buses["pd_kw"].sum() - 3715.0) < 1e-9
    assert abs(raw.buses["qd_kvar"].sum() - 2300.0) < 1e-9
    leaves = {n for n, d in graph.degree() if d == 1 and n != raw.root_bus}
    nonleaf_loads = raw.buses[
        (~raw.buses["bus_id"].isin(leaves | {raw.root_bus}))
        & ((raw.buses["pd_kw"] != 0.0) | (raw.buses["qd_kvar"] != 0.0))
    ]
    assert len(nonleaf_loads) > 0

