import numpy as np
from research_experiments.wzzt_paper.common import jobs,indices,noise_errors


def test_frozen_denominators_and_independent_split_replicates():
    rows=jobs();assert len(rows)==1204 and len({p['job_id'] for p in rows})==1204
    assert {name:sum(p['panel']==name for p in rows) for name in ('main','stress','topology','case33')}==dict(main=640,stress=400,topology=144,case33=20)
    replicas=[{100+10*b+i for b in range(10)} for i in range(3)]
    assert all(not replicas[i]&replicas[j] for i in range(3) for j in range(i))


def test_snapshot_sets_preserve_nested_physical_days_and_full_288():
    sets=[set(indices(N)) for N in (24,48,96,288)]
    assert all(a<=b for a,b in zip(sets,sets[1:])) and len(sets[-1])==96


def test_ar1_errors_have_declared_population_scale_and_correlation():
    rng=np.random.default_rng(20261029);x=noise_errors(rng.normal(size=(20000,2)),.8)
    assert abs(x.std()-1)<.025
    assert abs(np.corrcoef(x[:-1,0],x[1:,0])[0,1]-.8)<.025


def test_random_network_seeded_tree_has_zero_hidden_injections_and_legal_leaves():
    from research_experiments.wzzt_paper.networks import random_network
    a=random_network(16,1);b=random_network(16,1);g=a.to_networkx_graph()
    import networkx as nx
    assert nx.is_tree(g) and len(a.get_leaf_buses())==16
    assert all(d>=3 for v,d in g.degree if v!=a.root_bus and v not in a.get_leaf_buses())
    assert a.branches.equals(b.branches) and a.buses.loc[a.buses.bus_type=='hidden_internal','pd_kw'].eq(0).all()


def test_case33_resource_assignment_preserves_original_load_locations_and_totals():
    from research_experiments.wzzt_paper.networks import case33_network
    from rnj_wzzt.data.paper_style_case_bank import build_case_bank_resource_assignment
    n=case33_network();a=build_case_bank_resource_assignment(n);a['bus_id']=a.bus_id.map(n.terminal_to_original)
    terminals=n.buses[n.buses.bus_type=='observed_terminal']
    assert set(a.bus_id)==set(terminals.original_bus_id)
    assert abs(terminals.pd_kw.sum()-n.metadata['original_total_pd_kw'])<1e-8
    assert len(a)==32 and a.bus_id.is_unique
