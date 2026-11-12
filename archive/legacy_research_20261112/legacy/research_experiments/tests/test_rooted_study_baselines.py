"""Independent exact-tree and root-anchor checks for the experiment NJ control."""
import networkx as nx
import numpy as np
import pytest
from rnj_wzzt.scenario.simulation import _simulate_pool, _terminal_buses
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.reporting import _truth_nontrivial_clades
from research_experiments.rnj import rooted_study_baselines as module

@pytest.mark.parametrize('case',['paper15','soumalas11','flynn16','pengwah18'])
def test_exact_case_tree(case):
    net,_=_simulate_pool(case,8,0,1,0,0,scenario_suite='reference',root_observation='exact')
    labels=_terminal_buses(net)
    r,x=build_reduced_sensitivity_matrices(net,labels,voltage_model='squared-voltage')
    result=module.infer_classical_nj(r,x,labels,net.root_bus)
    assert rooted_clades(result.edges,net.root_bus,labels)==_truth_nontrivial_clades(net,labels)
    g=nx.Graph()
    g.add_weighted_edges_from(result.edges)
    assert nx.is_tree(g)
    geo=module.sensitivity_geometry(r,x,'RX_75R_25X')
    for i,a in enumerate(labels):
        assert nx.shortest_path_length(g,net.root_bus,a,weight='weight')==pytest.approx(geo.root_depths[i],abs=1e-9)
        for j,b in enumerate(labels):
            assert nx.shortest_path_length(g,a,b,weight='weight')==pytest.approx(geo.distance[i,j],abs=1e-9)


def test_multifurcating_root_and_negative_labels():
    r=np.diag([1.,2.,3.])
    result=module.infer_classical_nj(r,2*r,[-1,-2,-3],-4)
    assert rooted_clades(result.edges,-4,[-1,-2,-3])==set()
    assert len(result.edges)==3

@pytest.mark.parametrize('labels,root,factor',[([1,1],0,0),([1,2],1,0),([1,2],0,-.1),([1,2],0,float('nan'))])
def test_invalid_inputs(labels,root,factor):
    with pytest.raises(ValueError):
        module.infer_classical_nj(np.eye(2),np.eye(2),labels,root,collapse_factor=factor)
