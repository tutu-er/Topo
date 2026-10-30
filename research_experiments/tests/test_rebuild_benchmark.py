import numpy as np
import pytest

gp=pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.rebuild_benchmark.engine import Bank,search,regions,any_extension,audit_vector
from research_experiments.rebuild_benchmark.run import catalog


def test_frozen_data_partitions_and_counts():
    dev,test=catalog()
    assert len(dev)==16 and len(test)==140 and not set(dev)&set(test)
    assert len(set(test))==140


def test_original_fixed_wrong_tree_and_rebuild_share_original_loss():
    p=np.eye(3);z=np.array([0.,1.,1.]);y=p+np.outer(z,z)
    data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=p*0,drop_target=y)])
    pool=((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
    family=((0,),(1,),(2,),(0,1),(0,1,2));bank=Bank(data,pool)
    try:
        before=search(bank,family,np.eye(3),kind='forward',seconds=3.)
        after=search(bank,family,np.eye(3),kind='rebuild',seconds=3.)
        assert before['point']['objective']*9==pytest.approx(2.)
        assert before['stop_reason']=='structurally_complete'
        assert after['point']['objective']==pytest.approx(0.,abs=1e-8)
        assert any(set(s)=={1,2} for s in after['point']['supports'])
        assert all(a['warm_start']['valid'] and a['bound_consistent_with_known_upper'] for a in after['attempts'])
    finally:bank.close()


def test_large_polytomy_has_legal_bounded_regions():
    family=tuple((i,) for i in range(8))
    result=regions(family,8,4,np.eye(8))
    assert result and all(len(b)<=4 for _,b,_ in result)
    assert any(virtual for _,_,virtual in result)
    assert any_extension(family,8,()) is not None


def test_scoring_uses_labelled_clades_and_original_units():
    from research_experiments.rebuild_benchmark.analyze import score
    z=np.array([0.,1.,1.]);R=np.eye(3)+np.outer(z,z);X=.1*np.eye(3)
    data=core._prepare_scenarios([dict(P_terminal=np.eye(3),Q_terminal=np.zeros((3,3)),drop_target=R)])
    point=dict(supports=[(0,),(1,),(2,),(1,2)],r=[1.,1.,1.,1.],x=[.1,.1,.1,0.],objective=0.)
    truth=dict(R=R,X=X,clades={frozenset((11,12))})
    result=score(point,dict(train=data,validation=data,test=data),truth,[10,11,12],[(1,2)])
    assert result['clade_f1']==1. and result['exact']
    assert result['test_mae']==pytest.approx(0.)
    assert result['R_relative']==pytest.approx(0.) and result['X_relative']==pytest.approx(0.)
    assert score(None,{},truth,[10,11,12],[])['clade_f1']==0.


def test_full_pool_warm_start_reaches_a_candidate_after_the_first_four():
    from itertools import combinations
    p=np.eye(5);z=np.array([0.,0.,0.,1.,1.]);y=p+np.outer(z,z)
    data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=p*0,drop_target=y)])
    leaves=tuple((i,) for i in range(5));pool=[*leaves,*combinations(range(5),2)]
    bank=Bank(data,pool)
    try:
        result=search(bank,leaves,np.eye(5),kind='forward',seconds=3.,per_MIP=.1,warm_limit=None)
        first=result['attempts'][0]
        assert first['warm_start']['valid']
        assert first['warm_start']['objective']==pytest.approx(0.,abs=1e-9)
        assert first['accepted'] and result['point']['objective']==pytest.approx(0.,abs=1e-9)
    finally:bank.close()
