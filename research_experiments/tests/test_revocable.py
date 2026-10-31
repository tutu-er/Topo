import numpy as np
import pytest

pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.local_rebuild.matched import all_hierarchies
from research_experiments.revocable.engine import local_fit,clean_zero,search,neighborhoods
from research_experiments.rnj_count.engine import criterion


def make_bank(y,family):
    p=np.eye(len(y));data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=p*0,drop_target=y)])
    return FloorBank(data,family,dict(a=1.,b=1.,median_root_depth=1.))


def test_wrong_zero_boundary_is_released_and_crossing_truth_can_enter():
    p=np.eye(3);z=np.array([0.,1.,1.]);family=[(0,),(1,),(2,),(0,1),(0,1,2)]
    bank=make_bank(p+np.outer(z,z),family)
    try:
        old=bank.fit(family,preserve=True)
        point,event=local_fit(bank,old,tuple(range(3)),((0,),(1,),(2,)),penalty=.001,seconds=3.)
        assert event['accepted'] and (0,1) in event['released']
        assert (1,2) in point['supports'] and (0,1) not in point['supports']
        assert point['objective']==pytest.approx(0.,abs=1e-10)
        assert event['regularized_bound']<=criterion(point,3,.001)+1e-8
    finally:
        bank.close()


def test_verified_hard_boundary_stays_and_counts_in_the_local_capacity():
    p=np.eye(3);z=np.array([0.,1.,1.]);family=[(0,),(1,),(2,),(0,1),(0,1,2)]
    bank=make_bank(p+np.outer(z,z),family)
    try:
        old=bank.fit(family,preserve=True)
        point,event=local_fit(bank,old,tuple(range(3)),((0,),(1,),(2,)),penalty=.001,seconds=3.,hard=((0,1),))
        assert (0,1) in point['supports'] and (1,2) not in point['supports']
        assert not event['accepted']
    finally:
        bank.close()


def test_regularized_rebuild_matches_every_four_leaf_hierarchy():
    p=np.eye(4);z=np.array([1.,1.,0.,0.]);w=np.array([0.,0.,1.,1.])
    family=[(i,) for i in range(4)]+[(0,2),(0,1,2,3)]
    bank=make_bank(.3*p+.2*np.outer(z,z)+.01*np.outer(w,w),family)
    try:
        old=bank.fit(family,preserve=True);penalty=.003
        result,event=local_fit(bank,old,tuple(range(4)),tuple((i,) for i in range(4)),penalty=penalty,seconds=3.)
        base=[(i,) for i in range(4)]+[tuple(range(4))]
        exact=min(criterion(bank.fit([*base,*h],preserve=True),4,penalty) for h in all_hierarchies(4))
        assert criterion(result,4,penalty)==pytest.approx(exact,abs=1e-8)
    finally:
        bank.close()


def test_zero_cleanup_releases_only_provisional_supports():
    family=[(0,),(1,),(2,),(0,1),(0,1,2)];bank=make_bank(np.eye(3)*.3,family)
    try:
        old=bank.fit(family,preserve=True);point,removed=clean_zero(bank,old,hard=((0,1),))
        assert (0,1) in point['supports']
        point,removed=clean_zero(bank,old)
        assert (0,1) in removed and point['objective']==pytest.approx(old['objective'])
        assert (0,1,2) in point['supports']
        assert all((i,) in point['supports'] for i in range(3))
    finally:
        bank.close()


def test_subtree_size_is_not_a_releasability_limit():
    n=8;family=[(i,) for i in range(n)]+[(0,1,2),(3,4,5),(0,1,2,3,4,5),tuple(range(n))]
    point=dict(supports=family)
    candidates=neighborhoods(point,n,np.eye(n))
    assert any((0,1,2,3,4,5) not in blocks and set(v)>set((0,1,2,3,4,5)) and len(blocks)<=5
               for v,blocks,_ in candidates)


def test_six_terminal_wrong_boundary_can_be_replaced_in_a_four_block_milp():
    n=8;old_big=(0,1,2,3,4,5);truth_big=(3,4,5,6)
    blocks=((0,1,2),(3,4,5),(6,),(7,))
    family=[(i,) for i in range(n)]+[(0,1,2),(3,4,5),old_big,tuple(range(n))]
    y=np.eye(n)*.3
    for s,w in [((0,1,2),.2),((3,4,5),.2),(truth_big,.3)]:
        z=np.array([i in s for i in range(n)],float);y+=w*np.outer(z,z)
    bank=make_bank(y,family)
    try:
        old=bank.fit(family,preserve=True)
        point,event=local_fit(bank,old,tuple(range(n)),blocks,penalty=.001,seconds=3.)
        assert event['accepted'] and old_big in event['released']
        assert old_big not in point['supports'] and truth_big in point['supports']
        assert point['objective']==pytest.approx(0.,abs=1e-9)
    finally:
        bank.close()


def test_zero_full_set_stays_available_without_becoming_a_penalized_extension():
    p=np.eye(3);z=np.array([0.,1.,1.]);family=[(0,),(1,),(2,),(0,1),(0,1,2)]
    bank=make_bank(p+np.outer(z,z),family)
    try:
        result=search(bank,family,np.eye(3),kind='zero_only',coefficient=.003,seconds=3.)
        assert all((0,1,2) in point['supports'] for point in result['path'])
        assert (1,2) in result['point']['supports']
        assert not any((0,1,2) in event.get('added',[]) for event in result['attempts'])
    finally:
        bank.close()
