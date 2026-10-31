import numpy as np
import pytest
pytest.importorskip('gurobipy')
from itertools import combinations
from time import perf_counter
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.local_rebuild.model import canonical
from research_experiments.rebuild_benchmark.engine import solve,audit_vector
from research_experiments.wzzt_retune.engine import extension_start
from research_experiments.rnj_count.engine import criterion
from research_experiments.rnj_small_seed.engine import seed_family,frontier_model,exit_repair,repair_windows


def bank_for(n,atoms):
    p=np.eye(n);y=.2*p
    for C,r in atoms:
        z=np.array([i in C for i in range(n)],float);y+=r*np.outer(z,z)
    family=canonical([*((i,) for i in range(n)),tuple(range(n)),*[C for C,r in atoms]])
    data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=0*p,drop_target=y)])
    b=FloorBank(data,family,dict(a=1.,b=1.,median_root_depth=1.));b.seed_start_pool=family;b.joint_exit_seed_pool=family
    return b,family


def test_seed_selection_removes_rnj_ancestors_and_retains_disjoint_small_groups():
    bank,full=bank_for(7,[((0,1),.2),((0,1,2),.4),((3,4),.3),((0,1,2,3,4,5),.2)])
    try:
        point,event=seed_family(bank,full,mode='small3',reference=.01,deadline=perf_counter()+10)
        seeds=event['seeds'];assert len(seeds)<=2 and all(2<=len(C)<=3 for C in seeds)
        assert all(not set(C)&set(D) for C,D in combinations(seeds,2))
        assert (0,1,2,3,4,5) not in point['supports']
        assert all(r['deletion_cost']>=-1e-8 for r in event['rankings'])
    finally:bank.close()


def test_largest_frontier_model_matches_complete_single_union_enumeration():
    bank,_=bank_for(6,[((0,1,2),.4)])
    family=canonical([*((i,) for i in range(6)),tuple(range(6))]);blocks=tuple((i,) for i in range(6));penalty=.001
    try:
        current=bank.fit(family,preserve=True);model,a,lower=frontier_model(bank.scaled,family,tuple(range(6)),blocks,penalty,large=True)
        new=(0,1,2);start=extension_start(model,family,new,current,bank.lp.p,bank.lp.q,bank.lp.y);start[a]=0
        values,event=solve(bank,model,start,seconds=3.);assert event['audit']['valid']
        chosen=core._round_and_validate_support(values[model.z],family,6)
        fit=bank.fit([*family,chosen],preserve=True)
        exact=min([criterion(current,6,penalty),*[criterion(bank.fit([*family,C],preserve=True),6,penalty)
            for k in range(lower,6) for C in combinations(range(6),k)]])
        assert criterion(fit,6,penalty)==pytest.approx(exact,abs=1e-8)
    finally:bank.close()


def test_forward_atomic_group_cannot_be_split_but_exit_pool_opens_it():
    bank,full=bank_for(7,[((0,1),.2)])
    family=canonical([*((i,) for i in range(7)),tuple(range(7)),(0,1)]);blocks=((0,1),(2,),(3,),(4,),(5,),(6,))
    try:
        point=bank.fit(family,preserve=True);model,a,_=frontier_model(bank.scaled,family,tuple(range(7)),blocks,.001,large=False)
        illegal=extension_start(model,family,(1,2),point,bank.lp.p,bank.lp.q,bank.lp.y);illegal[a]=0
        assert not audit_vector(model,illegal)['valid']
        pool,windows=repair_windows(family,np.eye(7),7)
        assert (1,2) in pool and any({0,1,2}<=set(W) for W in windows)
    finally:bank.close()


def test_exit_replaces_wrong_seed_instead_of_merely_contracting_it():
    bank,_=bank_for(3,[((1,2),.4)]);initial=canonical([*((i,) for i in range(3)),(0,1),tuple(range(3))])
    try:
        current=bank.fit(initial,preserve=True)
        q,event=exit_repair(bank,initial,current,np.eye(3),reference=current['objective'],prefix_seconds=0.)
        assert q['accepted'] and q['audit']['valid']
        assert (1,2) in q['point']['supports'] and (0,1) not in q['point']['supports']
        assert q['point']['objective']<1e-8
    finally:bank.close()


def test_optional_known_feasible_start_is_retained_and_labelled():
    bank,true=bank_for(3,[((1,2),.4)]);initial=canonical([*((i,) for i in range(3)),(0,1),tuple(range(3))])
    try:
        current=bank.fit(initial,preserve=True);known=bank.fit(true,preserve=True)
        q,event=exit_repair(bank,initial,current,np.eye(3),reference=current['objective'],prefix_seconds=0.,warm_candidates=[known])
        assert q['warm_origin']=='complete_rnj'
        assert q['raw_score']<=q['warm_score']+1e-8 and q['score']<=q['before_score']+1e-8
        assert q['audit']['valid']
    finally:bank.close()


def test_long_budget_can_accept_an_incumbent_after_the_old_40_second_cutoff():
    bank,_=bank_for(3,[((1,2),.4)]);initial=canonical([*((i,) for i in range(3)),(0,1),tuple(range(3))])
    try:
        current=bank.fit(initial,preserve=True)
        q,event=exit_repair(bank,initial,current,np.eye(3),reference=current['objective'],prefix_seconds=41.,total_seconds=300.)
        assert q['accepted'] and 40.<q['incumbent_at']<=300.
        assert q['point']['objective']<1e-8
    finally:bank.close()
