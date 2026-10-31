import numpy as np
import pytest
pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.local_rebuild.matched import all_hierarchies
from research_experiments.local_rebuild.model import canonical
from research_experiments.rebuild_benchmark.engine import SCALE,reconstruction_start,solve
from research_experiments.joint_exit.model import build,joint_exit
from research_experiments.joint_exit.selection import choose
from research_experiments.rnj_count.engine import criterion


def bank_for(y,family):
    p=np.eye(len(y));data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=p*0,drop_target=y)])
    return FloorBank(data,family,dict(a=1.,b=1.,median_root_depth=1.))


def test_joint_domain_releases_crossing_old_boundary_and_matches_all_trees():
    n=4;p=np.eye(n);z=np.array([0.,1.,1.,0.]);family=[(i,) for i in range(n)]+[(0,1),tuple(range(n))]
    bank=bank_for(.3*p+.2*np.outer(z,z),family)
    try:
        old=bank.fit(family,preserve=True);pool=canonical([*family,*[s for h in all_hierarchies(n) for s in h]])
        penalty=.001;model,meta=build(bank.scaled,pool,penalty=penalty)
        values,event=solve(bank,model,reconstruction_start(bank,model,meta,old),seconds=3.)
        selected=[meta['candidates'][k] for k,a in enumerate(values[meta['selection']]) if a>.5]
        point=bank.fit([*meta['keep'],*selected],preserve=True)
        exact=min(criterion(bank.fit([*meta['keep'],*h],preserve=True),n,penalty) for h in all_hierarchies(n))
        assert criterion(point,n,penalty)==pytest.approx(exact,abs=1e-8)
        assert (1,2) in selected and (0,1) not in selected
        assert event['dual_bound']<=exact+1e-8
    finally:bank.close()


def test_hard_prior_is_never_released():
    n=3;family=[(i,) for i in range(n)]+[(0,1),tuple(range(n))]
    z=np.array([0.,1.,1.]);bank=bank_for(np.eye(n)+np.outer(z,z),family)
    try:
        old=bank.fit(family,preserve=True);model,meta=build(bank.scaled,[*family,(1,2)],penalty=.001,hard=((0,1),))
        values,event=solve(bank,model,reconstruction_start(bank,model,meta,old),seconds=3.)
        assert values[meta['selection']][meta['candidates'].index((0,1))]>.5
        assert values[meta['selection']][meta['candidates'].index((1,2))]<.5
    finally:bank.close()


def test_checkpoint_keeps_prior_feasible_solution_and_basis():
    n=3;family=[(i,) for i in range(n)]+[(0,1),tuple(range(n))];z=np.array([0.,1.,1.])
    bank=bank_for(np.eye(n)+np.outer(z,z),family)
    try:
        old=bank.fit(family,preserve=True)
        outputs,event=joint_exit(bank,family,old,np.eye(n),policy='expanded',coefficient=.003,
            reference=old['objective'],prefix_seconds=.1,budgets=(1.,3.))
        for checkpoint in outputs.values():
            assert checkpoint['audit']['valid']
            assert tuple(range(n)) in checkpoint['point']['supports']
            assert checkpoint['score']<=checkpoint['before_score']+1e-8
        assert (1,2) in outputs['3']['point']['supports']
    finally:bank.close()


def test_paired_simplicity_rule_uses_paired_prediction_errors():
    rows=[dict(index=0,validation=1.,internal=3,blocks=[.9,1.1]*12),
          dict(index=1,validation=1.0001,internal=2,blocks=[.9001,1.1001]*12),
          dict(index=2,validation=1.1,internal=1,blocks=[1.,1.2]*12)]
    assert choose(rows,'strict')['index']==0
    assert choose(rows,'paired')['index']==1


def test_rank_one_excitation_has_an_exact_distinct_topology_counterexample():
    from research_experiments.joint_exit.information import exact_rank_one_example
    example=exact_rank_one_example()
    assert example['P_rank']==1 and example['true_r']>0
    assert example['maximum_prediction_difference']<1e-12
    assert example['rich_excitation_prediction_difference']>.1


def test_only_audited_roundoff_is_projected_to_physical_bounds():
    from research_experiments.joint_exit.model import physical_weights
    weights,violation=physical_weights([-1e-12,.2,2.+1e-12])
    assert weights.tolist()==[0.,.2,2.] and violation<1e-7
    with pytest.raises(ValueError):physical_weights([-.001])
