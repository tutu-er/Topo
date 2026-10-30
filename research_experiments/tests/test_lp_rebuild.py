import numpy as np
import pytest

gp=pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.lp_rebuild.engine import FloorBank,local_families,criterion,signal_coefficients,select_output,enumerate_search,InfeasibleFamily
from research_experiments.local_rebuild.model import build_model
from research_experiments.rebuild_benchmark.engine import solve,reconstruction_start

CAL=dict(a=1.,b=1.,median_root_depth=1.)


def prepared(p,q,y):return core._prepare_scenarios([dict(P_terminal=p,Q_terminal=q,drop_target=y)])


def test_four_block_lp_enumeration_matches_joint_milp_with_floor_and_sparsity():
    rng=np.random.default_rng(31);p=rng.uniform(.1,1.,(14,4));q=rng.uniform(.1,.6,(14,4))
    z=np.array([1.,1.,0.,0.]);R=.2*np.eye(4)+.12*np.outer(z,z);X=.1*np.eye(4)+.07*np.outer(z,z)
    data=prepared(p,q,p@R.T+q@X.T+rng.normal(0,.001,(14,4)))
    family=tuple((i,) for i in range(4));blocks=family;bank=FloorBank(data,family,CAL,floor=.08)
    try:
        families,_=local_families(bank,family,tuple(range(4)),blocks,False)
        assert len(families)==26
        penalty=1e-4;points=[bank.fit(f) for f in families];best=min(criterion(x,4,penalty) for x in points)
        model,meta=build_model(bank.scaled,family,tuple(range(4)),blocks,bound=2.)
        for k in range(len(meta['candidates'])):
            model.rows.add({model.r.start+len(meta['keep'])+k:1.,model.x.start+len(meta['keep'])+k:1.,meta['selection'].start+k:-.08},lower=0.)
            model.variables.objective[meta['selection'].start+k]=penalty*1000.
        start=reconstruction_start(bank,model,meta,bank.fit(family))
        _,record=solve(bank,model,start,seconds=3.)
        assert record['audit']['valid'] and record['status']==2
        assert record['objective']==pytest.approx(best,abs=1e-8)
    finally:bank.close()


def test_depth_floor_can_be_satisfied_by_an_unexcited_reactive_coefficient():
    p=np.eye(3);z=np.array([1.,1.,0.]);data=prepared(p,p*0,.5*p+.02*np.outer(z,z))
    family=[(0,),(1,),(2,),(0,1)];bank=FloorBank(data,family,CAL,floor=.1)
    try:
        point=bank.fit(family);i=point['supports'].index((0,1))
        assert point['objective']==pytest.approx(0.,abs=1e-9)
        assert point['r'][i]==pytest.approx(.02,abs=1e-8)
        assert point['x'][i]>=.08-1e-8
    finally:bank.close()


def test_fixed_projection_really_lower_bounds_centered_signal_rms():
    rng=np.random.default_rng(5);p=rng.normal(size=(17,4));q=-2*p+rng.normal(size=p.shape)*.1
    s=(0,2);r,x=.3,.11;gr,gx=signal_coefficients(p,q,s,CAL)
    z=np.array([1.,0.,1.,0.]);effect=(r*p[:,s].sum(axis=1)+x*q[:,s].sum(axis=1))[:,None]*z
    effect-=effect.mean(axis=0)
    assert gr*r+gx*x<=np.sqrt(np.mean(effect**2))+1e-12


def test_signal_floor_survives_final_refit_and_cannot_hide_in_unexcited_x():
    p=np.eye(3);z=np.array([1.,1.,0.]);data=prepared(p,p*0,.5*p+.02*np.outer(z,z))
    leaves=[(0,),(1,),(2,)];bank=FloorBank(data,leaves,CAL,floor=1.,mode='signal',reference=.05)
    try:
        family=[*leaves,(0,1)];point=bank.fit(family)
        output=select_output(bank,[point],data,resolution=0.)['point'];i=output['supports'].index((0,1))
        gr,gx=signal_coefficients(p,p*0,(0,1),CAL)
        assert gx==0. and gr*output['r'][i]>=.05-1e-8
        assert output['objective']>1e-4
        assert max(q.get('response_floor_violation',0.) for q in bank.lp.solves)<=1e-7
    finally:bank.close()


def test_unexcited_infeasible_families_are_skipped_not_called_timeouts():
    p=np.ones((5,3));data=prepared(p,p*0,p*.1);leaves=[(0,),(1,),(2,)]
    bank=FloorBank(data,[*leaves,(0,1),(0,2),(1,2)],CAL,floor=1.,mode='signal',reference=.05)
    try:
        with pytest.raises(InfeasibleFamily):bank.fit([*leaves,(0,1)])
        result=enumerate_search(bank,leaves,np.eye(3),seconds=3.)
        assert result['stop_reason']=='neighborhood_scan_complete'
        assert result['point']['objective']==pytest.approx(0.,abs=1e-9)
        assert all(q['status'] in (2,3) for q in bank.lp.solves)
        assert any(q['status']==3 for q in bank.lp.solves)
    finally:bank.close()


def test_sparsity_grid_uses_positive_activation_floors():
    from research_experiments.lp_rebuild.run import configs
    assert len(configs())==11
    assert all(c['floor']>0 for c in configs().values() if c['sparsity']>0)
