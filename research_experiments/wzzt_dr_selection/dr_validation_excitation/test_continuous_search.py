"""Analytic checks of frozen validation tables and training-only separation."""
from types import SimpleNamespace
import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_validation_excitation import continuous_search as cs


def test_empirical90_uses_second_smallest_of_ten():
    assert cs.quantile_margin(np.arange(10)) == 1.
    assert cs.quantile_margin([10.,-5.]) == -5.


def synthetic():
    n=2
    p=np.ones((96,n))*.1;q=np.ones_like(p)*.02
    rh=np.array([[.5,.2],[.2,.4]]);xh=np.array([[.2,.1],[.1,.3]])
    y=p@rh.T+q@xh.T+np.linspace(-.01,.01,96)[:,None]
    hp=np.zeros((4,96,n,16));hq=np.zeros_like(hp)
    hp[2,:12,0,8]=1.;hq[2,:12,0,8]=.35
    hp[0,12:24,1,2]=-1.;hq[0,12:24,1,2]=-.35
    scenario=dict(scenario=0,days=[(p,q,y)]*4,multiplier=np.ones((4,16)))
    ctx=dict(source=dict(labels=[1,2],base_kw=200),r_h=rh,x_h=xh,hp=hp,hq=hq,
        past_train=(p[:3],q[:3],y[:3]),past_valid=(p[:3],q[:3],y[:3]),
        problem=SimpleNamespace(elasticities=np.ones(16)*.1))
    fits=[dict(family_id=0,has_target=False,r_matrix=np.diag([.7,.6]),x_matrix=np.diag([.3,.4])),
          dict(family_id=1,has_target=True,r_matrix=rh,x_matrix=xh)]
    return ctx,scenario,fits


def test_affine_validation_replays_exact_absolute_losses(monkeypatch):
    ctx,scenario,fits=synthetic()
    fixed=np.zeros(16);fixed[2]=.12
    monkeypatch.setattr(cs,'fit_families',lambda *a,**k:(fits,.1,[]))
    tables,_=cs.affine_tables(ctx,[scenario],fixed)
    amp=fixed.copy();amp[8]=.14
    for entry,fit in zip(tables[0]['families'],fits):
        p,q,y=ctx['past_valid'];parts=[p@fit['r_matrix'].T+q@fit['x_matrix'].T-y]
        for event in [2,3]:
            p,q,y=cs.event_data(ctx,scenario,event,amp)
            parts.append(p@fit['r_matrix'].T+q@fit['x_matrix'].T-y)
        actual=np.abs(np.vstack(parts)).mean()
        affine=entry['constant_loss']+np.abs(entry['residual_base']+
            entry['residual_price']@(amp[8:]/.1)).sum()/tables[0]['validation_size']
        assert affine == pytest.approx(actual,abs=1e-14)


def test_training_data_never_depends_on_validation_prices():
    ctx,scenario,_=synthetic()
    a=np.zeros(16);a[2]=.12
    b=a.copy();b[8]=.14
    for first,second in zip(cs.training_data(ctx,scenario,a),cs.training_data(ctx,scenario,b)):
        np.testing.assert_array_equal(first,second)


def test_acceptance_failure_keeps_full_scenario_mass():
    problem=SimpleNamespace(elasticities=np.ones(16)*.1)
    family=lambda f,target,c:dict(family_id=f,has_target=target,constant_loss=c,
        residual_base=np.array([]),residual_price=np.empty((0,8)))
    tables=[dict(scenario=s,validation_size=10,acceptance_margin=-1. if s<2 else 1.,
        families=[family(0,False,2.),family(1,True,1.)]) for s in range(10)]
    result=cs.evaluate_tables(problem,tables,np.zeros(16))
    assert result['denominator']==10 and result['passes']==8
    assert result['quantile_margin'] is None
