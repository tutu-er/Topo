"""Physical service, paired data, timing and independent accounting contracts."""
from copy import deepcopy

import numpy as np
import pytest

from . import ev_dynamic_study as study
from . import ev_fixed_train_design as design
from . import likelihood as lk


def test_saturated_baseline_and_half_amplitude_keep_battery_service():
    high=study.highlow_curve(6,3)
    constant=np.zeros_like(high);constant[:,0]=2.
    half=constant+.5*(high-constant)
    for u in (high,constant,half):
        op=study.operation_record(u)
        assert op['success'],op
        assert op['grid_energy_kwh']==pytest.approx(3.,abs=1e-7)
        assert op['energy_kwh'][-1]==pytest.approx(26.85,abs=1e-7)
        assert op['discharging_kwh']==pytest.approx(0.,abs=1e-8)
    assert np.var(half[:,0])==pytest.approx(.25*np.var(high[:,0]))
    assert np.count_nonzero(high[:,1:])==0
    assert np.sum(high[:,0]>0)==3


def test_continuous_process_and_equal_meter_budget():
    cfg=deepcopy(study.CONFIG);cfg['training_points']=8
    a=study.make_case(991701,1,cfg)
    bcfg=deepcopy(cfg);bcfg['public_voltage_repeat_counts']=[1,9]*3
    b=study.make_case(991701,1,bcfg)
    for field in ('p','q','source_p','source_q','root_v'):
        assert np.array_equal(a[field],b[field])
    for field in a['white']:
        assert np.array_equal(a['white'][field],b['white'][field])
    assert a['train'].t==8 and a['validation'].t==4 and a['future_start']==12
    assert np.array_equal(a['train'].p-a['tm'].bias_p,
                          b['train'].p-b['tm'].bias_p)
    assert sum(cfg['public_voltage_repeat_counts'])==sum(bcfg['public_voltage_repeat_counts'])==30
    assert np.all(a['fm'].v_sigma==a['fm'].v_sigma[0])
    assert np.ptp(a['p'][:,0])>0 and np.ptp(a['q'][:,0])>0


def test_call_accounting_counts_both_aliases_and_restores_after_failure():
    original=lk.ac_predict
    candidate=study.base.RadialCandidate.from_supports('star',3,())
    with pytest.raises(ValueError):
        with study.count_physics_calls() as counted:
            study.geometry.ac_predict(candidate,[],[],np.zeros((2,3)),np.zeros((2,3)),None,None,1.)
    assert counted['AC_batches']==1 and counted['AC_snapshots']==2 and counted['AC_failures']==1
    assert lk.ac_predict is original


def test_timing_reverse_preserves_signed_energy_and_throughput():
    u=study.highlow_curve(6,3);u[1,0]=-.5;u[3,0]=.5/(.95*.95)
    # The unmodified high/low starts with headroom in slot 3.
    forward=study.operation_record(u,discharge=True)
    reverse=study.operation_record(u[::-1],discharge=True)
    assert forward['success'] and reverse['success']
    for field in ('grid_energy_kwh','throughput_kwh','charging_kwh','discharging_kwh'):
        assert forward[field]==pytest.approx(reverse[field],abs=1e-7)


def test_complete_AC_price_matches_reoptimized_source_finite_difference():
    from .test_ev_fixed_train_design import context as example
    ctx,*_=example.__wrapped__()
    u=ctx['reference_kw'].copy();options=dict(multistart=2,max_nfev=180,tol=1e-12)
    result=study.complete_ac_prices(ctx,u,seed=64,fit_options=options)
    assert result['success']
    eps=.002;plus=u.copy();minus=u.copy();plus[1,0]+=eps;minus[1,0]-=eps
    high=design.evaluate_fixed_action(ctx,plus,seed=64,fit_options=options)
    low=design.evaluate_fixed_action(ctx,minus,seed=64,fit_options=options)
    assert high['success'] and low['success']
    fd=(high['mean']-low['mean'])/(2*eps)
    assert result['gradient_per_kw'][1,0]==pytest.approx(fd,rel=2e-4,abs=2e-5)


def test_trial_failure_is_preserved_and_hash_checked(monkeypatch):
    monkeypatch.setattr(study,'source_hashes',lambda:{'example':'fixed'})
    def broken(*args,**kwargs):raise ValueError('declared failure')
    monkeypatch.setattr(study,'_run_trial',broken)
    result=study.run_trial(31,0)
    assert result['status']=='driver_failure' and result['source_unchanged']
    assert result['failure']=='ValueError:declared failure'
    assert result['actual_solver_calls']['total_AC_batches']==0


def test_driver_never_passes_validation_or_future_to_planner(monkeypatch):
    from types import SimpleNamespace
    cfg=deepcopy(study.CONFIG);cfg.update(training_points=8,mean_policy_ablation=False,
        diagnostics=False,select_modes=['none','constant'])
    original=study.make_case(991710,1,cfg);changed=deepcopy(original)
    changed['p'][12:]+=.03;changed['q'][12:]+=.01
    changed['white']['v'][12:]*=-3
    changed['validation']=lk.Observations(**{**study.io.observations_record(changed['validation']),
        'p':changed['validation'].p+.02,'v':changed['validation'].v+.001})
    bank=[study.base.RadialCandidate.from_supports('star' if not family else
        'clade_'+''.join(map(str,family[0])),3,family) for family in study.base.FAMILIES]
    fits={b.name:SimpleNamespace(r=np.full(b.m,.004),x=np.full(b.m,.002),
        source_p=np.full(8,.1),source_q=np.full(8,.03),success=True,loss=1.) for b in bank}
    def fitting(candidates,obs,meter,*,trained=None,**kwargs):
        if trained is not None:
            assert trained is fits
            for name,f in trained.items():
                assert np.array_equal(f.r,fits[name].r) and np.array_equal(f.x,fits[name].x)
        return fits,dict(all_complete=True,fits=[dict(candidate=b.name,loss=1.) for b in bank],cost={})
    seen=[]
    def selector(training,candidates,trained,train_meter,future_meter,**kwargs):
        assert training is current[0]['train'] and trained is fits
        assert 'validation' not in kwargs and 'truth' not in kwargs
        seen.append((training.p.copy(),deepcopy(kwargs['forecast'])))
        const=np.zeros((6,3));const[:,0]=2.
        return dict(none=np.zeros_like(const),constant=const),dict(methods={})
    def actual(case,candidates,trained,validated,power,config,**kwargs):
        assert trained is fits
        return dict(status='complete',M=0.)
    monkeypatch.setattr(study,'source_hashes',lambda:{'example':'unchanged'})
    monkeypatch.setattr(study,'propose',lambda *a,**k:(bank,{}))
    monkeypatch.setattr(study.old,'fit_bank',fitting)
    monkeypatch.setattr(study.design,'select_fixed_controls',selector)
    monkeypatch.setattr(study,'actual_curve',actual)
    current=[original]
    monkeypatch.setattr(study,'make_case',lambda *a:current[0])
    first=study.run_trial(991710,1,cfg);current[0]=changed
    second=study.run_trial(991710,1,cfg)
    assert first['status']==second['status']=='complete'
    assert np.array_equal(seen[0][0],seen[1][0])
    assert study.io.serial(seen[0][1])==study.io.serial(seen[1][1])
    assert [m['power_kw'] for m in first['methods']]==[m['power_kw'] for m in second['methods']]
