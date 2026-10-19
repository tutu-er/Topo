"""Independent checks of quality inputs, physical execution and history units."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from . import information_study as study
from .information_context import quality_models
from .simulation import model, observe
from .single_site_milp import SingleSitePlan, constant_site_plan


def test_public_quality_repeat_changes_no_history_truth_or_future_standard_noise():
    cfg=study.configuration();a=study.make_path(930002,cfg,case='moderate',truth_family=((0,1),),quality='uniform')
    b=study.make_path(930002,cfg,case='moderate',truth_family=((0,1),),quality='spatial')
    for name in ('r','x','p','q','source_p','source_q'):
        assert np.array_equal(getattr(a[0],name),getattr(b[0],name))
    assert a[1]==b[1] and a[4]==b[4]
    assert all(np.array_equal(a[5][k],b[5][k]) for k in a[5])
    ha,_=observe(a[0],slice(0,12));hb,_=observe(b[0],slice(0,12))
    assert np.array_equal(ha.v,hb.v) and np.array_equal(ha.p,hb.p)


def test_future_execution_adds_EV_once_and_uses_actual_saved_sigma():
    cfg=study.configuration();path,seeds,case,q,perm,white,amp=study.make_path(930002,cfg,quality='temporal')
    history,_=observe(path,slice(0,12));future,joint,quality=quality_models(history,model(3),kind=q,row_permutation=perm)
    con=study.old.device_constraints(cfg);plan=constant_site_plan(0,4,3,con)['plan']
    observed,physical=study.future_observe(path,plan,future,white)
    assert np.allclose(observed.p-future.bias_p-future.p_sigma*white['p'],path.p[12:]+plan.p,atol=1e-14)
    for name in ('p','q','v','master_p','master_q'):
        bias=getattr(future,'bias_'+name) if name in ('p','q','v') else 0
        residual=(getattr(observed,name)-physical[name]-bias)/getattr(future,name+'_sigma')
        assert np.allclose(residual,white[name],atol=1e-9)
    assert quality['v_total_raw_reads']==60
    assert np.allclose(joint.v_sigma[:12],model(3).v_sigma)


def test_same_shape_cheap_rules_and_averaging_keep_independent_meter_bias():
    path,*_=study.make_path(933001,study.configuration(),quality='uniform')
    h,_=observe(path,slice(0,12));meters=[quality_models(h,model(3),kind=k) for k in study.QUALITY]
    assert all(np.array_equal(f.bias_v,model(3).bias_v) for f,j,q in meters)
    assert all(q['v_total_raw_reads']==60 for f,j,q in meters)
    assert all(np.array_equal(meters[0][0].p_sigma,f.p_sigma) for f,j,q in meters)


def fake_trial(hid,quality,margin):
    records=[]
    for m in study.METHODS:
        records.append(dict(method=m,status='complete',true_name='true',true_covered=True,
            true_gap=0.,signed_margin_full_SSE=margin if m=='information_designed' else 0.,
            losses={'true':0.,'wrong':margin},seconds=0.,ac_calls=0))
    return dict(history_id=str(hid),run_id=str(hid)+'_'+quality,quality=quality,case='moderate',
        complete=True,true_covered=True,records=records,selection=dict(seconds=0.,
            methods={m:dict(status='optimized') for m in study.METHODS}),seconds=0.,total_ac_calls=0)


def test_pilot_paired_SE_uses_H_units_after_quality_average():
    trials=[fake_trial(1,q,v) for q,v in zip(study.QUALITY,[1.,4.,7.])]
    trials += [fake_trial(2,q,v) for q,v in zip(study.QUALITY,[3.,6.,9.])]
    result=study.summarize(trials);pair=result['paired_effects'][0]
    assert result['run_count']==6 and result['independent_history_count']==2
    assert pair['paired_condition_count']==6 and pair['paired_count']==2
    assert pair['signed_margin_effect']['history_count']==2
    assert pair['signed_margin_effect']['effect']==pytest.approx(5.)
    assert pair['signed_margin_effect']['paired_standard_error']==pytest.approx(1.)


def test_optimizer_failure_is_reported_even_when_joint_fit_succeeds():
    trial=fake_trial(1,'uniform',1.)
    trial['selection']['methods']['information_designed']['status']='selection_failure_fallback'
    row=study.summarize([trial])['method_summary']['information_designed']
    assert row['complete_count']==1 and row['failed_count']==0 and row['selection_failures']==1


def test_select_interface_contains_no_future_truth_path_or_noise():
    import inspect
    parameters=inspect.signature(study.select_controls).parameters
    assert not any(name in parameters for name in ('truth','path','noise','white','future_observations'))


def test_actual_voltage_violation_is_physical_failure_and_infinite_calibration_score(monkeypatch):
    cfg=study.configuration();c=study.old.RadialCandidate.from_supports('star',3,()) if hasattr(study.old,'RadialCandidate') else None
    from .likelihood import RadialCandidate
    c=RadialCandidate.from_supports('star',3,())
    fit=SimpleNamespace(success=True)
    monkeypatch.setattr(study,'propose',lambda *a,**k:([c],{}))
    monkeypatch.setattr(study.old,'fit_all_candidates',lambda *a,**k:({'star':fit},{},dict(ac_calls=0,fit_calls=1,seconds=0.)))
    monkeypatch.setattr(study.old,'fit_record',lambda *a,**k:dict(name='star'))
    plan=constant_site_plan(0,4,3,study.old.device_constraints(cfg))['plan']
    choices={m:plan for m in study.METHODS}
    selection=dict(seconds=0.,selector_ac_calls=0,methods={m:dict(status='optimized') for m in study.METHODS})
    monkeypatch.setattr(study,'select_controls',lambda *a,**k:(choices,selection))
    physical=dict(voltage=np.full((4,4),.89,dtype=complex))
    future=SimpleNamespace()
    monkeypatch.setattr(study,'future_observe',lambda *a,**k:(future,physical))
    result=study.run_trial(938101,case='star',truth_family=(),quality='uniform')
    assert not result['complete'] and result['relative_score'] is None
    assert result['calibration_score_is_infinite']
    assert all(r['status']=='physical_failure' and not r['operational_voltage_ok'] for r in result['records'])




def test_search_exception_preserves_spent_AC_calls_without_double_setup(monkeypatch):
    from .test_information_context import fixture
    h,m,cs,fs=fixture()
    from .information_context import build_centres,quality_models
    centres,_=build_centres(h,cs,fs,m,refit=False)
    future,_,quality=quality_models(h,m)
    monkeypatch.setattr(study,'build_centres',lambda *a,**k:(centres,dict(ac_calls=0,fit_calls=0,reason='fixture')))
    class Counter:
        def __init__(self,*a,**k):
            self._ac_calls=2;self.initialization_ac_calls=2
    def failed_search(evaluator,*a,**k):
        evaluator._ac_calls+=5
        raise RuntimeError('after five real AC attempts')
    monkeypatch.setattr(study,'StructuredInformationEvaluator',Counter)
    monkeypatch.setattr(study,'optimize_price_curves',failed_search)
    def no_profile(*a,**k):raise ValueError('no profile calls')
    monkeypatch.setattr(study,'GradientPriceEvaluator',no_profile)
    cfg=study.configuration()
    plans,record=study.select_controls(h,cs,fs,m,future,quality,
        constraints=study.old.device_constraints(cfg),config=cfg)
    assert record['selector_ac_calls']==28  # four searches * (two setup + five attempted evaluations)
    assert all(record['searches'][tag]['partial_ac_calls_retained']==5 for tag in ('seeds','robust','nominal','uniform'))
    assert all(plans[method] is not None for method in study.METHODS)
