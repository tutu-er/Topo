"""Cheap independent driver tests: interfaces, isolation, failure and freezing."""
from dataclasses import replace
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from . import gradient_price_study as study
from .likelihood import RadialCandidate
from .single_site_milp import constant_site_plan


def candidates():
    return [RadialCandidate.from_supports(name,3,family) for name,family in
            [('star',()),('clade01',((0,1),)),('clade02',((0,2),)),('clade12',((1,2),))]]


def fake_fit(c,obs,loss=1.,success=True):
    fit=SimpleNamespace(r=np.full(c.m,.01),x=np.full(c.m,.006),p=obs.p.copy(),q=obs.q.copy(),
        source_p=np.full(obs.t,.1),source_q=np.full(obs.t,.03),bias_p=np.zeros(3),bias_q=np.zeros(3),bias_v=np.zeros(3),
        loss=float(loss),loglik=-.5*loss,success=success,predicted=dict(p=obs.p,q=obs.q,v=obs.v,
        master_p=obs.master_p,master_q=obs.master_q),diagnostics={})
    fit.summary=lambda:dict(loss=fit.loss,loglik=fit.loglik,success=fit.success,r=fit.r,x=fit.x,diagnostics={})
    return fit


def cheap_trial_components(monkeypatch):
    calls=[]
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'v1'})
    monkeypatch.setattr(study,'propose',lambda history,**kwargs:(candidates(),dict(uses_truth=False)))
    def bank(cs,obs,meter,**kwargs):
        fs={c.name:fake_fit(c,obs,i+1) for i,c in enumerate(cs)}
        return fs,{},dict(ac_calls=4,fit_calls=4,seconds=0.)
    monkeypatch.setattr(study,'fit_all_candidates',bank)
    class Evaluator:
        def __init__(self,history,cs,fs,meter,*,horizon,**kwargs):
            self.history=history;self.horizon=horizon;self.base_kva=20.;self.q_over_p=0.;self.reference_names=[c.name for c in cs]
    monkeypatch.setattr(study,'GradientPriceEvaluator',Evaluator)
    def optimize(evaluator,constraints,*,sites,settings):
        calls.append(dict(mode=settings.gradient_mode,history=evaluator.history.p.copy(),references=evaluator.reference_names))
        plans={j:constant_site_plan(j,evaluator.horizon,3,constraints)['plan'] for j in sites}
        evaluations={j:dict(score=float(j)) for j in plans}
        return dict(success=True,plan=plans[1],site_plans=plans,site_evaluations=evaluations,
            selected_site=1,score=1.,fit_calls=4,total_ac_calls=8,failures=[],settings=dict(mode=settings.gradient_mode))
    monkeypatch.setattr(study,'optimize_price_curves',optimize)
    return calls


def simple_trial(seed,pid='policy',*,complete=True):
    records=[]
    for method in study.METHODS:
        records.append(dict(method=method,status='complete' if complete else 'joint_fit_failure',
            losses=dict(truth=1.,wrong=2.),gaps=dict(truth=0.,wrong=1.),true_name='truth',true_gap=0.,
            true_covered=True,seconds=0.,ac_calls=0))
    return dict(history_id=str(seed),seed=seed,policy_id=pid,complete=complete,true_covered=True,
        relative_score=0. if complete else np.inf,case='star',seconds=0.,total_ac_calls=0,records=records,
        selection=dict(activated=True,methods={m:dict(status='optimized') for m in study.METHODS}))


def test_selector_signature_has_only_historical_information():
    parameters=inspect.signature(study.select_controls).parameters
    assert 'truth' not in parameters and 'path' not in parameters and 'future_noise' not in parameters
    assert set(parameters)=={'history','candidates','historical_fits','meter','constraints','horizon','settings','fit_options','gate_half_sse','reference_policy','seed'}


def test_complete_trial_runs_all_six_methods_and_four_candidate_fits(monkeypatch,tmp_path):
    calls=cheap_trial_components(monkeypatch)
    trial=study.run_trial(42,case='strong',truth_family=((0,1),),output=tmp_path)
    assert trial['complete'] and len(trial['records'])==6
    assert trial['independent_history_unit'] and not trial['selection_uses_future_noise']
    assert len(trial['historical_fits'])==4
    assert all(len(r['fits'])==4 for r in trial['records'])
    assert [c['mode'] for c in calls]==['full','pq_only']
    assert trial['selection']['methods']['optimized_fixed']['selected_site']==0
    assert trial['selection']['methods']['optimized_designed']['selected_site']==1
    loaded=study.read_json(tmp_path/'trial_42.json.gz')
    assert loaded['source_sha256']=={'fake.py':'v1'}
    assert len(loaded['history_observations']['p'])==12
    assert all(len(r['observations']['p'])==4 for r in loaded['records'])
    assert trial['records'][0]['validation']['energy_pu_hours']==0.
    assert all(r['validation']['energy_pu_hours']==pytest.approx(.15) for r in trial['records'][1:])


def test_future_noise_change_preserves_history_and_selected_controls(monkeypatch):
    calls=cheap_trial_components(monkeypatch)
    first=study.run_trial(43,case='weak',truth_family=((0,2),))
    original=study._make_path
    def changed(*a,**k):
        path,seeds,label=original(*a,**k)
        path.noise['v'][path.history_count:]+=.01
        return path,seeds,label
    monkeypatch.setattr(study,'_make_path',changed)
    second=study.run_trial(43,case='weak',truth_family=((0,2),))
    assert first['history_hash']==second['history_hash']
    assert first['seeds']==second['seeds']
    assert [study.serial(r['action']) for r in first['records']]==[study.serial(r['action']) for r in second['records']]
    assert first['records'][0]['observation_hash']!=second['records'][0]['observation_hash']
    assert np.array_equal(calls[0]['history'],calls[2]['history'])


def test_generator_policy_and_evaluation_noise_streams_are_distinct():
    path,seeds,label=study._make_path(1,study.configuration(),case='star',truth_family=())
    assert len(set(seeds.values()))==3 and label=='star'
    assert len(path.p)==16 and path.history_count==12
    assert len(path.r)==4


def test_gate_deactivates_without_search_when_history_gap_large(monkeypatch):
    cheap_trial_components(monkeypatch)
    path,_,_=study._make_path(11,study.configuration());history,_=study.observe(path,slice(0,12));cs=candidates()
    fits={c.name:fake_fit(c,history,100*i) for i,c in enumerate(cs)}
    def forbidden(*a,**k):raise AssertionError('gate should prevent search')
    monkeypatch.setattr(study,'optimize_price_curves',forbidden)
    choices,record=study.select_controls(history,cs,fits,study.model(3),constraints=study.device_constraints(study.configuration()),horizon=4)
    assert not record['activated'] and record['gate_reason']=='historical_gap_above_gate'
    assert all(np.array_equal(choices[m].p,choices['constant_fixed'].p) for m in study.METHODS[2:])


def test_historical_failure_remains_and_causes_constant_fallback(monkeypatch):
    cheap_trial_components(monkeypatch)
    path,_,_=study._make_path(12,study.configuration());history,_=study.observe(path,slice(0,12));cs=candidates()
    fits={c.name:fake_fit(c,history,i+1,success=i!=2) for i,c in enumerate(cs)}
    choices,record=study.select_controls(history,cs,fits,study.model(3),constraints=study.device_constraints(study.configuration()),horizon=4)
    assert not record['activated'] and record['gate_reason']=='history_fit_failure'
    assert choices['optimized_designed'] is choices['constant_fixed']


def test_failed_search_is_retained_and_falls_back(monkeypatch):
    cheap_trial_components(monkeypatch)
    def failed(*a,**k):raise RuntimeError('declared search failure')
    monkeypatch.setattr(study,'optimize_price_curves',failed)
    trial=study.run_trial(13,case='star')
    assert trial['selection']['activated']
    assert trial['selection']['methods']['optimized_designed']['status']=='selection_failure_fallback'
    assert 'declared search failure' in trial['selection']['searches']['full']['error']
    assert trial['complete']


def test_joint_fit_failure_maps_calibration_score_to_infinity(monkeypatch):
    cheap_trial_components(monkeypatch)
    original=study.fit_all_candidates
    def failed(cs,obs,meter,**kwargs):
        fs,failures,cost=original(cs,obs,meter,**kwargs)
        if obs.t==16:fs[cs[0].name].success=False;failures[cs[0].name]='declared failed fit'
        return fs,failures,cost
    monkeypatch.setattr(study,'fit_all_candidates',failed)
    trial=study.run_trial(14)
    assert not trial['complete'] and np.isinf(trial['relative_score'])
    assert len(trial['records'])==6
    assert all(r['status']=='joint_fit_failure' for r in trial['records'])


def test_fit_counter_keeps_exception_work_and_restores(monkeypatch):
    cs=candidates();history=SimpleNamespace()
    def physical(*a,**k):return {}
    monkeypatch.setattr(study.likelihood,'ac_predict',physical)
    def fail(*a,**k):study.likelihood.ac_predict();raise RuntimeError('AC defect')
    monkeypatch.setattr(study.likelihood,'fit_candidate',fail)
    fs,failures,cost=study.fit_all_candidates(cs,history,None)
    assert not fs and len(failures)==4 and cost['ac_calls']==4
    assert study.likelihood.ac_predict is physical


def test_calibration_uses_single_alpha_and_keeps_infinite_failure():
    rows=[simple_trial(i) for i in range(12)]
    gate=study.calibration_result(rows)
    assert gate['relative']['rank']==12 and not gate['relative']['vacuous']
    rows[0]=simple_trial(0,complete=False)
    gate=study.calibration_result(rows)
    assert gate['failed_history_count']==1 and gate['all_history_count']==12
    assert gate['relative']['vacuous']


def test_duplicate_or_mixed_calibration_histories_rejected():
    with pytest.raises(ValueError,match='duplicate'):study.calibration_result([simple_trial(1),simple_trial(1)])
    with pytest.raises(ValueError,match='mixed'):study.calibration_result([simple_trial(1),simple_trial(2,pid='other')])


def test_summary_rejects_overlap_and_policy_drift():
    calibration=study.calibration_result([simple_trial(i) for i in range(12)])
    with pytest.raises(ValueError,match='overlap'):study.summarize_trials([simple_trial(0)],calibration)
    with pytest.raises(ValueError,match='hash'):study.summarize_trials([simple_trial(20,pid='other')],calibration)


def test_summary_retains_full_failure_denominator():
    summary=study.summarize_trials([simple_trial(1),simple_trial(2,complete=False)])
    assert summary['all_history_count']==2 and summary['failed_history_count']==1
    assert all(r['all_history_count']==2 and r['failed_count']==1 for r in summary['method_summary'])


def test_formal_freeze_and_resume_source_drift_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'v1'})
    def trial(seed,*,config,output,**kwargs):
        pid=study.identity(study.configuration(config),study.source_hashes())
        result=simple_trial(seed,pid)
        if output is not None:study.write_json(output/f'trial_{seed}.json.gz',result,compressed=True)
        return result
    monkeypatch.setattr(study,'run_trial',trial)
    rows,summary=study.run_batch('calibration',tmp_path,count=12)
    assert len(rows)==12 and (tmp_path/'freeze.json').exists()
    rows2,_=study.run_batch('evaluation',tmp_path,count=2)
    assert len(rows2)==2
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'v2'})
    with pytest.raises(ValueError,match='changed'):study.run_batch('calibration',tmp_path,count=12)
    with pytest.raises(ValueError,match='changed|frozen'):study.run_batch('evaluation',tmp_path,count=3,start_seed=830000)


def test_evaluation_requires_calibration_freeze(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'v1'})
    with pytest.raises(ValueError,match='calibration freeze'):study.run_batch('evaluation',tmp_path,count=1)


def test_gzip_is_reproducible_and_preserves_failure_flag(tmp_path):
    value=dict(score=np.inf,is_infinite=True,array=np.array([1.,2.]))
    study.write_json(tmp_path/'a.json.gz',value,compressed=True)
    study.write_json(tmp_path/'b.json.gz',value,compressed=True)
    assert (tmp_path/'a.json.gz').read_bytes()==(tmp_path/'b.json.gz').read_bytes()
    assert study.read_json(tmp_path/'a.json.gz')==dict(score=None,is_infinite=True,array=[1.,2.])




def test_failed_evaluation_retains_all_and_uses_all_covered_denominator():
    gate=study.calibration_result([simple_trial(i) for i in range(12)])
    failed=simple_trial(20,complete=False)
    study.apply_trial_gate(failed,gate)
    assert not failed['any_method_false_rejection']
    assert all(r['retained_candidates']==['truth','wrong'] for r in failed['records'])
    summary=study.summarize_trials([failed],gate)
    assert summary['policy_false_rejection_denominator']==1
    assert all(r['false_rejection_denominator']==1 and r['incomplete_retained_all_count']==1 for r in summary['method_summary'])


def test_gate_rejects_wrong_competitor_and_preserves_truth():
    gate=study.calibration_result([simple_trial(i) for i in range(12)])
    trial=simple_trial(30)
    study.apply_trial_gate(trial,gate)
    assert all(r['only_truth_retained'] and r['wrong_candidates_removed']==1 for r in trial['records'])
    assert not trial['any_method_false_rejection']


def test_future_physics_failure_remains_in_every_method(monkeypatch):
    cheap_trial_components(monkeypatch)
    original=study.observe
    def broken(path,rows,action=None,**kwargs):
        if action is not None:raise RuntimeError('declared future AC failure')
        return original(path,rows,action,**kwargs)
    monkeypatch.setattr(study,'observe',broken)
    trial=study.run_trial(55)
    assert not trial['complete'] and np.isinf(trial['relative_score'])
    assert len(trial['records'])==6
    assert all(r['status']=='physical_failure' for r in trial['records'])
    assert trial['simulation_ac_calls']==7



def test_history_failure_scores_infinity_even_if_all_joint_fits_recover(monkeypatch):
    cheap_trial_components(monkeypatch)
    original=study.fit_all_candidates
    def bank(cs,obs,meter,**kwargs):
        fits,failures,cost=original(cs,obs,meter,**kwargs)
        if obs.t==12:fits[cs[0].name].success=False;failures[cs[0].name]='history failed'
        return fits,failures,cost
    monkeypatch.setattr(study,'fit_all_candidates',bank)
    trial=study.run_trial(56)
    assert trial['historical_failures'] and not trial['complete']
    assert np.isinf(trial['relative_score'])
    assert all(r['status']=='complete' for r in trial['records'])
    assert trial['selection']['gate_reason']=='history_fit_failure'


def test_full_internal_voltage_guard_survives_good_terminal_voltages(monkeypatch):
    cheap_trial_components(monkeypatch)
    original=study.observe
    def internal_violation(path,rows,action=None,**kwargs):
        obs,physical=original(path,rows,action,**kwargs)
        if action is not None:
            physical['voltage']=physical['voltage'].copy()
            physical['voltage'][0,0]=.8+0j
        return obs,physical
    monkeypatch.setattr(study,'observe',internal_violation)
    trial=study.run_trial(57)
    assert all(r['terminal_voltage_min']>.9 for r in trial['records'])
    assert all(r['voltage_min']==pytest.approx(.8) and not r['operational_voltage_ok'] for r in trial['records'])


def test_random_site_includes_variable_feasible_uniform_infeasible_site(monkeypatch):
    cheap_trial_components(monkeypatch)
    path,_,_=study._make_path(58,study.configuration());history,_=study.observe(path,slice(0,12));cs=candidates()
    fits={c.name:fake_fit(c,history,i+1) for i,c in enumerate(cs)}
    cap=np.full((4,3),.3);cap[1,1]=.05
    limits=replace(study.device_constraints(study.configuration()),p_charge_max=cap)
    assert not constant_site_plan(1,4,3,limits)['success']
    def search(evaluator,constraints,**kwargs):
        plan=constant_site_plan(0,4,3,constraints)['plan']
        return dict(success=True,selected_site=0,site_plans={0:plan},plan=plan,fit_calls=0,total_ac_calls=0)
    monkeypatch.setattr(study,'optimize_price_curves',search)
    seed=next(seed for seed in range(100) if np.random.default_rng(seed+901).choice([0,1,2])==1)
    choices,record=study.select_controls(history,cs,fits,study.model(3),constraints=limits,horizon=4,seed=seed)
    assert record['random_site']==1 and 1 in record['public_feasible_sites']
    assert choices['optimized_random'].site==1
    assert study.check_single_site_plan(choices['optimized_random'],limits)['feasible']
    assert record['methods']['optimized_random']['status']=='selection_failure_fallback'


def test_unresolved_includes_joint_failure_retaining_every_candidate():
    gate=study.calibration_result([simple_trial(i) for i in range(12)])
    summary=study.summarize_trials([simple_trial(20,complete=False)],gate)
    assert all(r['unresolved']==1 and r['failed_count']==1 for r in summary['method_summary'])



def supported_fixture(monkeypatch,losses):
    calls=cheap_trial_components(monkeypatch)
    path,_,_=study._make_path(59,study.configuration());history,_=study.observe(path,slice(0,12));cs=candidates()
    fits={c.name:fake_fit(c,history,loss) for c,loss in zip(cs,losses)}
    args=dict(constraints=study.device_constraints(study.configuration()),horizon=4)
    return calls,history,cs,fits,args


def test_default_reference_policy_keeps_all_equally_supported_candidates(monkeypatch):
    calls,history,cs,fits,args=supported_fixture(monkeypatch,[1.,1.,1.,1.])
    choices,record=study.select_controls(history,cs,fits,study.model(3),**args)
    assert record['activated'] and record['reference_policy']=='all_within_gate'
    assert record['supported_count']==4
    assert set(record['design_candidates'])=={c.name for c in cs}
    assert all(gap==0 for gap in record['all_candidate_historical_gaps'].values())
    assert all(set(call['references'])=={c.name for c in cs} for call in calls)
    assert study.DEFAULT_CONFIG['version']=='single_site_gradient_price_v2_supported'


def test_all_supported_uses_half_sse_gate_and_excludes_only_large_gap(monkeypatch):
    calls,history,cs,fits,args=supported_fixture(monkeypatch,[1.,2.,9.,20.])
    choices,record=study.select_controls(history,cs,fits,study.model(3),**args)
    assert record['activated'] and record['supported_count']==3
    assert set(record['design_candidates'])=={'star','clade01','clade02'}
    assert record['all_candidate_historical_gaps']['clade12']==pytest.approx(9.5)
    assert record['all_candidate_historical_gaps']['clade02']==pytest.approx(4.)
    assert all('clade12' not in call['references'] for call in calls)


def test_two_closest_is_explicit_comparison_and_logs_wider_support(monkeypatch):
    calls,history,cs,fits,args=supported_fixture(monkeypatch,[1.,1.,1.,1.])
    choices,record=study.select_controls(history,cs,fits,study.model(3),reference_policy='two_closest',**args)
    assert record['reference_policy']=='two_closest' and record['supported_count']==4
    assert record['design_candidates']==['clade01','clade02']
    assert all(call['references']==['clade01','clade02'] for call in calls)


def test_reference_selection_ignores_external_truth_annotations(monkeypatch):
    calls,history,cs,fits,args=supported_fixture(monkeypatch,[1.,1.,1.,1.])
    for i,c in enumerate(cs):fits[c.name].evaluation_is_true=(i==0)
    _,first=study.select_controls(history,cs,fits,study.model(3),**args)
    for i,c in enumerate(cs):fits[c.name].evaluation_is_true=(i==3)
    _,second=study.select_controls(history,cs,fits,study.model(3),**args)
    assert first['design_candidates']==second['design_candidates']
    assert first['all_candidate_historical_gaps']==second['all_candidate_historical_gaps']
    assert 'truth' not in inspect.signature(study.select_controls).parameters


def test_difficult_supported_reference_failure_is_kept_and_causes_fallback(monkeypatch):
    calls,history,cs,fits,args=supported_fixture(monkeypatch,[1.,2.,4.,12.])
    seen=[]
    def rejected(history,candidates,*a,**k):
        names=[c.name for c in candidates];seen.append(names)
        assert 'clade12' in names
        raise RuntimeError('necessary supported clade12 failed')
    monkeypatch.setattr(study,'GradientPriceEvaluator',rejected)
    _,record=study.select_controls(history,cs,fits,study.model(3),**args)
    assert len(record['design_candidates'])==4 and len(seen)==2
    assert record['methods']['optimized_designed']['status']=='selection_failure_fallback'
    assert 'clade12 failed' in record['searches']['full']['error']


def test_supported_policy_propagated_by_run_trial(monkeypatch):
    calls=cheap_trial_components(monkeypatch)
    trial=study.run_trial(60,config={'reference_policy':'two_closest'})
    assert trial['selection']['reference_policy']=='two_closest'
    assert len(trial['selection']['design_candidates'])==2
    assert all(len(call['references'])==2 for call in calls)


@pytest.mark.parametrize('policy',['unknown','all',''])
def test_invalid_reference_policy_rejected(monkeypatch,policy):
    calls,history,cs,fits,args=supported_fixture(monkeypatch,[1.,2.,4.,12.])
    with pytest.raises(ValueError,match='reference_policy'):
        study.select_controls(history,cs,fits,study.model(3),reference_policy=policy,**args)
