"""Formal wrapper tests use fake records only; no AC/selector execution."""
from copy import deepcopy
from fractions import Fraction
import inspect
import json
import numpy as np
import pytest
from . import ev_flexibility_formal as m


def record(seed=982001,case='moderate',quality='uniform',gap=1.):
    names=['star','clade_01']; losses={'star':gap,'clade_01':0.}
    return dict(history_id=str(seed),run_id=str(seed)+'_'+quality,case=case,quality=quality,
        source_sha256=m.development.source_hashes(),candidates=[dict(name=n) for n in names],
        historical_fits=[dict(name=n,success=True,loss=i) for i,n in enumerate(names)],historical_failures={},
        selector_uses_truth=False,selector_uses_future_noise=False,seconds=6.,total_ac_calls=10,gate_passed=True,
        selection=dict(activated=True,seconds=2.,methods={n:dict(status='information_optimized') for n in m.METHODS}),
        records=[dict(method=n,status='complete',true_name='star',true_covered=True,losses=dict(losses),
            signed_margin_full_SSE=-gap) for n in m.METHODS])


def fake_runner(calls,*,slow=False,failed_seed=None):
    def run(seed,case,family,quality):
        calls.append((seed,case,family,quality));r=record(seed,case,quality,gap=seed%5)
        if slow:r.update(seconds=900.,gate_passed=False);r['selection']['seconds']=30.
        if seed==failed_seed:r['records'][0]['status']='physical_failure'
        return r,None
    return run


def test_seed_ranges_and_label_streams_are_fixed_and_separate():
    assert m.CAL_SEEDS==tuple(range(982001,982040))
    assert m.EVAL_SEEDS==tuple(range(983001,983049))
    assert set(m.CAL_SEEDS).isdisjoint(m.EVAL_SEEDS)
    for seed in m.CAL_SEEDS+m.EVAL_SEEDS:
        label=m.labels_for_seed(seed)
        assert label==m.labels_for_seed(seed) and label['quality'] in m.QUALITY
        assert (label['family']==[])==(label['case']=='star')
        assert label['label_streams']['case']!=int(np.random.SeedSequence(seed).spawn(5)[0].generate_state(1)[0])
    assert set(inspect.signature(m.development.select).parameters)=={'history','candidates','fits','history_model','future_model','quality','seed'}


def test_freeze_has23_exact_sources_and87_predeclared_labels(tmp_path):
    freeze=m.initialize_freeze(tmp_path)
    assert len(freeze['source_sha256'])==23
    assert len(freeze['schedule']['calibration'])==39 and len(freeze['schedule']['evaluation'])==48
    assert m.validate_snapshot(m.read(tmp_path/'science_source_snapshot.json.gz'),freeze)==23
    assert m.initialize_freeze(tmp_path)==freeze
    assert freeze['generator_config']['population']['case_mass']==[.15,.15,.4,.3]
    assert freeze['methods']==list(m.METHODS)


def test_snapshot_tampering_fails(tmp_path):
    f=m.initialize_freeze(tmp_path);s=m.read(tmp_path/'science_source_snapshot.json.gz')
    first=next(iter(s['sources']));s['sources'][first]['bytes_base64']='YWJj'
    with pytest.raises(ValueError,match='hash mismatch'):m.validate_snapshot(s,f)


def test_resume_identity_environment_or_config_tampering_fails(tmp_path):
    m.initialize_freeze(tmp_path);saved=m.read(tmp_path/'freeze.json');saved['policy_config']['p_charge_max']=.31;m.save(tmp_path/'freeze.json',saved)
    with pytest.raises(ValueError,match='resume identity'):m.initialize_freeze(tmp_path)


def test_max_score_uses_all9_joint_truth_gaps_not_design_score():
    r=record();r['records'][3]['losses']['star']=7.
    r['selection']['methods']['charge_fixed']['predicted_score']=1e6
    result=m.calibration_score(r)
    assert result['score']==7. and not result['score_is_infinite']
    assert len(result['method_truth_gaps'])==9


@pytest.mark.parametrize('kind',['joint_fit','physical','coverage','historical','nonfinite','null'])
def test_failure_is_infinite_and_preserves_denominator(kind):
    r=record()
    if kind=='joint_fit':r['records'][0]['status']='joint_fit_failure'
    if kind=='physical':r['records'][0]['status']='physical_failure'
    if kind=='coverage':r['records'][0]['true_covered']=False
    if kind=='historical':r['historical_fits'][0]['success']=False
    if kind=='nonfinite':r['records'][0]['losses']['star']=float('nan')
    if kind=='null':r['records'][0]['losses']['star']=None
    score=m.calibration_score(r)
    assert score['score'] is None and score['score_is_infinite'] and score['reasons']
    t=m.tolerance_threshold([score]+[dict(score=1.,score_is_infinite=False)]*38)
    assert t['threshold_is_infinite'] and t['failed_history_count']==1


def test_unactivated_or_safe_fallback_uses_actual_successful_joint_losses():
    r=record(gap=3);r['selection']['activated']=False
    r['selection']['methods']['charge_fixed']['status']='constant_fallback'
    score=m.calibration_score(r)
    assert score['score']==3 and not score['score_is_infinite']
    gate=m.evaluate_gate(r,4)['charge_fixed']
    assert gate['controller_fallback'] and not gate['activated']


def test_rank39_maximum_and_exact_binomial_tail():
    scores=[dict(score=float(i),score_is_infinite=False) for i in range(39)]
    t=m.tolerance_threshold(scores)
    assert t['rank']==39 and t['threshold']==38.
    assert Fraction(t['binomial_tail_exact'])==Fraction(9,10)**39
    assert t['binomial_tail']==pytest.approx(.01642320326826066)
    assert t['binomial_tail']<.05
    with pytest.raises(ValueError,match='exactly39'):m.tolerance_threshold(scores[:38])


def test_gate_ties_infinity_and_failure_retain_all_without_truth_input():
    assert m.retain_candidates({'a':1,'b':3},['a','b'],2)==['a','b']
    assert m.retain_candidates({'a':1,'b':3},['a','b'],1.9)==['a']
    assert m.retain_candidates({'a':1,'b':3},['a','b'],float('inf'))==['a','b']
    assert m.retain_candidates({'a':1},['a','b'],1,complete=False)==['a','b']
    assert 'truth' not in inspect.signature(m.retain_candidates).parameters


def test_mutually_exclusive_partition_has_full_denominator_including_failures():
    rs=[record(seed=983001+i,gap=g) for i,g in enumerate((-10.,5.,1.,1.))]
    # Correct complete losses for the unique case.
    for row in rs[0]['records']:row['losses']={'star':0.,'clade_01':10.}
    rs[3]['records'][0]['status']='physical_failure'
    summary=m.evaluation_summary(rs,2,dict(policy_id='fake'))
    assert summary['n']==4 and summary['planned_n']==48 and not summary['complete']
    assert summary['methods']['none']['counts']==dict(unique_correct=1,false_rejection=1,safe_undecided=1,compute_failure=1)
    for v in summary['methods'].values():assert sum(v['counts'].values())==4


def test_margin_uses_only_complete_pairs_and_case_rows():
    rs=[record(seed=983001+i,case='star' if i==0 else 'weak') for i in range(3)]
    rs[0]['records'][m.METHODS.index('charge_fixed')]['status']='joint_fit_failure'
    summary=m.evaluation_summary(rs,2,dict(policy_id='fake'))
    primary=summary['comparisons'][0]
    assert primary['primary_full_n']==3 and primary['margin_complete_pairs']==2
    assert summary['by_case']['star']['margins']['charge_fixed_vs_constant_fixed']==[]
    assert len(summary['by_case']['weak']['margins']['charge_fixed_vs_constant_fixed'])==2


def test_partial_resume_runs_only_new_fixed_seeds_and_keeps_slow_H(tmp_path):
    f=m.initialize_freeze(tmp_path);calls=[];runner=fake_runner(calls,slow=True)
    first=m.run_phase(tmp_path,'calibration',f,limit=2,runner=runner)
    second=m.run_phase(tmp_path,'calibration',f,limit=3,runner=runner)
    assert [c[0] for c in calls]==[982001,982002,982003]
    assert len(first)==2 and len(second)==3
    assert all(not r['gate_passed'] for r in second)
    assert not any(m.calibration_score(r)['score_is_infinite'] for r in second)
    checkpoint=m.read(tmp_path/'calibration'/'checkpoint.json')
    assert checkpoint['completed_histories']==3 and checkpoint['planned_histories']==39
    assert not list(tmp_path.rglob('*.tmp'))


def test_corrupt_trial_identity_or_truth_leak_is_rejected(tmp_path):
    f=m.initialize_freeze(tmp_path);m.run_phase(tmp_path,'calibration',f,limit=1,runner=fake_runner([]))
    path=tmp_path/'calibration'/'trial_982001.json.gz';r=m.read(path);r['selector_uses_truth']=True;m.save(path,r)
    with pytest.raises(ValueError,match='isolation'):m.run_phase(tmp_path,'calibration',f,limit=1,runner=fake_runner([]))


def test_test_double_trial_cannot_resume_as_real(tmp_path):
    f=m.initialize_freeze(tmp_path);m.run_phase(tmp_path,'calibration',f,limit=1,runner=fake_runner([]))
    with pytest.raises(ValueError,match='test-double'):m.run_phase(tmp_path,'calibration',f,limit=1)


def test_callback_exception_creates9_failed_rows_and_continues(tmp_path):
    f=m.initialize_freeze(tmp_path);calls=[]
    def fail(seed,*a):calls.append(seed);raise RuntimeError('declared synthetic failure')
    rs=m.run_phase(tmp_path,'calibration',f,limit=2,runner=fail)
    assert calls==[982001,982002] and len(rs)==2
    assert all(len(r['records'])==9 and m.calibration_score(r)['score_is_infinite'] for r in rs)


def test_partial_all_phase_cannot_start_evaluation(tmp_path):
    calls=[];result=m.run_batch(tmp_path,phase='all',limit=2,runner=fake_runner(calls))
    assert len(calls)==2 and not result['complete'] and 'tolerance' not in result
    assert not (tmp_path/'evaluation').exists()


def test_infinite_calibration_stops_evaluation_and_keeps_all_candidates(tmp_path):
    calls=[];result=m.run_batch(tmp_path,phase='all',runner=fake_runner(calls,failed_seed=982004))
    assert len(calls)==39 and result['stage']=='uninformative_calibration'
    assert result['retain_all_candidates'] and not result['evaluation_started']
    assert not (tmp_path/'evaluation').exists()


def test_full_fake_pipeline_has39_then48_independent_units_and_finite_gate(tmp_path):
    calls=[];runner=fake_runner(calls);result=m.run_batch(tmp_path,phase='all',runner=runner)
    assert len(calls)==87 and [x[0] for x in calls]==list(m.CAL_SEEDS+m.EVAL_SEEDS)
    assert result['n']==48 and result['complete'] and result['tolerance']['threshold']==4
    assert not result['whole_research_complete'] and not result['old_threshold_applied']
    assert sum(result['quality_counts'].values())==48
    assert all(v['n']==48 and sum(v['counts'].values())==48 for v in result['methods'].values())
    again=m.run_batch(tmp_path,phase='evaluation',runner=runner)
    assert len(calls)==87 and again==result


def test_missing_method_or_duplicate_method_is_integrity_error():
    r=record();r['records'].pop()
    with pytest.raises(ValueError,match='missing required'):m.calibration_score(r)
    r=record();r['records'].append(deepcopy(r['records'][0]))
    with pytest.raises(ValueError,match='duplicate method'):m.calibration_score(r)


@pytest.mark.parametrize('kind',['failed','missing','invalid','runner'])
def test_required_H_fit_failure_retains_all_and_is_failure_for_all_methods(kind):
    r=record()
    if kind=='failed':r['historical_fits'][0]['success']=False
    if kind=='missing':r['historical_fits'].pop()
    if kind=='invalid':r['historical_fits'][0]['loss']=None
    if kind=='runner':r['runner_exception']='declared failure'
    gate=m.evaluate_gate(r,.01)
    assert all(g['state']=='compute_failure' and g['retained']==['star','clade_01'] for g in gate.values())
    assert all(g['signed_margin_full_SSE'] is None and g['margin_unavailable_reason']=='historical_fit_failure' for g in gate.values())


def test_active_context_fallback_is_design_failure_but_complete_AC_is_evaluated():
    r=record(gap=1.);r['selection']['methods']['charge_fixed']['status']='constant_fallback'
    gate=m.evaluate_gate(r,2.)['charge_fixed']
    assert gate['design_failure'] and gate['controller_fallback'] and not gate['not_activated']
    assert gate['state']=='safe_undecided' and gate['signed_margin_full_SSE']==-1.
    assert not m.calibration_score(r)['score_is_infinite']
    r['selection']['activated']=False
    gate=m.evaluate_gate(r,2.)['charge_fixed']
    assert not gate['design_failure'] and gate['not_activated']
    summary=m.evaluation_summary([r],2.,dict(policy_id='fake'))
    assert summary['methods']['charge_fixed']['not_activated']==1
    assert summary['methods']['charge_fixed']['design_failures']==0
