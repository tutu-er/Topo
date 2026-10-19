"""Independent safe-review score, failure and decision negative controls."""
from copy import deepcopy
import math
from fractions import Fraction
import pytest
from . import ev_flexibility_safe_review as review


def fixture(history_id='toy'):
    names=['a','b','c','d'];values={'a':0.,'b':1.,'c':2.,'d':3.}
    return dict(history_id=history_id,candidates=[{'name':n} for n in names],historical_fits=[dict(name=n,success=True,loss=0.) for n in names],historical_failures={},selection=dict(activated=True,methods={m:dict(status='information_optimized') for m in review.METHODS}),records=[dict(method=m,status='complete',losses=dict(values),true_name='a',true_covered=True,selection=dict(status='information_optimized')) for m in review.METHODS])


def test_failed_covered_method_score0_means_no_rejection_not_perfect_fit():
    t=fixture();t['records'][0].update(status='joint_fit_failure',losses={'a':12.})
    s=review.score_history(t);assert s['score']==0. and s['failed_methods']['none']
    assert s['method_raw_truth_gaps']['none'] is None
    d=review.decisions(t,0.)['none'];assert d['state']=='compute_failure' and d['retained']==['a','b','c','d']


def test_all_failed_known_coverage_does_not_produce_successful_identification():
    t=fixture()
    for row in t['records']:row['status']='physical_failure'
    assert review.score_history(t)['score']==0.
    assert {d['state'] for d in review.decisions(t,0.).values()}=={'compute_failure'}


@pytest.mark.parametrize('damage',['unknown','missing','conflicting'])
def test_unknown_or_conflicting_coverage_makes_score_infinite(damage):
    t=fixture()
    if damage=='unknown':t['records'][0]['true_covered']=None
    elif damage=='missing':t['records'][0]['true_name']='absent'
    else:t['records'][0]['true_name']='b'
    assert review.score_history(t)['score_is_infinite']


def test_historical_failure_retains_all_even_if_future_fits_complete():
    t=fixture();t['historical_fits'][0]['success']=False
    s=review.score_history(t);assert s['historical_failure'] and s['score']==0.
    assert all(d['state']=='compute_failure' and len(d['retained'])==4 for d in review.decisions(t,0.).values())


def test_retained_set_does_not_use_truth_labels():
    t=fixture();a=review.decisions(t,.5)['charge_fixed']['retained']
    for row in t['records']:row['true_name']='b'
    b=review.decisions(t,.5)['charge_fixed']['retained'];assert a==b==['a']
    assert review.decisions(t,.5)['charge_fixed']['state']=='false_rejection'


@pytest.mark.parametrize('value',[None,math.nan,math.inf,-1.])
def test_invalid_complete_losses_preserve_compute_failure_and_full_set(value):
    t=fixture();t['records'][0]['losses']['a']=value
    assert review.score_history(t)['failed_methods']['none']
    assert review.decisions(t,.1)['none']['retained']==['a','b','c','d']


def test_rank39_threshold_exact_tail_and_ties():
    scores=[dict(score_is_infinite=False,score=float(i),failure_retained=False) for i in range(39)]
    rule=review.threshold(scores);assert rule['threshold']==38.
    assert Fraction(rule['binomial_tail_exact'])==Fraction(9,10)**39
    assert rule['binomial_tail']<.05
    assert review.retained(['a','b'],{'a':0.,'b':1.},1.)==['a','b']
    scores[-1]['score_is_infinite']=True;assert review.threshold(scores)['threshold_is_infinite']
    with pytest.raises(ValueError):review.threshold(scores[:-1])


def test_duplicate_method_does_not_vacuously_pass_failure_scoring():
    t=fixture();t['records'][-1]['method']=t['records'][0]['method']
    with pytest.raises(ValueError):review.score_history(t)



def test_compact_only_summary_and_trial_review_do_not_need_original_raw(tmp_path):
    from .information_fit_recovery import write_json,read_json
    import shutil
    original=tmp_path/'raw';compact=tmp_path/'compact'
    original.mkdir();(original/'calibration').mkdir()
    write_json(original/'freeze.json',{'schema':'test'})
    write_json(original/'science_source_snapshot.json.gz',{'schema':'test'})
    summary={'tolerance':{'threshold':2.5},'scores':[{'score':2.5}]}
    write_json(original/'calibration_summary.json',summary)
    record={'state':[1.,2.],'local_scaled_singular_values':[1.,.1]}
    write_json(original/'calibration/trial_1.json.gz',record)
    review.export_compact(original,compact,phases=('calibration',))
    shutil.rmtree(original)
    assert review.reported_summary(compact,'calibration')==summary
    audit=review.numeric.verify_compact_manifest(compact)
    assert audit['success'] and audit['raw_files_available_and_verified']==0
    assert read_json(compact/'calibration/trial_1.json.gz')=={'state':[1.,2.]}
    with pytest.raises(ValueError,match='missing local frozen evaluation summary'):
        review.reported_summary(compact,'evaluation')



def test_cost_exceedance_retains_planned_H_and_distinguishes_time_from_fit_gate():
    rows=[dict(history_id='a',seconds=10.,selection={'seconds':2.},config={'condition_seconds_gate':90.,'controller_seconds_gate':15.},gate_passed=False),
          dict(history_id='b',seconds=100.,selection={'seconds':16.},config={'condition_seconds_gate':90.,'controller_seconds_gate':15.},gate_passed=False)]
    costs=review.cost_summary(rows)
    assert costs['planned_H']==2 and costs['saved_gate_exceedances']==2
    assert costs['controller_time_exceedances']==costs['condition_time_exceedances']==1
    assert costs['known_condition_duration_H']==2 and costs['total_condition_seconds']==110.


def test_compact_summary_equality_ignores_observation_metadata_only():
    a={'counts':{'n':39},'threshold':6.,'input_sha256':{'raw':'a'},'seconds':1.,'compact_manifest':None,'git_revision_observed_at_review':'a'}
    b={**a,'input_sha256':{'compact':'b'},'seconds':2.,'compact_manifest':{'success':True},'git_revision_observed_at_review':'b'}
    assert review.compare_scientific_summaries(a,b)['scientific_payload_exactly_equal']
    b['counts']={'n':38}
    with pytest.raises(ValueError,match='scientific results differ'):review.compare_scientific_summaries(a,b)
