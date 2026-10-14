from copy import deepcopy
import math
import pytest
from . import ev_policy_calibration as c


def frozen():
    return dict(config={'policy':'fixed-before-calibration','fit_budget':200},
        source_sha256={'selector.py':'1'*64}, frozen_before_trials=True)


def spec(**kwargs):
    return c.spec_from_freeze(frozen(), reference_id='declared-simulation-v1',
        candidates=('A','B'), methods=kwargs.pop('methods',('policy',)), **kwargs)


def record(i, truth='A', score=0., *, specification=None, stratum='declared_asset_reference',
           status='complete'):
    s=specification or spec(); losses={a:0. for a in s.candidates}; losses[truth]=float(score)
    return dict(**c.freeze_identity(frozen()), reference_id=s.reference_id,
        history_id=str(i), independent_unit_id=str(i),stratum=stratum,
        true_name=truth,candidate_names=list(s.candidates),
        methods={m:dict(status=status,losses=dict(losses)) for m in s.methods})


def calibration_records(s=None):
    s=s or spec()
    return [record(i, 'A', i, specification=s) for i in range(9)]+[
        record(i+20,'B',i,specification=s) for i in range(9)]


@pytest.mark.parametrize('n,alpha,rank,infinite',[(0,.1,1,True),(8,.1,9,True),
    (9,.1,9,False),(18,.05,19,True),(19,.05,19,False),(9,.2,8,False)])
def test_exact_small_sample_order_rank(n,alpha,rank,infinite):
    out=c.upper_order_quantile(range(n),alpha)
    assert (out['rank'],out['infinite'])==(rank,infinite)
    assert out['rank_marginal_error_bound']<=alpha
    if not infinite:assert out['threshold']==rank-1


def test_leave_one_out_rank_bound_and_ties_independently():
    # For an exchangeable ordering of a fixed batch, the held-out rank is uniform.
    for batch in (list(range(10)),[0]*10,[0,0,0,1,1,2,2,3,3,3]):
        rejections=0
        for j,new in enumerate(batch):
            q=c.upper_order_quantile(batch[:j]+batch[j+1:],.1)
            rejections+=new>q['threshold']
        assert rejections<=1


@pytest.mark.parametrize('scores',[[math.nan],[-1],[float('-inf')]])
def test_bad_scores_rejected(scores):
    with pytest.raises(ValueError):c.upper_order_quantile(scores)


def test_candidate_specific_null_not_true_oracle_at_application():
    rows=calibration_records()
    for r in rows[9:]:r['methods']['policy']['losses']['B']*=10
    cal=c.calibrate(rows,spec())
    trial=record(200,'A',50)
    out=c.evaluate([trial],cal)
    d=out['records'][0]['methods']['policy']
    assert d['thresholds']=={'A':8.,'B':80.}
    assert d['retained']==['B']
    # Change only the evaluation truth label: decisions remain the same.
    alternate=deepcopy(trial);alternate['true_name']='B'
    assert c.evaluate([alternate],cal)['records'][0]['methods']['policy']['retained']==['B']


def test_ties_kept_and_nontruth_exclusion_counted():
    cal=c.calibrate(calibration_records(),spec())
    trial=record(200,'A',8)
    assert c.evaluate([trial],cal)['records'][0]['methods']['policy']['retained']==['A','B']
    trial['methods']['policy']['losses']={'A':0.,'B':9.}
    out=c.evaluate([trial],cal)
    assert out['methods']['policy']['unique_correct']==1
    assert out['methods']['policy']['wrong_candidates_excluded']==1


def test_missing_null_cell_is_infinite_and_preserved():
    cal=c.calibrate(calibration_records()[:8],spec(),planned_history_ids=range(36))
    assert cal['histories_total']==8 and cal['histories_pending']==28
    assert cal['thresholds']['declared_asset_reference']['A']['policy']['infinite']
    assert cal['thresholds']['declared_asset_reference']['B']['policy']['n']==0
    out=c.evaluate([record(200,'A',1000)],cal)
    assert out['records'][0]['methods']['policy']['retained']==['A','B']


def test_failure_zero_score_matches_keep_all_including_denominator():
    rows=calibration_records()
    rows[0]['methods']['policy']=dict(status='fit_failure',losses={'A':None})
    cal=c.calibrate(rows,spec())
    assert cal['histories_total']==18 and cal['records'][0]['scores']['policy']==0
    bad=record(200,'A',999,status='operation_failure')
    out=c.evaluate([bad],cal)
    assert out['methods']['policy']['total']==1
    assert out['methods']['policy']['known_covered_denominator']==1
    assert out['methods']['policy']['operation_failures']==1
    assert out['methods']['policy']['false_rejections']==0
    assert out['methods']['policy']['unresolved']==1
    assert out['records'][0]['methods']['policy']['retained']==['A','B']


def test_familywise_uses_extra_max_and_fixed_method_family():
    s=spec(methods=('one','two'),familywise=True)
    rows=calibration_records(s)
    for r in rows:r['methods']['two']['losses'][r['true_name']]+=10
    cal=c.calibrate(rows,s)
    assert cal['thresholds']['declared_asset_reference']['A']['__family__']['threshold']==18
    assert cal['records'][0]['scores']=={'one':0.,'two':10.}
    trial=record(200,'A',18,specification=s)
    out=c.evaluate([trial],cal)
    assert out['totals']['any_method_false_rejections']==0
    trial['methods']['two']['losses']['A']=19.
    assert c.evaluate([trial],cal)['totals']['any_method_false_rejections']==1


def test_worst_stratum_cannot_use_hidden_truth_stratum_to_tighten():
    s=spec(strata=('low','high'))
    rows=[]
    for k,stratum in enumerate(s.strata):
        for r in calibration_records(s):
            r['history_id']=r['independent_unit_id']=f'{k}:{r["history_id"]}'
            r['stratum']=stratum
            r['methods']['policy']['losses'][r['true_name']]+=10*k
            rows.append(r)
    cal=c.calibrate(rows,s)
    out=c.evaluate([record(200,'A',15,specification=s,stratum='low')],cal)
    assert out['records'][0]['methods']['policy']['thresholds']['A']==18.
    assert out['totals']['any_method_false_rejections']==0
    with pytest.raises(ValueError):spec(stratum_mode='public')


@pytest.mark.parametrize('mutator',[lambda r:r.update(policy_id='other'),
    lambda r:r.update(config_sha256='other'),lambda r:r.update(reference_id='other'),
    lambda r:r.update(source_manifest_sha256='other'),lambda r:r.update(history_id=None),
    lambda r:r['methods']['policy']['losses'].update(A=float('nan')),
    lambda r:r['methods']['policy']['losses'].pop('B')])
def test_malformed_or_changed_policy_records_raise(mutator):
    r=record(1);mutator(r)
    with pytest.raises(ValueError):c.calibrate([r],spec())


def test_unit_duplicates_cal_eval_overlap_and_missing_records_raise():
    r=record(1);other=record(2);other['independent_unit_id']='1'
    with pytest.raises(ValueError):c.calibrate([r,other],spec())
    with pytest.raises(ValueError):c.calibrate([r],spec(),expected_history_ids=[1,2])
    cal=c.calibrate([r],spec())
    with pytest.raises(ValueError):c.evaluate([other],cal)
    with pytest.raises(ValueError):c.evaluate([r],cal)


@pytest.mark.parametrize('miss,outside',[(True,False),(False,True)])
def test_coverage_or_undeclared_reference_blocks_rejection(miss,outside):
    rows=calibration_records()
    if miss:rows[0]['candidate_miss']=True
    if outside:rows[0]['stratum']='undeclared'
    cal=c.calibrate(rows,spec())
    assert not cal['rejection_enabled']
    assert not cal['conditional_guarantee_available']
    assert c.evaluate([record(200,'A',1000)],cal)['records'][0]['methods']['policy']['retained']==['A','B']


def test_evaluation_candidate_miss_kept_separate_from_false_rejection():
    cal=c.calibrate(calibration_records(),spec())
    r=record(200);r['true_name']='outside-C';r['candidate_miss']=True
    out=c.evaluate([r],cal)
    assert out['totals']['candidate_miss_histories']==1
    assert out['methods']['policy']['known_covered_denominator']==0
    assert out['records'][0]['methods']['policy']['false_rejection'] is None
    assert out['methods']['policy']['unresolved']==1


def test_calibration_tampering_detected():
    cal=c.calibrate(calibration_records(),spec())
    cal['thresholds']['declared_asset_reference']['A']['policy']['threshold']=0
    with pytest.raises(ValueError):c.evaluate([record(200)],cal)


def raw_trial():
    fit=lambda name,loss:dict(name=name,loss=loss,success=True)
    return dict(config=deepcopy(frozen()['config']),source_sha256=frozen()['source_sha256'],
        seed=1,true_name='A',candidate_names=['A','B'],historical_failures={},
        historical_fits=[fit('A',0.),fit('B',1.)],
        rows=[dict(method='policy',status='complete',losses={'A':1.,'B':0.},
                   fits=[fit('A',1.),fit('B',0.)])])


def test_saved_trial_adapter_provenance_and_valid_fallback():
    trial=raw_trial();trial['rows'][0]['fallback_reason']='MILP timed out; feasible constant executed'
    out=c.adapt_saved_trial(trial,frozen(),spec())
    assert out['methods']['policy']['status']=='complete'
    cal=c.calibrate([out],spec())
    assert cal['records'][0]['scores']['policy']==1.
    trial['config']['fit_budget']=201
    with pytest.raises(ValueError):c.adapt_saved_trial(trial,frozen(),spec())


@pytest.mark.parametrize('change',[
    lambda t:t['historical_fits'][0].update(success=False),
    lambda t:t['rows'][0]['fits'][0].update(success=False),
    lambda t:t['rows'][0]['fits'].pop(),
    lambda t:t['rows'][0]['fits'][0].update(loss=2.),
    lambda t:t['rows'].clear()])
def test_saved_fit_and_missing_method_failures_preserved(change):
    trial=raw_trial();change(trial)
    out=c.adapt_saved_trial(trial,frozen(),spec())
    assert out['methods']['policy']['status']!='complete'
    assert c.calibrate([out],spec())['records'][0]['scores']['policy']==0.


def test_truth_family_adapter_uses_saved_bank_only_offline():
    trial=raw_trial();trial.pop('true_name');trial['truth']={'family':[[0,1]]}
    trial['history']={'p':[[1,1,1]]};trial['proposal']={'bank':[
        dict(name='A',supports=[[0],[1],[2],[0,1],[0,1,2]]),
        dict(name='B',supports=[[0],[1],[2],[0,1,2]])]}
    assert c.adapt_saved_trial(trial,frozen(),spec())['true_name']=='A'
    trial['truth']['family']=[[0,2]]
    assert c.adapt_saved_trial(trial,frozen(),spec())['candidate_miss']


def test_unfrozen_or_changed_source_provenance_raises():
    freeze=frozen();freeze['frozen_before_trials']=False
    with pytest.raises(ValueError):c.adapt_saved_trial(raw_trial(),freeze,spec())
    trial=raw_trial();trial['source_changed_during_trial']=True
    with pytest.raises(ValueError):c.adapt_saved_trial(trial,frozen(),spec())
    freeze=frozen();freeze['policy_id']='forged'
    with pytest.raises(ValueError):c.freeze_identity(freeze)


def test_complete_mode_document_matches_existing_driver_encoding():
    import hashlib,json
    payload=dict(config=frozen()['config'],prediction='ensemble',master_form='shared_control_square',
                 domain='asset',source_sha256=frozen()['source_sha256'])
    ident=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    freeze=dict(**payload,policy_config=payload,policy_id=ident,frozen_before_trials=True)
    s=c.spec_from_freeze(freeze,reference_id='declared-simulation-v1',candidates=('A','B'),methods=('policy',))
    trial=raw_trial();trial.update(policy_config=deepcopy(payload),policy_id=ident)
    assert c.adapt_saved_trial(trial,freeze,s)['policy_id']==ident
    trial['policy_config']['prediction']='center'
    with pytest.raises(ValueError):c.adapt_saved_trial(trial,freeze,s)


def test_top_level_modes_are_hashed_and_duplicate_fields_agree():
    freeze=dict(**frozen(),prediction='center',master_form='shared_control_square',domain='asset')
    before=c.freeze_identity(freeze)
    freeze['prediction']='ensemble'
    assert c.freeze_identity(freeze)['policy_id']!=before['policy_id']
    freeze['policy_config']={k:freeze[k] for k in ('config','prediction','master_form','domain','source_sha256')}
    freeze['policy_config']['source_sha256']={'selector.py':'2'*64}
    with pytest.raises(ValueError):c.freeze_identity(freeze)


def test_saved_truth_disagreement_is_a_provenance_error():
    trial=raw_trial();trial['rows'][0]['true_name']='B'
    with pytest.raises(ValueError):c.adapt_saved_trial(trial,frozen(),spec())
