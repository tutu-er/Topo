"""Small negative controls plus committed-CSV replay; no raw trials/fits."""
import copy
import json
import math
from fractions import Fraction
import pytest
from .calibration_tolerance_audit import METHODS,CANDIDATES,validate_phase,history_scores,tolerance_rank,retained,verify_original,re_gate,audit,read_csv


def fixture():
    methods=[];fits=[];meta=dict(history_id='toy',case='weak',quality='uniform',true_name='clade_01')
    losses=dict(zip(CANDIDATES,[0.,1.,2.,3.]))
    for m in METHODS:
        methods.append(dict(meta,method=m,status='complete',true_gap='0.0',retained_candidates='["clade_01"]',only_truth_retained='True',false_rejection='False',unresolved='False',wrong_candidates_removed='3'))
    for m in METHODS+('H',):
        for c in CANDIDATES:fits.append(dict(meta,method=m,name=c,success='True',loss=str(losses[c])))
    return methods,fits


@pytest.mark.parametrize('damage',['missing_method','missing_history','missing_candidate','duplicate_fit','nan_fit','blank_gap','nan_failed_fit'])
def test_missing_products_or_nonfinite_cells_cannot_vacuously_pass(damage):
    methods,fits=fixture();ids=['toy']
    if damage=='missing_method':methods.pop()
    elif damage=='missing_history':ids.append('missing')
    elif damage=='missing_candidate':fits.pop()
    elif damage=='duplicate_fit':fits.append(copy.deepcopy(fits[0]))
    elif damage=='nan_fit':fits[0]['loss']='NaN'
    elif damage=='blank_gap':methods[0]['true_gap']=''
    else:
        methods[0]['status']='joint_fit_failure';methods[0]['true_gap']='';fits[0]['success']='False';fits[0]['loss']='NaN'
    with pytest.raises(ValueError):validate_phase(methods,fits,ids)


def test_wrong_original_kept_and_flags_are_rejected_before_new_gate():
    methods,fits=fixture();methods[0]['retained_candidates']='["star"]'
    phase=validate_phase(methods,fits,['toy'])
    with pytest.raises(ValueError,match='original kept mismatch'):re_gate(phase,.5,1.5)
    methods,fits=fixture();methods[0]['wrong_candidates_removed']='2'
    with pytest.raises(ValueError,match='removed count'):verify_original(validate_phase(methods,fits,['toy']),.5)


def test_failed_finite_loss_is_not_a_success_and_score_infinity_is_not_false():
    methods,fits=fixture();methods[0].update(status='joint_fit_failure',true_gap='',retained_candidates=json.dumps(CANDIDATES),only_truth_retained='False',unresolved='True',wrong_candidates_removed='0')
    fits[0]['success']='False'  # finite loss is still a failed candidate.
    phase=validate_phase(methods,fits,['toy']);assert history_scores(phase)['toy']==math.inf
    result=re_gate(phase,.5,1.5)['all_eighty']
    assert result['old_score_exceeds_count']==result['new_score_exceeds_count']==1
    assert result['old_any_false_count']==result['new_any_false_count']==0
    assert result['methods']['none']['old']['compute_failure']==1
    assert retained(phase,'toy','none',.5)==set(CANDIDATES)
    methods[0]['status']='complete';methods[0]['true_gap']='0.0'
    with pytest.raises(ValueError,match='unsuccessful fit'):validate_phase(methods,fits,['toy'])


def test_original_unresolved_and_false_can_overlap_and_H_is_not_twelfth_method():
    methods,fits=fixture();methods[0].update(true_gap='2.0',retained_candidates='["clade_02","clade_12"]',only_truth_retained='False',false_rejection='True',unresolved='True',wrong_candidates_removed='1')
    new_losses={'clade_01':2.,'clade_02':0.,'clade_12':.2,'star':5.}
    for r in fits:
        if r['method']=='none':r['loss']=str(new_losses[r['name']])
        if r['method']=='H':r['loss']=str(1e10+new_losses[r['name']])
    phase=validate_phase(methods,fits,['toy']);verify_original(phase,.5)
    assert history_scores(phase)['toy']==2.
    result=re_gate(phase,.5,2.)['all_eighty']
    assert result['methods']['none']['old']['false_rejection']==1
    assert result['methods']['none']['new']['safe_undecided']==1


def test_ties_infinity_and_exact_parameters_preserve_strict_rejection():
    choice=tolerance_rank([0.]*39)
    assert choice['rank']==39 and choice['threshold']==0.
    assert choice['alpha_exact']=='1/10' and choice['delta_exact']=='1/20'
    tail=Fraction(9,10)**39;assert Fraction(choice['tail_exact'])==tail
    assert choice['tail_by_rank'][38]>.05>choice['tail_by_rank'][39]
    assert tolerance_rank([0.]*39,delta=tail)['rank']==39
    assert tolerance_rank([0.]*39,delta=tail-Fraction(1,10**50))['rank']==40
    assert tolerance_rank([0.]*35+[math.inf]*4)['threshold']==math.inf
    with pytest.raises(ValueError):tolerance_rank([math.nan]*39)
    methods,fits=fixture()
    for r in fits:r['loss']='0.0'
    for r in methods:r.update(retained_candidates=json.dumps(CANDIDATES),only_truth_retained='False',unresolved='True',wrong_candidates_removed='0')
    phase=validate_phase(methods,fits,['toy']);result=re_gate(phase,0.,math.inf)['all_eighty']
    assert result['old_any_false_count']==result['old_score_exceeds_count']==0
    assert result['new_any_false_count']==result['new_score_exceeds_count']==0


def test_mixture_risk_does_not_transfer_to_weak_case():
    weak_mass=Fraction(1,20);weak_risk=Fraction(1);other_risk=Fraction(0)
    mixture=weak_mass*weak_risk+(1-weak_mass)*other_risk
    assert mixture<=Fraction(1,10) and weak_risk>Fraction(1,10)
    # This counterexample invalidates any automatic case-risk interpretation.


def test_empty_csv_is_rejected(tmp_path):
    p=tmp_path/'empty.csv';p.write_text('history_id,method\n',encoding='utf-8')
    with pytest.raises(ValueError,match='empty'):read_csv(p,['history_id','method'])


def test_committed_compact_csvs_match_every_saved_diagnostic_number():
    result=audit()
    assert result['validation_success'] and result['raw_trials_read']==result['new_optimization_or_fit_calls']==0
    assert result['coverage']['cal_methods']==39*11 and result['coverage']['eval_candidate_fits']==80*11*4
    assert result['original_kept_and_flags_verified']==880 and result['diagnostic_numeric_comparison_passed']
    assert result['old_threshold']==2.9617106149485366 and result['tolerance']['threshold']==4.9258142507602045
    assert result['all_eighty']['old_any_false_count']==11 and result['all_eighty']['new_any_false_count']==6
    assert 'no subgroup/fixed-theta guarantee' in result['scope']
