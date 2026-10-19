"""Average-rank and fixed-denominator tests; no science model is imported."""
from copy import deepcopy
import pytest
from . import ev_original_rank_diagnostic as r


def summary():
    rows=[];paired=[]
    for seed in range(1,13):
        family=('star','cherry_01','cherry_02','cherry_12')[(seed-1)%4]
        for i,method in enumerate(r.METHODS):
            rows.append(dict(seed=seed,family=family,method=method,status='complete',fallback_to_constant=False,
                predicted_mean=float(i),actual_M=float(2*i),noiseless_joint_M=float(3*i)))
        for contrast,_,_ in r.CONTRASTS:
            paired.append(dict(seed=seed,family=family,contrast=contrast,delta_predicted_mean=float(seed),
                delta_actual_M=float(seed*2),delta_noiseless_joint_M=float(12-seed)))
    return dict(phase='evaluation',available_independent_histories=12,missing_histories=[],method_rows=rows,paired_rows=paired)


def test_average_ranks_keep_exact_ties():
    assert r.average_ranks([1.,1.,3.,4.])==[1.5,1.5,3.,4.]
    result=r.spearman([1.,1.,3.,4.],[1.,2.,3.,4.])
    assert result['rho']==pytest.approx(.9486832980505138)
    assert result['x_tied_observations']==2 and result['planned_pairs']==4


def test_constant_ranks_are_missing_correlation_not_zero():
    result=r.spearman([1.,1.,1.,1.],[1.,2.,3.,4.])
    assert result['rho'] is None and result['status']=='constant_rank_vector'
    assert result['valid_pairs']==4


def test_four_policies_per_history_and_one_correlation_per_contrast_not_pooled():
    result=r.analyze(summary())
    assert result['independent_histories']==12 and len(result['policy_points'])==48
    assert len(result['per_history'])==24 and len(result['contrast_correlations'])==16
    assert all(v['optimized']['planned_pairs']==4 for v in result['per_history'])
    assert all(v['planned_pairs']==12 for v in result['contrast_correlations'])
    assert result['within_history_summary']['actual_M']['optimized']['median']==pytest.approx(1.)
    assert result['no_p_values'] and result['no_pooled_method_history_sample']


def test_failed_random_contrast_preserves_twelve_planned_and_eleven_valid_histories():
    data=summary();point=next(p for p in data['paired_rows'] if p['contrast']=='full_vs_random' and p['seed']==5)
    point.update(delta_predicted_mean=None,delta_actual_M=None,delta_noiseless_joint_M=None)
    result=r.analyze(data);correlation=next(x for x in result['contrast_correlations'] if x['contrast']=='full_vs_random' and x['response']=='delta_actual_M')
    assert correlation['planned_pairs']==12 and correlation['valid_pairs']==11 and correlation['missing_histories']==[5]
    assert len([p for p in result['contrast_points'] if p['contrast']=='full_vs_random' and p['response']=='delta_actual_M'])==12


def test_fallback_uses_executed_score_and_keeps_ties():
    data=summary();constant=next(v for v in data['method_rows'] if v['seed']==1 and v['method']=='constant')
    for row in data['method_rows']:
        if row['seed']==1 and row['method'] in ('charge_mobile','v2g_mobile'):
            row.update(fallback_to_constant=True,predicted_mean=constant['predicted_mean'],
                actual_M=constant['actual_M'],noiseless_joint_M=constant['noiseless_joint_M'])
    result=r.analyze(data);record=next(v for v in result['per_history'] if v['seed']==1 and v['response']=='actual_M')
    assert record['optimized']['x_tied_observations']==2
    bad=deepcopy(data);next(v for v in bad['method_rows'] if v['seed']==1 and v['method']=='charge_mobile')['predicted_mean']=100.
    with pytest.raises(ValueError,match='actually executed'):r.analyze(bad)


def test_within_history_failed_policy_keeps_four_planned_points():
    data=summary();point=next(v for v in data['method_rows'] if v['seed']==1 and v['method']=='v2g_fixed')
    point.update(actual_M=None,noiseless_joint_M=None,status='fit_failure')
    result=r.analyze(data);record=next(v for v in result['per_history'] if v['seed']==1 and v['response']=='actual_M')
    assert record['optimized']['valid_pairs']==3 and record['optimized']['planned_pairs']==4
    assert len([v for v in result['policy_points'] if v['seed']==1])==4
