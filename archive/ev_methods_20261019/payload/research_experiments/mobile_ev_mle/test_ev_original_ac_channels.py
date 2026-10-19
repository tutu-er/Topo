"""Arithmetic/unit tests for saved-fit channel decomposition, with no fitting."""
from copy import deepcopy
import csv

import numpy as np
import pytest

from . import ev_original_ac_channels as channels


def fixture(scale=1.,baseline=10.):
    H=N=2;sigmas=dict(p=.01,q=.02,v=.001,master_p=.03,master_q=.04)
    history={k:np.full((H,3) if k in ('p','q','v') else H,1. if k=='v' else .1)
             for k in channels.CHANNELS}
    future={k:np.full((N,3) if k in ('p','q','v') else N,1. if k=='v' else .1)
            for k in channels.CHANNELS}
    predicted={k:np.concatenate((history[k],future[k]))+sigmas[k]*scale for k in sigmas}
    fit=dict(name='candidate',success=True,predicted=predicted,loss=44*scale*scale)
    hfit=dict(success=True,loss=baseline)
    meter={k+'_sigma':v for k,v in sigmas.items()}
    return fit,hfit,history,future,meter


def test_joint_sse_partition_history_penalty_and_physical_rms_units():
    fit,hfit,H,N,m=fixture()
    r=channels.candidate_channels(fit,hfit,H,N,m,m,20.)
    assert r['historical_SSE']==pytest.approx(22)
    assert r['future_SSE']==pytest.approx(22)
    assert r['historical_penalty']==pytest.approx(12)
    assert r['added_half_SSE']==pytest.approx(17)
    assert r['future_V_SSE']==pytest.approx(6)
    assert r['future_master_P_SSE']==pytest.approx(2)
    assert r['future_V_RMS_pu']==pytest.approx(.001)
    assert r['future_P_RMS_kw']==pytest.approx(.2)
    assert r['future_Q_RMS_kvar']==pytest.approx(.4)
    assert r['future_master_P_RMS_kw']==pytest.approx(.6)
    assert r['future_master_Q_RMS_kvar']==pytest.approx(.8)
    assert r['future_V_white_RMS']==pytest.approx(1)


def test_signed_margin_has_exact_channel_and_historical_penalty_decomposition():
    f,h,H,N,m=fixture();wrong=channels.candidate_channels(f,h,H,N,m,m,20)
    f,h,H,N,m=fixture(.5,3);own=channels.candidate_channels(f,h,H,N,m,m,20)
    r=channels.pair_decomposition(own,wrong)
    assert r['M']==pytest.approx(33)
    assert r['historical_difference']==pytest.approx(16.5)
    assert r['historical_baseline_margin']==pytest.approx(7)
    assert r['historical_penalty_difference']==pytest.approx(9.5)
    assert r['future_V_difference']==pytest.approx(4.5)


def test_negative_historical_penalty_is_retained_for_attained_local_fits():
    f,h,H,N,m=fixture(baseline=30)
    r=channels.candidate_channels(f,h,H,N,m,m,20)
    assert r['historical_penalty']==pytest.approx(-8)
    assert r['added_half_SSE']==pytest.approx(7)


def test_missing_or_failed_candidate_does_not_become_zero_distance():
    f,h,H,N,m=fixture();r=channels.candidate_channels(f,None,H,N,m,m,20)
    assert r['available'] and not r['historical_baseline_success']
    assert r['historical_penalty'] is None and r['added_half_SSE'] is None
    missing=channels.candidate_channels(None,h,H,N,m,m,20)
    assert not missing['available'] and 'joint_SSE' not in missing
    assert not channels.pair_decomposition(r,missing)['available']
    failed=dict(r,fit_success=False)
    assert not channels.pair_decomposition(r,failed)['available']


def test_channel_partition_detects_changed_joint_loss():
    f,h,H,N,m=fixture();f['loss']+=1
    with pytest.raises(ValueError,match='partition'):
        channels.candidate_channels(f,h,H,N,m,m,20)


def test_guard_fallback_resource_comes_from_selected_constant_plan():
    constant=np.array([[2,0,0],[2,0,0.]])
    mobile=np.array([[0,6,0],[0,6,0.]])
    trial=dict(config=dict(dt_hours=.25),selection=dict(constant_operation=dict(u_kw=constant,moves=0,
        travel_energy_kwh=0.,driver_cost=0.),methods=dict(v2g_mobile=dict(guard=dict(accepted=False),
        operation=dict(u_kw=mobile,moves=1,travel_energy_kwh=.2,driver_cost=30.)))))
    r=channels.selected_resource(trial,'v2g_mobile',dict(power_kw=constant))
    assert r['moves']==0 and r['drive_kwh']==0 and r['remote_throughput_kwh']==0
    trial['selection']['methods']['v2g_mobile']['guard']['accepted']=True
    r=channels.selected_resource(trial,'v2g_mobile',dict(power_kw=mobile))
    assert r['moves']==1 and r['drive_kwh']==pytest.approx(.2)
    assert r['remote_throughput_kwh']==pytest.approx(3.)


def test_existing_noiseless_csv_is_independently_crosschecked(tmp_path):
    f,h,H,N,m=fixture();r=channels.candidate_channels(f,h,H,N,m,m,20)
    row=dict(seed=1,method='constant',candidate='candidate',**r)
    fields=['seed','method','candidate','joint_SSE','historical_SSE','added_half_SSE',
        'future_V_SSE','future_P_SSE','future_Q_SSE','future_master_P_SSE','future_master_Q_SSE']
    path=tmp_path/'noiseless.csv'
    with path.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader();w.writerow({k:row[k] for k in fields})
    assert channels.crosscheck_csv([row],path)['rows_checked']==1
    unavailable=dict(seed=1,method='failed_method',candidate='candidate',available=False,reason='operation_failure')
    audit=channels.crosscheck_csv([row,unavailable],path)
    assert audit['expected_candidate_rows']==2 and len(audit['omitted_unavailable_rows'])==1
    bad=deepcopy(row);bad['future_V_SSE']+=1
    with pytest.raises(ValueError,match='CSV'):
        channels.crosscheck_csv([bad],path)


def test_within_history_spearman_preserves_ties_and_uses_one_history():
    np.testing.assert_allclose(channels.midranks([3,1,1,4]),[3,1.5,1.5,4])
    rows=[dict(method=str(i),predicted=x,M=y) for i,(x,y) in enumerate([(1,4),(1,4),(2,3),(3,1)])]
    r=channels.within_history_spearman(rows,'predicted','M')
    assert r['available'] and r['spearman']==pytest.approx(-1)
    assert r['planned_methods']==r['valid_method_pairs']==4
    rows.append(dict(method='failed',predicted=None,M=None))
    r=channels.within_history_spearman(rows,'predicted','M')
    assert r['planned_methods']==5 and r['valid_method_pairs']==4 and r['missing_methods']==['failed']


def test_constant_or_missing_rank_correlations_remain_unavailable():
    rows=[dict(method=str(i),x=1.,y=float(i)) for i in range(9)]
    r=channels.within_history_spearman(rows,'x','y')
    assert not r['available'] and r['reason']=='constant_predictor' and r['valid_method_pairs']==9
    rows=[dict(method='missing',x=None,y=2)]
    r=channels.within_history_spearman(rows,'x','y')
    assert not r['available'] and r['reason']=='fewer_than_two_available_method_pairs'
