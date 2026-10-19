"""Predeclared physical positions, paired noise and unit audit checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from .ev_physical_spatial_fit_audit import (CONFIG,generated_history,candidates,
    observed,fit_metrics,fit_record,serial)
from .likelihood import ac_predict


def test_independent_histories_have_expected_physical_EV_scale_and_meter_units():
    a=generated_history(987201);b=generated_history(987202)
    assert a['case'].base_kw==1000.
    assert a['meter'].p_sigma*1000==pytest.approx(.6)
    assert a['meter'].q_sigma*1000==pytest.approx(.4)
    assert a['meter'].master_p_sigma*1000==pytest.approx(1.)
    assert not np.array_equal(a['history'].p,b['history'].p)
    assert np.max(abs(a['source_p']*1000-10))<=2.
    assert np.max(abs(a['source_q']*1000-3))<=1.
    assert a['history'].source_p is None
    assert np.min(a['h_correlation_p'][~np.eye(3,dtype=bool)])>.9


def test_predeclared_bank_covers_true_supports_without_truth_argument():
    bank=candidates(); assert len(bank)==4
    assert [c.name for c in bank]==CONFIG['candidate_bank']
    true=generated_history(987201)['case'].candidate
    assert next(c for c in bank if c.name=='clade_02').supports==true.supports


def test_paired_future_noise_has_only_physical_response_differences():
    g=generated_history(987201);case=g['case'];h=12
    noEV=ac_predict(case.candidate,case.r,case.x,g['p'][h:],g['q'][h:],g['source_p'][h:],g['source_q'][h:],g['root_v'][h:])
    total=g['p'][h:].copy();total[:,2]+=np.array(CONFIG['curve_kw'])/1000.
    withEV=ac_predict(case.candidate,case.r,case.x,total,g['q'][h:],g['source_p'][h:],g['source_q'][h:],g['root_v'][h:])
    noise={k:v[h:] for k,v in g['noise'].items()}
    a=observed(noEV,g['meter'],noise,slice(None));b=observed(withEV,g['meter'],noise,slice(None))
    for channel in ('p','q','v','master_p','master_q'):
        assert getattr(b,channel)-getattr(a,channel)==pytest.approx(withEV[channel]-noEV[channel],abs=3e-16)
    assert b.p[:,2]-a.p[:,2]==pytest.approx([.006,0.,.006,0.])


def test_margin_loglik_identity_and_true_line_unit_mapping():
    g=generated_history(987201);case=g['case'];bank=candidates();fits={}
    for c,loss in zip(bank,(12.,10.,8.,11.)):
        r=case.r.copy() if c.name=='clade_02' else np.full(c.m,.003)
        x=case.x.copy() if c.name=='clade_02' else np.full(c.m,.002)
        fits[c.name]=SimpleNamespace(success=True,loss=loss,loglik=100.-.5*loss,
            r=r,x=x,p=g['p'][:12],q=g['q'][:12],source_p=g['source_p'][:12],source_q=g['source_q'][:12])
    result=fit_metrics(fits,bank,g,g['p'][:12],g['q'][:12],g['source_p'][:12],g['source_q'][:12])
    assert result['full_SSE_signed_margin']==2.
    assert result['half_SSE_loglik_margin']==result['actual_loglik_margin']==1.
    assert result['true_candidate_impedance_error']['r_rmse_ohm']==0.
    assert result['source_error_by_candidate']['clade_02']['N_p_rmse_kw'] is None
    fits['clade_01'].success=False
    failed=fit_metrics(fits,bank,g,g['p'][:12],g['q'][:12],g['source_p'][:12],g['source_q'][:12])
    assert failed['full_SSE_signed_margin'] is None


def test_failed_numeric_fit_can_be_saved_as_null_without_deleting_failure():
    assert serial(dict(loss=float('inf'),r=np.array([np.nan]),status='failed'))==dict(loss=None,r=[None],status='failed')
