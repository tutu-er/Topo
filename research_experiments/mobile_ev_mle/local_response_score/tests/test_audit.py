from copy import deepcopy
import numpy as np
import pytest

from research_experiments.mobile_ev_mle.local_rx_identifiability.cases import candidate_bank, public_domain
from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle.local_response_score import scoring
from research_experiments.mobile_ev_mle.local_response_score.ac_holdout import fit_fixed
from research_experiments.mobile_ev_mle.local_response_score.audit import (
    _data, _profile, _fixed_covariance, _geometry, _audit_profile, _audit_ac, audit_qp,
)


def fixture():
    candidate=candidate_bank()[1]
    domain=public_domain();domain['power_caps']=np.array([.15,.12,.1])
    rng=np.random.default_rng(9701)
    p=rng.uniform(.015,.05,(24,3));q=rng.uniform(.005,.02,(24,3))
    r=np.full(candidate.m,.025);x=np.full(candidate.m,.017)
    physical=lk.ac_predict(candidate,r,x,p,q,root_v=np.ones(len(p)))
    sigma=dict(p=np.full(3,.0005),q=np.full(3,.0002),v=np.full(3,.0005))
    obs={key:np.asarray(value)+rng.normal(size=p.shape)*sigma[key]
         for key,value in dict(p=p,q=q,v=physical['v']).items()}
    return candidate,domain,r,x,obs,sigma


def test_independent_response_profile_and_covariance_recompute():
    candidate,domain,r,x,obs,sigma=fixture()
    data=_data(obs,sigma)
    value=_profile(candidate,data,r,x)
    saved=scoring.score_fixed(candidate,obs,sigma,r,x)
    np.testing.assert_allclose(value['loss'],saved['loss'],rtol=1e-12)
    _audit_profile(candidate,obs,sigma,saved,r,x)
    for weight in ('nominal','upper'):
        expected=scoring.fixed_covariance(candidate,scoring.prepare(obs,sigma),domain,weight=weight)
        np.testing.assert_allclose(_fixed_covariance(candidate,data,domain,weight),expected,atol=1e-18)
    changed=deepcopy(saved);changed['loss']+=1.
    with pytest.raises(AssertionError,match='profile loss'):
        _audit_profile(candidate,obs,sigma,changed,r,x)


def test_independent_sector_qp_geometry_and_dual_recompute():
    candidate,domain,_,_,obs,sigma=fixture()
    original=scoring._qp_geometry(candidate,scoring.prepare(obs,sigma),domain)
    active,lower,upper,a,rhs=_geometry(candidate,domain,_data(obs,sigma)['d'])
    for got,expected in zip((active,lower,upper,a,rhs),
                            (original[1],original[3],original[4],original[5],original[0]['linear_rhs'])):
        np.testing.assert_allclose(got,expected,atol=1e-15)
    saved=scoring.fit_qp(candidate,obs,sigma,domain,weight='upper')
    checked=audit_qp(candidate,obs,sigma,domain,saved)
    assert checked['lower_bound']<=checked['primal']+1e-6
    bad=deepcopy(saved);bad['certificate']['multiplier'][0]=-1.
    with pytest.raises(AssertionError,match='nonnegative'):
        audit_qp(candidate,obs,sigma,domain,bad)
    bad=deepcopy(saved);bad['certificate']['raw_lower_bound']+=1.
    with pytest.raises(AssertionError,match='tangent lower'):
        audit_qp(candidate,obs,sigma,domain,bad)


def test_independent_full_ac_residual_and_fixed_line_boundary():
    candidate,domain,r,x,obs,sigma=fixture()
    fitted=fit_fixed(candidate,obs,sigma,domain,r,x)
    checked=_audit_ac(candidate,obs,sigma,domain,fitted,expected_lines=(r,x))
    assert checked['max_ohm_residual']<1e-10
    assert checked['public_remainder_covered']
    changed=deepcopy(fitted);changed['loss']+=1.
    with pytest.raises(AssertionError,match='SSE'):
        _audit_ac(candidate,obs,sigma,domain,changed,expected_lines=(r,x))
    with pytest.raises(AssertionError,match='fixed R'):
        _audit_ac(candidate,obs,sigma,domain,fitted,expected_lines=(r*1.001,x))
    changed=deepcopy(fitted);changed['projected_gradient']+=.01
    with pytest.raises(AssertionError,match='projected gradient'):
        _audit_ac(candidate,obs,sigma,domain,changed,expected_lines=(r,x))


def test_same_sign_magnitude_box_changes_do_not_make_a_confidence_claim():
    candidate,domain,r,x,obs,sigma=fixture()
    fitted=fit_fixed(candidate,obs,sigma,domain,r,x)
    checked=_audit_ac(candidate,obs,sigma,domain,fitted,expected_lines=(r,x))
    assert checked['meter_remainder_coverage_claim_eligible']==checked['meter_rectangle_member']
