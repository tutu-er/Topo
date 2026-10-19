"""Independent physical identities and bound direction checks."""
import numpy as np
import pytest

from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle.local_rx_identifiability.cases import candidate_bank, public_domain
from research_experiments.mobile_ev_mle.local_response_score.remainder import (
    contrast_matrix, exact_remainder, remainder_bound,
)


def _inputs(candidate, seed=51, count=11):
    rng = np.random.default_rng(seed)
    domain = public_domain()
    domain.update(power_caps=np.array([.1, .09, .08]))
    phi = rng.uniform(np.arctan(domain['ratio_min']), np.arctan(domain['ratio_max']), candidate.m)
    mag = rng.uniform(domain['z_min'], .65*domain['z_max'], candidate.m)
    r, x = mag*np.cos(phi), mag*np.sin(phi)
    p = rng.uniform(-.025, .06, (count, 3))
    q = rng.uniform(-.01, .025, (count, 3))
    b = rng.uniform(.995, 1.02, count)
    physical = lk.ac_predict(candidate, r, x, p, q, root_v=b)
    assert np.min(abs(physical['voltage'])) >= domain['v_min']
    return domain, r, x, p, q, b, physical


@pytest.mark.parametrize('n', [2, 3, 5])
def test_contrasts_orthonormal(n):
    d = contrast_matrix(n)
    np.testing.assert_allclose(d @ np.ones(n), 0., atol=1e-15)
    np.testing.assert_allclose(d @ d.T, np.eye(n-1), atol=1e-15)


@pytest.mark.parametrize('candidate', candidate_bank(), ids=lambda c: c.name)
def test_exact_distflow_contrasts_include_pair_loss_propagation(candidate):
    _, r, x, p, q, _, physical = _inputs(candidate)
    result = exact_remainder(candidate, r, x, p, q, physical=physical)
    assert result['identity_max_abs'] < 1e-12
    np.testing.assert_allclose(result['exact_contrast'],
        result['linear_contrast']+result['contrast_remainder'], atol=1e-12)


@pytest.mark.parametrize('candidate', candidate_bank(), ids=lambda c: c.name)
def test_public_domain_bound_contains_legal_ac_explanations(candidate):
    for seed in range(3):
        domain, r, x, p, q, _, physical = _inputs(candidate, seed+67)
        actual = exact_remainder(candidate, r, x, p, q, physical=physical)
        bound = remainder_bound(candidate, domain)
        value = actual['contrast_remainder']
        assert np.all(value >= bound['contrast_lower'])
        assert np.all(value <= bound['contrast_upper'])
        assert bound['alpha_meter'] == 0.
        assert bound['common_support_edges_cancelled'] == [0]


@pytest.mark.parametrize('candidate', candidate_bank(), ids=lambda c: c.name)
def test_meter_conditioned_bound_uses_no_truth_and_fixed_line_scope(candidate):
    domain, r, x, p, q, _, physical = _inputs(candidate)
    meter = dict(sigma_p=np.full(3, .0005), sigma_q=np.full(3, .0003), sigma_v=np.full(3, .0005))
    # Values stay within the simultaneous rectangle by construction.  The
    # API receives readings only; exact values enter this independent audit.
    observed = dict(p=p+meter['sigma_p'], q=q-meter['sigma_q'], v=physical['v']+.5*meter['sigma_v'])
    public = remainder_bound(candidate, domain)
    observed_bound = remainder_bound(candidate, domain, observed, meter)
    fixed_bound = remainder_bound(candidate, domain, observed, meter, fixed_r=r, fixed_x=x)
    value = exact_remainder(candidate, r, x, p, q, physical=physical)['contrast_remainder']
    assert np.all(value >= observed_bound['contrast_lower'])
    assert np.all(value <= observed_bound['contrast_upper'])
    assert np.all(value >= fixed_bound['contrast_lower'])
    assert np.all(value <= fixed_bound['contrast_upper'])
    assert observed_bound['alpha_meter'] == .01
    assert np.all(observed_bound['radius_about_center'] <= public['radius_about_center'])
    assert np.all(fixed_bound['radius_about_center'] <= observed_bound['radius_about_center'])
    assert 'fixed_lines_only' in fixed_bound['scope']


def test_common_trunk_changes_do_not_change_local_remainder_at_same_terminal_phasors():
    candidate = candidate_bank()[0]
    domain, r, x, p, q, _, physical = _inputs(candidate)
    reference = exact_remainder(candidate, r, x, p, q, physical=physical)
    # A legal gauge transform changes the external slack magnitude and the
    # common trunk while preserving all local current/voltage magnitudes.
    rr, xx = r.copy(), x.copy()
    rr[0] *= .8
    xx[0] *= .8
    new_root = abs(physical['voltage'][:, 0]+complex(rr[0], xx[0])*physical['branch_current'][:, 0])
    changed = exact_remainder(candidate, rr, xx, p, q, root_v=new_root)
    np.testing.assert_allclose(changed['contrast_remainder'], reference['contrast_remainder'], atol=1e-14)


def test_invalid_observed_rectangle_preserves_failure():
    candidate = candidate_bank()[0]
    domain, _, _, p, q, _, physical = _inputs(candidate)
    observed = dict(p=p, q=q, v=physical['v']+1.)
    with pytest.raises(ValueError, match='do not intersect'):
        remainder_bound(candidate, domain, observed, dict(p=.0005, q=.0005, v=.0005))


def test_inconsistent_supplied_physical_state_fails():
    candidate = candidate_bank()[0]
    _, r, x, p, q, _, physical = _inputs(candidate)
    wrong = {**physical, 'v': physical['v']+.01*np.arange(3)[None, :]}
    with pytest.raises(ArithmeticError, match='violates'):
        exact_remainder(candidate, r, x, p, q, physical=wrong)
