import numpy as np

from ...local_rx_identifiability.cases import make_case
from ... import likelihood as lk
from ..ac_holdout import fit_fixed


def test_ac_fixed_observation_only_and_descent():
    case = make_case('paper15', 'ordinary', 32991, total=16)
    candidate = case['candidates'][0]
    sigma = {k:case['meter']['sigma_'+k] for k in ('p','q','v')}
    # Public-domain centre rather than true lines.
    z = np.sqrt(case['domain']['z_min']*case['domain']['z_max'])
    phi = np.arctan(.6)
    r = np.full(candidate.m, z*np.cos(phi)); x = np.full(candidate.m, z*np.sin(phi))
    fit = fit_fixed(candidate, case['observed'], sigma, case['domain'], r, x)
    assert fit['qualified'], fit['projected_gradient']
    assert fit['lines_updated'] is False and fit['truth_used'] is False
    assert np.array_equal(fit['r'], r) and np.array_equal(fit['x'], x)
    values = [a['loss'] for a in fit['trace']]+[fit['loss']]
    assert np.max(np.diff(values)) < 1e-9
    physical = lk.ac_predict(candidate, r, x, fit['p'], fit['q'], root_v=fit['root_v'])
    loss = sum(np.sum(((fit[k]-case['observed'][k])/sigma[k])**2) for k in ('p','q','v'))
    assert abs(loss-fit['loss']) < 1e-9
    assert np.max(abs(physical['v']-fit['v'])) < 1e-12


def test_ac_noiseless_fixed_correct_lines_zero_loss():
    case = make_case('paper11', 'ordinary', 32991, total=8)
    obs = {k:case['truth'][k] for k in ('p','q','v')}
    sigma = {k:case['meter']['sigma_'+k] for k in ('p','q','v')}
    fit = fit_fixed(case['candidates'][0], obs, sigma, case['domain'],
                    case['truth']['r'], case['truth']['x'])
    assert fit['qualified']
    assert fit['loss'] < 1e-10
