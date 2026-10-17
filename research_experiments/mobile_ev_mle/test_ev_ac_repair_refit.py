"""Physical replay, shared lines, real source Q, and retained failed fits."""
from dataclasses import replace
import json

import numpy as np
import pytest

from . import ev_ac_repair_refit as repair
from . import likelihood as lk
from .ev_fixed_train_fit import subset_observations, subset_model


OPTIONS = dict(r_bounds=(.002, .04), x_bounds=(.0003, .02),
    ratio_bounds=(.12, .5), source_p_bounds=(-.2, .2), source_q_bounds=(-.2, .2),
    voltage_bounds=(.85, 1.15), multistart=1, max_nfev=200, tol=1e-10)


def fixture():
    rng = np.random.default_rng(713); T = 16; n = 3
    candidate = lk.RadialCandidate.from_supports('star', n, [])
    r = np.array([.009, .014, .018, .023])
    x = r*np.array([.19, .26, .34, .42])
    p = rng.uniform(.025, .19, (T, n))
    q = rng.uniform(.015, .095, (T, n))
    # The future includes a genuine active-power probing block, already in P.
    p[11:, 1] += np.array([.03, .12, -.01, .08, .02])
    sp = rng.uniform(.01, .06, T); sq = rng.uniform(.025, .065, T)
    root = rng.uniform(.998, 1.003, T)
    model = lk.MeasurementModel(.0008, .0006, .0001, .0005, .0004,
        bias_p=np.array([.0002, -.0001, .0004]),
        bias_q=np.array([-.0002, .0003, .0001]),
        bias_v=np.array([.0001, .0002, -.0001]))
    phy = lk.ac_predict(candidate, r, x, p, q, sp, sq, root)
    obs = lk.Observations(p+model.bias_p, q+model.bias_q,
        phy['v']+model.bias_v, root, phy['master_p'], phy['master_q'])
    ds, es = slice(0, 11), slice(11, None)
    D, E = (subset_observations(obs, s) for s in (ds, es))
    dm, em = (subset_model(model, obs, s) for s in (ds, es))
    return (candidate,), D, dm, E, em, (r, x, sp, sq)


def test_noiseless_full_ac_source_q_and_shared_parameter_replay():
    bank, D, dm, E, em, truth = fixture()
    history = repair.fit_history(bank, D, dm, OPTIONS, seed=33)
    assert history.complete, history.record()
    result = repair.evaluate_refits(bank, D, dm, E, em, history, OPTIONS,
        seed=37, bayes_centers={'star': dict(r=truth[0]*1.08, x=truth[1]*1.08)})
    assert result['complete'], result
    row = result['comparisons']['star']
    assert [r['label'] for r in row['starts']] == ['history', 'bayes']
    assert all(len(r['optimizer']['diagnostics']['restarts']) == 1 for r in row['starts'])
    assert row['joint']['loss'] <= row['fixed']['loss']+1e-8
    state = row['joint']['state']; r = np.asarray(state['r']); x = np.asarray(state['x'])
    sp = np.asarray(state['source_p']); sq = np.asarray(state['source_q'])
    assert len(sp) == len(sq) == D.t+E.t
    assert np.mean(sq) > .02  # Reactive source is not suppressed or merged into P.
    assert np.all(x/r >= OPTIONS['ratio_bounds'][0]-1e-12)
    assert np.all(x/r <= OPTIONS['ratio_bounds'][1]+1e-12)
    parts = []
    for block, meter, slc, label in ((D, dm, slice(0,D.t), 'history'),
                                   (E, em, slice(D.t,None), 'future')):
        phy = lk.ac_predict(bank[0], r, x, block.p-meter.bias_p,
            block.q-meter.bias_q, sp[slc], sq[slc], block.root_v)
        residual = np.c_[(phy['v']+meter.bias_v-block.v)/meter.v_sigma,
            (phy['master_p']-block.master_p)/meter.master_p_sigma,
            (phy['master_q']-block.master_q)/meter.master_q_sigma]
        parts.append(float(np.sum(residual**2)))
        assert parts[-1] == pytest.approx(row['joint']['blocks'][label]['loss'], abs=1e-10)
    assert sum(parts) == pytest.approx(row['joint']['loss'], abs=1e-10)
    assert sum(parts) < 1e-7
    assert row['joint']['ac_equation_residual'] < 1e-8
    assert row['joint']['power_balance_residual'] < 1e-8
    json.dumps(history.record(), allow_nan=False)
    json.dumps(result, allow_nan=False)


def test_failed_joint_optimizer_retains_valid_plan_and_all_candidates(monkeypatch):
    bank, D, dm, E, em, _ = fixture()
    history = repair.fit_history(bank, D, dm, OPTIONS, seed=44)
    original = repair.fit_rx_inputs
    def fail_joint(*args, **kwargs):
        if kwargs.get('fixed_r') is None:
            raise RuntimeError('synthetic exhausted joint solver')
        return original(*args, **kwargs)
    monkeypatch.setattr(repair, 'fit_rx_inputs', fail_joint)
    result = repair.evaluate_refits(bank, D, dm, E, em, history, OPTIONS, seed=45)
    assert not result['complete']
    assert result['keep_all_on_failure'] and result['retained_on_failure'] == ['star']
    row = result['comparisons']['star']
    assert row['joint']['usable'] and not row['joint']['success']
    assert row['joint']['optimization_failure']
    assert row['joint']['selected_origin'] == 'witness:history'
    assert row['joint']['loss'] == pytest.approx(row['fixed']['loss'])
    assert 'synthetic exhausted joint solver' in row['starts'][0]['error']


def test_unavailable_history_and_changed_cache_are_explicit(monkeypatch):
    bank, D, dm, E, em, _ = fixture()
    history = repair.fit_history(bank, D, dm, OPTIONS, seed=51)
    with pytest.raises(ValueError, match='cache differs'):
        repair.evaluate_refits(bank, replace(D, v=D.v+.0002), dm,
            E, em, history, OPTIONS)
    history.fits.clear()
    result = repair.evaluate_refits(bank, D, dm, E, em, history, OPTIONS)
    assert not result['complete']
    assert result['retained_on_failure'] == ['star']
    assert 'history fit unavailable' in result['failures']['star']
    assert result['losses']['joint'] == {}


def test_invalid_bayesian_center_cannot_escape_original_domain():
    bank, D, dm, E, em, _ = fixture()
    history = repair.fit_history(bank, D, dm, OPTIONS, seed=61)
    out = repair.evaluate_refits(bank, D, dm, E, em, history, OPTIONS, seed=62,
        bayes_centers={'star': dict(r=np.full(4,.1), x=np.full(4,.001))})
    row = out['comparisons']['star']
    assert row['starts'][1]['label'] == 'bayes'
    assert not row['starts'][1]['success']
    assert 'original box and ratio cone' in row['starts'][1]['error']
    assert row['joint']['line_domain_feasible']
    # An optional illegal start is recorded; a fully converged legal history
    # start remains available, without widening the electrical domain.
    assert row['joint_no_worse_than_fixed']
