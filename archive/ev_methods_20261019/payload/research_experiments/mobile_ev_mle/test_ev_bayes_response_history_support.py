"""Observed-only fixed-line support checks on a small complete AC record."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from . import ev_bayes_response_history_support as tool
from .ev_bayes_small_audit import _physics
from .likelihood import RadialCandidate, MeasurementModel


def history_record():
    n, H, V = 2, 2, 2
    c = RadialCandidate.from_supports('star', n)
    r, x = np.ones(c.m)*.01, np.ones(c.m)*.002
    latent = np.array([[.1, .13, .03, .04, .1, .04], [.12, .1, .04, .03, .09, .03],
        [.11, .1, .03, .04, .11, .04], [.13, .12, .04, .03, .1, .03]])
    model = MeasurementModel(p_sigma=.01, q_sigma=.012, v_sigma=.001,
        master_p_sigma=.008, master_q_sigma=.009, bias_p=.002, bias_q=-.001, bias_v=.0001)
    physical = _physics(c, r, x, latent)
    observed = dict(p=physical['p']+model.bias_p, q=physical['q']+model.bias_q,
        v=physical['v']+model.bias_v, root_v=np.ones(H+V),
        master_p=physical['master_p'], master_q=physical['master_q'], source_p=None, source_q=None)
    blocks = [{k: None if v is None else np.asarray(v)[slc] for k, v in observed.items()} for slc in (slice(0, H), slice(H, None))]
    posterior_r, posterior_x = r*1.2, x*1.15
    eta = np.c_[np.log(posterior_r), np.log(posterior_x/posterior_r)].ravel()
    return dict(spec=dict(index=0, seed=2026110100), status='complete', candidate_names=['star'],
        observed_training=blocks[0], observed_evidence=blocks[1], train_meter=asdict(model), evidence_meter=asdict(model),
        training_result=dict(fits=[dict(name='star', supports=c.supports, r=r, x=x)]),
        prediction=dict(joint=dict(options=dict(r_bounds=[1e-6, .1], x_bounds=[1e-6, .1],
            source_p_bounds=[-.5, .5], source_q_bounds=[-.5, .5], voltage_bounds=[.85, 1.15], ratio_bounds=[.12, .4]),
            candidate_updates=dict(star=dict(TV_posterior_future=dict(mean=eta))))))


def test_join_broadcasts_meter_on_each_original_block():
    raw = history_record()
    raw['train_meter']['v_sigma'] = [[.001, .002], [.003, .004]]
    raw['evidence_meter']['v_sigma'] = [.005, .006]
    observed, meter, H, V = tool.history_inputs(raw)
    assert (H, V) == (2, 2) and observed.p.shape == (4, 2)
    np.testing.assert_allclose(meter.v_sigma, [[.001, .002], [.003, .004], [.005, .006], [.005, .006]])
    assert meter.bias_p.shape == (4, 2)
    assert observed.source_p is None and meter.source_p_sigma is None


def test_real_source_fit_keeps_lines_and_independent_sse_identity():
    raw = history_record(); result = tool.support_record(raw, multistart=1, max_nfev=40)
    assert result['success'], result['failures']
    count = result['denominators']
    assert count['line_center_fit_slots_expected'] == count['fit_slots_successful'] == 2
    assert count['source_solver_calls'] == 2
    row = result['candidates'][0]
    assert row['centers']['train_ML']['SSE'] < 1e-14
    assert row['SSE_increase_bounded_minus_ML'] > 0
    for label, center in row['centers'].items():
        assert center['ratio_domain']['feasible']
        assert center['independent_replay']['saved_loss_identity_error'] < 1e-12
        blocks = center['independent_replay']['blocks']
        assert abs(blocks['training']['SSE']+blocks['evidence']['SSE']-blocks['all']['SSE']) < 1e-14
        assert center['diagnostics']['line_parameters_fixed']
        assert center['diagnostics']['source_node'] == 1


def test_no_future_or_true_topology_dependency_and_same_restart_seeds():
    raw = history_record(); calls = []
    actual = tool.source_fit
    def watched(*args, **kwargs):
        calls.append(kwargs['seed'])
        return actual(*args, **kwargs)
    first = tool.support_record(raw, multistart=1, max_nfev=40, fitter=watched)
    poisoned = deepcopy(raw)
    poisoned['true_name'] = 'malicious-unused'
    poisoned['true_future_inputs_evaluation_only'] = object()
    poisoned['methods'] = object()
    second = tool.support_record(poisoned, multistart=1, max_nfev=40, fitter=watched)
    assert first['success'] and second['success']
    assert len(set(calls)) == 1
    assert first['candidates'][0]['SSE_increase_bounded_minus_ML'] == second['candidates'][0]['SSE_increase_bounded_minus_ML']
    assert not second['topology_truth_used'] and not second['future_used']


def test_failed_candidate_and_failed_source_solver_keep_denominators():
    raw = history_record(); raw['candidate_names'].append('missing')
    def failed(*args, **kwargs): raise RuntimeError('kept optimizer failure')
    result = tool.support_record(raw, fitter=failed)
    count = result['denominators']
    assert not result['success'] and count['line_center_fit_slots_expected'] == 4
    assert count['fit_slots_attempted'] == 2 and count['fit_slots_successful'] == 0
    assert count['fit_slots_failed'] == 4 and len(result['candidates']) == 2
    assert len(result['failures']) == 3


def test_ratio_violation_is_a_failure_without_silent_projection():
    raw = history_record(); raw['prediction']['joint']['options']['ratio_bounds'] = [.3, .4]
    result = tool.support_record(raw)
    assert not result['success']
    assert result['denominators']['fit_slots_attempted'] == 2
    assert result['denominators']['source_solver_calls'] == 0
    assert all('ratio domain' in f['failure'] for f in result['failures'])
