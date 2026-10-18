import copy

import numpy as np
import pytest

from ..local_rx_identifiability.audit import _write
from .audit import audit_fit, audit_directory, _inputs, _traces, _sha, _expected
from .diagnostics import derivative_blocks, diagnose
from .test_solver import fixture


def point(mode='U'):
    candidate, theta, obs, sigma, domain, root, local = fixture(24)
    kwargs = dict(root_known=None, root_meter=None)
    if mode == 'K':
        kwargs['root_known'] = root
        local = local[:, :6]
    if mode == 'R':
        kwargs['root_meter'] = (root+.0002, .0005)
    r, x = np.exp(theta[:4])*np.cos(theta[4:]), np.exp(theta[:4])*np.sin(theta[4:])
    fitted = dict(r=r.tolist(), x=x.tolist(), p=obs['p'].tolist(), q=obs['q'].tolist(),
                  root_v=root.tolist(), theta=theta.tolist(), local=local.tolist(),
                  status='unqualified', qualified=False)
    value = derivative_blocks(candidate, obs, sigma, fitted, **kwargs)
    residual, a, c = (value[k] for k in ('residual', 'shared', 'state'))
    # At this interior supplied test point the projected gradient is the raw
    # gradient (or smaller than its distance to the broad physical bounds).
    ga, gc = np.einsum('tri,tr->i', a, residual), np.einsum('tri,tr->ti', c, residual)
    low = np.r_[np.full(4, np.log(domain['z_min'])), np.full(4, np.arctan(domain['ratio_min']))]
    high = np.r_[np.full(4, np.log(domain['z_max'])), np.full(4, np.arctan(domain['ratio_max']))]
    ll = np.c_[(-np.array(domain['power_caps'])-obs['p'])/sigma['p'],
               (-np.array(domain['power_caps'])-obs['q'])/sigma['q']]
    lu = np.c_[(np.array(domain['power_caps'])-obs['p'])/sigma['p'],
               (np.array(domain['power_caps'])-obs['q'])/sigma['q']]
    if mode != 'K':
        ll = np.c_[ll, np.full(24, (domain['root_min']-1.)/.01)]
        lu = np.c_[lu, np.full(24, (domain['root_max']-1.)/.01)]
    pg = max(np.max(abs(np.clip(-ga, low-theta, high-theta))),
             np.max(abs(np.clip(-gc, ll-local, lu-local))))
    loss = float(np.sum(residual**2))
    trace = [loss+1., loss]
    fitted.update(loss=loss, J=loss/2, residual=residual.tolist(), trace=trace,
                  projected_gradient=float(pg),
                  attempts=[dict(loss=loss, trace=trace, accepted_steps=1, physical=True)])
    diagnostics = diagnose(candidate, obs, sigma, domain, fitted, **kwargs)
    fitted['information'] = dict(rank=diagnostics['profile_information']['rank'])
    return candidate, obs, sigma, domain, fitted, kwargs, diagnostics


@pytest.mark.parametrize('mode,rank', [('U', 6), ('R', 8), ('K', 8)])
def test_saved_points_replay_in_all_boundary_models_without_fitting(mode, rank):
    candidate, obs, sigma, domain, fitted, kwargs, diagnostics = point(mode)
    issues = []
    result = audit_fit(candidate, fitted, obs, sigma, domain, issues, 'test',
                       saved_diagnostics=diagnostics, **kwargs)
    assert not [i for i in issues if i['severity'] == 'error'], issues
    assert result['profile_rank'] == rank
    assert result['independent_voltage_max_error'] < 2e-10
    assert result['complete_raw_sse'] == pytest.approx(fitted['loss'])
    assert sum(i['code'] == 'scientific_fit_not_qualified' for i in issues) == 1


def test_tampered_root_loss_and_trace_are_detected():
    candidate, obs, sigma, domain, fitted, kwargs, diagnostics = point('R')
    fitted['loss'] += 2.
    fitted['attempts'][0]['trace'] = [1., 2.]
    issues = []
    audit_fit(candidate, fitted, obs, sigma, domain, issues, 'tampered',
              saved_diagnostics=diagnostics, **kwargs)
    codes = {item['code'] for item in issues}
    assert 'loss_mismatch' in codes and 'complete_loss_mismatch' in codes
    assert 'attempt_trace_not_decreasing' in codes


def test_chosen_attempt_must_be_lowest_legal_loss():
    _, _, _, _, fitted, _, _ = point()
    fitted['attempts'].append(dict(loss=-1., trace=[0., -1.], accepted_steps=1, physical=True))
    issues = []; _traces(fitted, issues, 'tampered')
    assert 'chosen_loss_not_best_legal_attempt' in {item['code'] for item in issues}


def test_unmeasured_input_cannot_silently_receive_boundary():
    spec = dict(network='paper15', seed=33001, N=288, boundary='U')
    inference = dict(observed={k:np.ones((288, 3)).tolist() for k in ('p', 'q', 'v')},
                     root_known=np.ones(288).tolist(), root_meter=None, truth_used=False,
                     interval_minutes=15, information_rule='K is ideal')
    issues = []; _inputs(inference, spec, issues, 'tampered')
    assert 'unmeasured_boundary_trajectory_exposed' in {item['code'] for item in issues}


def test_scientific_failure_is_preserved_as_warning():
    candidate, obs, sigma, domain, _, kwargs, _ = point()
    issues = []
    result = audit_fit(candidate, dict(status='failed', qualified=False, attempts=[]),
                       obs, sigma, domain, issues, 'failed', **kwargs)
    assert not result['replayed'] and result['failure_preserved']
    assert all(i['severity'] == 'warning' for i in issues)


def test_full_main_matrix_cannot_be_silently_replaced_by_subset(tmp_path):
    from pathlib import Path
    protocol = Path(__file__).with_name('protocol.md')
    config = dict(tasks=_expected(False)[:2], pilot=False, independent_random_families=1,
                  protocol_sha256=_sha(protocol), source_manifest={})
    _write(tmp_path/'configuration.json', config)
    result = audit_directory(tmp_path)
    assert 'configuration_slot_matrix' in {i['code'] for i in result['issues']}
    config['representative_replay'] = True
    _write(tmp_path/'configuration.json', config)
    subset = audit_directory(tmp_path)
    assert subset['representative_replay']
    assert not {'configuration_slot_matrix', 'representative_slot_matrix'} & {i['code'] for i in subset['issues']}
    assert subset['configured_candidate_slots'] == 8
    assert subset['independent_random_families'] == 1
    # Missing saved records remain errors in either mode.
    assert subset['errors'] > 0 and not subset['audit_pass']


def test_source_bytes_must_match_frozen_configuration(tmp_path, monkeypatch):
    from pathlib import Path
    from . import audit
    source = tmp_path/'restored_source'; source.mkdir()
    (source/'scientific.py').write_text('original=1\n', encoding='utf-8')
    monkeypatch.setattr(audit, 'SOURCE_ROOT', source)
    config = dict(tasks=_expected(True), pilot=True, independent_random_families=1,
                  protocol_sha256=_sha(Path(__file__).with_name('protocol.md')),
                  source_manifest={'scientific.py':'0'*64})
    _write(tmp_path/'configuration.json', config)
    result = audit_directory(tmp_path)
    assert 'source_hash_mismatch' in {item['code'] for item in result['issues']}
