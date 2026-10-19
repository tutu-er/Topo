"""Small independent denominator/noise/hash checks for the outside-freeze auditor."""
import base64
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from . import ev_fixed_train_trial_review as review
from .test_ev_fixed_train_review import fixture
from .information_fit_recovery import write_json


def bank_fixture():
    candidate, observations, meter, row = fixture()
    saved = dict(fits=[row], failures={}, all_complete=True)
    config = dict(training=dict(r_bounds=[.001, .01], x_bounds=[.001, .01]),
                  source=dict(source_p_bounds=[-.5, .5], source_q_bounds=[-.5, .5]))
    return candidate, observations, meter, row, saved, config


def test_bank_preserves_failed_candidate_in_denominator():
    c, obs, meter, row, saved, cfg = bank_fixture()
    bank = {c.name: c, 'missing': c}
    with pytest.raises(ValueError, match='disappeared'):
        review._review_bank(saved, bank, obs, meter, {c.name: row}, cfg, training=True)
    saved.update(all_complete=False, failures={'missing': 'computation failure'})
    out = review._review_bank(saved, bank, obs, meter, {c.name: row}, cfg, training=True)
    assert out['replayed_states'] == 1
    assert out['declared_failures'] == {'missing': 'computation failure'}
    assert not out['declared_complete']


def test_fixed_bank_uses_exact_training_lines():
    c, obs, meter, row, saved, cfg = bank_fixture()
    trained = deepcopy(row)
    trained['r'][0] += 1e-15
    with pytest.raises(ValueError, match='trained R changed'):
        review._review_bank(saved, {c.name: c}, obs, meter, {c.name: trained}, cfg)


def test_negative_margin_is_kept_and_nearest_competitor_can_switch():
    assert review._margin({'a': 3., 'b': 1., 'c': 4.}, 'a') == -2.
    assert review._margin({'a': 3., 'b': 8., 'c': 4.}, 'a') == 1.
    assert review._margin({'a': 3.}, 'not-in-bank') is None


def test_white_replay_rejects_changed_noise_and_uses_fixed_bias_once():
    c, obs, meter, row, saved, cfg = bank_fixture()
    physical = review.forward_physics(c, row['r'], row['x'], row['p'], row['q'],
                                     row['source_p'], row['source_q'], obs.root_v)
    expected = {}
    for name in review.CHANNELS:
        mean = physical[name] + (np.asarray(getattr(meter, 'bias_'+name)) if name in ('p', 'q', 'v') else 0)
        expected[name] = (getattr(obs, name)-mean)/getattr(meter, name+'_sigma')
    out = review._white(physical, obs, meter, expected, 'toy')
    assert max(out.values()) == 0.
    expected['v'][0, 0] += .01
    with pytest.raises(ValueError, match='whitened noise'):
        review._white(physical, obs, meter, expected, 'toy')


def source_fixture(tmp_path):
    name = 'ev_fixed_train_fit.py'
    raw = (review.HERE/name).read_bytes()
    hashes = {name: hashlib.sha256(raw).hexdigest()}
    freeze = dict(config=dict(toy=True), source_sha256=hashes)
    freeze['policy_id'] = review.identity(dict(config=freeze['config'], source_sha256=hashes))
    write_json(tmp_path/'source_snapshot.json.gz', dict(source_sha256=hashes,
               source_base64={name: base64.b64encode(raw).decode()}))
    return freeze


def test_snapshot_matches_current_frozen_bytes(tmp_path):
    freeze = source_fixture(tmp_path)
    out = review.source_audit(tmp_path, freeze)
    assert out['success'] and out['source_count'] == 1


def test_snapshot_byte_tamper_is_rejected(tmp_path):
    freeze = source_fixture(tmp_path)
    name = next(iter(freeze['source_sha256']))
    write_json(tmp_path/'source_snapshot.json.gz', dict(source_sha256=freeze['source_sha256'],
               source_base64={name: base64.b64encode(b'changed').decode()}))
    with pytest.raises(ValueError, match='snapshot source'):
        review.source_audit(tmp_path, freeze)


def test_policy_digest_tamper_is_rejected(tmp_path):
    freeze = source_fixture(tmp_path)
    freeze['config']['toy'] = False
    with pytest.raises(ValueError, match='policy digest'):
        review.source_audit(tmp_path, freeze)
