"""Paired-index/hash tamper checks; fixtures generate meters but never fit/select."""
from copy import deepcopy

import numpy as np
import pytest

from . import ev_fixed_train_training_support_review as review
from .ev_fixed_train_training_support import make_training_inputs, condition_config
from .ev_fixed_train_fit import subset_model
from .gradient_price_study import observations_record
from .information_context import meter_to_record


@pytest.fixture
def fixture():
    data = make_training_inputs(990101, 0, 32)
    case = data['case']; cfg = condition_config(32)
    truth = dict(supports=case['candidate'].supports, **{k: case[k] for k in
                 ('r', 'x', 'p', 'q', 'source_p', 'source_q', 'root_v', 'assets')})
    common = dict(config=cfg, truth=truth, seeds=case['seeds'],
                  validation=observations_record(data['validation']),
                  validation_meter=meter_to_record(data['validation_meter']),
                  future_white=data['future_white'], future_meter=meter_to_record(case['fm']))
    weak = deepcopy(common)
    weak['training_meter'] = meter_to_record(subset_model(case['hm'], case['history'], slice(0, 8)))
    record = deepcopy(common)
    record.update(training_length=32, training_indices=list(range(32)),
        training_index_space='new rich96 stream prefix', validation_indices=[8, 9, 10, 11],
        training_generation=data['generation'], training_truth=data['training_truth'],
        training_white=data['training_white'], training_physical=data['training_physical'],
        training=observations_record(data['training']), training_meter=meter_to_record(data['training_meter']))
    return review.plain(record), review.plain(weak)


def test_rich32_remains_32_and_full96_hashes_match(fixture):
    record, weak = fixture
    result = review.replay_rich_inputs(record, weak)
    assert result['success'] and result['training_length'] == 32
    assert max(result['independent_forward_max_abs_error'].values()) < 2e-10


def test_original_future_offset_is_12_even_with_32_training_points(fixture):
    record, weak = fixture
    holdout = review._holdout(record)
    assert len(holdout['future_p']) == 6
    assert holdout['future_p'] == record['truth']['p'][12:]


def test_relabelling_rich32_as_eight_is_rejected(fixture):
    record, weak = fixture
    record['training_indices'] = list(range(8))
    with pytest.raises(ValueError, match='rich training prefix'):
        review.replay_rich_inputs(record, weak)


def test_paired_validation_cannot_change(fixture):
    record, weak = fixture
    record['validation']['v'][0][0] += 1e-9
    with pytest.raises(ValueError, match='validation/future/asset/device changed'):
        review.replay_rich_inputs(record, weak)


def test_future_source_cannot_change_across_training_conditions(fixture):
    record, weak = fixture
    record['truth']['source_p'][12] += 1e-12
    with pytest.raises(ValueError, match='original latent paths changed'):
        review.replay_rich_inputs(record, weak)


def test_full96_noise_hash_tamper_is_rejected(fixture):
    record, weak = fixture
    record['training_generation']['rich96_white_sha256'] = '0'*64
    with pytest.raises(ValueError, match='rich96 noise hash'):
        review.replay_rich_inputs(record, weak)


def test_prefix_measured_PQ_tamper_is_rejected(fixture):
    record, weak = fixture
    record['training']['p'][0][0] += 1e-12
    with pytest.raises(ValueError, match='measured rich prefix'):
        review.replay_rich_inputs(record, weak)
