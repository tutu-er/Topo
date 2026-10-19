"""Saved-state review checks, using a small fixture without any fitting."""
from copy import deepcopy
import base64
import gzip
from hashlib import sha256
import json

import numpy as np
import pytest

from . import ev_systematic_review as review
from .likelihood import RadialCandidate


def _serial(value):
    return json.loads(json.dumps(value, default=lambda x: np.asarray(x).tolist()))


@pytest.fixture
def record():
    n, H, V, T = 3, 3, 1, 2
    bank = [RadialCandidate.from_supports('star', n, ()),
            RadialCandidate.from_supports('clade_01', n, ((0, 1),))]
    meter = dict(p_sigma=.0006, q_sigma=.0004, v_sigma=.00012,
                 master_p_sigma=.001, master_q_sigma=.001,
                 bias_p=[-.00015, 0., .00015], bias_q=[.0001, 0., -.0001],
                 bias_v=[-.00004, 0., .00004])
    cfg = dict(base_kw=20., source_interface=1, ratio_bounds=[.05, 1.5],
               training=dict(r_bounds=[1e-6, .1], x_bounds=[1e-6, .1]),
               source=dict(source_p_bounds=[-.5, .5], source_q_bounds=[-.5, .5]))
    count = H + V + T
    p = np.tile([.12, .14, .105], (count, 1)) + np.arange(count)[:, None] * .001
    q = .25 * p
    true = dict(r=np.full(bank[0].m, .003), x=np.full(bank[0].m, .001),
                source_p=np.full(count, .05), source_q=np.full(count, .02))
    physical = review.physical.forward_physics(bank[0], true['r'], true['x'], p, q,
        true['source_p'], true['source_q'], np.ones(count))
    all_obs = {key: physical[key] for key in ('p', 'q', 'v', 'root_v', 'master_p', 'master_q')}
    for key in ('p', 'q', 'v'):
        all_obs[key] = all_obs[key] + meter['bias_' + key]
    slices = (slice(0, H), slice(H, H + V), slice(H + V, count))
    observations = [{key: value[part] for key, value in all_obs.items()} for part in slices]
    blocks = list(zip(('training', 'validation', 'future'), observations, [meter] * 3))
    zero = np.zeros((T, n))
    action = sha256(np.asarray(zero, dtype='<f8').tobytes()).hexdigest()
    result = dict(case_id='fixture', seed=901, phase='development', index=0, config=cfg,
        status='complete', true_name='star', candidate_miss=False,
        candidates=[dict(name=c.name, supports=c.supports) for c in bank],
        training=observations[0], validation=observations[1], training_meter=meter,
        validation_meter=meter, future_meter=meter,
        training_result=dict(fits=[], failures={}, all_complete=True), history_prefits={},
        future_records={}, common_cost=dict(training_seconds=1., fixed_history_seconds=.2,
            joint_history_seconds=.3), methods=[])
    comparisons = {}
    for candidate in bank:
        state = dict(r=np.full(candidate.m, .003), x=np.full(candidate.m, .001),
                     source_p=np.full(count, .05), source_q=np.full(count, .02))
        def partial(length, begin=0):
            return {**state, 'source_p': state['source_p'][begin:begin + length],
                    'source_q': state['source_q'][begin:begin + length]}
        training = partial(H)
        train_score = review.replay_state(candidate, training, blocks[:1], cfg)
        trained = dict(**training, name=candidate.name, success=True,
                       loss=train_score['loss'], loglik=train_score['loglik'])
        result['training_result']['fits'].append(trained)
        validation = partial(V, H)
        val_score = review.replay_state(candidate, validation, blocks[1:2], cfg)
        valfit = dict(**validation, success=True, loss=val_score['loss'], loglik=val_score['loglik'])
        prestate = partial(H + V)
        prescore = review.replay_state(candidate, prestate, blocks[:2], cfg)
        fixed_pre = dict(**prescore, success=True, training_fit=trained, validation_fit=valfit)
        joint_pre = dict(**prescore, success=True, state=prestate, optimizer=None)
        result['history_prefits'][candidate.name] = dict(fixed=fixed_pre, joint=joint_pre)
        score = review.replay_state(candidate, state, blocks, cfg)
        fixed_post = dict(**score, success=True, state=state)
        joint_post = dict(**score, success=True, state=state, optimizer=None)
        comparisons[candidate.name] = dict(
            fixed=dict(pre=fixed_pre, post=fixed_post, fixed_total=score['loss'],
                       fixed_holdout=sum(score['blocks'][x]['loss'] for x in ('validation', 'future')),
                       future_increment=score['blocks']['future']['loss']),
            joint=dict(pre=joint_pre, post=joint_post, joint_total=score['loss'],
                       profile_increment=score['loss'] - prescore['loss']),
            checks=dict(source_interface=1, shared_theta_across_all_joint_blocks=True,
                        PQ_direct_inputs=True, EV_already_in_total_PQ=True),
            cost=dict(fixed_future_seconds=.1, joint_post_seconds=.2))
    result['future_records'][action] = dict(observations=observations[2], power_kw=zero,
                                          comparisons=comparisons, failures={})
    for name in review.METHODS:
        inference = {}
        for mode in review.MODES:
            side = 'joint' if mode == 'joint_total' else 'fixed'
            losses = {key: value[side][mode] for key, value in comparisons.items()}
            own = 2 * (.3 if side == 'joint' else .1)
            inference[mode] = dict(losses=losses, complete=True, M=losses['clade_01'] - losses['star'],
                seconds=own, total_seconds=1. + .05 + own + .2 + (.3 if side == 'joint' else 0.))
        result['methods'].append(dict(name=name, power_kw=zero, action_key=action, planning_failure=None,
            planning_seconds=.05, operation=dict(success=True), inference=inference))
    return _serial(result)


def test_complete_states_and_all_denominators_pass(record):
    result = review.review_trial(record)
    assert result['success'] and result['status'] == 'passed'
    assert result['saved_states_replayed'] == 12
    assert result['AC_block_replays'] == 24
    assert result['method_mode_checks'] == 30
    assert result['PQ_once_checks'] == 1


@pytest.mark.parametrize('change', ['loss', 'theta', 'cost', 'PQ', 'source'])
def test_tampered_values_are_rejected(record, change):
    action = next(iter(record['future_records']))
    future = record['future_records'][action]
    if change == 'loss':
        record['methods'][0]['inference']['fixed_total']['losses']['star'] += 1.
    elif change == 'theta':
        future['comparisons']['star']['fixed']['post']['state']['r'][0] *= 1.01
    elif change == 'cost':
        record['methods'][0]['inference']['joint_total']['total_seconds'] -= .1
    elif change == 'PQ':
        # Keep the original none observation as baseline but give another
        # saved action a different P reading without its EV injection.
        changed = deepcopy(future)
        changed['observations']['p'][0][0] += .01
        power = np.array(changed['power_kw']); power[0, 0] = .5
        changed['power_kw'] = power.tolist()
        newkey = sha256(np.asarray(power, dtype='<f8').tobytes()).hexdigest()
        record['future_records'][newkey] = changed
    else:
        future['comparisons']['star']['joint']['post']['state']['source_p'][0] = 1.
    result = review.review_trial(record)
    assert not result['success'] and result['status'] == 'failed'
    assert result['errors']


def test_training_failure_keeps_missing_reason_and_case_denominator():
    row = dict(case_id='train8', seed=902, status='pipeline_failure', failure='required training fits incomplete',
        methods=[dict(name=name, planning_failure='training failure', operation=dict(success=False),
                      inference={mode: dict(complete=False) for mode in review.MODES}) for name in review.METHODS])
    result = review.review_trial(row)
    assert not result['success'] and result['status'] == 'incomplete'
    assert result['available_checks_passed']
    assert result['saved_states_replayed'] == 0 and result['missing']
    assert all('training' in item['reason'] for item in result['missing'])
    assert result['recorded_inference_failures'] == 30


def test_missing_future_is_incomplete_not_a_complete_review_pass(record):
    record['future_records'] = {}
    record['status'] = 'partial_failure'
    result = review.review_trial(record)
    assert result['available_checks_passed'] and not result['success']
    assert result['status'] == 'incomplete' and result['missing']


def test_duplicate_method_slot_is_not_silently_collapsed(record):
    record['methods'].append(deepcopy(record['methods'][0]))
    result = review.review_trial(record)
    assert not result['success']
    assert any(item['check'] == 'method denominator' for item in result['errors'])


def test_mobile_planning_cost_pays_shared_ancestors_only_once():
    own = dict(none=0., constant=0., highlow=0., square_fixed=1., square_mobile=2.,
               scalar_fixed=3., scalar_mobile=4., nearest_fixed=5., gradac_fixed=6., scalar_hist_fixed=7.)
    deps = {name: [] for name in review.METHODS}
    deps.update(square_mobile=['square_fixed'], scalar_fixed=['square_fixed'],
                nearest_fixed=['square_fixed'], gradac_fixed=['square_fixed'], scalar_hist_fixed=['square_fixed'],
                scalar_mobile=['square_fixed', 'square_mobile', 'scalar_fixed'])
    expected = dict(none=0., constant=0., highlow=0., square_fixed=1.1, square_mobile=3.1,
                    scalar_fixed=4.3, scalar_mobile=10.3, nearest_fixed=6.3, gradac_fixed=7.3, scalar_hist_fixed=8.3)
    metadata = {name: dict(dependencies=deps[name], planning_actual_seconds=own[name],
                           planning_seconds=expected[name]) for name in review.METHODS}
    methods = {name: dict(planning_seconds=expected[name]) for name in review.METHODS}
    record = dict(selection=dict(methods=metadata, setup_seconds=.1, context_seconds=.2))
    count = {'planning_dependency_checks': 0}
    review._planning(record, methods, count)
    assert count['planning_dependency_checks'] == 10
    methods['scalar_mobile']['planning_seconds'] += 1.  # double-paid square ancestor
    with pytest.raises(ValueError, match='deduplicated planning'):
        review._planning(record, methods, count)


def test_directory_compact_roundtrip_source_hash_and_expected_count(record, tmp_path):
    payload = b'independent fixture science bytes\n'
    hashes = {'fixture.py': sha256(payload).hexdigest()}
    record.update(policy_id='fixture-policy', source_sha256=hashes)
    directory = tmp_path / 'compact'
    directory.mkdir()
    review.write(directory / 'freeze.json', dict(policy_id='fixture-policy', source_sha256=hashes))
    review.write(directory / 'source_snapshot.json.gz', dict(source_sha256=hashes,
        source_base64={'fixture.py': base64.b64encode(payload).decode()}))
    review.write(directory / 'case_000_901.json.gz', record)
    result = review.review_directories([directory], expected_count=1)
    assert result['success'] and result['case_denominator'] == 1
    assert result['independent_tool_sha256'] and result['source_audits'][0]['source_count'] == 1
    assert not review.review_directories([directory], expected_count=2)['success']
    review.write(directory / 'case_001_901.json.gz', record)
    duplicate = review.review_directories([directory], expected_count=2)
    assert duplicate['case_denominator'] == 2 and not duplicate['success']
    assert duplicate['status_counts'] == {'passed': 1, 'failed': 1}


def test_corrupt_snapshot_fails_without_discarding_reviewed_case(record, tmp_path):
    hashes = {'fixture.py': sha256(b'original').hexdigest()}
    record.update(policy_id='fixture-policy', source_sha256=hashes)
    review.write(tmp_path / 'freeze.json', dict(policy_id='fixture-policy', source_sha256=hashes))
    review.write(tmp_path / 'source_snapshot.json.gz', dict(source_sha256=hashes,
        source_base64={'fixture.py': base64.b64encode(b'changed').decode()}))
    review.write(tmp_path / 'case_000_901.json.gz', record)
    result = review.review_directories([tmp_path], expected_count=1)
    assert not result['success'] and result['case_denominator'] == 1
    assert not result['source_audits'][0]['success']
    assert result['status_counts'] == {'passed': 1}
