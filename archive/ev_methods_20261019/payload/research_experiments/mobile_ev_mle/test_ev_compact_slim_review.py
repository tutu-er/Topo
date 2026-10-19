"""Slim scope, provenance and arithmetic checks; no fit or selector invoked."""
from copy import deepcopy
import gzip
import inspect
import json

import numpy as np
import pytest

from . import ev_compact_delivery as delivery
from . import ev_compact_slim_review as review
from .test_ev_compact_review import prediction_fixture


def prediction():
    saved, history, meter, candidates, hfits, config, contexts, _ = prediction_fixture()
    return delivery.slim_prediction(saved), candidates, hfits, history, meter, config, contexts


def test_prediction_summary_arithmetic_keeps_scope_and_uses_no_ac_or_fit(monkeypatch):
    from . import likelihood
    def forbidden(*args, **kwargs):
        raise AssertionError('new model/fit called')
    monkeypatch.setattr(likelihood, 'ac_predict', forbidden)
    monkeypatch.setattr(likelihood, 'fit_candidate', forbidden)
    audit = review.audit_prediction_summary(*prediction())
    assert audit == dict(predicted_fit_summaries_checked=4, directed_tasks=2,
                         failed_reference_cells=0, failed_directed_tasks=0,
                         predicted_fit_AC_states_replayed=0, success=True)


@pytest.mark.parametrize('change', [
    lambda s: s['pairs'][0].update(added=1.),
    lambda s: s['records'][0]['fits'][0].update(loglik=1.),
    lambda s: s['records'][0]['fits'][0].update(half_loss=1.),
    lambda s: s['pairs'][0].update(weight=.9),
    lambda s: s['records'][0]['forecast_state']['p'].__setitem__((0, 0), .8),
    lambda s: s['records'][0].update(delivery_prediction_fits_are_summaries=False),
    lambda s: s['records'].pop(),
    lambda s: s['pairs'].pop(),
    lambda s: s.update(uses_truth=True),
])
def test_prediction_summary_tampering_rejected(change):
    args = list(prediction()); change(args[0])
    with pytest.raises(ValueError):
        review.audit_prediction_summary(*args)


def test_predicted_failure_keeps_reference_candidate_and_task_denominators():
    args = list(prediction()); saved = args[0]
    saved['success'] = False
    saved['records'][0].update(success=False, fits=[], failures={'A': 'failed', 'B': 'failed'})
    for pair in saved['pairs']:
        if pair['a'] == 'A':
            pair['success'] = False
    audit = review.audit_prediction_summary(*args)
    assert not audit['success'] and audit['directed_tasks'] == 2
    assert audit['predicted_fit_summaries_checked'] == 2
    assert audit['failed_reference_cells'] == audit['failed_directed_tasks'] == 1


def test_forecast_exception_retains_all_context_and_pair_rows():
    args = list(prediction()); saved = args[0]
    saved['success'] = False
    saved['records'][0].update(success=False, fits=[], failures={'__forecast_or_joint__': 'AC did not converge'})
    for pair in saved['pairs']:
        if pair['a'] == 'A':
            pair['success'] = False
    audit = review.audit_prediction_summary(*args)
    assert not audit['success'] and audit['directed_tasks'] == 2
    assert audit['predicted_fit_summaries_checked'] == 2


def make_directory(tmp_path, *, planned=(1,)):
    source = tmp_path / 'raw'; source.mkdir(); target = tmp_path / 'slim'
    methods = list(review.full.METHODS)
    config = dict(methods=methods, development_seeds=list(planned), calibration_seeds=[201], evaluation_seeds=[301])
    hashes = {f'source_{i}.py': 'a' * 64 for i in range(72)}
    policy = dict(config=config, prediction='center', master_form='shared_control_square', domain='asset', source_sha256=hashes)
    policy_id = review.digest(json.dumps(policy, sort_keys=True).encode())
    freeze = dict(**policy, policy_config=policy, policy_id=policy_id, phase='development', seeds=list(planned))
    trial = dict(**policy, policy_config=policy, policy_id=policy_id, phase='development', seed=1,
                 selection=dict(methods={}, predicted_AC={}, all_prediction_records=[]),
                 rows=[dict(method=name, status='operation_or_design_failure') for name in methods])
    (source / 'freeze.json').write_text(json.dumps(freeze), encoding='utf-8')
    (source / 'source_snapshot.json.gz').write_bytes(gzip.compress(b'{}', mtime=0))
    (source / 'trial_1.json.gz').write_bytes(gzip.compress(json.dumps(trial).encode(), mtime=0))
    manifest = delivery.export_directory(source, target, allow_partial=len(planned) > 1)
    return source, target, freeze, manifest


def fake_audit(trial):
    counters = dict(historical_fit_states_replayed=4, historical_explanation_states_replayed=0,
                    actual_noisy_fit_states_replayed=0, actual_noiseless_fit_states_replayed=0,
                    saved_fit_states_replayed=4, physical_future_states_replayed=0,
                    declared_failed_saved_fit_states=0, unavailable_failed_saved_fit_states=0,
                    actual_unique_noisy_actions=0, actual_unique_noiseless_actions=0,
                    planned_methods=10, saved_methods=10)
    return dict(seed=trial['seed'], success=True, **counters, prediction_summary_audits=[],
                operation_audit=dict(operation_ledgers_replayed=0, inheritance_checks=[]))


def patch_replay(monkeypatch, freeze):
    monkeypatch.setattr(review.source_restore, 'audit_directory', lambda directory: dict(
        success=True, checked_count=72, policy_identity=review.cal.freeze_identity(freeze)))
    monkeypatch.setattr(review, 'audit_trial', fake_audit)


def test_directory_metadata_offline_without_original_raw_and_explicit_scope(tmp_path, monkeypatch):
    source, target, freeze, manifest = make_directory(tmp_path); patch_replay(monkeypatch, freeze)
    (source / 'trial_1.json.gz').unlink()
    output = tmp_path / 'review.json'; result = review.review_directory(target, output)
    assert result['success'] and result['audited_histories'] == 1
    assert not result['complete_raw_replay'] and not result['original_raw_content_independently_read']
    assert result['new_fit_calls'] == result['selector_calls'] == result['optimizer_calls'] == 0
    assert result['totals']['saved_fit_states_replayed'] == 4
    assert result['original_raw_provenance'][0]['sha256'] == manifest['trial_files'][0]['raw_sha256']
    assert json.loads(output.read_text())['scope'].startswith('saved actual AC')
    with pytest.raises(FileExistsError):
        review.review_directory(target, output)


@pytest.mark.parametrize('change', [
    lambda m: m.update(artifact_schema='unknown'),
    lambda m: m.update(complete_raw_replay_replacement=True),
    lambda m: m['trial_files'][0].update(slim_file='../escape.json.gz'),
    lambda m: m['trial_files'][0].update(slim_file='C:\\escape.json.gz'),
    lambda m: m['trial_files'].append(deepcopy(m['trial_files'][0])),
    lambda m: m.update(exported_seeds=[]),
    lambda m: m.update(raw_bytes=1),
])
def test_manifest_scope_paths_duplicates_and_sizes_rejected(tmp_path, change):
    _, target, freeze, manifest = make_directory(tmp_path); change(manifest)
    with pytest.raises(ValueError):
        review.validate_manifest(manifest, freeze)


def test_slim_compressed_hash_and_extra_trial_are_not_ignored(tmp_path, monkeypatch):
    _, target, freeze, manifest = make_directory(tmp_path); patch_replay(monkeypatch, freeze)
    trial = target / manifest['trial_files'][0]['slim_file']; trial.write_bytes(trial.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='compressed byte'):
        review.review_directory(target)
    trial.write_bytes(trial.read_bytes()[:-7])
    (target / 'trial_8.slim.json.gz').write_bytes(b'extra')
    with pytest.raises(ValueError, match='exactly match'):
        review.review_directory(target)


@pytest.mark.parametrize('change', [
    lambda t: t['delivery'].update(source_raw_sha256='b' * 64),
    lambda t: t['delivery'].update(source_raw_bytes=1),
    lambda t: t['delivery'].update(complete_raw_replay_replacement=True),
    lambda t: t.update(policy_id='0' * 64),
    lambda t: t.update(phase='evaluation'),
    lambda t: t['delivery'].update(saved_method_rows=[]),
    lambda t: t['delivery'].update(policy_config_ref='../elsewhere.json'),
])
def test_inner_metadata_failures_remain_in_historical_denominator(tmp_path, monkeypatch, change):
    _, target, freeze, manifest = make_directory(tmp_path); patch_replay(monkeypatch, freeze)
    row = manifest['trial_files'][0]; path = target / row['slim_file']
    trial = review._decode(path.read_bytes(), compressed=True); change(trial)
    data = gzip.compress(json.dumps(trial).encode(), mtime=0); path.write_bytes(data)
    row['slim_sha256'] = review.digest(data); row['slim_bytes'] = len(data); manifest['slim_bytes'] = len(data)
    (target / 'delivery_manifest.json').write_text(json.dumps(manifest))
    result = review.review_directory(target)
    assert not result['success'] and result['available_independent_histories'] == 1
    assert result['audited_histories'] == 0 and result['seeds'] == [1] and len(result['review_errors']) == 1


def test_partial_manifest_missing_history_stays_explicit(tmp_path, monkeypatch):
    _, target, freeze, _ = make_directory(tmp_path, planned=(1, 2)); patch_replay(monkeypatch, freeze)
    result = review.review_directory(target)
    assert result['success'] and result['partial'] and result['missing_histories'] == [2]
    assert result['planned_independent_histories'] == 2 and result['available_independent_histories'] == 1


def test_frozen_phase_seed_overlap_is_rejected(tmp_path):
    _, _, freeze, manifest = make_directory(tmp_path)
    freeze['config']['evaluation_seeds'] = [1]
    with pytest.raises(ValueError, match='seed overlap'):
        review.validate_manifest(manifest, freeze)


def test_duplicate_json_key_is_not_silently_accepted():
    with pytest.raises(ValueError, match='duplicate JSON key'):
        review._decode(b'{"key":1,"key":2}')


def test_actual_replay_uses_independent_arithmetic_and_scope_names_omissions():
    source = inspect.getsource(review)
    assert 'old.replay_fits' in source and 'old.forward_physics' in source
    assert 'predicted_fit_AC_residuals=False' in source and 'nonbest_QP_states=False' in source
    assert 'same_witness_AC_gates=False' in source and 'global_MILP_witness_certificates=False' in source
    import ast
    tree = ast.parse(source)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert not {'fit_candidate', 'ac_predict', 'solve', 'run_trial', 'make_case'} & calls
