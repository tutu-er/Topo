"""Prepare an append-only Pengwah candidate bank from its longer E search.

No identification, parameter fitting, procurement, AC calculation, or new-day
scoring is performed here. The original 16-case input index remains unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ... import io
from ...observations import load, fingerprint
from ...history import freeze_history
from ...identification.algorithm import SearchConfig, _output
from ...identification.competition import competition
from ...identification.optimization import errors
from ..flexibility_sweep_20261113.fitting import pack_score


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked(ref):
    actual = io.ref(ref['path'])
    if actual['sha256'] != ref['sha256']:
        raise ValueError('Locked source changed: '+ref['path'])
    return Path(ref['path'])


def prepare(index_path, diagnostic_path, out):
    index_path, diagnostic_path, out = map(lambda p: Path(p).resolve(),
                                          (index_path, diagnostic_path, out))
    if out.exists():
        raise FileExistsError('Preserve previous enriched-bank preparation')
    source = io.read(index_path)
    matches = [(i, row) for i, row in enumerate(source['inputs'])
               if row['case'] == 'pengwah18' and row['start_date'] == '2024-10-14']
    if len(matches) != 1:
        raise ValueError('Require the one predeclared Pengwah autumn input')
    original_index, item = matches[0]
    if original_index != 7:
        raise ValueError('The paired random-stream index was locked to original input 7')
    checked(source['resource_context'])
    original_path = checked(item['files']['101_native.json'])
    train_path = checked(item['files']['train.npz'])
    validation_path = checked(item['files']['validation.npz'])
    original = io.read(original_path)
    train, validation = load(train_path), load(validation_path)
    original_frozen = freeze_history(original, validation)
    diagnostic = io.read(diagnostic_path)
    protocol_path = diagnostic_path.with_name('protocol.json')
    diagnostic_protocol = io.read(protocol_path)
    if diagnostic_protocol['input']['101_native.json'] != item['files']['101_native.json']:
        raise ValueError('Longer E search did not originate from this frozen native result')
    if diagnostic_protocol['training_fingerprint'] != fingerprint(train):
        raise ValueError('Longer E search used different training observations')
    if diagnostic['budget_seconds'] != 60. or diagnostic['native']['status'] != 2:
        raise ValueError('Require the saved, completed 60-second E-pool diagnostic')
    point = deepcopy(diagnostic['closed_point'])
    original_E = next(e for e in reversed(original['events']) if e['stage'] == 'E')
    if not set(map(tuple, point['supports'])) <= set(map(tuple, original_E['pool'])):
        raise ValueError('Added point contains supports outside the original E pool')
    if content_hash(diagnostic_protocol['original_warm']) != content_hash(original_E['point']):
        raise ValueError('Longer E search did not use the original incumbent')
    if not np.isclose(errors(train, point)['mse'], point['mse'], rtol=1e-12, atol=1e-18):
        raise ValueError('Added training-fit point fails independent observed-loss replay')
    config = SearchConfig(**original['config'])
    old_points = [row['point'] for row in original['frozen_candidates']]
    selected, mae_view, regenerated = _output([*old_points, point], train, validation,
                                             original['reference_mse'], config)
    added_index = len(old_points)
    added = regenerated[-1]
    added['source_stage'] = 'original_E_pool_independent_60s'
    enriched = deepcopy(original)
    enriched['frozen_candidates'].append(added)
    enriched['selected'] = selected
    enriched['selected_mse_view'] = deepcopy(selected)
    enriched['selected_mae_view'] = mae_view
    enriched['handoff']['parameter_rule'] = (
        'Original four frozen candidates retained verbatim; append one training-fitted point '
        'from the original finite E pool after an independent longer search; no refit here')
    enrichment = dict(original_native=io.ref(original_path), source_search=io.ref(diagnostic_path),
        source_search_protocol=io.ref(protocol_path), added_native_index=added_index,
        original_points_preserved=len(old_points), added_point_sha256=content_hash(point),
        added_candidate_origin='saved 60-second finite-pool search closed_point; not a truth-family QP',
        original_search_seconds=original.get('total_seconds'),
        additional_search_budget_seconds=diagnostic['budget_seconds'],
        additional_search_wall_seconds=diagnostic['search_wall_seconds'],
        additional_total_seconds=diagnostic['total_seconds'],
        original_runtime_fields_scope='Unchanged top-level config/timing/MIQP/path/events describe the original 30-second fit only',
        scoring='Original strict validation MSE and original numerical tie; no new-day observations read',
        truth_files_read=False, new_day_values_read=False, parameter_refits=0,
        original_history_reused=True, separate_supplement=True)
    enriched['candidate_enrichment'] = enrichment
    enriched['competition'] = competition(enriched, validation)
    for row in enriched['competition']['models']:
        if row['index'] == added_index:
            row['source_stage'] = added['source_stage']
    if enriched['frozen_candidates'][:added_index] != original['frozen_candidates']:
        raise AssertionError('An original candidate field was changed')
    frozen = freeze_history(enriched, validation)
    np.testing.assert_array_equal(frozen.H[:added_index], original_frozen.H)
    np.testing.assert_array_equal(frozen.sse[:added_index], original_frozen.sse)
    score = pack_score(enriched, train, validation)
    if selected['index'] != original['selected']['index']:
        raise AssertionError('The predeclared natural wrong-history condition changed')
    out.mkdir(parents=True)
    shutil.copyfile(original_path, out/'original_101_native.json')
    shutil.copyfile(diagnostic_path, out/'source_search_60s.json')
    shutil.copyfile(Path(__file__), out/'enriched_pool.py')
    io.write(out/'101_native.json', enriched)
    io.write(out/'score.json', score)
    entry = deepcopy(item)
    entry['files']['101_native.json'] = io.ref(out/'101_native.json')
    entry['original_input_index'] = original_index
    entry['candidate_enrichment'] = dict(native=io.ref(out/'101_native.json'),
                                         score=io.ref(out/'score.json'), **enrichment)
    output_index = deepcopy(source)
    output_index['inputs'] = [entry]
    output_index['input_count'] = 1
    output_index['original_source_index'] = io.ref(index_path)
    output_index['supplement_scope'] = (
        'One already-used Pengwah autumn history and physical day; only its candidate pool is enlarged. '
        'Separate 4-forecast by 8-execution diagnostic; not an independent new history or replacement of the main 16.')
    io.write(out/'qualified_enriched_input.json', output_index)
    receipt = dict(status='complete', prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_index=io.ref(index_path), source_search=io.ref(diagnostic_path),
        native=io.ref(out/'101_native.json'), original_native_copy=io.ref(out/'original_101_native.json'),
        score=io.ref(out/'score.json'), output_index=io.ref(out/'qualified_enriched_input.json'),
        source=io.ref(__file__), original_input_index=original_index,
        old_points=len(old_points), new_points=len(enriched['frozen_candidates']),
        old_distinct_topologies=len(set(original_frozen.topology_ids)),
        new_distinct_topologies=len(set(frozen.topology_ids)),
        original_candidate_prefix_exact=True, original_matrix_prefix_bitwise_equal=True,
        original_history_sse_prefix_bitwise_equal=True, original_selected_index=original['selected']['index'],
        enriched_selected_index=selected['index'], added_index=added_index,
        old_validation_mse=original['selected']['validation_mse'],
        added_validation_mse=added['validation_mse'],
        validation_tie_physical=original['reference_mse']*config.validation_tie,
        score_independent_verification=score['independent_verification'],
        no_truth_or_new_day_scoring=True, procurement_started=False, original_files_unchanged=True)
    # Recheck every original file that this preparation actually parsed.
    checked(item['files']['101_native.json']); checked(item['files']['train.npz'])
    checked(item['files']['validation.npz']); checked(source['resource_context'])
    io.write(out/'receipt.json', receipt)
    io.write(out/'complete.json', dict(receipt=io.ref(out/'receipt.json')))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.index, args.diagnostic, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'old_points', 'new_points',
                                             'enriched_selected_index', 'added_index')}, ensure_ascii=False))
