"""Replay saved actual AC states and operations from a compact delivery.

This is a scope-limited offline audit. Predicted-fit residuals, nonbest QP
states and global MILP witnesses are omitted by the exporter and are not
reconstructed here. No fitting, selection, optimization or raw-local path is
used. Run as a module from a checkout containing the 72 frozen sources.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import ev_compact_review as full
from . import ev_compact_source_restore as source_restore
from . import ev_original_review as old
from . import ev_policy_calibration as cal
from .information_replay import _meter, _obs
from .likelihood import RadialCandidate, concat_observations

HERE = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = source_restore.DEFAULT_DIRECTORY
SCHEMA = 'ev_compact_slim_trace_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _decode(data, *, compressed=False):
    return json.loads(gzip.decompress(data) if compressed else data,
                      object_pairs_hook=_unique_object)


def _sha(value, label):
    old.require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
                label + ' requires lowercase SHA-256')


def validate_manifest(manifest, freeze):
    old.require(manifest.get('artifact_schema') == SCHEMA, 'delivery manifest schema')
    old.require(manifest.get('complete_raw_replay_replacement') is False,
                'slim manifest incorrectly claims complete raw replay')
    planned = freeze['seeds']
    old.require(all(type(seed) is int for seed in planned) and len(set(planned)) == len(planned),
                'duplicate/noninteger frozen historical seeds')
    old.require(manifest['phase'] == freeze['phase'] and manifest['planned_seeds'] == sorted(planned),
                'manifest phase/planned seed mismatch')
    phase = freeze['phase']; config = freeze['config']
    old.require(phase in ('development', 'calibration', 'evaluation')
                and sorted(planned) == sorted(config[phase + '_seeds']), 'frozen phase seed plan')
    seed_sets = [set(config[key + '_seeds']) for key in ('development', 'calibration', 'evaluation')]
    old.require(all(seed_sets[i].isdisjoint(seed_sets[j]) for i in range(3) for j in range(i)),
                'development/calibration/evaluation seed overlap')
    rows = manifest['trial_files']
    names = set(); seeds = set()
    for row in rows:
        seed = row['seed']; name = row['slim_file']
        old.require(type(seed) is int and seed in planned and seed not in seeds,
                    'unplanned/duplicate manifest historical sample')
        old.require(name == f'trial_{seed}.slim.json.gz' and name not in names,
                    'noncanonical/duplicate slim filename')
        for key in ('raw_sha256', 'slim_sha256'):
            _sha(row[key], 'manifest ' + key)
        old.require(row.get('raw_unchanged') is True and isinstance(row.get('raw_path'), str),
                    'manifest raw provenance declaration')
        old.require(type(row['raw_bytes']) is int and row['raw_bytes'] > 0
                    and type(row['slim_bytes']) is int and row['slim_bytes'] > 0,
                    'manifest byte sizes')
        names.add(name); seeds.add(seed)
    missing = sorted(set(planned) - seeds)
    old.require(manifest['exported_seeds'] == sorted(seeds) and manifest['missing_seeds'] == missing
                and manifest['partial'] == bool(missing), 'manifest exported/missing seed bookkeeping')
    old.require(manifest['raw_bytes'] == sum(row['raw_bytes'] for row in rows)
                and manifest['slim_bytes'] == sum(row['slim_bytes'] for row in rows),
                'manifest total byte sizes')
    sources = manifest['source_files']
    old.require(len(sources) == 2 and {r['name'] for r in sources} ==
                {'freeze.json', 'source_snapshot.json.gz'}, 'delivery frozen file coverage')
    for row in sources:
        _sha(row['sha256'], 'frozen delivery file')
        old.require(type(row['bytes']) is int and row['bytes'] > 0
                    and row.get('byte_identical') is True, 'frozen file byte provenance')
    _sha(manifest['exporter_sha256'], 'manifest exporter')
    return {row['slim_file']: row for row in rows}


def validate_trial_metadata(trial, manifest, row, freeze):
    old.require(trial.get('artifact_schema') == SCHEMA, 'slim trial schema')
    old.require(cal.freeze_identity(trial) == cal.freeze_identity(freeze), 'slim trial/freeze policy identity')
    old.require(trial['seed'] == row['seed'] and trial['phase'] == freeze['phase']
                and trial['domain'] == freeze['domain'], 'slim phase/seed/domain')
    delivery = trial['delivery']
    old.require(delivery.get('complete_raw_replay_replacement') is False
                and delivery.get('source_raw_is_immutable') is True, 'slim replay/provenance scope')
    old.require(delivery['source_raw_sha256'] == row['raw_sha256']
                and delivery['source_raw_bytes'] == row['raw_bytes']
                and delivery['source_raw_path'] == row['raw_path'], 'slim raw provenance differs from manifest')
    old.require(delivery['exporter_sha256'] == manifest['exporter_sha256']
                and delivery['exporter_dependencies_sha256'] == manifest['exporter_dependencies_sha256'],
                'slim exporter provenance differs from manifest')
    old.require(delivery['policy_config_ref'] == 'freeze.json', 'slim policy configuration reference')
    methods = list(full.method_names(trial))
    saved = [r['method'] for r in trial['rows']]
    old.require(delivery['declared_methods'] == methods and delivery['saved_method_rows'] == saved
                and delivery['missing_planned_method_rows'] == sorted(set(methods) - set(saved)),
                'slim method/failure denominator provenance')
    old.require(row['method_rows'] == len(saved) and row['declared_methods'] == methods,
                'manifest method denominator')


def _summary_loglik(fit, history, meter, horizon):
    """Only check saved loss -> Gaussian normalization, not omitted residuals."""
    if fit.get('loss') is None:
        old.require(not fit['success'], 'successful predicted summary has no SSE')
        return
    value = fit['loss']
    old.require(np.isfinite(value) and value >= 0, 'predicted summary finite nonnegative SSE')
    if fit.get('half_loss') is not None:
        old.close(fit['half_loss'], .5 * value, 'predicted summary half SSE')
    if fit.get('loglik') is not None:
        normalizer = 0.
        for channel in old.CHANNELS:
            observed = getattr(history, channel)
            if observed is None:
                continue
            shape = (history.t + horizon,) + old.arr(observed).shape[1:]
            sigma = np.broadcast_to(getattr(meter, channel + '_sigma'), shape)
            old.require(np.isfinite(sigma).all() and np.all(sigma > 0), 'predicted summary Gaussian sigma')
            normalizer += np.log(2 * np.pi * sigma**2).sum()
        old.close(fit['loglik'], -.5 * (value + normalizer), 'predicted summary loglik normalization')
    old.require(not fit.get('global_MLE_certified', False), 'predicted local fit labelled global')


def audit_prediction_summary(saved, candidates, hfits, history, jm, config, contexts):
    """Check saved losses, tasks and weights; never reconstruct predicted AC."""
    old.require(not saved['uses_truth'] and not saved['uses_future_noise'], 'prediction information declaration')
    old.require(saved.get('source_interface', 1) == 1 and saved.get('source_interface_shared', True)
                and not saved.get('alternative_futures_stacked', False), 'prediction source/future declarations')
    power = old.arr(saved['power_kw']); T = config['horizon']; n = history.n
    old.require(power.shape == (T, n), 'prediction command dimensions')
    names = list(candidates); pairs = [(a, b) for a in names for b in names if a != b]
    lookup = {}; seen = set(); summary_states = 0; failed_cells = 0
    for record in saved['records']:
        cid = record.get('ensemble_id'); a = record['a']; cell = (cid, a)
        old.require(cell not in seen and cid in contexts and a in candidates, 'prediction summary cell coverage')
        seen.add(cell)
        failed_cells += not bool(record['success'])
        old.require(record.get('delivery_prediction_fits_are_summaries') is True
                    and 'predicted_observations' not in record, 'prediction summary omission scope')
        context = contexts[cid]; background = context['forecast_background'][a]
        forecast = record['forecast_state']
        for key in ('q', 'source_p', 'source_q', 'root_v'):
            old.close(forecast[key], background[key], 'saved H-only forecast:' + key, atol=1e-12)
        old.close(forecast['p'], old.arr(background['p']) + old.arr(context['control_gain']) * power / config['base_kw'],
                  'saved forecast total P once', atol=1e-12)
        fits = record.get('fits') or []; failures = record.get('failures') or {}
        fitnames = [fit['name'] for fit in fits]
        old.require(len(set(fitnames)) == len(fitnames) and set(fitnames) <= set(names),
                    'predicted fit summary candidate coverage')
        global_failure = '__forecast_or_joint__' in failures
        old.require(set(failures) <= set(names) | {'__forecast_or_joint__'}, 'unknown prediction failure key')
        if global_failure:
            old.require(not record['success'], 'forecast exception labelled successful')
        else:
            old.require(set(fitnames) | set(failures) == set(names), 'predicted fit summary failure denominator')
        if record['success']:
            old.require(set(fitnames) == set(names) and not failures, 'complete predicted summary candidate coverage')
        for fit in fits:
            _summary_loglik(fit, history, jm, T); summary_states += 1
        lookup[cell] = {fit['name']: fit for fit in fits}
        if 'joint_observation_count' in record:
            old.require(record['joint_observation_count'] == history.t + T
                        and record['history_observation_count'] == history.t
                        and record['future_observation_count'] == T, 'predicted summary H+one future count')
    old.require(seen == {(cid, a) for cid in contexts for a in names}, 'missing prediction summary cell')
    seen_pairs = set(); scores = {key: [] for key in ('added', 'joint', 'self_relative')}; weights = []
    for pair in saved['pairs']:
        cid = pair.get('ensemble_id'); a = pair['a']; b = pair['b']; cell = (cid, a, b)
        old.require(cell not in seen_pairs and cid in contexts and (a, b) in pairs, 'prediction summary pair coverage')
        seen_pairs.add(cell); weight = contexts[cid]['weight'] / len(pairs)
        if 'weight' in pair:
            old.close(pair['weight'], weight, 'prediction summary task weight', atol=1e-12)
        if not pair['success']:
            continue
        own = lookup[cid, a][a]; other = lookup[cid, a][b]
        old.require(own['success'] and other['success'] and b in hfits and a in hfits,
                    'successful prediction task lacks successful fit')
        expected = dict(added=.5 * (other['loss'] - hfits[b]['loss']),
                        joint=.5 * (other['loss'] - hfits[a]['loss']),
                        self_relative=.5 * (other['loss'] - own['loss']))
        for key, value in expected.items():
            old.close(pair[key], value, 'predicted summary task:' + key); scores[key].append(value)
        weights.append(weight)
    old.require(seen_pairs == {(cid, a, b) for cid in contexts for a, b in pairs},
                'missing directed prediction summary task')
    if saved['success']:
        old.require(len(weights) == len(seen_pairs), 'successful prediction summary incomplete')
        for key, values in scores.items():
            if 'mean_' + key + '_score' in saved:
                old.close(saved['mean_' + key + '_score'], old.arr(weights) @ old.arr(values), 'predicted summary mean:' + key)
            if 'worst_' + key + '_score' in saved:
                old.close(saved['worst_' + key + '_score'], min(values), 'predicted summary worst:' + key)
        old.close(saved['score'], old.arr(weights) @ old.arr(scores['added']), 'saved prediction score')
    return dict(predicted_fit_summaries_checked=summary_states, directed_tasks=len(seen_pairs),
                failed_reference_cells=failed_cells,
                failed_directed_tasks=sum(not bool(pair['success']) for pair in saved['pairs']),
                predicted_fit_AC_states_replayed=0, success=bool(saved['success']))


def audit_operations_and_selection(trial, cache):
    selection = trial['selection']; rows = {row['method']: row for row in trial['rows']}
    opconfig = full.operation_config(trial); operations = {}; inherited = []; replay_count = 0
    constant = selection.get('constant_operation') or {}
    if constant.get('feasible'):
        operations['constant'] = old.replay_operation_record(constant, full.service_config(opconfig, 'charge_fixed'))
        replay_count += 1
        if rows.get('constant', {}).get('power_kw') is not None:
            old.close(rows['constant']['power_kw'], constant['u_kw'], 'constant actual curve')
    for name, record in selection['methods'].items():
        cfg = full.service_config(opconfig, name); operation = record.get('operation')
        if operation and operation.get('feasible'):
            operations[name] = old.replay_operation_record(operation, cfg); replay_count += 1
            if rows.get(name, {}).get('power_kw') is not None:
                old.close(rows[name]['power_kw'], operation['u_kw'], 'selected actual curve')
        if name not in full.DESIGNS:
            continue
        old.require(not record['random_used'], 'random used by primary method')
        cap = record.get('cap'); checked = []; initial_keys = set()
        for initial in record['initial_candidates']:
            old.require('random' not in initial['origin'], 'random initial candidate')
            if not initial['operation'].get('feasible'):
                old.require(initial['AC_score'] is None, 'infeasible initial scored'); continue
            old.replay_operation_record(initial['operation'], cfg); replay_count += 1
            key = full.key(initial['operation']['u_kw']); initial_keys.add(key)
            old.require(key in cache, 'initial plan has no prediction summary')
            value = full.score(cache[key], cap)
            if value is None:
                old.require(initial['AC_score'] is None, 'failed initial prediction scored')
            else:
                old.close(initial['AC_score'], value, 'initial prediction summary score')
                checked.append((initial['origin'], key, value))
        if checked:
            old.close(record['initial_best_score'], max(row[2] for row in checked), 'best initial summary score')
        if record.get('raw_operation', {}).get('feasible'):
            old.replay_operation_record(record['raw_operation'], cfg); replay_count += 1
            rawkey = full.key(record['raw_operation']['u_kw']); prediction = record['raw_prediction']
            old.require(rawkey == full.key(prediction['power_kw']) and rawkey in cache
                        and old.identity(prediction) == old.identity(cache[rawkey]), 'raw prediction summary cache')
            value = full.score(prediction, cap)
            if record.get('raw_proxy_gate') and value is not None:
                checked.append(('MILP', rawkey, value))
        if operation and operation.get('feasible') and record.get('selected_score') is not None:
            finalkey = full.key(operation['u_kw']); old.require(finalkey in cache, 'selected prediction summary absent')
            value = full.score(cache[finalkey], cap); old.close(record['selected_score'], value, 'selected summary score')
            old.require(checked and value >= max(row[2] for row in checked) - 1e-8, 'selection loses checked score')
            old.require(any(origin == record['selected_origin'] and key == finalkey for origin, key, _ in checked),
                        'selected source/curve mismatch')
            old.require(record['guard']['accepted'] == (record['selected_origin'] == 'MILP'), 'selection guard/source mismatch')
            for parent in full.INHERITS[name]:
                parentop = selection['methods'].get(parent, {}).get('operation', {})
                if parentop.get('feasible'):
                    parentkey = full.key(parentop['u_kw']); old.require(parentkey in initial_keys, 'parent plan omitted')
                    parent_score = full.score(cache[parentkey], cap)
                    if parent_score is not None:
                        old.require(value >= parent_score - 1e-8, 'inheritance summary score regression')
                        inherited.append(dict(method=name, parent=parent, final=value, parent_score=parent_score))
    siblings = ('refined_v2g_mobile', 'capped_v2g_mobile')
    if all(name in selection['methods'] for name in siblings):
        sets = [{full.key(row['operation']['u_kw']) for row in selection['methods'][name]['initial_candidates']
                 if row['operation'].get('feasible')} for name in siblings]
        old.require(sets[0] == sets[1], 'mean/cap sibling initial plans differ')
    return dict(final_operations=operations, operation_ledgers_replayed=replay_count,
                inheritance_checks=inherited, proxy_gate_decision_independently_replayed=False)


def audit_trial(trial):
    old.require(trial.get('artifact_schema') == SCHEMA, 'slim trial schema')
    config = full.physical_config(trial); history = _obs(trial['history']); selection = trial['selection']
    old.require(trial.get('source_interface', 1) == 1, 'actual source interface')
    full.operation_config(trial)
    hm, fm, jm = (_meter(trial['measurement_models'][key]) for key in ('hm', 'fm', 'jm'))
    old.require(not selection['uses_truth'] and not selection['uses_future_noise']
                and not selection['random_used_by_primary'], 'selector information declarations')
    names = [row['name'] for row in trial['proposal']['bank']]
    old.require(len(names) == 4 and len(set(names)) == 4 and not trial['proposal']['uses_truth'], 'candidate bank')
    candidates = {row['name']: RadialCandidate.from_supports(row['name'], history.n, row['supports'])
                  for row in trial['proposal']['bank']}
    hfits = {row['name']: row for row in trial['historical_fits']}
    historical = old.replay_fits(trial['historical_fits'], candidates, history, hm, trial['historical_failures'],
                                complete=not trial['historical_failures'], config=config, domain=trial['domain'])
    view = dict(trial, config=config); assets = old.asset_unit_audit(view)
    truth = trial['truth']; H = history.t; n = history.n; T = config['horizon']
    tc = RadialCandidate.from_supports('independent_truth', n, truth['family'])
    trueH = old.forward_physics(tc, truth['r'], truth['x'], old.arr(truth['p'])[:H], old.arr(truth['q'])[:H],
                               old.arr(truth['source_p'])[:H], old.arr(truth['source_q'])[:H], old.arr(truth['root_v'])[:H])
    for channel in old.CHANNELS:
        if getattr(history, channel) is not None:
            old.close(getattr(history, channel), trueH[channel] + np.asarray(getattr(hm, channel + '_sigma'))
                      * old.arr(truth['white'][channel])[:H]
                      + (getattr(hm, 'bias_' + channel) if channel in ('p', 'q', 'v') else 0),
                      'true historical measurement:' + channel, atol=2e-10)
    repeats = old.arr(config['public_voltage_repeat_counts'])[:, None]
    old.close(fm.v_sigma, np.broadcast_to(old.arr(hm.v_sigma) / np.sqrt(repeats), (T, n)), 'public V precision')
    for channel in ('p', 'q', 'master_p', 'master_q'):
        target = old.arr(getattr(fm, channel + '_sigma'))
        old.close(target, np.broadcast_to(old.arr(getattr(hm, channel + '_sigma'))
                  / np.sqrt(config['other_future_repeat_count']), target.shape), 'public future precision:' + channel)
    for channel in old.CHANNELS:
        if getattr(history, channel) is None:
            continue
        shape = old.arr(getattr(history, channel)).shape; future_shape = (T,) + shape[1:]
        old.close(getattr(jm, channel + '_sigma'), np.concatenate((np.broadcast_to(getattr(hm, channel + '_sigma'), shape),
                  np.broadcast_to(getattr(fm, channel + '_sigma'), future_shape))), 'H plus one future meter:' + channel)
    rows = {row['method']: row for row in trial['rows']}; expected = set(full.method_names(trial))
    old.require(len(rows) == len(trial['rows']), 'duplicate method row')
    if not selection.get('failures', {}):
        old.require(set(rows) == expected, 'complete method denominator')
    else:
        old.require({'none', 'constant'} <= set(rows) <= expected, 'failed method denominator')
    cache = {}; prediction_audits = []; explanation_count = 0
    if selection.get('all_prediction_records'):
        contexts, _ = full._prediction_contexts(selection, history, hm, hfits, config)
        if selection['prediction'] == 'ensemble':
            for a, bank in selection['context']['history_explanations'].items():
                for entry in bank:
                    replay = old.replay_fits([entry['fit']], {a: candidates[a]}, history, hm, {},
                                            complete=True, config=config, domain=trial['domain'])
                    explanation_count += replay['replayed_states']
                    old.require(entry['fit']['loss'] <= hfits[a]['loss']
                                + 2 * trial['config']['ensemble']['history_gate_half_sse'] + 1e-7,
                                'history explanation cost gate')
        for saved in selection['all_prediction_records']:
            key = full.key(saved['power_kw']); old.require(key not in cache, 'duplicate prediction summary cache')
            cache[key] = saved
            prediction_audits.append(audit_prediction_summary(saved, candidates, hfits, history, jm, config, contexts))
        old.require(selection['unique_predicted_actions'] == len(cache), 'prediction summary cache count')
        old.require(selection['predicted_fit_calls'] == sum(record.get('fit_calls', 0) for record in cache.values()),
                    'prediction summary fit call count')
        old.require(selection['predicted_ac_calls'] == sum(record.get('ac_calls', 0) for record in cache.values()),
                    'prediction summary AC call count')
        for saved in selection['predicted_AC'].values():
            key = full.key(saved['power_kw'])
            old.require(key in cache and old.identity(saved) == old.identity(cache[key]), 'named prediction summary cache')
    operation_audit = audit_operations_and_selection(trial, cache)
    family = tuple(map(tuple, truth['family']))
    true_names = [a for a, c in candidates.items() if tuple(s for s in c.supports if 1 < len(s) < n) == family]
    actual_states = 0; mean_states = 0; physical_states = 0; actual_cache = {}; mean_cache = {}; method_rows = []
    actual_failed = 0; mean_failed = 0; actual_unavailable = 0; mean_unavailable = 0
    for name, row in rows.items():
        old.require(row['candidate_miss'] == (len(true_names) != 1), 'candidate miss label')
        if row.get('physical') is not None:
            old.future_audit(view, row, fm); physical_states += 1
        if row.get('observations') is not None and 'fits' in row:
            obs = _obs(row['observations'])
            replay = old.replay_fits(row['fits'], candidates, concat_observations(history, obs), jm, row['failures'],
                                    complete=row['status'] == 'complete', config=config, domain=trial['domain'])
            actual_states += replay['replayed_states']; key = old.control_key(row['power_kw'])
            actual_failed += replay['failed_states']
            actual_unavailable += replay['saved_states'] - replay['replayed_states']
            if row['cache_hit']:
                old.require(key in actual_cache and row['fits'] == actual_cache[key]['fits'], 'actual future fit cache')
            else:
                old.require(key not in actual_cache, 'duplicate actual action refitted'); actual_cache[key] = row
            if row['status'] == 'complete':
                losses = {fit['name']: fit['loss'] for fit in row['fits']}
                old.require(losses == row['losses'], 'actual saved SSE')
                ranking = sorted(losses, key=lambda a: (losses[a], a))
                old.require(row['winner'] == ranking[0] and row['candidate_ranking'] == ranking, 'actual candidate ranking')
                for a in candidates:
                    for b in candidates:
                        if a != b:
                            old.close(row['pair_signed_loss_differences'][a + '__to__' + b],
                                      .5 * (losses[b] - losses[a]), 'actual likelihood pair gap')
                if true_names:
                    true = true_names[0]; margin = min(value for a, value in losses.items() if a != true) - losses[true]
                    old.close(row['margin'], margin, 'actual SSE margin'); old.close(row['loglik_margin'], margin / 2, 'actual LL margin')
        mean = row.get('noiseless_AC_refit')
        if mean:
            obs = _obs(mean['future_mean_observations'])
            for channel in old.CHANNELS:
                if getattr(obs, channel) is not None:
                    old.close(getattr(obs, channel), old.arr(row['physical'][channel])
                              + (getattr(fm, 'bias_' + channel) if channel in ('p', 'q', 'v') else 0),
                              'actual noiseless mean:' + channel, atol=2e-10)
            replay = old.replay_fits(mean['fits'], candidates, concat_observations(history, obs), jm, mean['failures'],
                                    complete=mean['all_candidates'], config=config, domain=trial['domain'])
            mean_states += replay['replayed_states']
            mean_failed += replay['failed_states']
            mean_unavailable += replay['saved_states'] - replay['replayed_states']
            for detail in mean['candidate_distances']:
                old.noiseless_distance_audit(detail, mean['fits'], hfits, obs, fm, H)
            old.require([detail['candidate'] for detail in mean['candidate_distances']] == list(candidates),
                        'noiseless candidate/failure denominator')
            key = old.control_key(row['power_kw'])
            if key in mean_cache:
                old.require(mean['fits'] == mean_cache[key]['fits'], 'noiseless actual cache')
            else:
                mean_cache[key] = mean
        method_rows.append(dict(method=name, status=row['status'], candidate_miss=row['candidate_miss'],
                                noisy_candidate_failures=row.get('failures', {}),
                                noiseless_candidate_failures=(mean or {}).get('failures', {})))
    count = historical['replayed_states'] + explanation_count + actual_states + mean_states
    return dict(seed=trial['seed'], success=True, independent_histories=1, methods=method_rows,
                planned_methods=len(expected), saved_methods=len(rows), missing_methods=sorted(expected - set(rows)),
                historical_fit_states_replayed=historical['replayed_states'],
                historical_explanation_states_replayed=explanation_count,
                actual_noisy_fit_states_replayed=actual_states, actual_noiseless_fit_states_replayed=mean_states,
                declared_failed_saved_fit_states=historical['failed_states'] + actual_failed + mean_failed,
                unavailable_failed_saved_fit_states=historical['saved_states'] - historical['replayed_states']
                                                    + actual_unavailable + mean_unavailable,
                saved_fit_states_replayed=count, physical_future_states_replayed=physical_states,
                actual_unique_noisy_actions=len(actual_cache), actual_unique_noiseless_actions=len(mean_cache),
                asset_unit_audit=assets, prediction_summary_audits=prediction_audits,
                operation_audit=operation_audit,
                coverage=dict(historical_saved_AC_states=True, actual_noisy_and_noiseless_saved_AC_states=True,
                              actual_future_physics_and_paired_noise=True, fit_SSE_loglik_and_units=True,
                              saved_initial_raw_final_operation_ledgers=True, selection_origin_and_inherited_summary_scores=True,
                              predicted_fit_loss_summary_arithmetic=True, predicted_fit_AC_residuals=False,
                              nonbest_QP_states=False, best_QP_states=False, same_witness_AC_gates=False,
                              global_MILP_witness_certificates=False),
                skipped=['predicted AC fit state/residual replay: omitted from slim',
                         'nonbest QP state/residual replay: omitted from slim',
                         'best QP state/matrix replay: outside this actual-AC audit',
                         'same-witness gate residual replay: vectors omitted from slim',
                         'global MILP witness certificates: witness arrays omitted from slim'],
                new_fit_calls=0, selector_calls=0, optimizer_calls=0,
                scope='scope-limited slim replay; not a complete raw-trial replay')


def review_directory(directory=DEFAULT_DIRECTORY, output=None):
    directory = Path(directory).resolve()
    evidence = {name: (directory / name).read_bytes() for name in
                ('delivery_manifest.json', 'freeze.json', 'source_snapshot.json.gz')}
    manifest = _decode(evidence['delivery_manifest.json']); freeze = _decode(evidence['freeze.json'])
    identity = cal.freeze_identity(freeze); rows = validate_manifest(manifest, freeze)
    actual_files = {path.name for path in directory.glob('trial_*.json.gz')}
    old.require(actual_files == set(rows), 'delivery trial files do not exactly match manifest')
    for row in manifest['source_files']:
        raw = evidence[row['name']]
        old.require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'delivery frozen file hash/size')
    sources = source_restore.audit_directory(directory)
    old.require(sources['success'] and sources['checked_count'] == 72
                and sources['policy_identity'] == identity, '72 frozen source bytes/policy mismatch')
    audits = []; errors = []; hashes = {}; raw_provenance = []
    for name, row in sorted(rows.items()):
        path = directory / name; data = path.read_bytes(); before = digest(data)
        old.require(len(data) == row['slim_bytes'] and before == row['slim_sha256'], 'slim compressed byte hash/size')
        trial = _decode(data, compressed=True)
        try:
            validate_trial_metadata(trial, manifest, row, freeze)
            audits.append(audit_trial(trial))
        except (ValueError, KeyError, TypeError, IndexError, RuntimeError, FloatingPointError) as error:
            errors.append(dict(seed=row['seed'], error=type(error).__name__ + ':' + str(error)))
        old.require(digest(path.read_bytes()) == before, 'slim changed during replay')
        hashes[name] = before
        raw_provenance.append(dict(seed=row['seed'], sha256=row['raw_sha256'], bytes=row['raw_bytes'],
                                   path=row['raw_path'], independently_read=False))
        del trial, data
    for name, data in evidence.items():
        old.require((directory / name).read_bytes() == data, 'delivery metadata changed during replay: ' + name)
    totals = {key: sum(trial[key] for trial in audits) for key in
              ('historical_fit_states_replayed', 'historical_explanation_states_replayed',
               'actual_noisy_fit_states_replayed', 'actual_noiseless_fit_states_replayed',
               'saved_fit_states_replayed', 'physical_future_states_replayed',
               'declared_failed_saved_fit_states', 'unavailable_failed_saved_fit_states',
               'actual_unique_noisy_actions', 'actual_unique_noiseless_actions', 'planned_methods', 'saved_methods')}
    totals['operation_ledgers_replayed'] = sum(trial['operation_audit']['operation_ledgers_replayed'] for trial in audits)
    totals['inheritance_score_checks'] = sum(len(trial['operation_audit']['inheritance_checks']) for trial in audits)
    totals['predicted_fit_summaries_checked'] = sum(audit['predicted_fit_summaries_checked'] for trial in audits
                                                   for audit in trial['prediction_summary_audits'])
    totals['prediction_directed_tasks_checked'] = sum(audit['directed_tasks'] for trial in audits
                                                    for audit in trial['prediction_summary_audits'])
    totals['failed_prediction_reference_cells'] = sum(audit['failed_reference_cells'] for trial in audits
                                                      for audit in trial['prediction_summary_audits'])
    totals['failed_prediction_directed_tasks'] = sum(audit['failed_directed_tasks'] for trial in audits
                                                     for audit in trial['prediction_summary_audits'])
    result = dict(success=not errors, artifact_schema='ev_compact_slim_review_v1',
                  scope='saved actual AC/operation replay plus prediction-loss-summary arithmetic; incomplete raw scope',
                  complete_raw_replay=False, policy_id=identity['policy_id'], phase=freeze['phase'],
                  planned_independent_histories=len(freeze['seeds']), available_independent_histories=len(rows),
                  audited_histories=len(audits), seeds=sorted(row['seed'] for row in rows.values()),
                  missing_histories=manifest['missing_seeds'], partial=manifest['partial'],
                  trial_sha256=hashes, freeze_sha256=digest(evidence['freeze.json']),
                  delivery_manifest_sha256=digest(evidence['delivery_manifest.json']),
                  source_snapshot_sha256=digest(evidence['source_snapshot.json.gz']), source_audit=sources,
                  original_raw_provenance=raw_provenance, original_raw_content_independently_read=False,
                  trials=audits, totals=totals, review_errors=errors,
                  new_fit_calls=0, selector_calls=0, optimizer_calls=0,
                  reviewer=dict(name=Path(__file__).name, sha256=digest(Path(__file__).read_bytes()),
                                bytes_base64=base64.b64encode(Path(__file__).read_bytes()).decode('ascii'),
                                science_source=False,
                                dependencies_sha256={Path(module.__file__).name: digest(Path(module.__file__).read_bytes())
                                                     for module in (full, old, source_restore, cal)}))
    if output is not None:
        with Path(output).open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = review_directory(args.directory, args.output)
    print(json.dumps({key: result[key] for key in
                      ('success', 'scope', 'complete_raw_replay', 'audited_histories', 'missing_histories', 'totals', 'review_errors')}))
    if not result['success']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
