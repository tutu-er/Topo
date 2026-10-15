"""Independent paired-rich-training evidence replay, outside the freeze.

The training index space is a new rich96 stream; validation and future retain
their original weak-study index space.  Training length is never relabelled as
eight.  No fitter, source optimizer, selector, QP/MILP solve, or case generator
is called.  Saved-state AC and operation checks reuse the independent v2
auditor.  A matched frozen forward AC call checks the exact rich96 observation
hash; a separate branch sweep checks the corresponding physical outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from . import likelihood as lk
from . import ev_fixed_train_trial_review as old
from .information_fit_recovery import read_json, write_json, exact_candidate
from .information_replay import _obs, _meter
from .ev_original_review import require, close, forward_physics, replay_operation_record
from .ev_fixed_train_design import build_fixed_contexts

HERE = Path(__file__).resolve().parent
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def _exact(actual, expected, name):
    require(plain(actual) == plain(expected), name)


def _holdout(record):
    truth = record['truth']
    return dict(validation=record['validation'], validation_meter=record['validation_meter'],
        future_p=np.asarray(truth['p'])[12:].tolist(), future_q=np.asarray(truth['q'])[12:].tolist(),
        future_source_p=np.asarray(truth['source_p'])[12:].tolist(),
        future_source_q=np.asarray(truth['source_q'])[12:].tolist(),
        future_root_v=np.asarray(truth['root_v'])[12:].tolist(), future_white=record['future_white'],
        future_meter=record['future_meter'], r=truth['r'], x=truth['x'], assets=truth['assets'],
        device=record['config']['physics']['device'])


def replay_rich_inputs(record, weak):
    """Reconstruct declared rich RNG stream/hash without a case generator."""
    length = record['training_length']; cfg = record['config']; generation = record['training_generation']
    rich = generation['config']; count = rich['maximum_sequence_length']; base_kw = cfg['physics']['base_kw']
    require(length in rich['lengths'] and count == 96 and length in (8, 32, 96), 'rich length domain')
    require(record['training_indices'] == list(range(length))
            and generation['training_prefix_indices'] == list(range(length)), 'rich training prefix indices')
    require(record['training_index_space'] == 'new rich96 stream prefix'
            and record['validation_indices'] == [8, 9, 10, 11], 'separate rich and original index spaces')
    require(cfg['training_points'] == length and cfg['validation_points'] == 4 and cfg['horizon'] == 6,
            'rich train/validation/future dimensions')
    require(generation['training_length'] == length and not generation['training_EV_applied']
            and generation['validation_future_unchanged'] and not generation['generator_reads_future_for_training']
            and not generation['truth_used_by_selector'], 'rich information/generation scope')
    _exact(record['seeds'], weak['seeds'], 'original independent seed streams changed')
    _exact(record['truth'], weak['truth'], 'paired assets or original latent paths changed')
    _exact(record['config']['physics'], weak['config']['physics'], 'physical/device comparison conditions changed')
    holdout = _holdout(record); _exact(holdout, _holdout(weak), 'original validation/future/asset/device changed')
    require(identity(holdout) == generation['paired_validation_future_asset_sha256'], 'paired holdout/asset hash')
    seeds = {k: (int(weak['seeds'][k])+rich['seed_offset']) % rich['seed_modulus']
             for k in ('background', 'source', 'noise')}
    _exact(seeds, generation['rich_seeds'], 'new rich random stream seeds')
    _exact(generation['original_seeds'], weak['seeds'], 'generation original seed map')
    rng = np.random.default_rng(seeds['background'])
    p = (rich['P_mean_kw']+rng.uniform(-rich['P_independent_halfwidth_kw'], rich['P_independent_halfwidth_kw'], (count, 3)))/base_kw
    q = (rich['Q_mean_kvar']+rng.uniform(-rich['Q_independent_halfwidth_kvar'], rich['Q_independent_halfwidth_kvar'], (count, 3)))/base_kw
    sr = np.random.default_rng(seeds['source'])
    sp = (rich['source_P_mean_kw']+sr.uniform(-rich['source_P_halfwidth_kw'], rich['source_P_halfwidth_kw'], count))/base_kw
    sq = (rich['source_Q_mean_kvar']+sr.uniform(-rich['source_Q_halfwidth_kvar'], rich['source_Q_halfwidth_kvar'], count))/base_kw
    root = np.full(count, rich['training_root_v'])
    truth = dict(p=p, q=q, source_p=sp, source_q=sq, root_v=root)
    require(identity(plain(truth)) == generation['rich96_physical_inputs_sha256'], 'rich96 physical input hash')
    nr = np.random.default_rng(seeds['noise'])
    white = {k: nr.standard_normal((count, 3) if k in ('p', 'q', 'v') else count) for k in CHANNELS}
    require(identity(plain(white)) == generation['rich96_white_sha256'], 'rich96 noise hash')
    for k in truth: _exact(np.asarray(record['training_truth'][k]), truth[k][:length], 'training true rich prefix '+k)
    for k in CHANNELS: _exact(np.asarray(record['training_white'][k]), white[k][:length], 'training white rich prefix '+k)
    require(p.min()*base_kw >= rich['P_mean_kw']-rich['P_independent_halfwidth_kw']
            and p.max()*base_kw <= rich['P_mean_kw']+rich['P_independent_halfwidth_kw'], 'training P amplitude')
    require(q.min()*base_kw >= rich['Q_mean_kvar']-rich['Q_independent_halfwidth_kvar']
            and q.max()*base_kw <= rich['Q_mean_kvar']+rich['Q_independent_halfwidth_kvar'], 'training Q amplitude')
    candidate = exact_candidate(dict(name='independent_rich_truth', supports=record['truth']['supports']), 3)
    branch = forward_physics(candidate, record['truth']['r'], record['truth']['x'], p, q, sp, sq, root)
    matched = lk.ac_predict(candidate, record['truth']['r'], record['truth']['x'], p, q, sp, sq, root)
    errors = {}
    for k in CHANNELS:
        close(matched[k], branch[k], 'rich96 independent/matched forward '+k, atol=2e-10, rtol=1e-10)
        errors[k] = float(np.max(abs(np.asarray(matched[k])-np.asarray(branch[k]))))
    meter = _meter(record['training_meter']); train = _obs(record['training'])
    require(train.t == length, 'saved rich observation length')
    # Original historical meter precision and calibration are constant in
    # time. Verify this before extending them to the full rich96 sequence.
    full = {}; selected = {}
    for k in CHANNELS:
        shape = (length, 3) if k in ('p', 'q', 'v') else (length,)
        sigma = np.broadcast_to(getattr(meter, k+'_sigma'), shape)
        require(np.array_equal(sigma, np.broadcast_to(sigma[0], shape)), 'time-varying rich training sigma')
        weak_meter = _meter(weak['training_meter'])
        weak_sigma = np.broadcast_to(getattr(weak_meter, k+'_sigma'), (8, 3) if k in ('p', 'q', 'v') else (8,))
        _exact(sigma[0], weak_sigma[0], 'rich/weak training precision '+k)
        values = matched[k]+sigma[0]*white[k]
        if k in ('p', 'q', 'v'):
            bias = np.broadcast_to(getattr(meter, 'bias_'+k), shape)
            require(np.array_equal(bias, np.broadcast_to(bias[0], shape)), 'time-varying rich training bias')
            values += bias[0]
        full[k] = values; selected[k] = values[:length]
    full.update(root_v=root, source_p=None, source_q=None)
    selected.update(root_v=root[:length], source_p=None, source_q=None)
    require(identity(plain(full)) == generation['rich96_observations_sha256'], 'rich96 complete observation hash')
    _exact(selected, record['training'], 'measured rich prefix generated with fresh training noise')
    require(identity(record['training']) == generation['selected_prefix_observations_sha256'], 'saved prefix observation hash')
    for k in ('p', 'q', 'v', 'master_p', 'master_q', 'root_v'):
        close(record['training_physical'][k], branch[k][:length], 'saved rich physical prefix '+k, atol=2e-10, rtol=1e-10)
    noise = old._white({k: branch[k][:length] for k in ('p', 'q', 'v', 'master_p', 'master_q', 'root_v')},
                       train, meter, {k: white[k][:length] for k in CHANNELS}, 'rich training')
    return dict(success=True, rich96_inputs_hash=generation['rich96_physical_inputs_sha256'],
        rich96_white_hash=generation['rich96_white_sha256'], rich96_observation_hash=generation['rich96_observations_sha256'],
        selected_prefix_hash=generation['selected_prefix_observations_sha256'], paired_holdout_hash=identity(holdout),
        training_length=length, independent_forward_max_abs_error=errors, training_noise_error=noise,
        independent_branch_forward_calls=1, matched_forward_calls_for_exact_hash=1,
        full_observation_hash_scope='matched frozen forward arithmetic, separately checked against independent branch AC')


def review_condition(record, weak):
    require(record['schema'] == 'ev_fixed_train_training_support_trial_v1' and record['source_unchanged'], 'rich schema/source drift')
    require(not record['validation_used_by_selector'] and not record['truth_used_by_selector']
            and not record['future_noise_used_by_selector'] and not record['statistically_calibrated']
            and not record['statistical_threshold_applied'], 'rich information/statistical scope')
    require(record['independent_development_units'] == 4
            and record['number_of_training_conditions_is_not_independent_sample_size'], 'paired statistical units')
    require(old.identity(dict(config=record['config'], source_sha256=record['source_sha256'])) == record['policy_id'], 'condition policy digest')
    generation = replay_rich_inputs(record, weak); cfg = record['config']; length = record['training_length']
    train, val = _obs(record['training']), _obs(record['validation'])
    tm, vm, fm = map(_meter, (record['training_meter'], record['validation_meter'], record['future_meter']))
    bank = {r['name']: exact_candidate(r, train.n) for r in record['candidates']}
    require(len(bank) == len(record['candidates']) == 4 and not record['proposal']['uses_truth'], 'rich candidate bank')
    truth = record['truth']; true = exact_candidate(dict(name='independent_truth', supports=truth['supports']), train.n)
    matches = [k for k, c in bank.items() if c.supports == true.supports]
    require(record['true_name'] == (matches[0] if len(matches) == 1 else None)
            and record['candidate_miss'] == (len(matches) != 1), 'true candidate mapping')
    trained = {r['name']: r for r in record['training_result']['fits']}
    train_audit = old._review_bank(record['training_result'], bank, train, tm, trained, cfg, training=True)
    counts = Counter(saved_fit_states=train_audit['saved_states'], replayed_fit_states=train_audit['replayed_states'])
    result = dict(seed=record['seed'], condition=record['condition'], training_length=length,
                  success=True, status=record['status'], generation=generation, training=train_audit)
    if not record['training_result']['all_complete']:
        require(record['status'] == 'training_failure', 'rich training failure retained')
        return dict(result, counts=dict(counts), preserved_training_failure=True)
    val_audit = old._review_bank(record['validation_result'], bank, val, vm, trained, cfg)
    counts.update(saved_fit_states=val_audit['saved_states'], replayed_fit_states=val_audit['replayed_states'])
    val_M = old._margin(val_audit['losses'], record['true_name']) if record['validation_result']['all_complete'] else None
    old._compare_optional(record['validation_M'], val_M, 'rich validation margin')
    err = record['training_parameter_error']
    if record['true_name'] in trained:
        fitted = trained[record['true_name']]
        for k in ('r', 'x'):
            close(err[k.upper()+'_relative'], np.linalg.norm(np.asarray(fitted[k])-np.asarray(truth[k]))/np.linalg.norm(truth[k]),
                  'true candidate training parameter error '+k)
    result.update(validation=val_audit, validation_M=val_M)
    if not record['selector_executed']:
        require(record['selection'] is None and not record['methods'], 'training-only condition unexpectedly has selected actions')
        return dict(result, counts=dict(counts), selector_executed=False)
    selection = record['selection']; scfg = selection['configuration']; T, n = cfg['horizon'], train.n
    require([m['name'] for m in record['methods']] == cfg['methods'], 'rich method order/denominator')
    require(all(selection[k] is False for k in ('uses_validation', 'uses_truth', 'uses_future_noise', 'random_used_by_primary'))
            and selection['line_parameters_unchanged'], 'rich selector information/fixed-line flags')
    actual = []; powers = {}; curves = set(); noise_errors = []
    for method in record['methods']:
        name = method['name']; power = method['power_kw']
        if power is None:
            require(method['status'] == 'selection_failure', 'rich action failure status')
            actual.append(dict(name=name, preserved_selection_failure=True)); continue
        power = np.asarray(power, float); require(power.shape == (T, n), 'rich action dimensions')
        powers[name] = power
        if name == 'none':
            close(power, np.zeros_like(power), 'rich noEV control', atol=0, rtol=0); operation = None
        else:
            saved_op = selection['constant_operation'] if name == 'constant' else selection['methods'][name]['operation']
            _exact(method['operation'], saved_op, 'rich method/selector saved operation')
            close(saved_op['u_kw'], power, 'rich executed power')
            operation = replay_operation_record(saved_op, old._service(scfg, name, T, n))
        phy = forward_physics(true, truth['r'], truth['x'], np.asarray(truth['p'])[12:]+power/scfg['base_kw'],
            np.asarray(truth['q'])[12:], np.asarray(truth['source_p'])[12:], np.asarray(truth['source_q'])[12:],
            np.asarray(truth['root_v'])[12:])
        for k in ('p', 'q', 'v', 'master_p', 'master_q', 'root_v'):
            close(method['physical'][k], phy[k], 'rich future physical EV once '+k, atol=3e-10, rtol=1e-10)
        kinds = {}
        for kind, obskey in (('noisy', 'observations'), ('noiseless', 'noiseless_observations')):
            obs = _obs(method[obskey]); expected = {k: np.asarray(record['future_white'][k]) if kind == 'noisy'
                                                  else np.zeros_like(record['future_white'][k]) for k in CHANNELS}
            error = old._white(phy, obs, fm, expected, 'rich '+name+' '+kind); noise_errors.extend(error.values())
            audited = old._review_bank(method[kind], bank, obs, fm, trained, cfg)
            counts.update(saved_fit_states=audited['saved_states'], replayed_fit_states=audited['replayed_states'])
            losses = {k: val_audit['losses'][k]+v for k, v in audited['losses'].items()} if (
                method[kind]['all_complete'] and record['validation_result']['all_complete']) else {}
            field = 'combined_loss' if kind == 'noisy' else 'combined_noiseless_loss'
            require(set(method[field]) == set(losses), 'rich combined candidate coverage')
            for k, value in losses.items(): close(method[field][k], value, 'rich validation + future SSE')
            old._compare_optional(method['M' if kind == 'noisy' else 'noiseless_M'], old._margin(losses, record['true_name']), 'rich combined M')
            future_M = old._margin(audited['losses'], record['true_name']) if method[kind]['all_complete'] else None
            old._compare_optional(method['future_M' if kind == 'noisy' else 'noiseless_future_M'], future_M, 'rich future M')
            kinds[kind] = audited
        old._compare_optional(method['delta_M_from_validation'], None if method['M'] is None or val_M is None
                             else method['M']-val_M, 'rich added M')
        key = np.round(power, 10).astype('<f8').tobytes()
        require(method['actual_fit_cache_hit'] == (key in curves), 'rich actual cache accounting'); curves.add(key)
        close(method['grid_net_kwh'], scfg['dt_hours']*power.sum(), 'rich net energy')
        close(method['grid_throughput_kwh'], scfg['dt_hours']*abs(power).sum(), 'rich throughput')
        actual.append(dict(name=name, status=method['status'], M=method['M'], noiseless_M=method['noiseless_M'],
                           operation=operation, kinds=kinds))
    require(record['unique_actual_curves'] == len(curves), 'rich actual unique curve denominator')
    context = build_fixed_contexts(train, tuple(bank.values()), {k: old._restore(v) for k, v in trained.items()}, tm, fm,
        -scfg['command_power_kw'], scfg['command_power_kw'], np.asarray(selection['constant_operation']['u_kw']),
        base_kw=scfg['base_kw'], **cfg['source'])
    require(context['success'] and context['diagnostics']['training_line_sha256'] == selection['context']['training_line_sha256'],
            'rich fixed training-line derivative snapshot')
    qps = {}; gates = {}; predictions = {}; seen = set()
    for name, entry in selection['methods'].items():
        if name not in old.DESIGNS: continue
        require(not entry['random_used'] and all('random' not in r['origin'] for r in entry['initial_candidates']), 'rich random entered primary ancestry')
        if entry.get('raw') is not None:
            qps[name] = old._replay_exchange(entry['raw'], context, old._service(scfg, name, T, n), cfg)
            counts['QP_states'] += qps[name]['QP_states']
            if entry.get('proxy_AC_check') is not None and entry['raw'].get('u') is not None:
                gate = old._same_witness_review(entry['proxy_AC_check'], np.asarray(entry['raw']['u']).reshape(T, n), context, bank, trained, fm)
                gates[name] = gate; counts['same_witness_AC_pairs'] += gate['replayed_pairs']
                require(entry['raw_proxy_gate'] == gate['gate_passed'], 'rich raw AC gate')
            if entry.get('raw_prediction') is not None and entry['raw'].get('u') is not None:
                prediction = entry['raw_prediction']; key = old.identity(prediction)
                if key not in seen:
                    seen.add(key); audited = old._predictive_review(prediction, np.asarray(entry['raw']['u']).reshape(T, n), context, bank, trained, fm)
                    predictions[name+'_raw'] = audited; counts['predictive_AC_fit_states'] += audited['replayed_states']
    for name, prediction in selection['predicted_AC'].items():
        key = old.identity(prediction)
        if key in seen: continue
        require(name in powers, 'rich selected prediction/action link')
        seen.add(key); audited = old._predictive_review(prediction, powers[name], context, bank, trained, fm)
        predictions[name] = audited; counts['predictive_AC_fit_states'] += audited['replayed_states']
    if all(k in selection['methods'] for k in old.PRIMARY):
        a, b = (selection['methods'][k] for k in old.PRIMARY)
        require(a['initial_actions_sha256'] == b['initial_actions_sha256']
                and a['configured_exchange_budget'] == b['configured_exchange_budget'], 'rich mean/cap fairness')
    result.update(counts=dict(counts), actual_methods=actual, QP=qps, same_witness_AC=gates, predictive_AC=predictions,
        future_white_max_abs_error=max(noise_errors, default=0.), selector_executed=True,
        derivative_rebuild_AC_calls=context['diagnostics']['ac_calls'],
        prediction_scope='only selected and raw predictions with explicit saved actions; anonymous unused cache entries not replayed',
        no_new_fit=True, no_new_QP_or_MILP_solve=True, no_case_generator=True)
    return result


def review_directory(directory, weak_directory, output=None, *, lengths=(8, 32), seeds=tuple(range(990101, 990105))):
    directory, weak_directory = Path(directory), Path(weak_directory)
    planned = {(length, seed) for length in lengths for seed in seeds}; seen = set(); conditions = []; errors = []; counts = Counter()
    sources = {}; raw_hashes = {}; weak_hashes = {}; nested = {}
    for length in lengths:
        sub = directory/('rich'+str(length)); freeze = read_json(sub/'freeze.json')
        sources[str(length)] = old.source_audit(sub, freeze)
        for path in sorted(sub.glob('trial_*.json.gz')):
            record = read_json(path); seed = record['seed']; key = length, seed
            require(key in planned and key not in seen, 'unexpected/duplicate rich paired condition'); seen.add(key)
            relative = str(path.relative_to(directory)).replace('\\', '/'); raw_hashes[relative] = sha(path)
            weakpath = weak_directory/('trial_'+str(seed)+'.json.gz'); weak_hashes[weakpath.name] = sha(weakpath)
            try:
                require(record['training_length'] == length and record['source_sha256'] == freeze['source_sha256']
                        and record['policy_id'] == freeze['policy_id'], 'rich condition/freeze identity')
                audit = review_condition(record, read_json(weakpath)); conditions.append(audit); counts.update(audit['counts'])
                nested[key] = record
            except (ValueError, RuntimeError, KeyError, TypeError, IndexError, AssertionError) as error:
                errors.append(dict(training_length=length, seed=seed, error=type(error).__name__+': '+str(error)))
    prefix_checks = []
    for seed in seeds:
        available = sorted(length for length in lengths if (length, seed) in nested)
        for short, long in zip(available[:-1], available[1:]):
            a, b = nested[short, seed], nested[long, seed]
            for field in ('training', 'training_truth', 'training_white'):
                for name, values in a[field].items():
                    if values is None: require(b[field][name] is None, 'rich nested absent channel')
                    else: _exact(values, np.asarray(b[field][name])[:short], 'rich nested '+field+' '+name)
            for hashfield in ('rich96_physical_inputs_sha256', 'rich96_white_sha256', 'rich96_observations_sha256',
                              'paired_validation_future_asset_sha256'):
                require(a['training_generation'][hashfield] == b['training_generation'][hashfield], 'rich regimes share full96/holdout hash')
            prefix_checks.append(dict(seed=seed, short=short, long=long, prefix_exact=True))
    require(raw_hashes == {name: sha(directory/name) for name in raw_hashes}, 'rich raw changed during review')
    require(weak_hashes == {name: sha(weak_directory/name) for name in weak_hashes}, 'weak parent raw changed during review')
    for length in lengths:
        sub = directory/('rich'+str(length))
        require(old.source_audit(sub, read_json(sub/'freeze.json')) == sources[str(length)], 'rich source changed during review')
    result = dict(success=not errors and seen == planned, available_conditions_checked=not errors,
        planned_conditions=len(planned), audited_conditions=len(conditions), independent_units=len({r['seed'] for r in conditions}),
        independent_unit_meaning='one paired asset/history; multiple training regimes are paired conditions',
        missing_conditions=[dict(training_length=L, seed=s) for L, s in sorted(planned-seen)], review_errors=errors,
        counts=dict(counts), nested_prefix_checks=prefix_checks, source_audits=sources,
        condition_sha256_before=raw_hashes, condition_sha256_after=raw_hashes, parent_weak8_sha256_before=weak_hashes,
        parent_weak8_sha256_after=weak_hashes, reviewer_sha256=sha(__file__),
        reused_trial_reviewer_sha256=sha(Path(old.__file__)), conditions=conditions,
        scope='independent actual AC/operation/noise replay; exact paired holdout and rich96 input hashes; matched-source QP reconstruction; no calibrated rejection or global AC claim')
    if output is not None: write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--weak-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[8, 32])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(990101, 990105)))
    args = parser.parse_args()
    result = review_directory(args.directory, args.weak_directory, args.output, lengths=args.lengths, seeds=args.seeds)
    print(json.dumps({k: result[k] for k in ('success', 'available_conditions_checked', 'planned_conditions', 'audited_conditions',
                                           'independent_units', 'missing_conditions', 'review_errors', 'counts')}))


if __name__ == '__main__': main()
