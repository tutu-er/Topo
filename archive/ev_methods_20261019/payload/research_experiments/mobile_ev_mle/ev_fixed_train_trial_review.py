"""Outside-freeze replay of saved train/validation/future EV evidence.

Only saved states are replayed.  No training, source fit, policy, QP/MILP solve,
or simulation case generator is called.  The original independent branch
sweep checks AC; frozen AC derivatives rebuild the declared first-order model.
The latter is a matched-source reconstruction, not an independent derivative
proof.  Saved solver best bounds are checked for direction and bookkeeping;
this module does not verify a branch-and-bound tree.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .information_fit_recovery import read_json, write_json, exact_candidate
from .information_replay import _obs, _meter
from .ev_fixed_train_review import review_fixed_fit
from .ev_fixed_train_design import build_fixed_contexts
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .ev_original_review import (forward_physics, require, close, replay_operation_record,
                                replay_qp, witness_box_upper, polynomial, master_backend_audit)

HERE = Path(__file__).resolve().parent
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q')
OUTPUTS = ('v', 'master_p', 'master_q')
PRIMARY = ('refined_v2g_mobile', 'capped_v2g_mobile')
DESIGNS = ('charge_fixed', 'v2g_fixed', 'charge_mobile', 'v2g_mobile', *PRIMARY)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_audit(directory, freeze):
    hashes = freeze['source_sha256']
    saved = read_json(directory/'source_snapshot.json.gz')
    require(saved['source_sha256'] == hashes and set(saved['source_base64']) == set(hashes),
            'frozen source/snapshot coverage')
    rows = []
    for name, expected in sorted(hashes.items()):
        raw = base64.b64decode(saved['source_base64'][name], validate=True)
        require(hashlib.sha256(raw).hexdigest() == expected, 'snapshot source: '+name)
        path = ((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else (HERE/name)).resolve()
        require(path.is_relative_to(HERE.parents[1].resolve()), 'source outside repository')
        current = sha(path)
        require(current == expected, 'current frozen source changed: '+name)
        rows.append(dict(name=name, sha256=current))
    require(identity(dict(config=freeze['config'], source_sha256=hashes)) == freeze['policy_id'],
            'freeze policy digest')
    return dict(success=True, source_count=len(rows), rows=rows,
                snapshot_sha256=sha(directory/'source_snapshot.json.gz'))


def _service(config, method, T, n):
    mobile = method in ('quality_mobile', 'random_mobile', 'charge_mobile', 'v2g_mobile', *PRIMARY)
    discharge = method in ('quality_mobile', 'random_mobile', 'v2g_fixed', 'v2g_mobile', *PRIMARY)
    access = np.ones((T, n), bool)
    if not mobile:
        access[:, 1:] = False
    return MobileServiceConfig(n_slots=T, n_sites=n, dt_hours=config['dt_hours'],
        p_charge_max_kw=config['command_power_kw'], p_discharge_max_kw=config['command_power_kw'],
        allow_discharge=discharge, battery_capacity_kwh=config['capacity_kwh'],
        initial_soc=config['initial_energy_kwh']/config['capacity_kwh'],
        terminal_energy_kwh=config['terminal_energy_kwh'], eta_charge=config['eta_charge'],
        eta_discharge=config['eta_discharge'], access=access, mobile=mobile,
        travel_slots=config['travel_slots'], setup_slots=config['setup_slots'],
        travel_energy_kwh=config['drive_kwh'], max_moves=config['max_moves'],
        max_throughput_kwh=config['throughput_kwh'], travel_cost=config['travel_worker_minutes'],
        setup_cost=config['setup_worker_minutes'], driver_budget=config['driver_budget_minutes'],
        cost_units='worker_minutes', min_arrival_throughput_kwh=config['min_arrival_throughput_kwh'])


def _restore(row):
    return SimpleNamespace(**{k: np.asarray(v, float) if k in ('r', 'x', 'p', 'q', 'source_p', 'source_q')
                              else v for k, v in row.items()})


def _margin(losses, true):
    return None if true not in losses else min(v for name, v in losses.items() if name != true)-losses[true]


def _compare_optional(saved, value, name):
    if value is None:
        require(saved is None, name+' should be unavailable')
    else:
        close(saved, value, name)


def _white(physical, observations, meter, expected, label):
    errors = {}
    for channel in CHANNELS:
        observed = getattr(observations, channel)
        require(observed is not None, 'missing observed channel: '+channel)
        mean = physical[channel]
        if channel in ('p', 'q', 'v'):
            mean = mean + np.asarray(getattr(meter, 'bias_'+channel))
        sigma = np.broadcast_to(getattr(meter, channel+'_sigma'), observed.shape)
        white = (observed-mean)/sigma
        error = float(np.max(abs(white-np.asarray(expected[channel]))))
        require(error <= 1e-7, label+' paired whitened noise: '+channel)
        errors[channel] = error
    close(observations.root_v, physical['root_v'], label+' known root', atol=0, rtol=0)
    require(observations.source_p is None and observations.source_q is None, 'unmetered source became measured')
    return errors


def _review_bank(saved, bank, observations, meter, training_rows, config, *, training=False):
    rows = saved['fits']; names = [r['name'] for r in rows]; failures = saved['failures']
    require(len(names) == len(set(names)) and set(names) <= set(bank), 'fit bank duplicate/unknown candidate')
    require(set(names) | set(failures) == set(bank), 'fit failure disappeared from denominator')
    require(bool(saved['all_complete']) == (set(names) == set(bank) and not failures
            and all(r['success'] for r in rows)), 'fit completeness flag')
    audits = []; losses = {}
    for row in rows:
        name = row['name']; finite = all(np.isfinite(np.asarray(row.get(k), float)).all()
                for k in ('r', 'x', 'p', 'q', 'source_p', 'source_q', 'loss', 'loglik'))
        if not finite or not all(k in row.get('predicted', {}) for k in CHANNELS):
            require(not row['success'] and name in failures, 'successful fit has no replayable finite state')
            audits.append(dict(candidate=name, finite_state_unavailable=True, declared_fit_success=False))
            continue
        trained = training_rows[name]
        audited = review_fixed_fit(bank[name], observations, meter, row,
            trained_r=None if training else trained['r'], trained_x=None if training else trained['x'])
        for key in ('r', 'x'):
            lower, upper = config['training'][key+'_bounds']
            require(np.all(np.asarray(row[key]) >= lower-1e-12)
                    and np.all(np.asarray(row[key]) <= upper+1e-12), 'line physical bounds')
        for key in ('source_p', 'source_q'):
            lower, upper = config['source'][key+'_bounds']
            require(np.all(np.asarray(row[key]) >= lower-1e-12)
                    and np.all(np.asarray(row[key]) <= upper+1e-12), 'source physical bounds')
        require(not row['success'] or (audited['voltage_min'] >= .85-1e-10
                and audited['voltage_max'] <= 1.15+1e-10), 'successful fit voltage domain')
        audits.append(audited); losses[name] = audited['full_SSE']
    return dict(declared_complete=saved['all_complete'], saved_states=len(rows),
                replayed_states=sum(not a.get('finite_state_unavailable', False) for a in audits),
                declared_failures=failures, audits=audits, losses=losses)


def _model(pair, lower, upper):
    return dict(name=pair.name, c=pair.c, B=pair.B, A0=pair.A0, At=pair.At,
        domain=dict(physical_lower=pair.lower, physical_upper=pair.upper, base_center=pair.base_center,
                    shift=pair.shift, action_lower=lower, action_upper=upper))


def _replay_exchange(saved, context, cfg, protocol):
    if saved.get('u') is None:
        require(not saved['completed'], 'missing MILP action labelled complete')
        return dict(available=False, status=saved['status'], QP_states=0)
    require(saved['required_pairs'] == [p.name for p in context['pairs']], 'all directed pairs retained')
    require(saved['aggregation'] == 'mean' and saved['score_scale'] == .5
            and not saved['full_curve_enumeration'] and not saved['global_nonlinear_AC_optimum_certified'],
            'MILP aggregation/score/scope')
    domain = build_mobile_service_domain(cfg); op = domain.as_dict()
    low, high = op['control_lower'], op['control_upper']
    models = [_model(p, low, high) for p in context['pairs']]
    K = len(models); weights = np.array([saved['weights'][m['name']] for m in models])
    close(weights, np.full(K, 1/K), 'equal planning task weights', atol=1e-12)
    require(all(saved['offsets'][m['name']] == 0 for m in models), 'training cost should not be recounted')
    cap = saved.get('score_caps')
    caps = np.full(K, np.inf) if cap is None else np.array([cap[m['name']] for m in models])
    entries = []; qp_count = 0; fallback_count = 0
    for index, e in enumerate(saved['evaluations']):
        require(domain.check(e['operation_x'])['feasible'], 'QP evaluation operation infeasible')
        close(np.asarray(e['operation_x'])[domain.indices.u].ravel(), e['u'], 'QP evaluation u=Ux')
        require(len(e['qp_results']) == K, 'all competing QPs must remain')
        qs = [replay_qp(m, e['u'], q) for m, q in zip(models, e['qp_results'])]
        qp_count += len(qs); fallback_count += sum(q.get('declared_exception_fallback', False) for q in qs)
        lo = .5*np.array([q['lower'] for q in qs]); hi = .5*np.array([q['upper'] for q in qs])
        close(e['pair_score_lower'], lo, 'pair lower'); close(e['pair_score_upper'], hi, 'pair upper')
        clo, chi = np.minimum(lo, caps), np.minimum(hi, caps)
        for key, value in (('mean_score_interval', [weights@lo, weights@hi]),
                           ('worst_score_interval', [min(lo), min(hi)]),
                           ('capped_pair_score_lower', clo), ('capped_pair_score_upper', chi),
                           ('capped_mean_score_interval', [weights@clo, weights@chi]),
                           ('capped_worst_score_interval', [min(clo), min(chi)])):
            close(e[key], value, key)
        interval = [float(weights@clo), float(weights@chi)]
        close(e['score_lower'], interval[0], 'selected capped/mean lower')
        close(e['score_upper'], interval[1], 'selected capped/mean upper')
        require(e['all_qp_converged'] == all(q['success'] for q in qs), 'QP convergence denominator')
        entries.append(dict(index=index, lower=interval[0], upper=interval[1],
                            all_qp_converged=all(q['success'] for q in qs)))
    require(saved['qp_calls'] == qp_count, 'QP state count')
    lower = max(e['lower'] for e in entries)
    close(saved['objective_lower'], lower, 'best checked QP lower')
    best = saved['best_evaluation_index']
    close(saved['u'], saved['evaluations'][best]['u'], 'best retained action')
    close(entries[best]['lower'], lower, 'retained action attains lower')
    expected_zeros = {(t*cfg.n_sites+j, t*cfg.n_sites+k) for t in range(cfg.n_slots)
                      for j in range(cfg.n_sites) for k in range(j+1, cfg.n_sites)}
    require(set(map(tuple, saved['zero_product_pairs'])) == expected_zeros, 'single-car zero products')
    initial_counts = saved['iterations'][0]['witnesses_before'] if saved['iterations'] else saved['witness_counts']
    initial = []; witness_count = 0
    for k, model in enumerate(models):
        values = []
        for j, witness in enumerate(saved['witnesses'][k]):
            # Here all source witnesses have exactly zero slope.  Full-box
            # feasibility is independently checked by interval arithmetic.
            intercept, slope = np.asarray(witness['physical_intercept']), np.asarray(witness['physical_slope'])
            require(np.array_equal(slope, np.zeros_like(slope)), 'fixed-source witness slope changed')
            require(np.all(intercept >= context['pairs'][k].lower-1e-12)
                    and np.all(intercept <= context['pairs'][k].upper+1e-12), 'fixed source witness outside box')
            upper, complete_box = witness_box_upper(model, witness, expected_zeros, 0., .5,
                                                    control_lower=low, control_upper=high)
            require(complete_box, 'fixed source must be legal over full control box')
            if j < initial_counts[k]:
                values.append(min(upper, caps[k]))
            witness_count += 1
        initial.append(min(values))
    close(saved['initial_pair_coarse_upper'], initial, 'initial valid witness/cap upper', atol=2e-5)
    previous = float(weights@np.asarray(initial))
    close(saved['initial_fixed_witness_upper'], previous, 'initial aggregate upper', atol=2e-5)
    masters = []
    for iteration in saved['iterations']:
        master = iteration['master']
        close(iteration['previous_upper'], previous, 'retained master upper sequence', atol=2e-5)
        if master.get('u') is not None:
            require(domain.check(master['operation_x'])['feasible'], 'saved MILP incumbent operation')
            close(np.asarray(master['operation_x'])[domain.indices.u].ravel(), master['u'], 'master power link')
        audit = master_backend_audit(master, dict(exchange=protocol['exchange'])) if master.get('scale') is not None else None
        if master.get('bound_consistent') and master.get('surrogate_incumbent') is not None:
            require(master['objective_upper'] >= master['surrogate_incumbent']-1e-8, 'MILP upper/incumbent direction')
        for rec in master.get('shared_square_witnesses', []):
            k, j = rec['pair'], rec['witness']
            a, C, terms = polynomial(models[k], saved['witnesses'][k][j])
            require(not terms, 'source-only model has unexpected bilinear terms')
            close(rec['kappa'], float(np.asarray(a, np.longdouble)@np.asarray(a, np.longdouble)), 'shared square constant')
            d = np.asarray((np.asarray(C, np.longdouble)**2).sum(axis=0), float)
            error = float(d@(((high-low)/master['control_segments'])**2/4))
            close(rec['square_chord_cost_error_upper'], error, 'shared-square chord error')
            require(rec['upward_cost_guard'] >= 0, 'upward coefficient guard sign')
            close(rec['pwl_score_error_upper'], .5*(error+rec['upward_cost_guard']), 'PWL half-SSE error')
        if iteration.get('upper_update_accepted'):
            require(master['bound_consistent'], 'untrusted master bound accepted')
            proposed = min(previous, float(master['objective_upper']))
            close(iteration['global_upper'], proposed, 'accepted numeric master bound')
            require(proposed >= iteration['global_lower']-1e-8, 'upper below attained lower')
            previous = proposed
        masters.append(dict(status=master['status'], backend=audit,
            square_coefficients_checked=len(master.get('shared_square_witnesses', [])),
            numeric_best_bound=master.get('solver_best_bound')))
    close(saved['objective_upper'], previous, 'final retained upper', atol=2e-5)
    gap = saved['objective_upper']-saved['objective_lower']
    close(saved['gap'], gap, 'proxy gap'); close(saved['raw_gap'], gap, 'signed proxy gap')
    target = protocol['exchange']['absolute_gap']+protocol['exchange']['relative_gap']*max(
        1e-12, abs(saved['objective_upper']), abs(saved['objective_lower']))
    close(saved['target_gap'], target, 'stopping tolerance')
    if saved['completed']:
        require(gap >= 0 and gap <= target+1e-8 and entries[best]['all_qp_converged'], 'completed proxy gap')
    else:
        require(saved['unfinished'], 'unfinished result lost')
    return dict(available=True, status=saved['status'], completed=saved['completed'],
                lower=saved['objective_lower'], upper=saved['objective_upper'], gap=gap,
                QP_states=qp_count, QP_exception_fallbacks=fallback_count,
                legal_fixed_witnesses=witness_count, masters=masters,
                scope='independent QP primal/tangent/cap and retained-bound arithmetic; solver tree not independently certified')


def _white_output(physical, meter):
    return np.concatenate([(np.asarray(physical[k]) + (np.asarray(meter.bias_v) if k == 'v' else 0))
                           .ravel()/np.broadcast_to(getattr(meter, k+'_sigma'), np.shape(physical[k])).ravel()
                           for k in OUTPUTS])


def _same_witness_review(saved, power, context, bank, training_rows, meter):
    """Independently replay AC at the exact source vector used by each QP."""
    power = np.asarray(power, float).reshape(context['control_shape']); T, n = power.shape
    require(len(saved['pairs']) == len(saved['qp_results']) == len(context['pairs']), 'same-witness pair count')
    p = context['background']['p']+power/context['base_kw']; q = context['background']['q']
    rows = []
    for pair, model, entry, qp in zip(context['pairs'], context['models'], saved['pairs'], saved['qp_results']):
        require(entry['name'] == pair.name, 'same-witness task order')
        if not entry['success']:
            require(not entry['gate_passed'], 'failed same-witness replay passed gate')
            rows.append(dict(name=pair.name, preserved_failure=True, gate_passed=False)); continue
        source = np.asarray(entry['proxy_source'], float)
        close(source, qp['physical_parameter'], 'AC witness must use the actual QP source')
        qp_audit = replay_qp(_model(pair, context['control_lower'], context['control_upper']), power.ravel(), qp)
        a, b = model['a'], model['b']; ar, br = training_rows[a], training_rows[b]
        refsource = np.asarray(model['reference_source']).reshape(T, 2)
        aa = forward_physics(bank[a], ar['r'], ar['x'], p, q, refsource[:, 0], refsource[:, 1], context['root_v'])
        source_grid = source.reshape(T, 2)
        bb = forward_physics(bank[b], br['r'], br['x'], p, q, source_grid[:, 0], source_grid[:, 1], context['root_v'])
        exact = _white_output(bb, meter)-_white_output(aa, meter)
        residual = pair.c+pair.B@power.ravel()+pair.A0@(source-pair.base_center)
        close(entry['proxy_residual'], residual, 'same-witness proxy residual')
        close(entry['complete_AC_residual'], exact, 'same-witness complete AC response')
        close(entry['proxy_SSE'], residual@residual, 'same-witness proxy SSE')
        close(entry['complete_AC_witness_SSE'], exact@exact, 'same-witness complete AC SSE')
        error = exact-residual; channel = {}
        for name, values in (('v', error[:T*n]), ('master_p', error[T*n:T*(n+1)]),
                             ('master_q', error[T*(n+1):])):
            channel[name] = dict(SSE=float(values@values), RMS=float(np.sqrt(np.mean(values**2))),
                                 max_abs=float(np.max(abs(values))))
            for key, value in channel[name].items():
                close(entry['channel_error'][name][key], value, 'same-witness channel '+name+' '+key)
        rms = max(v['RMS'] for v in channel.values())
        close(entry['max_channel_RMS'], rms, 'same-witness maximum channel RMS')
        voltage = all(np.min(x['v']) >= .85 and np.max(x['v']) <= 1.15 for x in (aa, bb))
        box = bool(np.all(source >= pair.lower-1e-10) and np.all(source <= pair.upper+1e-10))
        gate = bool(qp_audit['success'] and voltage and box and rms <= saved['gate'])
        require(entry['qp_converged'] == qp_audit['success'] and entry['voltage_feasible'] == voltage
                and entry['source_box_feasible'] == box and entry['gate_passed'] == gate,
                'same-witness feasibility/gate decision')
        rows.append(dict(name=pair.name, gate_passed=gate, max_channel_RMS=rms,
                         complete_AC_witness_SSE=float(exact@exact), QP=qp_audit))
    require(saved['success'] == all(not r.get('preserved_failure', False) for r in rows)
            and saved['gate_passed'] == all(r['gate_passed'] for r in rows), 'aggregate same-witness gate')
    return dict(success=saved['success'], gate_passed=saved['gate_passed'], pairs=rows,
                replayed_pairs=sum(not r.get('preserved_failure', False) for r in rows))


def _predictive_review(prediction, power, context, bank, training_rows, meter):
    require(not prediction['uses_validation'] and not prediction['uses_truth']
            and not prediction['uses_future_noise'], 'predictive information leak flag')
    if not prediction['success']:
        return dict(success=False, declared_failures=prediction['failures'], replayed_states=0)
    power = np.asarray(power, float).reshape(context['control_shape']); T, n = power.shape
    p = context['background']['p']+power/context['base_kw']; q = context['background']['q']
    references = {}
    for name, c in bank.items():
        f = training_rows[name]; source = context['forecasts'][name]
        references[name] = forward_physics(c, f['r'], f['x'], p, q, source['source_p'], source['source_q'], context['root_v'])
    require([r['name'] for r in prediction['pairs']] == [p.name for p in context['pairs']], 'predictive directed pair denominator')
    rows = []; added = []
    for pair, model, entry in zip(context['pairs'], context['models'], prediction['pairs']):
        a, b = model['a'], model['b']; phy = references[a]; f = training_rows[b]
        require(entry['success'] and entry['line_parameters_fixed'] and entry['source_interface'] == 1,
                'predictive fit/source scope')
        obs = _obs(dict(p=(p+np.asarray(meter.bias_p)).tolist(), q=(q+np.asarray(meter.bias_q)).tolist(),
            v=(phy['v']+np.asarray(meter.bias_v)).tolist(), master_p=phy['master_p'].tolist(),
            master_q=phy['master_q'].tolist(), root_v=context['root_v'].tolist(), source_p=None, source_q=None))
        cp, cq = obs.p-np.asarray(meter.bias_p), obs.q-np.asarray(meter.bias_q)
        rival = forward_physics(bank[b], f['r'], f['x'], cp, cq, entry['source_p'], entry['source_q'], obs.root_v)
        state = dict(entry['solver'], name=b, supports=bank[b].supports, p=cp, q=cq,
            source_p=entry['source_p'], source_q=entry['source_q'], bias_p=meter.bias_p,
            bias_q=meter.bias_q, bias_v=meter.bias_v,
            predicted={k: rival[k]+(np.asarray(getattr(meter, 'bias_'+k)) if k in ('p', 'q', 'v') else 0)
                       for k in CHANNELS})
        audited = review_fixed_fit(bank[b], obs, meter, state, trained_r=f['r'], trained_x=f['x'])
        close(entry['joint'], audited['full_SSE'], 'predictive attained SSE')
        close(entry['added'], .5*audited['full_SSE'], 'predictive distance')
        model_view = _model(pair, context['control_lower'], context['control_upper'])
        qp = replay_qp(model_view, power.ravel(), entry['proxy_QP'])
        close(entry['proxy_to_AC_distance_difference'], audited['full_SSE']-qp['upper'], 'proxy vs local AC cost')
        added.append(.5*audited['full_SSE']); rows.append(dict(name=pair.name, AC=audited, QP=qp))
    weights = np.full(len(rows), 1/len(rows)); values = np.asarray(added)
    for key, value in (('mean', weights@values), ('cap20_mean', weights@np.minimum(values, 20)), ('worst', min(values))):
        close(prediction[key], value, 'predictive score: '+key)
    return dict(success=True, replayed_states=len(rows), mean=float(weights@values),
                cap20_mean=float(weights@np.minimum(values, 20)), pairs=rows)


def review_trial(trial):
    cfg = trial['config']; T = cfg['horizon']; cut = cfg['training_points']; vcount = cfg['validation_points']
    require(cut == 8 and vcount == 4 and T == 6, 'declared 8/4/6 split')
    require(trial['training_indices'] == list(range(cut))
            and trial['validation_indices'] == list(range(cut, cut+vcount)), 'training/validation indices')
    require(trial['source_unchanged'] and not trial['validation_used_by_selector'], 'source or validation leak flag')
    require(identity(dict(config=cfg, source_sha256=trial['source_sha256'])) == trial['policy_id'], 'trial policy identity')
    train, val = _obs(trial['training']), _obs(trial['validation'])
    tm, vm, fm = map(_meter, (trial['training_meter'], trial['validation_meter'], trial['future_meter']))
    require(train.t == cut and val.t == vcount, 'saved split dimensions')
    bank = {r['name']: exact_candidate(r, train.n) for r in trial['candidates']}
    require(len(bank) == len(trial['candidates']) == 4, 'four candidate bank')
    require(not trial['proposal']['uses_truth'], 'candidate proposer uses truth')
    truth = trial['truth']; true = exact_candidate(dict(name='independent_truth', supports=truth['supports']), train.n)
    correct = [name for name, c in bank.items() if c.supports == true.supports]
    require(trial['true_name'] == (correct[0] if len(correct) == 1 else None)
            and trial['candidate_miss'] == (len(correct) != 1), 'offline true candidate mapping')
    total = cut+vcount+T
    require(np.asarray(truth['p']).shape == (total, train.n), 'truth time denominator')
    # Recreate only the already specified noise stream, not loads/assets or a
    # new case. All methods must use the same saved future noise realization.
    streams = np.random.SeedSequence(trial['seed']).spawn(5)
    seeds = {name: int(stream.generate_state(1)[0]) for name, stream in zip(('asset', 'background', 'source', 'noise', 'policy'), streams)}
    require(seeds == trial['seeds'], 'saved independent random-stream seeds')
    rng = np.random.default_rng(seeds['noise'])
    white = {k: rng.standard_normal((total, train.n) if k in ('p', 'q', 'v') else total) for k in CHANNELS}
    for k in CHANNELS:
        close(trial['future_white'][k], white[k][cut+vcount:], 'saved future white stream: '+k, atol=0, rtol=0)
    noise_errors = {}; physical_states = 0
    for label, obs, meter, start, stop in (('training', train, tm, 0, cut), ('validation', val, vm, cut, cut+vcount)):
        phy = forward_physics(true, truth['r'], truth['x'], np.asarray(truth['p'])[start:stop],
            np.asarray(truth['q'])[start:stop], np.asarray(truth['source_p'])[start:stop],
            np.asarray(truth['source_q'])[start:stop], np.asarray(truth['root_v'])[start:stop])
        noise_errors[label] = _white(phy, obs, meter, {k: white[k][start:stop] for k in CHANNELS}, label)
        physical_states += 1
    training_rows = {r['name']: r for r in trial['training_result']['fits']}
    train_audit = _review_bank(trial['training_result'], bank, train, tm, training_rows, cfg, training=True)
    counts = Counter(saved_fit_states=train_audit['saved_states'], replayed_fit_states=train_audit['replayed_states'])
    if not trial['training_result']['all_complete']:
        require(trial['status'] == 'training_failure', 'training failure status')
        return dict(seed=trial['seed'], success=True, preserved_training_failure=True, training=train_audit,
                    counts=dict(counts), actual_methods=[], scope='finite training states and retained training failure only')
    val_audit = _review_bank(trial['validation_result'], bank, val, vm, training_rows, cfg)
    counts.update(saved_fit_states=val_audit['saved_states'], replayed_fit_states=val_audit['replayed_states'])
    val_M = _margin(val_audit['losses'], trial['true_name']) if trial['validation_result']['all_complete'] else None
    _compare_optional(trial['validation_M'], val_M, 'validation margin')
    selection = trial['selection']; scfg = selection['configuration']
    require(all(selection[k] is False for k in ('uses_validation', 'uses_truth', 'uses_future_noise', 'random_used_by_primary')),
            'selector leak/random-primary flags')
    require(selection['context']['source_interface'] == 1 and selection['line_parameters_unchanged'], 'selector fixed source/line scope')
    require([m['name'] for m in trial['methods']] == cfg['methods'], 'method order/denominator')
    method_audits = []; curves = {}; named_power = {}; selection_methods = selection['methods']
    for method in trial['methods']:
        name = method['name']; power = method['power_kw']
        if power is None:
            require(method['status'] == 'selection_failure', 'missing action status')
            method_audits.append(dict(name=name, preserved_selection_failure=True)); continue
        power = np.asarray(power, float); require(power.shape == (T, train.n), 'method control dimensions')
        named_power[name] = power
        if name == 'none':
            close(power, np.zeros_like(power), 'noEV control', atol=0, rtol=0); operation = None
        else:
            saved_op = selection['constant_operation'] if name == 'constant' else selection_methods[name]['operation']
            close(saved_op['u_kw'], power, 'selected/executed power')
            operation = replay_operation_record(saved_op, _service(scfg, name, T, train.n))
        phy = forward_physics(true, truth['r'], truth['x'], np.asarray(truth['p'])[cut+vcount:]+power/scfg['base_kw'],
            np.asarray(truth['q'])[cut+vcount:], np.asarray(truth['source_p'])[cut+vcount:],
            np.asarray(truth['source_q'])[cut+vcount:], np.asarray(truth['root_v'])[cut+vcount:])
        physical_states += 1
        for k in ('p', 'q', 'v', 'master_p', 'master_q', 'root_v'):
            close(method['physical'][k], phy[k], 'true physical EV once: '+k, atol=3e-10, rtol=1e-10)
        kind_audits = {}; combined = {}
        for kind, obskey in (('noisy', 'observations'), ('noiseless', 'noiseless_observations')):
            obs = _obs(method[obskey]); expected = {k: white[k][cut+vcount:] if kind == 'noisy' else np.zeros_like(white[k][cut+vcount:]) for k in CHANNELS}
            noise_errors[name+'_'+kind] = _white(phy, obs, fm, expected, name+' '+kind)
            audited = _review_bank(method[kind], bank, obs, fm, training_rows, cfg)
            counts.update(saved_fit_states=audited['saved_states'], replayed_fit_states=audited['replayed_states'])
            loss = {k: val_audit['losses'][k]+v for k, v in audited['losses'].items()} if (
                method[kind]['all_complete'] and trial['validation_result']['all_complete']) else {}
            combined[kind] = loss
            field = 'combined_loss' if kind == 'noisy' else 'combined_noiseless_loss'
            require(set(method[field]) == set(loss), 'combined loss candidate coverage')
            for k, value in loss.items(): close(method[field][k], value, 'validation + future SSE identity')
            future_M = _margin(audited['losses'], trial['true_name']) if method[kind]['all_complete'] else None
            _compare_optional(method['future_M' if kind == 'noisy' else 'noiseless_future_M'], future_M, 'future margin')
            _compare_optional(method['M' if kind == 'noisy' else 'noiseless_M'], _margin(loss, trial['true_name']), 'combined margin')
            kind_audits[kind] = audited
        _compare_optional(method['delta_M_from_validation'], None if method['M'] is None or val_M is None else method['M']-val_M,
                          'margin change after added data')
        key = np.round(power, 10).astype('<f8').tobytes()
        require(method['actual_fit_cache_hit'] == (key in curves), 'actual curve cache flag')
        curves[key] = True
        method_audits.append(dict(name=name, status=method['status'], M=method['M'], noiseless_M=method['noiseless_M'],
                                 operation=operation, kinds=kind_audits))
    require(trial['unique_actual_curves'] == len(curves), 'unique actual curve denominator')
    require(selection['model_task_count'] == len(bank)*(len(bank)-1), 'all-pair planning denominator')
    baseline = np.asarray(selection['constant_operation']['u_kw'])
    contexts = build_fixed_contexts(train, tuple(bank.values()), {k: _restore(v) for k, v in training_rows.items()}, tm, fm,
        -scfg['command_power_kw'], scfg['command_power_kw'], baseline, base_kw=scfg['base_kw'], **cfg['source'])
    require(contexts['success'], 'matched-source derivative reconstruction failed')
    require(contexts['diagnostics']['training_line_sha256'] == selection['context']['training_line_sha256'], 'planning training-line snapshot')
    require(contexts['diagnostics']['R_X_coordinates_in_competitor_QP'] == 0
            and contexts['diagnostics']['source_coordinates'] == 2*T, 'source-only QP dimension')
    qp_audits = {}; prediction_audits = {}; witness_audits = {}; seen_pred = set()
    for name, rec in selection_methods.items():
        if name not in DESIGNS: continue
        require(not rec['random_used'] and all('random' not in x['origin'] for x in rec['initial_candidates']), 'random used in primary ancestry')
        cap = rec['cap']; initials = [x['AC_score'] for x in rec['initial_candidates'] if x['AC_score'] is not None and x['operation']['feasible']]
        if initials: close(rec['initial_best_score'], max(initials), 'deterministic initial best')
        if rec.get('selected_score') is not None:
            require(not initials or rec['selected_score'] >= max(initials)-1e-10, 'checked parent inheritance degraded')
            pred = selection['predicted_AC'][name]
            score = pred['mean' if cap is None else 'cap20_mean']
            close(rec['selected_score'], score, 'selected full AC score')
        if rec.get('raw') is not None:
            qp_audits[name] = _replay_exchange(rec['raw'], contexts, _service(scfg, name, T, train.n), cfg)
            counts['QP_states'] += qp_audits[name]['QP_states']
            if rec.get('proxy_AC_check') is not None and rec['raw'].get('u') is not None:
                audit = _same_witness_review(rec['proxy_AC_check'], np.asarray(rec['raw']['u']).reshape(T, train.n),
                                             contexts, bank, training_rows, fm)
                witness_audits[name] = audit; counts['same_witness_AC_pairs'] += audit['replayed_pairs']
                require(rec['raw_proxy_gate'] == audit['gate_passed'], 'raw AC gate flag')
            if rec.get('raw_prediction') is not None and rec['raw'].get('u') is not None:
                pred = rec['raw_prediction']; key = identity(pred)
                if key not in seen_pred:
                    seen_pred.add(key); audit = _predictive_review(pred, np.asarray(rec['raw']['u']).reshape(T, train.n), contexts, bank, training_rows, fm)
                    prediction_audits[name+'_raw'] = audit; counts['predictive_AC_fit_states'] += audit['replayed_states']
    for name, pred in selection['predicted_AC'].items():
        key = identity(pred)
        if key in seen_pred: continue
        require(name in named_power, 'saved selected prediction lacks executed action link')
        seen_pred.add(key); audit = _predictive_review(pred, named_power[name], contexts, bank, training_rows, fm)
        prediction_audits[name] = audit; counts['predictive_AC_fit_states'] += audit['replayed_states']
    if all(k in selection_methods for k in PRIMARY):
        a, b = (selection_methods[k] for k in PRIMARY)
        require(a['initial_actions_sha256'] == b['initial_actions_sha256']
                and a['configured_exchange_budget'] == b['configured_exchange_budget'], 'mean/cap sibling fairness')
    return dict(seed=trial['seed'], success=True, status=trial['status'], source_interface=1,
        training=train_audit, validation=val_audit, validation_M=val_M, actual_methods=method_audits,
        counts=dict(counts), physical_states_replayed=physical_states,
        noise_max_abs_error=max(v for c in noise_errors.values() for v in c.values()),
        same_white_noise_errors=noise_errors, QP=qp_audits, predictive_AC=prediction_audits,
        same_witness_AC=witness_audits,
        derivative_rebuild_AC_calls=contexts['diagnostics']['ac_calls'],
        prediction_scope='selected and raw-MILP predictions with explicit saved action links; anonymous all_prediction_records are not all replayable',
        selection_scope='frozen training-only API and saved input flags checked; no counterfactual policy rerun',
        no_new_fit=True, no_new_QP_or_MILP_solve=True, no_case_generator=True)


def review_directory(directory, output=None, *, planned_seeds=None):
    directory = Path(directory); freeze = read_json(directory/'freeze.json')
    source_before = source_audit(directory, freeze)
    paths = sorted(directory.glob('trial_*.json.gz')); hashes = {p.name: sha(p) for p in paths}
    planned = list(freeze['config']['development_seeds'] if planned_seeds is None else planned_seeds)
    histories, errors, counts = [], [], Counter(); seen = set()
    for path in paths:
        trial = read_json(path); seed = trial['seed']
        require(seed not in seen and seed in planned, 'unexpected/duplicate independent history'); seen.add(seed)
        try:
            require(trial['source_sha256'] == freeze['source_sha256'] and trial['policy_id'] == freeze['policy_id'], 'trial/freeze source identity')
            audit = review_trial(trial); histories.append(audit); counts.update(audit['counts'])
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError, AssertionError) as error:
            errors.append(dict(seed=seed, path=path.name, error=type(error).__name__+': '+str(error)))
    require(hashes == {p.name: sha(p) for p in paths}, 'review mutated saved trials')
    source_after = source_audit(directory, freeze)
    require(source_after == source_before, 'frozen sources changed during review')
    result = dict(success=not errors and set(planned) == seen, available_trials_checked=not errors,
        planned_seeds=planned, available_seeds=sorted(seen), missing_seeds=sorted(set(planned)-seen),
        planned_histories=len(planned), audited_histories=len(histories), independent_histories=len(histories),
        review_errors=errors, counts=dict(counts), histories=histories,
        source_audit=source_before, trial_sha256_before=hashes, trial_sha256_after=hashes,
        freeze_sha256=sha(directory/'freeze.json'), reviewer_sha256=sha(__file__),
        scope='independent saved-state AC, exact fixed-input/line and source-interface checks; matched-source QP reconstruction; no statistical calibration or global AC certificate')
    if output is not None: write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    result = review_directory(args.directory, args.output, planned_seeds=args.seeds)
    print(json.dumps({k: result[k] for k in ('success', 'available_trials_checked', 'planned_histories',
                                           'audited_histories', 'missing_seeds', 'review_errors', 'counts')}))


if __name__ == '__main__':
    main()
