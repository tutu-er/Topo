"""Independent conditional Gaussian checks for a completed multi-network panel.

Only frozen planning arrays and historical observations enter the calculation.
No future AC observations, production evidence evaluator, optimizer or identifier
is called. All eight policies share the same simulated power and voltage errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from scipy.stats import binom

from ..distribution_gate_20261114 import probability as prob
from . import audit

POLICIES = audit.POLICIES
MODES = audit.MODES
LAWS = ('fixed', 'forecast')


def generators(seed, entry_index, forecast_index):
    """Independent histories; common path arrays across policies and targets."""
    return [np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(entry_index), int(forecast_index), component]))
        for component in range(3)]


def simulate_target(model, powers, forecast, target, *, draws, seed,
                    entry_index, forecast_index, batch_size):
    """Save full-native SSE, bounded in memory to one conditional target."""
    if draws < 2 or batch_size < 1:
        raise ValueError('At least two independent draws and a positive batch size required')
    T, channels = forecast.shape
    n, H = model['n'], model['H']
    if channels != 2*n or set(powers) != set(POLICIES):
        raise ValueError('Power shape or eight-policy contract differs')
    # Equal actions often produce eight identical nominal power arrays. Keep
    # every policy in the saved output, but evaluate each exact array once.
    # Equality is exact (including dtype), never an allclose approximation.
    groups = []
    for pi, policy in enumerate(POLICIES):
        for group in groups:
            other = powers[POLICIES[group[0]]]
            if powers[policy].dtype == other.dtype and np.array_equal(powers[policy], other):
                group.append(pi)
                break
        else:
            groups.append([pi])
    result = np.empty((len(LAWS), len(POLICIES), draws, len(H)))
    power_rng, node_rng, root_rng = generators(seed, entry_index, forecast_index)
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws-start)
        xi = power_rng.normal(size=(count, T, 2*n))*.05*abs(forecast)[None]
        error = (node_rng.normal(size=(count, T, n))*model['independent_sd']
                 + root_rng.normal(size=(count, T, 1))*model['root_sd'])
        for li, law in enumerate(LAWS):
            for group in groups:
                pi = group[0]
                policy = POLICIES[pi]
                X = powers[policy][None] + (xi if law == 'forecast' else 0.)
                if len(X) == 1:
                    X = np.broadcast_to(X, (count, T, 2*n))
                Y = X@model['means'][target].T + error
                # Iterating frozen points avoids native_count*batch*T*N memory.
                for j, h in enumerate(H):
                    residual = Y-X@h.T
                    result[li, pi, start:start+count, j] = (
                        np.square(residual).sum(axis=(1, 2))+model['history'][j])
                for duplicate in group[1:]:
                    result[li, duplicate, start:start+count] = result[li, pi, start:start+count]
    if not np.isfinite(result).all():
        raise ArithmeticError('Non-finite simulated native SSE')
    return result


def quote_provenance(policy, plans):
    parent = policy.removesuffix('_rotation')
    quote = plans.get(parent, {}) if parent in MODES else {}
    is_new = policy in MODES
    return dict(parent_quote_mode=parent if parent in MODES else None,
        new_gate_feasible=quote.get('new_gate_feasible') if is_new else None,
        anchor_fallback=quote.get('anchor_fallback', False) if is_new else False,
        zero_fallback=quote.get('zero_fallback', False) if is_new else False,
        executed_source=quote.get('executed_source') if is_new else policy,
        parent_new_gate_feasible=quote.get('new_gate_feasible') if quote else None,
        parent_anchor_fallback=quote.get('anchor_fallback', False),
        parent_zero_fallback=quote.get('zero_fallback', False))


def summarize_target(model, forecast, powers, total, target, plans, gate_cache):
    """Recompute selection, pair events and theoretical bounds from full SSE."""
    draws, T, n = total.shape[2], len(forecast), model['n']
    i = model['representatives'][target]
    foreign = [j for j, f in enumerate(model['families']) if f != model['families'][i]]
    family_labels = np.asarray(model['families'], dtype=str)
    rows, events = [], {}
    winners = np.empty((len(LAWS), len(POLICIES), draws), dtype=np.int64)
    family_success = np.empty_like(winners, dtype=bool)
    representative_success = np.empty_like(winners, dtype=bool)

    def gates(policy, mode):
        key = (policy, mode)
        if key not in gate_cache:
            gate_cache[key] = audit.exact_gates(model, forecast, powers[policy], mode)
        return gate_cache[key]

    for li, law in enumerate(LAWS):
        for pi, policy in enumerate(POLICIES):
            all_sse = total[li, pi]
            winner = prob.select_native(model, all_sse, T*n)
            family = family_labels[winner] == family_labels[i]
            # Fixed-input MGF receives its own conservative bound; the exact
            # normal tail is separately reported for every fixed-input action.
            parent = policy.removesuffix('_rotation')
            theory_mode = (('mgf_fixed' if parent == 'mgf_fixed' else 'normal_fixed')
                           if law == 'fixed' else 'mgf_forecast')
            check = gates(policy, theory_mode)
            tie = check['tie']
            margins = (all_sse[:, foreign]-all_sse[:, i, None])/(2*model['sigma2'])
            representative = np.all(margins >= tie, axis=1)
            pairs = [r for r in check['rows'] if r['target'] == target]
            if [r['competitor_native'] for r in pairs] != foreign:
                raise ArithmeticError('Foreign native comparator order differs')
            union = min(1., sum(r['pair_failure'] for r in pairs))
            failed = int((~representative).sum())
            pair_rows = []
            for k, pair in enumerate(pairs):
                count = int((margins[:, k] < tie).sum())
                mean = float(margins[:, k].mean())
                variance = float(margins[:, k].var(ddof=1))
                pair_rows.append(dict(pair, observed_failure_count=count,
                    observed_failure_rate=count/draws,
                    observed_failure_95_CI=prob.interval(count, draws),
                    observed_logodds_mean=mean, observed_logodds_variance=variance,
                    mean_error_in_standard_errors=(None if pair['variance'] <= 0 else
                        float((mean-pair['mean'])/np.sqrt(pair['variance']/draws))),
                    one_sided_bound_exceedance_tail=float(binom.sf(count-1, draws, pair['pair_failure']))))
            row = dict(law=law, target_representative=target, target_native=i,
                target_topology_id=model['families'][i], policy=policy, n=draws,
                target_family_winner_count=int(family.sum()), target_family_winner_rate=float(family.mean()),
                target_family_winner_95_CI=prob.interval(int(family.sum()), draws),
                representative_beats_all_foreign_count=int(representative.sum()),
                representative_success_rate=float(representative.mean()),
                representative_success_95_CI=prob.interval(int(representative.sum()), draws),
                comparison_count=len(foreign), single_family_vacuous=not foreign,
                theory_mode=theory_mode, theoretical_union_failure_upper=union,
                theoretical_representative_success_lower=1-union,
                one_sided_union_bound_exceedance_tail=float(binom.sf(failed-1, draws, union)),
                executed_policy_gate_feasible=check['feasible'],
                declared_law_matched=((policy == 'mgf_forecast') == (law == 'forecast'))
                    if policy in MODES else None,
                pairs=pair_rows, **quote_provenance(policy, plans))
            if law == 'fixed':
                row['fixed_MGF_union_failure_upper'] = min(1., gates(policy, 'mgf_fixed')['target_failure_sums'][str(target)])
                row['fixed_exact_normal_union_failure_upper'] = min(1., gates(policy, 'normal_fixed')['target_failure_sums'][str(target)])
            rows.append(row)
            events[law, policy] = (family, representative)
            winners[li, pi], family_success[li, pi], representative_success[li, pi] = winner, family, representative
    comparisons = []
    for law in LAWS:
        for policy in POLICIES:
            for baseline in ('no_dfs', 'anchor'):
                if policy == baseline:
                    continue
                for event_index, event_name in enumerate(('target_family_winner', 'representative_beats_all_foreign')):
                    a, b = events[law, policy][event_index], events[law, baseline][event_index]
                    comparisons.append(dict(law=law, target_representative=target,
                        policy=policy, baseline=baseline, event=event_name, n=draws,
                        gained=int(np.sum(a & ~b)), lost=int(np.sum(~a & b)),
                        mean_difference=float(np.mean(a.astype(float)-b))))
    return rows, comparisons, dict(winner_native=winners, family_success=family_success,
        representative_success=representative_success)


def forecast_probability(folder, entry, protocol, out, draws, seed, batch_size):
    seal = prob.read(folder/'seal.json')
    if not (folder/'complete.json').exists():
        raise ValueError('Completed forecast required before model simulation')
    for value in seal.values():
        if isinstance(value, dict) and 'path' in value and 'sha256' in value:
            prob.checked(value)
    if seal['score'] != entry['score'] or seal['validation'] != entry['files']['validation.npz']:
        raise ValueError('Sealed history differs from indexed history')
    planner = prob.arrays(prob.checked(seal['planner']))
    plans = prob.read(prob.checked(seal['plans']))
    if set(planner) != {'forecast', *POLICIES} or set(plans) != set(MODES):
        raise ValueError('Unexpected strategy contract')
    score = prob.read(prob.checked(entry['score']))
    val = prob.arrays(prob.checked(entry['files']['validation.npz']))
    native = prob.read(prob.checked(entry['native_copy']))
    model = prob.build_model(score, val, native)
    config = prob.read(prob.checked(seal['resource_case']))
    resource_spec = prob.read(prob.checked(entry['resource_spec']))
    base = float(entry['base_kw'])
    if not np.isfinite(base) or base <= 0 or float(resource_spec['base_kw_kvar']) != base:
        raise ValueError('Explicit base differs from historical resource specification')
    forecast = planner['forecast']
    power = {p: forecast+prob.shift(config, planner[p], model['labels'], base) for p in POLICIES}
    draw = int(folder.name.removeprefix('forecast'))
    if not 0 <= draw < int(protocol['forecasts']):
        raise ValueError('Forecast index outside sealed protocol')
    out.mkdir(parents=True, exist_ok=False)
    rows, comparisons, samples, cache = [], [], [], {}
    for target in model['targets']:
        total = simulate_target(model, power, forecast, target, draws=draws, seed=seed,
            entry_index=entry['index'], forecast_index=draw, batch_size=batch_size)
        local_rows, local_comparisons, events = summarize_target(
            model, forecast, power, total, target, plans, cache)
        path = out/f'target{target:03d}_samples.npz'
        np.savez_compressed(path, total_sse=total, policies=np.asarray(POLICIES), laws=np.asarray(LAWS),
            representative_native_indices=np.asarray(model['representatives']),
            conditional_target=np.asarray(target), **events)
        rows.extend(local_rows)
        comparisons.extend(local_comparisons)
        samples.append(prob.ref(path))
    result = dict(**audit.identity(entry), entry_index=entry['index'], forecast=draw,
        source_seal=prob.ref(folder/'seal.json'), score=entry['score'], validation=entry['files']['validation.npz'],
        native_copy=entry['native_copy'], source_complete=prob.ref(folder/'complete.json'),
        targets=model['targets'], native_points=len(model['H']), native_families=len(set(model['families'])),
        representative_native_indices=model['representatives'], root_voltage_error_sd=model['root_sd'],
        independent_voltage_error_sd=model['independent_sd'], samples=samples, rows=rows, comparisons=comparisons,
        conditioning='Each historical target K is assumed as the mean; nominal planned delivery. Model simulation, not physical AC recovery.')
    prob.write(out/'report.json', result)
    return result


def run(source, out, draws=2048, seed=2026111503, batch_size=64):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if draws < 2 or batch_size < 1:
        raise ValueError('At least two draws and a positive batch size required')
    completion = prob.read(source/'complete.json')
    protocol = prob.read(source/'protocol.json')
    entries = prob.read(prob.checked(protocol['inputs']))
    if tuple(protocol['policies']) != POLICIES or tuple(protocol['modes']) != MODES:
        raise ValueError('Panel strategy contract differs')
    if len({e['id'] for e in entries}) != len(entries) or len({e['index'] for e in entries}) != len(entries):
        raise ValueError('Repeated history identity or random-stream index')
    if not completion.get('source_unchanged', False):
        raise ValueError('Source run reports changed code')
    out.mkdir(parents=True)
    sources = [Path(__file__), Path(prob.__file__), Path(audit.__file__)]
    for path in sources:
        shutil.copyfile(path, out/('source_'+path.name))
    own = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_run=prob.ref(source/'protocol.json'),
        source_complete=prob.ref(source/'complete.json'), sources=[prob.ref(p) for p in sources],
        draws=draws, seed=seed, batch_size=batch_size, phase=protocol['phase'], smoke=protocol.get('smoke', False),
        expected_forecasts=len(entries)*int(protocol['forecasts']),
        input_qualification_ledger=protocol['qualification_ledger'], preparation_failures=protocol['preparation_failures'],
        planned_histories=protocol['planned_histories'], runnable_histories=len(entries),
        stream='PCG64 SeedSequence[seed, entry.index, forecast, component:0 power,1 node,2 root]; streams continue across batches, reset identically per conditional target; shared across eight policies and both laws',
        law='Fixed P or independent 5% Gaussian P/Q error scaled by frozen forecast, shared-root plus independent-node Gaussian squared-voltage errors; time independent; nominal requested delivery',
        reporting='Conditional model simulation only. Binomial intervals assume iid simulation paths; all multiple comparisons are exploratory and unadjusted. One-family certificates are vacuous and do not assess true-topology coverage.',
        environment=dict(python=sys.version, numpy=np.__version__))
    prob.write(out/'protocol.json', own)
    results, failures = [], []
    for entry in entries:
        for draw in range(int(protocol['forecasts'])):
            folder = source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            try:
                item = forecast_probability(folder, entry, protocol,
                    out/'histories'/entry['id']/f'forecast{draw:02d}', draws, seed, batch_size)
                results.append(item)
            except Exception as exc:
                failures.append(dict(**audit.identity(entry), forecast=draw, error=repr(exc), traceback=traceback.format_exc()))
            prob.write(out/'progress.json', dict(completed=len(results), failed=len(failures),
                expected=own['expected_forecasts'], last_id=entry['id'], last_forecast=draw))
    unchanged = all(prob.ref(p)['sha256'] == r['sha256'] for p, r in zip(sources, own['sources'], strict=True))
    answer = dict(status='complete' if not failures and unchanged else 'failed', protocol=prob.ref(out/'protocol.json'),
        source_unchanged=unchanged, results=results, failures=failures,
        expected_forecasts=own['expected_forecasts'], completed_forecasts=len(results),
        row_groups=sum(len(r['rows']) for r in results))
    prob.write(out/'report.json', answer)
    prob.write(out/'complete.json', dict(status=answer['status'], report=prob.ref(out/'report.json'),
        completed_forecasts=len(results), failed_forecasts=len(failures), source_unchanged=unchanged))
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2026111503)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    report = run(args.run, args.out, args.draws, args.seed, args.batch_size)
    print(dict(status=report['status'], forecasts=report['completed_forecasts'],
        groups=report['row_groups'], failures=len(report['failures'])))
    raise SystemExit(0 if report['status'] == 'complete' else 1)
