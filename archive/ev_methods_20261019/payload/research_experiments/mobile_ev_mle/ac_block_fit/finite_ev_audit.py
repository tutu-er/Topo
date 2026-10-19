"""Fixed-endpoint full AC distances and derivatives at finite EV power.

Only the last arrived history row is perturbed. No fitter, forecast, simulator
truth, plan selector, or source re-estimation enters this numerical audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .. import likelihood as lk
from ..ev_fixed_train_fit import subset_observations, subset_model
from ..ev_rx_fit import _RxMap
from . import benchmark as b


def finite_response(candidate, data, meter, state, *, base_kw, power_kw,
                    site=0, counter=None):
    """h and dh/d(P,Q) from a fresh AC solve at a declared positive EV load.

    ``h`` stacks terminal voltage magnitudes and root P/Q. Derivatives are
    per kW/kvar; EV power factor is one, so its active derivative is P-site.
    The same fitted source P/Q and R/X are held exactly fixed.
    """
    if not np.isfinite([base_kw, power_kw]).all() or base_kw <= 0:
        raise ValueError("finite power and positive base_kw required")
    if isinstance(site, bool) or int(site) != site or not 0 <= site < data.n:
        raise ValueError("site must identify an observed terminal")
    ids = np.array([data.t-1])
    local = subset_observations(data, ids)
    lm = subset_model(meter, data, ids)
    p = local.p-lk._array(lm.bias_p, local.p.shape, "bias_p")
    q = local.q-lk._array(lm.bias_q, local.q.shape, "bias_q")
    p = p.copy(); p[0, int(site)] += power_kw/base_kw
    sp = np.asarray(state['source_p'], float)[ids]
    sq = np.asarray(state['source_q'], float)[ids]
    if counter is not None:
        counter['full_AC_replays'] = counter.get('full_AC_replays', 0)+1
    physical = lk.ac_predict(candidate, state['r'], state['x'], p, q,
                             sp, sq, local.root_v)
    if counter is not None:
        counter['exact_derivative_evaluations'] = counter.get('exact_derivative_evaluations', 0)+1
    derivatives = lk._response_derivatives(candidate, physical, estimate_source=True)
    columns = slice(2*candidate.m, 2*candidate.m+2*data.n)
    jac = np.concatenate((derivatives['v'][:, :, columns],
        derivatives['master_p'][:, None, columns],
        derivatives['master_q'][:, None, columns]), axis=1)/base_kw
    sigma = np.column_stack((lk._array(lm.v_sigma, p.shape, 'v_sigma'),
        lk._array(lm.master_p_sigma, (1,), 'master_p_sigma'),
        lk._array(lm.master_q_sigma, (1,), 'master_q_sigma')))
    mean = np.column_stack((physical['v']+lk._array(lm.bias_v, p.shape, 'bias_v'),
                            physical['master_p'], physical['master_q']))
    voltage = np.abs(physical['voltage'])
    return dict(mean=mean, sigma=sigma, response_PQ_derivative_per_kw=jac,
        active_derivative_per_kw=jac[:, :, int(site)],
        voltage_PQ_derivative_per_kw=jac[:, :data.n, :],
        voltage_range=[float(voltage.min()), float(voltage.max())],
        ac_equation_residual=float(physical['ac_equation_residual']),
        power_balance_residual=float(physical['power_balance_residual']),
        working_history_row=int(ids[0]), site=int(site), power_kw=float(power_kw),
        base_kw=float(base_kw), P_input=p, Q_input=q, source_p=sp, source_q=sq,
        R_X_refitted=False, source_refitted=False, future_used=False, truth_used=False)


def pair_distance(first, second):
    """Exact D and active derivative for two fixed models at the SAME action."""
    for key in ('site', 'power_kw', 'working_history_row', 'base_kw'):
        if first[key] != second[key]:
            raise ValueError('paired response actions differ: '+key)
    if not np.array_equal(first['sigma'], second['sigma']):
        raise ValueError('paired response meter precisions differ')
    sigma = np.asarray(first['sigma'])
    delta = (np.asarray(first['mean'])-second['mean'])/sigma
    direction = (np.asarray(first['active_derivative_per_kw'])
                 -second['active_derivative_per_kw'])/sigma
    return dict(distance=.5*float(np.sum(delta*delta)),
                active_derivative_per_kw=float(np.sum(delta*direction)),
                whitened_mean_difference=delta,
                whitened_active_response_difference_per_kw=direction)


def difference(value, reference, *, rtol=1e-3, floor=1e-12):
    value, reference = np.asarray(value, float), np.asarray(reference, float)
    delta = value-reference
    absolute = float(np.linalg.norm(delta))
    norm = float(np.linalg.norm(reference))
    relative = absolute/max(norm, floor)
    return dict(absolute_l2=absolute, max_abs=float(np.max(abs(delta), initial=0.)),
        reference_l2=norm, relative=relative, relative_tolerance=rtol,
        near_zero_reference=bool(norm <= floor), agreement=bool(relative <= rtol))


def summarize(rows, plan):
    """Keep every declared candidate and pair, including unavailable responses."""
    table = {(r['network'], r['candidate'], r['algorithm'], r['power_kw']): r for r in rows}
    manifest = plan['endpoint_manifest']
    networks = sorted({r['network'] for r in manifest})
    comparisons, pairs = [], []
    rtol, floor = plan['relative_tolerance'], plan['relative_denominator_floor']
    for power in plan['powers_kw']:
        for network in networks:
            names = sorted({r['candidate'] for r in manifest if r['network'] == network})
            for name in names:
                ref = table[(network, name, 'ref_jac', power)]
                new = table[(network, name, 'block_jac_gap', power)]
                entry = dict(network=network, candidate=name, power_kw=power,
                    both_common_qualified=bool(ref['common_qualified'] and new['common_qualified']),
                    available=bool(ref['available'] and new['available']))
                if entry['available']:
                    for field in ('mean', 'active_derivative_per_kw', 'voltage_PQ_derivative_per_kw'):
                        entry[field+'_comparison'] = difference(new['response'][field], ref['response'][field],
                                                                rtol=rtol, floor=floor)
                comparisons.append(entry)
            for left, right in combinations(names, 2):
                entry = dict(network=network, candidates=[left, right], power_kw=power,
                    all_four_common_qualified=all(table[(network, name, method, power)]['common_qualified']
                        for name in (left, right) for method in ('ref_jac', 'block_jac_gap')))
                for method in ('ref_jac', 'block_jac_gap'):
                    rr = [table[(network, name, method, power)] for name in (left, right)]
                    if all(r['available'] for r in rr):
                        entry[method] = pair_distance(*(r['response'] for r in rr))
                entry['available'] = all(m in entry for m in ('ref_jac', 'block_jac_gap'))
                if entry['available']:
                    for field in ('distance', 'active_derivative_per_kw'):
                        entry[field+'_comparison'] = difference([entry['block_jac_gap'][field]],
                            [entry['ref_jac'][field]], rtol=rtol, floor=floor)
                pairs.append(entry)
    primary = [p for p in pairs if p['all_four_common_qualified']]
    qualified = [c for c in comparisons if c['both_common_qualified']]
    def count(entries, field):
        available = [r for r in entries if r['available']]
        return dict(denominator=len(entries), available=len(available),
            passed=sum(r[field+'_comparison']['agreement'] for r in available),
            max_relative=max((r[field+'_comparison']['relative'] for r in available), default=None),
            max_absolute=max((r[field+'_comparison']['absolute_l2'] for r in available), default=None),
            near_zero_references=sum(r[field+'_comparison']['near_zero_reference'] for r in available))
    return dict(scope='finite-power full AC numerical consistency, fixed historical endpoints; no future policy benefit evaluated',
        total_endpoint_conditions=len(rows), endpoint_failures=sum(not r['available'] for r in rows),
        candidate_comparisons=comparisons, pair_comparisons=pairs,
        all_pair_distance=count(pairs, 'distance'), all_pair_active_derivative=count(pairs, 'active_derivative_per_kw'),
        primary_pair_distance=count(primary, 'distance'), primary_pair_active_derivative=count(primary, 'active_derivative_per_kw'),
        qualified_candidate_active_response=count(qualified, 'active_derivative_per_kw'),
        qualified_candidate_voltage_PQ_response=count(qualified, 'voltage_PQ_derivative_per_kw'),
        failed_and_unqualified_denominators_retained=True, optimizers_started=0,
        future_measurements_read=False, true_RX_read=False)


def run(output):
    output = Path(output); plan_path = output/'plan.json'
    plan = json.loads(plan_path.read_text())
    if (output/'completed.json').exists():
        raise FileExistsError('completed numerical audit already exists')
    whitelist = Path(plan['whitelist'])
    if sha256(whitelist.read_bytes()).hexdigest() != plan['whitelist_sha256']:
        raise ValueError('history whitelist changed')
    # Every source file is verified before the first physical replay.
    loaded = []
    for item in plan['endpoint_manifest']:
        path = Path(item['file'])
        if sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('saved endpoint changed: '+str(path))
        envelope = json.loads(path.read_text())
        loaded.append((item, envelope['task'], envelope['worker']))
    if len(loaded) != plan['expected_endpoints'] or len(loaded)*len(plan['powers_kw']) != plan['expected_endpoint_power_conditions']:
        raise ValueError('predeclared numerical denominator differs')
    package = b.load_history_package(whitelist)
    counter = {}; rows = []; began = perf_counter()
    for entry, task, worker in loaded:
        candidate, data, meter, options, _, _ = b._reconstruct(package['items'][entry['network']], task)
        state = {key: np.asarray(worker['attained_state'][key], float)
                 for key in ('r', 'x', 'source_p', 'source_q')}
        common = bool(entry['original_success'] and worker['solver_success']
                      and entry['common_pg'] <= plan['common_pg_target'])
        mapping = _RxMap(candidate.m, options['r_bounds'], options['x_bounds'], options['ratio_bounds'])
        source_ok = True
        for key in ('source_p', 'source_q'):
            low, high = lk._bounds(options[key+'_bounds'], (data.t,), key+'_bounds')
            source_ok &= bool(np.all(state[key] >= low-1e-12) and np.all(state[key] <= high+1e-12))
        state_domain_ok = bool(source_ok and mapping.contains(state['r'], state['x']))
        for power in plan['powers_kw']:
            clock = perf_counter()
            row = dict(network=entry['network'], candidate=entry['candidate'], algorithm=entry['algorithm'],
                power_kw=power, common_qualified=common, common_pg=entry['common_pg'],
                endpoint_sha256=entry['sha256'], available=False)
            try:
                response = finite_response(candidate, data, meter, state,
                    base_kw=package['items'][entry['network']]['base_kw'], power_kw=power,
                    site=plan['site'], counter=counter)
                physical = bool(state_domain_ok and response['ac_equation_residual'] <= 2e-10
                    and response['power_balance_residual'] <= 2e-10
                    and response['voltage_range'][0] >= options['voltage_bounds'][0]
                    and response['voltage_range'][1] <= options['voltage_bounds'][1]
                    and np.isfinite(response['mean']).all()
                    and np.isfinite(response['response_PQ_derivative_per_kw']).all())
                row.update(available=physical, response=response, physical_pass=physical,
                           same_original_state_domain_pass=state_domain_ok)
                if not physical:
                    row['failure'] = 'full AC physical/domain/nonfinite audit failed'
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                row['failure'] = type(error).__name__+': '+str(error)
            row['seconds'] = perf_counter()-clock
            rows.append(row)
    result = summarize(rows, plan)
    result.update(endpoint_conditions=rows, counters=counter, seconds=perf_counter()-began,
        source_sha256={path.name:sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), Path(__file__).with_name('test_finite_ev_audit.py'))},
        dependency_source_sha256={str(path):sha256(path.read_bytes()).hexdigest() for path in
            (Path(lk.__file__), Path(b.__file__),
             Path(sys.modules[subset_observations.__module__].__file__),
             Path(sys.modules[_RxMap.__module__].__file__))},
        plan_sha256=sha256(plan_path.read_bytes()).hexdigest(), whitelist_sha256=plan['whitelist_sha256'])
    b._write(output/'result.json', result)
    b._write(output/'completed.json', dict(status='complete',
        endpoint_conditions=len(rows), expected_endpoint_conditions=plan['expected_endpoint_power_conditions'],
        endpoint_failures=result['endpoint_failures'], source_sha256=result['source_sha256'],
        result_sha256=sha256((output/'result.json').read_bytes()).hexdigest(),
        optimizers_started=0, future_measurements_read=False, true_RX_read=False,
        seconds=result['seconds']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({key:result[key] for key in ('total_endpoint_conditions', 'endpoint_failures',
        'primary_pair_distance', 'primary_pair_active_derivative',
        'qualified_candidate_active_response', 'qualified_candidate_voltage_PQ_response', 'counters', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
