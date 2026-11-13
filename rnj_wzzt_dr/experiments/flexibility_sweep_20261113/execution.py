"""Paired original-AC execution after sealing every procurement policy."""
from __future__ import annotations

import atexit
import os
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import load
from ...dr.selection_mse import choose_mse
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence, solve
from ..linear_ac_population_20261113.study import evaluate
from .fitting import calibrated_means
from .simulation import runtime, background, forecast, response, observe


WORK = {}


def initialize(run):
    import gurobipy as gp
    out = Path(run)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    atexit.register(env.dispose)
    WORK.update(out=out, protocol=io.read(out/'protocol.json'), env=env, module=runtime(out))


def native_selection(score, observation, truth):
    power = np.column_stack((observation.p, observation.q))
    bank = score['native_bank']
    loss = np.array([m['history_sse']+np.square(observation.y-power@np.asarray(m['H']).T).sum()
                     for m in bank])
    chosen = choose_mse(loss/(score['cells']+observation.y.size), [m['internal'] for m in bank],
        K=1/score['reference_mse'], epsilon_scaled=score['validation_tie'])['selected']
    selected = bank[chosen]
    a, b = set(map(tuple, selected['active_clades'])), set(map(tuple, truth['truth_clades']))
    return dict(native_map_correct=selected['topology_id'] == truth['truth_id'],
                native_missing=len(b-a), native_extra=len(a-b), native_selected_index=chosen)


def margins(log_probability, target, runner):
    other = [j for j in range(len(log_probability)) if j != target]
    return dict(target_runner_logodds=None if runner is None else float(log_probability[target]-log_probability[runner]),
                target_minimum_logodds=None if not other else float(log_probability[target]-max(log_probability[other])))


def fixed_rotation(case):
    """One candidate-independent, energy-feasible shifting/curtailment pattern."""
    direction = np.where((np.arange(24)[:, None]//2+np.arange(case.R)[None, :]) % 2 == 0, 1., -1.)
    pattern, receipt = case.linear_optimum(direction, least_fee=True)
    if case.fee(pattern) <= 0:
        raise ArithmeticError('The fixed rotation has no payable activation')
    return pattern, receipt


def run_case(task):
    out, protocol = WORK['out'], WORK['protocol']
    entry, phase, days = task['entry'], task['phase'], task['days']
    directory = out/phase/entry['id']
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory/'complete.json'
    if complete.exists():
        return io.read(complete)
    if (directory/'started.json').exists():
        raise RuntimeError('An unfinished execution lock must be reviewed before resuming')
    io.write(directory/'started.json', dict(pid=os.getpid(), entry=entry, days=days))
    began = perf_counter()
    if io.ref(entry['score']['path'])['sha256'] != entry['score']['sha256']:
        raise ValueError('Frozen candidate score changed')
    s = io.read(entry['score']['path'])
    H = np.asarray([p['H'] for p in s['parameters']])
    target = s['map_index']
    competitors = [j for j in range(len(H)) if j != target]
    runner = max(competitors, key=lambda j: s['models'][j]['log_probability']) if competitors else None
    validation = load(entry['observed_refs']['validation']['path'])
    means = calibrated_means(s, validation)
    case = CustomerCase(io.read(out/'resource_configs'/f"{entry['adoption']}.json"))
    pattern, rotation_audit = fixed_rotation(case)
    truth = io.read(out/'offline_truth.json')
    rows = []
    for day in range(days):
        d = directory/f'day{day:03d}'
        d.mkdir(exist_ok=True)
        replicate = (100000 if phase == 'development' else 200000)+entry['index']*100+day
        common = dict(id=entry['id'], phase=phase, day=day, replicate=replicate,
            adoption=entry['adoption'], history_points=entry['history_points'],
            voltage_sigma=entry['voltage_sigma'], profile=entry['profile'], history_seed=entry['history_seed'],
            historical_p_map=s['p_map'], target=target, runner=runner,
            qualification=entry['qualification'], historical_truth_present=entry['truth_present'])
        controls, plans = {'no_dfs': np.zeros(case.shape)}, {}
        try:
            b = background(WORK['module'], case, entry, replicate)
            prediction = forecast(b, replicate)
            if competitors:
                evidence = CandidateEvidence(case, H, target, competitors, prediction, s['sigma2'],
                                             b['network']['base_kw'], means)
            for k, gamma in enumerate(protocol['gamma']):
                name = f'g{k}'
                tick = perf_counter()
                try:
                    if competitors:
                        result, point = solve(evidence, gamma, seconds=protocol['solver_seconds'], env=WORK['env'])
                    else:
                        result, point = dict(feasible=True, status='single_candidate_zero', cost_yuan=0.,
                            lower_bound_yuan=0., global_optimality_verified=True, seconds=0., source='zero'), np.zeros(case.shape)
                except Exception as exc:
                    result, point = dict(feasible=False, status='solver_failed_zero_fallback',
                        error=repr(exc), traceback=traceback.format_exc(), seconds=perf_counter()-tick,
                        cost_yuan=None, lower_bound_yuan=None, global_optimality_verified=False), None
                result['gamma'] = gamma
                controls['optimized_'+name] = np.zeros(case.shape) if point is None else point
                fee = 0. if point is None else case.fee(point)
                fraction = min(1., fee/case.fee(pattern))
                controls['rotation_'+name] = fraction*pattern
                result.update(rotation_planned_fee_yuan=case.fee(controls['rotation_'+name]),
                              rotation_same_planned_fee=abs(case.fee(controls['rotation_'+name])-fee) <= 1e-7)
                plans[name] = result
            # Every control is frozen before either true execution voltage or
            # actual response multiplier is opened to the evaluator.
            np.savez_compressed(d/'planner.npz', forecast=prediction, **controls)
            io.write(d/'plans.json', plans)
            io.write(d/'policy_seal.json', dict(planner=io.ref(d/'planner.npz'), plans=io.ref(d/'plans.json'),
                score=entry['score'], all_other_candidates_constrained=competitors,
                planner_receives_truth=False, planner_receives_true_lines_or_voltage=False))
            eta = response(case.R, replicate)
            np.savez_compressed(d/'physical_background.npz', P=b['p'], Q=b['q'], root=b['root'],
                meter_p=b['noise'][0], meter_q=b['noise'][1], meter_v=b['noise'][2],
                meter_root=b['noise'][3], response_multiplier=eta)
            io.write(d/'generation.json', b['metadata'])
        except Exception as exc:
            rows.append(dict(common, strategy='all', status='planning_or_background_failed',
                             error=repr(exc), traceback=traceback.format_exc()))
            io.write(directory/'results.json', rows)
            continue
        for name, u in controls.items():
            plan = None if name == 'no_dfs' else plans[name.split('_')[1]]
            info = dict(common, strategy=name, planning_feasible=None if plan is None else plan['feasible'],
                planned_fee_yuan=case.fee(u), settled_fee_yuan=case.fee(u*eta[None]),
                delivered_abs_kwh=float(case.dt*np.abs(u*eta[None]).sum()),
                max_requested_kw=float(np.max(abs(u))), max_delivered_kw=float(np.max(abs(u*eta[None]))))
            try:
                observation, audit, physical = observe(WORK['module'], b, u, case, eta)
                values = evaluate(s, observation, truth, target)
                ll = values.pop('log_probability')
                values.update(margins(ll, target, runner))
                values.update(native_selection(s, observation, truth))
                info.update(status='complete', **values, **audit)
                np.savez_compressed(d/(name+'.npz'), P=observation.p, Q=observation.q, V=observation.voltage,
                    root=observation.root, labels=observation.labels, log_probability=ll,
                    true_P=physical[0], true_Q=physical[1], true_V=physical[2])
            except Exception as exc:
                info.update(status='AC_or_scoring_failed', error=repr(exc), traceback=traceback.format_exc())
            rows.append(info)
        io.write(directory/'results.json', rows)
    receipt = dict(id=entry['id'], phase=phase, days=days, status='complete',
        rows=len(rows), failed_rows=sum(r['status'] != 'complete' for r in rows),
        seconds=perf_counter()-began, results=io.ref(directory/'results.json'), fixed_rotation_audit=rotation_audit)
    io.write(complete, receipt)
    return receipt
