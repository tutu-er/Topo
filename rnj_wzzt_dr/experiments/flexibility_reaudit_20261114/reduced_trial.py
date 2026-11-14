"""One additional exact-reformulation quote; never run AC or alter the main batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np

from ... import io
from ...observations import load
from ..absolute_identification_20261114.solver import SymmetricRecovery
from ..reactive_flex_20261114.domain import PQIncrementCase
from . import reduced_solver


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed sealed input: '+ref['path'])
    return Path(ref['path'])


def prepare(run_root, output, *, seconds=180., forecast=0, input_id=None):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not (run_root/'complete.json').exists():
        raise RuntimeError('Wait until the full long-budget batch is complete')
    source_protocol = io.read(run_root/'protocol.json')
    entries = io.read(checked(source_protocol['inputs']))
    selected = [e for e in entries if e['resource_variant'] == 'eligible_sites'
        and (input_id is None or e['original_id'] == input_id)]
    if len(selected) != 1: raise ValueError('Select exactly one eligible_sites input')
    entry = selected[0]; folder = run_root/'fraction0.10'/entry['id']/f'forecast{forecast:02d}'
    seal = io.read(folder/'seal.json')
    for key in ('planner', 'plans', 'score', 'protocol'): checked(seal[key])
    original_plans = io.read(seal['plans']['path'])
    prior = original_plans['symmetric_guard']
    if not prior.get('feasible'):
        raise ValueError('Specified original180 quote has no verified incumbent to retain')
    output.mkdir(parents=True)
    package = Path(__file__).resolve().parents[2]
    source_paths = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, '__file__', None) and str(m.__file__).endswith('.py')
        and Path(m.__file__).resolve().is_relative_to(package)})
    sources = []
    for path in source_paths:
        sources.append(io.ref(path)); destination = output/'source_snapshot'/path.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        phase='Single additional optimization-only exact-reformulation comparison',
        main_run=io.ref(run_root/'complete.json'), main_protocol=io.ref(run_root/'protocol.json'),
        entry=entry, forecast_index=forecast, base_kw=200., solver_seconds=seconds,
        mode='symmetric_guard', planner=seal['planner'], plans=seal['plans'],
        previous_best=prior, previous_shared_points=io.ref(folder/'shared_initial_points.npz'),
        previous_shared_receipt=io.ref(folder/'shared_initial_receipt.json'), sources=sources,
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        budget_scope='Additional solve budget includes loading/auditing saved seeds and case/model construction; original30 and original180 work is already spent and excluded',
        proof_scope='Equivalent buffered QCQP for the same frozen representative bank, resources, forecast, target set, risk and guards',
        not_part_of_main_recovery_denominator=True, new_AC=False, new_observations=False,
        altered_main_batch=False)
    io.write(output/'protocol.json', protocol)
    return protocol


def solve(output):
    import gurobipy as gp
    output = Path(output).resolve(); protocol = io.read(output/'protocol.json')
    if (output/'started.json').exists() or (output/'complete.json').exists():
        raise RuntimeError('Preserve existing or unfinished one-quote trial')
    for ref in protocol['sources']: checked(ref)
    for name in ('planner', 'plans', 'previous_shared_points'): checked(protocol[name])
    entry = protocol['entry']
    for name in ('score', 'resource_context'): checked(entry[name])
    checked(entry['files']['validation.npz'])
    env_began = perf_counter(); env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    environment_setup_seconds = perf_counter()-env_began
    io.write(output/'started.json', dict(created_utc=datetime.now(timezone.utc).isoformat()))
    began = perf_counter()
    try:
        score = io.read(entry['score']['path']); context = io.read(entry['resource_context']['path'])
        validation = load(entry['files']['validation.npz']['path'])
        with np.load(protocol['planner']['path'], allow_pickle=False) as data:
            forecast = data['forecast'].copy(); previous = data['symmetric_guard'].copy()
        with np.load(protocol['previous_shared_points']['path'], allow_pickle=False) as data:
            seeds = {name:data[name].copy() for name in data.files}
        seeds['original180_best_symmetric'] = previous.copy()
        case = PQIncrementCase(context, score['labels'], forecast, protocol['base_kw'],
            q_resources=entry['q_resources'], fraction=.1)
        evidence = SymmetricRecovery(case, score, validation, forecast, protocol['base_kw'], 'symmetric_guard')
        previous_direct = evidence.verify_expansion(previous); previous_domain = case.audit(previous)
        if not previous_direct['feasible'] or not previous_domain['passed']:
            raise ArithmeticError('Original180 incumbent fails unchanged original checks')
        np.savez_compressed(output/'loaded_initial_points.npz', **seeds)
        seed_and_case_seconds = perf_counter()-began
        receipt, point = reduced_solver.plan(case, score, validation, forecast, protocol['base_kw'],
            mode='symmetric_guard', seconds=max(.001, protocol['solver_seconds']-seed_and_case_seconds),
            env=env, seed_points=seeds)
        elapsed = perf_counter()-began
        prior_cost = case.fee(previous)
        receipt.update(charged_total_seconds=elapsed, seed_and_case_seconds=seed_and_case_seconds,
            charged_total_budget_exceeded=elapsed > protocol['solver_seconds']+.05,
            environment_setup_seconds_outside_quote=environment_setup_seconds,
            initial_points=io.ref(output/'loaded_initial_points.npz'), previous_180_cost_yuan=prior_cost,
            previous_180_direct=previous_direct, previous_180_resource_audit=previous_domain,
            no_new_AC=True, not_part_of_main_recovery_denominator=True)
        if point is not None:
            direct, domain = evidence.verify_expansion(point), case.audit(point)
            if not direct['feasible'] or not domain['passed']:
                raise ArithmeticError('Reduced result fails original checks')
            np.savez_compressed(output/'plan.npz', forecast=forecast, action=point)
            receipt.update(plan=io.ref(output/'plan.npz'),
                action_identical_to_original_180=bool(np.allclose(point, previous, atol=1e-8, rtol=1e-8)),
                max_P_action_difference_kw=float(np.max(np.abs(point[:, :case.P_R]-previous[:, :case.P_R]), initial=0.)),
                max_Q_action_difference_kvar=float(np.max(np.abs(point[:, case.P_R:]-previous[:, case.P_R:]), initial=0.)),
                fee_improvement_over_original_180_yuan=prior_cost-case.fee(point),
                original_180_incumbent_retained_or_improved=case.fee(point) <= prior_cost+1e-7)
        io.write(output/'receipt.json', receipt)
        io.write(output/'complete.json', dict(status='complete', receipt=io.ref(output/'receipt.json'),
            source_unchanged=all(io.ref(r['path'])['sha256'] == r['sha256'] for r in protocol['sources'])))
        return receipt
    except Exception as exc:
        io.write(output/'failure.json', dict(error=repr(exc), seconds=perf_counter()-began))
        raise
    finally:
        env.dispose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180.); parser.add_argument('--forecast', type=int, default=0)
    parser.add_argument('--input-id'); parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if not args.resume: prepare(args.run, args.out, seconds=args.seconds, forecast=args.forecast, input_id=args.input_id)
    if not args.prepare_only:
        result = solve(args.out)
        print({key:result.get(key) for key in ('status', 'source', 'cost_yuan', 'lower_bound_yuan',
            'relative_gap', 'charged_total_seconds', 'fee_improvement_over_original_180_yuan')})
