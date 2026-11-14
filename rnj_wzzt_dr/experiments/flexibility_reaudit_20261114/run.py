"""Longer-budget, paired resource audit using the frozen reactive experiment.

Only this isolated entry point changes runtime dispatch: previous sealed
feasible actions are added to the shared seed portfolio. Previous voltages,
truth, and execution outcomes are never read by the seed adapter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import hashlib
import json

import numpy as np

from ... import io
from ..measured_pq_flex_20261114 import runner as base
from ..reactive_flex_20261114 import runner as reactive


ORIGINAL_PORTFOLIO = reactive.build_portfolio


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def score_hash(score):
    return hashlib.sha256(json.dumps(score, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed prior sealed input: '+ref['path'])
    return Path(ref['path'])


def previous_points(case, score, forecast):
    """Read matching prior plans, map shared device identities, audit the new box."""
    root = Path(base.WORK['p']['previous_run']['path'])
    protocol = io.read(checked(base.WORK['p']['previous_protocol']))
    entries = io.read(checked(protocol['inputs']))
    current_score, current_forecast = score_hash(score), array_hash(forecast)
    new_ids = {(r['id'], int(r['node_label']), unit):i
        for i, (r, unit) in enumerate(zip(case.resources, case.action_units, strict=True))}
    points, records = {}, []
    for entry in entries:
        if entry.get('resource_variant') != 'eligible_sites':
            continue
        old_score = io.read(checked(entry['score']))
        if score_hash(old_score) != current_score:
            continue
        for folder in sorted((root/'fraction0.10'/entry['id']).glob('forecast*')):
            seal = io.read(folder/'seal.json')
            with np.load(checked(seal['planner']), allow_pickle=False) as saved:
                if array_hash(saved['forecast']) != current_forecast:
                    continue
                plans = io.read(checked(seal['plans']))
                old_config = io.read(folder/'resource_case.json')
                old_ids = [(r['id'], int(r['node_label']), unit)
                    for r, unit in zip(old_config['resources'], old_config['action_units'], strict=True)]
                for mode in protocol['modes']:
                    prior = plans[mode]
                    record = dict(mode=mode, planner=seal['planner'], plans=seal['plans'],
                        resource_config=io.ref(folder/'resource_case.json'), original_entry=entry['id'],
                        original_status=prior['status'], original_feasible=prior['feasible'],
                        original_cost_yuan=prior.get('cost_yuan'),
                        original_charged_seconds=prior.get('charged_total_seconds'),
                        current_score_hash=current_score, matching_forecast_sha256=current_forecast,
                        accepted=False)
                    if not prior['feasible']:
                        record['reason'] = 'Previous quote was not feasible'
                        records.append(record); continue
                    old_action = saved[mode].copy(); mapped = np.zeros(case.shape)
                    unavailable = []
                    for j, identity in enumerate(old_ids):
                        if identity in new_ids:
                            mapped[:, new_ids[identity]] = old_action[:, j]
                        elif np.max(abs(old_action[:, j]), initial=0.) > 1e-12:
                            unavailable.append(identity)
                    if unavailable:
                        record.update(reason='Active old device absent from current domain', unavailable=unavailable)
                    else:
                        audit = case.audit(mapped)
                        record.update(resource_audit=audit, mapped_cost_yuan=case.fee(mapped))
                        if audit['passed']:
                            name = 'previous30_'+mode
                            points[name] = mapped
                            record.update(accepted=True, seed_name=name,
                                reason='Shared devices mapped; current domain verified; optimizer still rechecks its evidence constraints')
                        else:
                            record['reason'] = 'Mapped action violates current resource contract'
                    records.append(record)
    return points, records


def extended_portfolio(case, score, forecast, base_kw, **kwargs):
    began = perf_counter(); budget = float(base.WORK['p'].get('shared_seed_seconds', 10.))
    prior, records = previous_points(case, score, forecast)
    loaded_seconds = perf_counter()-began
    options = dict(kwargs, seconds=max(.001, budget-loaded_seconds))
    points, receipt = ORIGINAL_PORTFOLIO(case, score, forecast, base_kw, **options)
    points.update(prior)
    elapsed = perf_counter()-began
    receipt.update(seconds=elapsed, budget_seconds=budget, budget_exceeded=elapsed > budget+.05,
        points=len(points), previous_plan_loading_seconds=loaded_seconds,
        previous_plan_records=records, previous_plans_added=len(prior),
        source='shared_resource_portfolio_plus_matching_previous_feasible_actions',
        previous_compute_cost='Prior 30s phase was already spent and is not included in the additional 180s budget; original charged times retained per prior quote',
        charge_rule='All current prior-plan loading, mapping, auditing and fresh seed generation charged to each method')
    return points, receipt


def initialize(out):
    base.initialize(out)
    reactive.build_portfolio = extended_portfolio


def prepare(args):
    protocol = reactive.prepare(args.index, args.ac_index, args.placements, args.out,
        seed=args.seed, forecasts=args.forecasts, repeats=args.repeats,
        seconds=args.seconds, workers=args.workers)
    prior = Path(args.previous_run).resolve()
    if not (prior/'complete.json').exists():
        raise ValueError('Prior phase must have a completion receipt')
    protocol.update(previous_run=dict(path=str(prior)), previous_protocol=io.ref(prior/'protocol.json'),
        previous_complete=io.ref(prior/'complete.json'),
        shared_seed_seconds=10.,
        source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        phase='paired re-evaluation of previously used backgrounds and forecast/response/meter streams',
        interpretation=f"{protocol['original_histories']} fixed previously evaluated historical input(s); "
            f"{len(protocol['placement_variants'])} resource variants share {args.forecasts*args.repeats} "
            'previously used conditional paths per history. This is a resource/budget comparison, not fresh-stream confirmation.',
        solver_scope='Same representative-model constraints; longer additional compute with prior plans retained when mapped feasible. All-native repair deferred.',
        budget_comparison_scope=f'{args.seconds:g}s additional planning phase includes current seed work; it excludes already-spent previous 30s phase',
        previous_plan_policy='Load only previous eligible_sites forecasts with identical frozen score and forecast; map exact device IDs/nodes/units, audit domain, then optimizer audits evidence')
    io.write(args.out/'protocol.json', protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path); parser.add_argument('--ac-index', type=Path)
    parser.add_argument('--placements', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous-run', type=Path, default=Path('outputs/reactive_flex_confirmation_20261114_v1'))
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--forecasts', type=int, default=4); parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=180.); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); args.out = args.out.resolve()
    if not args.resume: prepare(args)
    if not args.prepare_only:
        reactive.initialize = initialize
        completed = reactive.run(args.out)
        print({key: value for key, value in completed.items() if key != 'receipts'})


if __name__ == '__main__':
    main()
