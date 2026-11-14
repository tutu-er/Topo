"""Compare sealed quote receipts across budgets, retaining every missing/failing task.

Only identical forecasts, frozen scores, numerical resource domains and model
requirements enter the same-problem cost comparison. Resource changes remain
listed separately. No optimization or AC is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ... import io


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def resource_signature(config):
    rows = []
    for r, unit in zip(config['resources'], config['action_units'], strict=True):
        hours = r.get('available_clock_hours')
        if hours is None or hours == list(range(24)): hours = [0, 24]
        row = dict(id=r['id'], node=int(r['node_label']), unit=unit, kind=r['kind'])
        if unit == 'kvar':
            row.update(capacity=r['capacity_kvar'], hours=hours,
                positive_fee=r['fee_positive_yuan_per_kvarh'], negative_fee=r['fee_negative_yuan_per_kvarh'],
                hourly_fee_multiplier=r.get('hourly_fee_multiplier', [1.]*24),
                reservation=r.get('reservation_fee_yuan_per_kvar_day', 0.))
        else:
            row.update(capacity=r['contract_kw'], qratio=r['q_over_p'],
                positive_fee=r['fee_positive_yuan_per_kwh'], negative_fee=r['fee_negative_yuan_per_kwh'])
        rows.append(row)
    return digest(dict(resources=sorted(rows, key=lambda r:r['id']), pricing=config['pricing'],
        time='08:00 start, 24 one-hour decisions, four quarter-hours each',
        P_dynamics='EV cumulative +/-2h and zero daily energy; thermal rho=.98 +/-1h with zero terminal; forecast PV cap'))


def collect(root, partial=False):
    root = Path(root).resolve(); protocol = io.read(root/'protocol.json')
    if not partial and not (root/'complete.json').exists():
        raise ValueError('Incomplete run; use --partial to preserve pending tasks explicitly: '+str(root))
    entries = io.read(protocol['inputs']['path']); rows = []
    completion_path = root/'complete_fraction0.10.json'
    completion = io.read(completion_path) if completion_path.exists() else {}
    for entry in entries:
        score_path = Path(entry['score']['path']); score = io.read(score_path)
        if io.ref(score_path)['sha256'] != entry['score']['sha256']:
            raise ValueError('Frozen score hash changed')
        for draw in range(protocol['forecasts']):
            folder = root/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            plans = io.read(folder/'plans.json') if (folder/'plans.json').exists() else {}
            if (folder/'seal.json').exists():
                seal = io.read(folder/'seal.json')
                for field in ('planner', 'plans', 'score', 'protocol'):
                    if io.ref(seal[field]['path'])['sha256'] != seal[field]['sha256']:
                        raise ValueError('Changed sealed quote '+str(folder)+'/'+field)
            config = io.read(folder/'resource_case.json') if (folder/'resource_case.json').exists() else None
            seeds = io.read(folder/'shared_initial_receipt.json') if (folder/'shared_initial_receipt.json').exists() else {}
            forecast_hash = None
            if (folder/'planner.npz').exists():
                with np.load(folder/'planner.npz', allow_pickle=False) as data:
                    forecast_hash = array_digest(data['forecast'])
            for mode in protocol['modes']:
                receipt = plans.get(mode, {}); initial = receipt.get('initial', {})
                row = dict(run=str(root), input=entry.get('original_id', entry['id']),
                    variant=entry.get('resource_variant', 'unspecified'), forecast=draw, mode=mode,
                    budget_seconds=protocol['solver_seconds'],
                    original_task_failures=[failure for failure in completion.get('failed_tasks', [])
                        if failure.get('id') == entry['id'] and failure.get('forecast_index') == draw],
                    budget_scope=protocol.get('budget_comparison_scope', 'Original phase; current seed generation charged'),
                    status=receipt.get('status', 'missing_quote'), native_status=receipt.get('native_status'),
                    feasible=bool(receipt.get('feasible', False)), source=receipt.get('source'),
                    cost_yuan=receipt.get('cost_yuan'), lower_bound_yuan=receipt.get('lower_bound_yuan'),
                    relative_gap=receipt.get('relative_gap'), global_optimality_verified=receipt.get('global_optimality_verified', False),
                    lower_bound_withdrawn=receipt.get('lower_bound_withdrawn', False),
                    native_infeasibility_withdrawn=receipt.get('native_infeasibility_withdrawn', False),
                    infeasibility_proved=receipt.get('infeasibility_proved', False),
                    zero_fallback=receipt.get('zero_fallback'),
                    charged_total_seconds=receipt.get('charged_total_seconds'),
                    budget_exceeded=receipt.get('charged_total_budget_exceeded'),
                    initial_cost_yuan=initial.get('cost_yuan'), initial_feasible=initial.get('feasible'),
                    initial_audit_seconds=initial.get('audit_seconds'),
                    reported_initial_improvement_yuan=receipt.get('improvement_from_verified_initial_yuan'),
                    seed_seconds=seeds.get('seconds'), seed_budget_seconds=seeds.get('budget_seconds'),
                    seed_budget_exceeded=seeds.get('budget_exceeded'),
                    seed_LP_status_counts=dict(Counter(str(lp.get('status')) for lp in seeds.get('LPs', []))),
                    previous_plan_records=seeds.get('previous_plan_records', []),
                    receipt=io.ref(folder/'plans.json') if plans else None,
                    problem=dict(forecast=forecast_hash, frozen_score=digest(score),
                        resources=resource_signature(config) if config else None,
                        requirements=dict(mode=mode, target_mass=receipt.get('target_mass'),
                            targets=receipt.get('targets'), risk=receipt.get('individual_comparison_risk'),
                            pair_count=receipt.get('pair_count'), guard_count=receipt.get('guard_count'),
                            constrained_bank=receipt.get('constrained_bank', 'representative'),
                            numerical_gate_buffer=receipt.get('numerical_gate_buffer', 1e-8))))
                if row['feasible'] and row['cost_yuan'] is None:
                    row['receipt_inconsistency'] = 'Feasible without reported feasible cost'
                rows.append(row)
    return dict(root=str(root), protocol=io.ref(root/'protocol.json'),
        complete=(root/'complete.json').exists(), rows=rows,
        original_failed_tasks=completion.get('failed_tasks', []),
        original_execution_failed_rows=completion.get('execution_failed_rows'))


def compare(base, new):
    identity = lambda r:(r['input'], r['variant'], r['forecast'], r['mode'])
    before = {identity(r):r for r in base['rows']}; after = {identity(r):r for r in new['rows']}
    result = []
    for key in sorted(set(before) | set(after)):
        old, now = before.get(key), after.get(key)
        row = dict(key=list(key), old=old, new=now, same_problem=False)
        if old is None or now is None:
            row['comparison_status'] = 'missing_matching_variant_or_task'
        elif old['status'] == 'missing_quote' or now['status'] == 'missing_quote':
            row['comparison_status'] = 'missing_quote'
        else:
            changes = [field for field in old['problem'] if old['problem'][field] != now['problem'][field]]
            row.update(problem_differences=changes, same_problem=not changes,
                comparison_status='same_problem' if not changes else 'different_problem_not_budget_effect')
            if not changes and old['feasible'] and now['feasible']:
                improvement = old['cost_yuan']-now['cost_yuan']
                row.update(feasible_cost_improvement_yuan=improvement,
                    relative_feasible_cost_improvement=None if old['cost_yuan'] == 0 else improvement/old['cost_yuan'],
                    old_upper_bound_retained=improvement >= -1e-7,
                    lower_bound_change_yuan=None if old['lower_bound_yuan'] is None or now['lower_bound_yuan'] is None
                        else now['lower_bound_yuan']-old['lower_bound_yuan'])
            elif not changes:
                row['feasibility_transition'] = [old['status'], now['status']]
        result.append(row)
    paired = [r for r in result if 'feasible_cost_improvement_yuan' in r]
    summary = dict(base_run=base['root'], new_run=new['root'],
        old_quote_count=len(base['rows']), new_quote_count=len(new['rows']),
        old_status=dict(Counter(r['status'] for r in base['rows'])),
        new_status=dict(Counter(r['status'] for r in new['rows'])),
        comparison_status=dict(Counter(r['comparison_status'] for r in result)),
        same_problem_feasible_pairs=len(paired),
        improved_cost_pairs=sum(r['feasible_cost_improvement_yuan'] > 1e-7 for r in paired),
        worse_cost_pairs=sum(r['feasible_cost_improvement_yuan'] < -1e-7 for r in paired),
        mean_feasible_cost_improvement_yuan=None if not paired else float(np.mean([r['feasible_cost_improvement_yuan'] for r in paired])),
        source_counts=dict(Counter(str(r['source']) for r in new['rows'])),
        original_failed_tasks=new['original_failed_tasks'],
        original_execution_failed_rows=new['original_execution_failed_rows'],
        note='Same problem does not imply independent from-scratch timing: inspect prior-plan records and additional-phase budget scope.')
    return dict(summary=summary, comparisons=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--new', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    base = collect(args.base, args.partial)
    result = dict(source=io.ref(__file__), base_protocol=base['protocol'],
        bounds_scope='Solver-reported bounds for the implemented QCQP with its numerical buffer; no bound for AC recovery or an unbuffered wider problem is inferred',
        comparisons=[compare(base, collect(path, args.partial)) for path in args.new])
    args.out.mkdir(parents=True); io.write(args.out/'report.json', result)
    lines = ['# 求解预算比较', '', '全部报价、不可行、未决和缺失记录均保留；仅相同数学问题比较费用。', '']
    for item in result['comparisons']:
        s = item['summary']
        lines += [f"- `{Path(s['new_run']).name}`：同问题可行配对 {s['same_problem_feasible_pairs']}，"
            f"降费 {s['improved_cost_pairs']}，升费 {s['worse_cost_pairs']}；新状态 `{s['new_status']}`。"]
    lines += ['', '载入旧30秒方案的180秒运行属于追加计算；旧阶段成本另留在收据，不能称独立从零180秒对照。', '']
    (args.out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps([item['summary'] for item in result['comparisons']], ensure_ascii=False))


if __name__ == '__main__':
    main()
