"""Read-only complete denominators and actual device use for the resource audit."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from ... import io


def analyze(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    completion = io.read(run/'complete.json')
    if completion['status'] != 'complete':
        raise ValueError('A complete run is required')
    groups, baselines, devices = defaultdict(list), {}, defaultdict(set)
    for path in sorted(run.glob('fraction0.10/*/forecast*/results.json')):
        config = io.read(path.parent/'resource_case.json')
        rows = io.read(path)
        for row in rows:
            key = (row['id'], row['forecast_index'], row['repeat'])
            groups[row['resource_variant'], row['strategy']].append(row)
            if row['strategy'] == 'no_dfs':
                baselines[key] = row
        with np.load(path.parent/'planner.npz', allow_pickle=False) as plans:
            for strategy in {row['strategy'] for row in rows}:
                u = plans[strategy]
                for j, resource in enumerate(config['resources']):
                    if np.max(abs(u[:, j]), initial=0.) > 1e-4:
                        devices[rows[0]['resource_variant'], strategy].add(
                            (config['action_units'][j], resource['node_label']))
    records = []
    for (variant, strategy), rows in sorted(groups.items()):
        good = [r for r in rows if r['status'] == 'complete']
        paired = [(r, baselines[(r['id'], r['forecast_index'], r['repeat'])]) for r in good]
        paired = [(r,b) for r,b in paired if b['status'] == 'complete']
        def mean(name):
            return float(np.mean([r[name] for r in good])) if good else None
        records.append(dict(variant=variant, strategy=strategy, paths=len(rows), completed=len(good),
            failed=len(rows)-len(good), paired=len(paired),
            recovered=sum(r['native_map_correct'] for r in good),
            full_support_exact=sum(r['native_full_support_exact'] for r in good),
            added=sum(r['native_map_correct'] and not b['native_map_correct'] for r,b in paired),
            lost=sum(b['native_map_correct'] and not r['native_map_correct'] for r,b in paired),
            representative_native_correctness_disagreements=sum(r['map_correct']!=r['native_map_correct'] for r in good),
            mean_true_structure_weight=mean('p_true'),
            correct_weight_at_least90=sum(r['p_true'] >= .9 for r in good),
            wrong_weight_at_least90=sum(r['p_max'] >= .9 and not r['map_correct'] for r in good),
            settled_activation_yuan=mean('settled_fee_yuan'),
            delivered_abs_kwh=mean('delivered_abs_kwh'),
            delivered_abs_kvarh=mean('delivered_abs_kvarh'),
            active_requested_device_nodes=[dict(unit=unit,node=node) for unit,node in sorted(devices[variant,strategy])],
            voltage_min=min((r['all_bus_voltage_min'] for r in good),default=None),
            voltage_max=max((r['all_bus_voltage_max'] for r in good),default=None),
            voltage_excursion_paths=sum(r['voltage_outside_095_105_cells']>0 for r in good)))
    result=dict(source=io.ref(__file__),run=io.ref(run/'complete.json'),protocol=io.ref(run/'protocol.json'),
        interpretation='One previously used physical background, paired known forecast/meter streams; weights over structure representatives, final choice over all native points.',
        active_action_reporting_threshold=1e-4,records=records)
    out.mkdir(parents=True)
    io.write(out/'results.json',result)
    print(records)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.run,args.out)
