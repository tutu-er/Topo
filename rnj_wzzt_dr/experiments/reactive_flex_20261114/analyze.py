"""Summarize paired placements, keeping native P and Q quantities separate."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import numpy as np
from ... import io
from ..measured_pq_flex_20261114.analyze import analyze as base_analyze, stats
from ..measured_pq_20261114.physical import offline_truth
from ..posterior_mse_20261112.scoring import topology_id


def analyze(run, out):
    run,out=Path(run).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Keep completed summaries')
    out.mkdir(parents=True)
    base=base_analyze(run,out/'full')
    rows=io.read(out/'full/rows_with_bias_split.json')
    entries=io.read(run/'inputs.json');p=io.read(run/'protocol.json')
    truths={}
    for entry in entries:
        score=io.read(entry['score']['path']);labels=score['labels']
        supports=offline_truth(entry['case'],labels)
        family=topology_id(tuple(sorted(tuple(sorted(labels[i] for i in group))
                    for group in supports if 1<len(group)<len(labels))))
        truths[entry['id']]={i for i,m in enumerate(score['native_bank']) if m['topology_id']==family}
    groups=defaultdict(list)
    for row in rows:groups[row['resource_variant'],row['strategy']].append(row)
    summaries={}
    for (variant,strategy),selected in groups.items():
        complete=[r for r in selected if r['status']=='complete']
        true_gain=[];pair_identity=[]
        for r in complete:
            pairs=[e for e in r['evidence'].get('target_pairs',[]) if e['competitor'] in truths[r['id']]]
            if pairs and not r['historical_target_correct']:
                # One frozen true parameter point in this enriched input.
                if len(pairs)!=1:raise ValueError('Specify the true parameter comparison when multiple exist')
                true_gain.append(-pairs[0]['actual_logodds_change'])
                pair_identity.append((pairs[0]['competitor'],pairs[0]['target']))
        record=dict(paths=len(selected),complete=len(complete),failed=len(selected)-len(complete),
            recovered=sum(r.get('native_map_correct',False) for r in complete),
            truth_probability=stats([r['p_true'] for r in complete]),
            true_probability_at_least90=sum(r['p_true']>=.9 for r in complete),
            historical_target_correct_count=sum(r['historical_target_correct'] for r in complete),
            true_vs_historical_logodds_gain=stats(true_gain),true_vs_historical_pairs=sorted(set(pair_identity)),
            quantity={key:stats([r['delivered_quantities'][key] for r in complete]) for key in
                ('active_abs_kwh','reactive_abs_kvarh','active_yuan','reactive_yuan','total_yuan')},
            voltage_min=min((r['all_bus_voltage_min'] for r in complete),default=None),
            voltage_max=max((r['all_bus_voltage_max'] for r in complete),default=None),
            voltage_excursion_paths=sum(r['voltage_outside_095_105_cells']>0 for r in complete),
            root_energy_value_change_yuan=stats([r['root_energy_value_change_yuan'] for r in complete]))
        summaries[variant+'/'+strategy]=record
    quotes=defaultdict(list)
    for file in run.glob('fraction*/*/forecast*/plans.json'):
        variant=next(e['resource_variant'] for e in entries if e['id']==file.parent.parent.name)
        for mode,q in io.read(file).items():quotes[variant+'/'+mode].append(q)
    quote_summary={key:dict(n=len(q),status=dict(Counter(r['status'] for r in q)),
        source=dict(Counter(r.get('source','none') for r in q)),
        native_status=dict(Counter(str(r.get('native_status')) for r in q)),
        cost=stats([r.get('cost_yuan') for r in q]),lower=stats([r.get('lower_bound_yuan') for r in q]),
        gap=stats([r.get('relative_gap') for r in q]),
        seconds=stats([r.get('charged_total_seconds') for r in q]),
        initial_cost=stats([r.get('initial',{}).get('cost_yuan') for r in q]),
        initial_improvement=stats([r.get('improvement_from_verified_initial_yuan') for r in q]),
        proved_optimal=sum(r.get('global_optimality_verified',False) for r in q),
        budget_exceeded=sum(r['charged_total_budget_exceeded'] for r in q)) for key,q in quotes.items()}
    result=dict(run=io.ref(run/'complete.json'),protocol=io.ref(run/'protocol.json'),source=io.ref(__file__),
        full_analysis=io.ref(out/'full/SUMMARY.json'),fixed_network_day_histories=p['original_histories'],
        configuration_count=len(p['placement_variants']),forecasts=p['forecasts'],repeats=p['repeats'],
        paired_random_paths_per_history=p['forecasts']*p['repeats'],strategy_rows=len(rows),
        interpretation='Positions share forecast, meter and per-resource-index response streams; one physical background is not multiple independent field days.',
        summaries=summaries,quotes=quote_summary,source_unchanged=base['source_unchanged'])
    io.write(out/'SUMMARY.json',result)
    io.write(out/'complete.json',dict(summary=io.ref(out/'SUMMARY.json')))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=analyze(a.run,a.out)
    print({k:dict(recovered=v['recovered'],paths=v['paths'],mean_true_weight=v['truth_probability']['mean'])
           for k,v in r['summaries'].items()})
