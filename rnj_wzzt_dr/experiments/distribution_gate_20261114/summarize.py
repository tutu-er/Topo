"""Compact paired results; preserves failed quotes and conditional-law scopes."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def compact(audit_path,probability_path):
    a,p=read(audit_path),read(probability_path)
    if a['status']!='passed' or p['status']!='complete':
        raise ValueError('Finished independently audited results required')
    groups={}
    for name,g in a['groups'].items():
        groups[name]=dict(n=g['rows'],correct=g['native_correct'],full_support_exact=g['native_full_support_exact'],
            true_high=g['true_high'],wrong_high=g['wrong_high'],mean_true_weight=g['mean_true_weight']['mean'],
            planned_total_yuan=g['planned_cost']['total_yuan']['mean'],
            settled_total_yuan=g['settled_cost']['total_yuan']['mean'],
            settled_activation_yuan=g['settled_cost']['activation_yuan']['mean'],
            reserved_P_kw=g['planned_cost']['reserved_P_kw']['mean'],
            reserved_Q_kvar=g['planned_cost']['reserved_Q_kvar']['mean'],
            delivered_P_kwh=g['delivered']['P']['absolute_energy']['mean'],
            delivered_Q_kvarh=g['delivered']['Q']['absolute_energy']['mean'],
            voltage_range=[g['all_bus_voltage']['minimum'],g['all_bus_voltage']['maximum']],
            voltage_excursion_paths=g['voltage_excursion_paths'],
            clean_increment=g['evidence_increment']['clean_logodds']['mean'],
            observed_increment=g['evidence_increment']['observed_logodds']['mean'])
    probability=[]
    for item in p['results']:
        for row in item['rows']:
            probability.append({k:row[k] for k in ('forecast','law','target_native','policy','n',
                'target_family_winner_count','target_family_winner_rate','representative_success_rate',
                'theoretical_representative_success_lower','new_gate_feasible','anchor_fallback')})
    quotes=[]
    for q in a['quotes']:
        quotes.append({k:q.get(k) for k in ('forecast','policy','status','new_gate_feasible','anchor_fallback',
            'source','optimizer_invoked','native_status','cost_yuan','lower_bound_yuan','relative_gap',
            'new_quote_wall_seconds','total_method_seconds','infeasibility_proved')})
    return dict(audit_source=ref(audit_path),probability_source=ref(probability_path),
        audited_rows=a['audited_rows'],failed_tasks=len(a['original_failed_tasks']),
        quote_exceptions=a['original_quote_exceptions'],execution_failed_rows=a['original_execution_failed_rows'],
        groups=groups,comparisons=a['comparisons'],quotes=quotes,probability=probability,
        per_forecast={k:dict(correct={s:g['native_correct'] for s,g in v['groups'].items()},
            pair_moments=v['pair_moments']) for k,v in a['per_forecast'].items()},
        electrical_maximum_errors=a['electrical_maximum_errors'])


def cross_compare(left_audit,right_audit,policy='normal_fixed'):
    left,right=read(left_audit),read(right_audit)
    def lookup(data):
        return {(r['forecast'],r['repeat']):r for r in data['records'] if r['strategy']==policy}
    l,r=lookup(left),lookup(right)
    if l.keys()!=r.keys():raise ValueError('Paired world identities differ')
    # Reuse the already independently scored paired comparison, without new fits.
    from ..flexibility_stability_20261114.analyze import paired
    return paired(list(l.values()),list(r.values()))


def run(first,second,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    native=compact(first[0],first[1]);line=compact(second[0],second[1])
    result=dict(native=native,line=line,line_normal_vs_native_normal=cross_compare(second[0],first[0]),
        scope='Two development replays on the same one physical day, four forecasts and128 paired response/meter paths. Declared Gaussian simulations are separate; no independent-day confirmation.')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    rows=['|运行/策略|内部完整恢复|错误且权重≥90%|平均真权重|实际总费，元|实际P，kWh|实际Q，kvarh|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for label,data in [('native',native),('line',line)]:
        for name,g in data['groups'].items():
            rows.append(f"|{label}/{name}|{g['correct']}/{g['n']}|{g['wrong_high']}|{g['mean_true_weight']:.6f}|{g['settled_total_yuan']:.6f}|{g['delivered_P_kwh']:.6f}|{g['delivered_Q_kvarh']:.6f}|")
    (out/'table.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-audit',type=Path,required=True);parser.add_argument('--native-probability',type=Path,required=True)
    parser.add_argument('--line-audit',type=Path,required=True);parser.add_argument('--line-probability',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=run((args.native_audit,args.native_probability),(args.line_audit,args.line_probability),args.out)
    print(dict(native_rows=result['native']['audited_rows'],line_rows=result['line']['audited_rows'],out=str(args.out)))
