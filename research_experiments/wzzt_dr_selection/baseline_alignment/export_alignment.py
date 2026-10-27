"""Summarize finished default-baseline paths without running any optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(family):return [sorted(s) for s in sorted(family,key=lambda s:(len(s),tuple(sorted(s))))]
def score(labels,r,x,truth,n):
    active={frozenset(s) for s,a,b in zip(labels,r,x) if a>1e-9 or b>1e-9}
    p={s for s in active if 1<len(s)<n};t={s for s in truth if 1<len(s)<n}
    return {'internal_matched':len(p&t),'internal_true':len(t),'internal_predicted':len(p),
        'missing':serial(t-p),'extra':serial(p-t),'complete_effective_family':active==truth}
def path_rows(points,truth,n):
    rows=[]
    for i,p in enumerate(points):
        fam={frozenset(s) for s in p['support_labels']}
        blocked=[t for t in truth if 1<len(t)<n and any(bool(t&s) and not(t<=s or s<=t) for s in fam)]
        rows.append({'index':i,'supports':p['support_labels'],'r_values':p['r_values'],'x_values':p['x_values'],
            'train_mae':p['train_mae'],'validation_mae':p['validation_mae'],'validation_se':p['validation_se'],
            'recovery':score(p['support_labels'],p['r_values'],p['x_values'],truth,n),
            'true_internal_blocks_excluded_by_current_family':serial(blocked)})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=a.repo.resolve();root=a.runs.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'rnj_wzzt_core'))
    from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import _plain,_write_json
    n0=root/'N0_default_reference';n1=root/'N1_iflex_no_reward'
    if not (n0/'completion.json').exists() or not (n1/'summary.json').exists():raise ValueError('Wait for both recorded baseline conditions; do not relabel partial paths complete')
    with (n0/'prepared_case.pkl').open('rb') as f:case,_=pickle.load(f)
    truth=set(case.truth_clades)|{frozenset([t]) for t in case.terminals}
    f0=read(n0/'fit.json');f1=read(n1/'wzzt.json');s0=read(n0/'cli/metrics.json');s1=read(n1/'summary.json')
    records=[]
    for name,fit,selection in [('N0',f0,read(n0/'selection.json')),('N1',f1,read(n1/'rnj.json'))]:
        summary=s0['milp'][0] if name=='N0' else fit['summary']
        chosen=fit['selected_path_index'] if name=='N0' else summary['selected_path_index']
        rows=path_rows(fit['path'],truth,15)
        record={'condition':name,'training_rows':288 if name=='N0' else fit['training_time_rows'],
            'validation_rows':288 if name=='N0' else fit['validation_time_rows'],
            'rnj':selection,'selected_path_index':chosen,'selected':rows[chosen],
            'path':rows,'attempts':fit['attempted_extensions'],
            'stop_reason':fit['stop_reason'] if name=='N0' else summary['stop_reason'],
            'status':('incomplete' if fit['stop_reason'] in {'extension_not_proven_optimal','extension_without_incumbent','bound_expansion_limit'} else 'complete') if name=='N0' else fit['status'],
            'maximum_path_internal_matches':max(r['recovery']['internal_matched'] for r in rows),
            'minimum_path_train_mae':min(r['train_mae'] for r in rows),
            'minimum_path_validation_mae':min(r['validation_mae'] for r in rows)}
        records.append(record)
    files=[n0/'protocol.json',n0/'fit.json',n0/'selection.json',n0/'cli/metrics.json',n0/'completion.json',
        n1/'protocol.json',n1/'wzzt.json',n1/'summary.json',n1/'evaluation.json']
    result={'scope':'Two no-reward baseline conditions; no price comparison; same current production algorithm defaults',
        'condition_denominator':2,'completed_searches':sum(r['status']=='complete' for r in records),
        'full_recoveries':sum(r['selected']['recovery']['complete_effective_family'] for r in records),
        'records':records,'inputs_unchanged_in_N1':s1['source_input_hashes_unchanged'],
        'retained_private_file_sha256':{str(p):sha(p) for p in files},
        'exporter_sha256':sha(Path(__file__)),'new_optimizer_calls_by_export':0}
    _write_json(out/'alignment_results.json',_plain(result))
    _write_json(out/'normal_cli_metrics.json',s0)
    _write_json(out/'normal_cli_protocol.json',read(n0/'protocol.json'))
    _write_json(out/'iflex_protocol.json',read(n1/'protocol.json'))
    print(json.dumps({r['condition']:{'status':r['status'],'stop':r['stop_reason'],'recovery':r['selected']['recovery']} for r in records},indent=2))

if __name__=='__main__':main()
