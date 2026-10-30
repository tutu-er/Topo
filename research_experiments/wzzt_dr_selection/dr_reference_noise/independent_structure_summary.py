"""Independent effective-support counter for frozen outputs; no selection/solve."""
import argparse,json
from pathlib import Path
from collections import defaultdict,Counter

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def effective(p):return {tuple(s) for s,r,x in zip(p['supports'],p['r'],p['x'],strict=True) if max(r,x)>1e-9}
def summary(gen,run,out):
    target=read(Path(gen)/'evaluator_truth.json');cases={c['id']:c for c in target['cases']}
    results={p.stem:read(p) for p in (Path(run)/'records').glob('*.json') if not p.stem.endswith('_current_state')}
    rows=[];groups=defaultdict(list)
    for e in target['evaluations']:
        case=cases[e['case_id']];truth=set(map(tuple,e['target_family']));result=results[e['snapshot_id']]
        for name,m in result['methods'].items():
            row=dict(case=e['case_id'],snapshot=e['snapshot_id'],policy=e['policy'],method=name,status=m['status'],
                noise=case['noise_id'],c=case['truth']['c'],b=case['truth']['b'],pair=case['truth']['pair'],payment=e['actual_payment'])
            if m['status']=='complete':
                for part,p in (('raw',m['selected']['selected_uncontracted']),('final',m['selected'])):
                    support=effective(p);row[part+'_missing']=sorted(truth-support);row[part+'_false']=sorted(support-truth)
                    row[part+'_error']=len(support^truth);row[part+'_recovered']=support==truth
            rows.append(row);groups[(row['noise'],row['b'],row['c'],row['policy'],name)].append(row)
    table=[]
    for key,values in groups.items():
        valid=[v for v in values if v['status']=='complete']
        table.append(dict(noise=key[0],b=key[1],c=key[2],policy=key[3],method=key[4],planned=len(values),completed=len(valid),
            raw_error=sum(v['raw_error'] for v in valid),final_error=sum(v['final_error'] for v in valid),
            recovered=sum(v['final_recovered'] for v in valid)))
    by={(r['case'],r['policy'],r['method']):r for r in rows}
    contrast=defaultdict(Counter)
    for r in rows:
        if r['policy']!='pair_random' or r['status']!='complete':continue
        zero=by[(r['case'],'zero_sham',r['method'])]
        if zero['status']!='complete':contrast[r['method']]['unresolved']+=1;continue
        diff=r['final_error']-zero['final_error'];contrast[r['method']]['improved' if diff<0 else 'worse' if diff>0 else 'same']+=1
    output=dict(rows=rows,table=table,paid_vs_zero={k:dict(v) for k,v in contrast.items()},
        evaluation_rows=len(rows),statuses=dict(Counter(r['status'] for r in rows)),new_solvers=0)
    Path(out).write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','table')},indent=2))
    for t in table:
        if t['method'] in ('current_rebuilt_rnj','current_pool_wzzt','expanded_pool_wzzt','all_legal_family_reference'):
            print(json.dumps(t))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True);p.add_argument('--execution',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();summary(a.generation,a.execution,a.output)
