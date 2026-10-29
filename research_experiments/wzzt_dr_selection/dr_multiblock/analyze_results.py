"""Post-fit summaries only; no planning or identifier calls."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np

METHODS=('rnj_bank','baseline25','multistart_original_rule')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def analyze(root):
    root=Path(root);summary=read(root/'summary.json');planning=read(root/'planning.json')
    records=[read(p) for p in sorted((root/'records').glob('*.json'))]
    groups=summary['groups'];comparisons=[]
    for law in ('local','common'):
        single=[g for g in groups if g['law']==law and g['policy'].startswith('single_') and g['complete']]
        for method in METHODS:
            best=min(single,key=lambda g:(g['models'][method]['mean_errors'],g['mean_payment'],g['policy'])) if len(single)==41 else None
            selected=planning['selected'][law]['single']['plan']
            selected_single=next((g for g in single if g['plan']==selected),None)
            baseline=next((g for g in single if not any(g['plan'])),None)
            variants=[('no_reward',baseline),('planned_single',selected_single),('posthoc_best_fixed_single',best)]
            variants += [(name,next((g for g in groups if g['law']==law and g['policy']==name and g['complete']),None))
                         for name in ('stochastic_sync','stochastic_free','mean_free')]
            for label,g in variants:
                if g is None:continue
                rr=[r for r in records if r['phase']=='main' and r['law']==law and r['policy']==g['policy'] and r['status']=='evaluated']
                comparisons.append(dict(law=law,method=method,comparison=label,policy=g['policy'],plan=g['plan'],
                    **g['models'][method],mean_payment=g['mean_payment'],denominator=len(rr),
                    mean_participation_rank=float(np.mean([np.linalg.matrix_rank(np.asarray(r['participation'])[:2]) for r in rr])),
                    mean_successful_training_offers=float(np.mean([np.asarray(r['participation'])[:2].sum() for r in rr])),
                    origins=dict(Counter(r['evaluation']['models'][method]['selected_origin'] for r in rr))))
    controls=[dict(key=r['key'],case=r['case'],policy=r['policy'],plan=r['plan'],status=r['status'],
        models={m:r.get('evaluation',{}).get('models',{}).get(m,{}) for m in METHODS}) for r in records if r['phase']=='control']
    unique={r['observation_key']:r for r in records if r['status']=='evaluated'}
    different=[];mechanisms=[]
    for key,r in unique.items():
        e=r['evaluation']['models']
        errors=[e[m]['final']['all_block_errors'] for m in METHODS]
        if len(set(errors))>1:different.append(dict(key=key,representative=r['key'],errors=dict(zip(METHODS,errors))))
        for name,path in r['evaluation']['paths'].items():
            if path['accepted_steps']:
                mechanisms.append(dict(key=key,representative=r['key'],path=name,**path))
    result=dict(status=summary['status'],main_cells=sum(r['phase']=='main' for r in records),
        control_cells=len(controls),unique_evaluated_fits=len(unique),comparisons=comparisons,controls=controls,
        model_difference_unique_fits=different,accepted_paths=mechanisms,
        primary_endpoint='all effective block false + missing count',
        best_single_rule='fixed schedule minimizing 8-path mean errors, then actual mean payment, then policy order; posthoc reference only',
        conditional_scope='Synthetic independent DR participation paths given fixed natural load and zero meter noise',
        source_hashes={str(root/n):sha(root/n) for n in ('protocol.json','planning.json','summary.json')},
        analysis_source_sha256=sha(__file__))
    (root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=analyze(a.root)
    print(json.dumps({k:r[k] for k in ('status','main_cells','control_cells','unique_evaluated_fits')}))
    print(json.dumps([x for x in r['comparisons'] if x['method']=='multistart_original_rule'],ensure_ascii=False))
