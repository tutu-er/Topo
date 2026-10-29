"""Post-fit structure attribution; no fitting, price choice or simulation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def analyze(root):
    root=Path(root);summary=read(root/'summary.json')
    if summary['status']!='complete':raise RuntimeError('analysis requires the complete frozen batch')
    rows=[read(p) for p in sorted((root/'records').glob('*.json'))]
    unique={}
    for row in rows:
        if row['status']=='evaluated':unique.setdefault(row['observation_key'],row)
    details=[];accepted=[];crossing=[];lp_count=0
    for key,row in unique.items():
        fit=read(root/'fits'/key/'result.json');lp_count+=len(fit['LP_solves'])
        base=row['evaluation']['models']['rnj_bank']['final']
        item=dict(observation_key=key,reference_record=row['key'],models={})
        for name,info in row['evaluation']['models'].items():
            item['models'][name]=dict(errors=info['final']['all_block_errors'],complete=info['final']['complete_effective_family'],
                origin=info['selected_origin'],path_index=info.get('path_index'),
                effective_supports=sorted([s for s,r,x in zip(fit['models'][name]['support_labels'],fit['models'][name]['r'],fit['models'][name]['x']) if max(r,x)>1e-9]),
                errors_relative_to_RNJ=info['final']['all_block_errors']-base['all_block_errors'])
        details.append(item)
        for name,path in fit['searches'].items():
            for j,decision in enumerate(path['decisions']):
                if j+1>=len(path['path']):continue
                winner=None
                for c in decision['candidates']:
                    if winner is None or c['objective']<winner['objective']-1e-12:winner=c
                event=dict(observation_key=key,path=name,step=j+1,current=decision['current'],
                           candidate=winner['candidate'],removed=winner['removed'],objective=winner['objective'],
                           actual_acceptance_threshold=1e-10+.001*decision['current'])
                accepted.append(event)
                if winner['removed']:crossing.append(event)
    collisions=[]
    for row in rows:
        if row['status']!='evaluated':continue
        z=np.asarray(row['participation'],dtype=int)[:2]
        groups={}
        for i in range(5):groups.setdefault(tuple(z[:,i]),[]).append(101+i)
        duplicate=[v for v in groups.values() if len(v)>1]
        if not duplicate:raise RuntimeError('five-node binary two-event signature pigeonhole violated')
        collisions.append(dict(key=row['key'],training_signature_groups=[dict(signature=[int(a) for a in k],nodes=v) for k,v in groups.items()],
                               identical_training_profile_groups=duplicate,distinct_signatures=len(groups)))
    controls=[]
    for r in rows:
        if r['phase']=='control':controls.append(dict(key=r['key'],case=r['case'],seed=r['planning_seed'],policy=r['policy'],
           status=r['status'],models={k:v['final'] for k,v in r.get('evaluation',{}).get('models',{}).items()}))
    rescue=[]
    normal_controls={(r['planning_seed'],r['policy']):r for r in rows if r['phase']=='control' and r['case']=='normal' and r['status']=='evaluated'}
    for row in rows:
        if row['phase']!='main' or row['law']!='local' or row['status']!='evaluated':continue
        control=normal_controls.get((row['planning_seed'],row['policy']))
        if control and not control['evaluation']['models']['multistart_original_rule']['final']['complete_effective_family'] and row['evaluation']['models']['multistart_original_rule']['final']['complete_effective_family']:
            rescue.append(dict(random_record=row['key'],all_participate_control=control['key'],plan=row['plan'],
                 random_participation=row['participation'],control_participation=control['participation'],
                 random_payment=row['physical']['payment_currency'],control_payment=control['physical']['payment_currency'],
                 random_errors=0,control_errors=control['evaluation']['models']['multistart_original_rule']['final']['all_block_errors'],
                 role='same frozen price and budget, conditional random-vs-complete participation contrast; not population superiority or WZZT-specific benefit'))
    deltas={}
    for method in ('baseline25','multistart_original_rule'):
        vv=[d['models'][method]['errors_relative_to_RNJ'] for d in details]
        same=sum(d['models'][method]['effective_supports']==d['models']['rnj_bank']['effective_supports'] for d in details)
        deltas[method]=dict(better=sum(v<0 for v in vv),same_errors=sum(v==0 for v in vv),worse=sum(v>0 for v in vv),
                           same_effective_family=same,denominator=len(vv))
    # Average over two planning draws; the same eight response paths are reused.
    grouped=[]
    for law in ('local','common'):
        for policy in ('no_reward','single','sync','free','mean_free','rotation'):
            gg=[g for g in summary['groups'] if g['law']==law and g['policy']==policy]
            grouped.append(dict(law=law,policy=policy,planning_draws=len(gg),response_paths_per_draw=8,
                mean_errors_across_planning_draws=float(np.mean([g['models']['multistart_original_rule']['mean_errors'] for g in gg])),
                full_records=sum(g['models']['multistart_original_rule']['full_count'] for g in gg),record_denominator=16,
                mean_payment=float(np.mean([g['mean_payment'] for g in gg])),
                scope='two planner draws share eight response paths; 16 records are not 16 independent response realizations'))
    out=dict(status='complete',record_count=len(rows),main_count=summary['main'],control_count=summary['controls'],
        failed=summary['failed'],wall_limited=summary['wall_limited'],unique_fits=len(details),LP_count=lp_count,
        source_sha256={name:sha(root/name) for name in ('protocol.json','summary.json','history_audit.json')},analysis_source_sha256=sha(__file__),
        historical_unbounded_LPs=len(read(root/'history_audit.json')['LP_solves']),
        structural_attribution=deltas,unique_fit_details=details,accepted_steps=accepted,accepted_crossing_steps=crossing,
        selected_greedy_models=sum(v['path_index'] is not None and v['path_index']>0 for d in details for v in d['models'].values()),
        selected_origins=dict(Counter(v['origin'] for d in details for v in d['models'].values())),
        grouped_exploration=grouped,condition_groups=summary['groups'],controls=controls,
        complete_recovery_despite_all_participate_failure=rescue,
        training_profile_collisions=dict(records_checked=len(collisions),records_with_identical_profile_pairs=len(collisions),records=collisions,
            theorem='Five common-baseline nodes and two binary whole-event responses have at most four training spatial signatures. Some pair has identical training P and Q in every realization.',
            indistinguishable_pair_witness='For compatible diagonal H=diag(h), choose any duplicate pair i,j and 0<a<min(h_i,h_j)/2. Hprime=H+a*z_ij*z_ij^T-2a*(e_i*e_i^T+e_j*e_j^T) is a nonnegative tree with an internal pair and identical training outputs.',
            scope='Uniform training identifiability over the compatible model class only; does not imply the actual true tree always fails or that both validation events cannot separate the pair.'),
        limitations=['Randomized exploration control, not a new WZZT price optimizer.',
            'Eight independent response paths per law conditional on fixed load/noiseless linear voltage.',
            'Two planner draws do not exhaust all price ties; no posthoc replacement of chosen prices.',
            'Mean and stochastic objectives have independently sampled tie representatives; their difference also includes location draws.',
            'Zero and weak weights retained in the declared separate all-participate controls.',
            'Historical whole-family possibility is not topology probability or a guarantee for greedy search.'])
    (root/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('record_count','main_count','control_count','failed','wall_limited','unique_fits','LP_count','structural_attribution','selected_greedy_models')}))
    print(json.dumps(grouped,ensure_ascii=False))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
