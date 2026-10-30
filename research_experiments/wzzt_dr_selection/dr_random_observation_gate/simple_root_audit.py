"""Saved-fit RNJ+optional-universal-root counterfactual; no new fitting."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib,json,pickle
import numpy as np
from evaluate_gate import read,write,score,matrices,observed_arrays,METHODS

BASE=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
GEN=BASE/'wzzt_dr_random_observation_generation_v2'
EXE=BASE/'wzzt_dr_random_observation_execution_v1'
OUT=BASE/'evaluation_v1'
def canonical(values):return tuple(sorted(set(tuple(sorted(s)) for s in values),key=lambda s:(len(s),s)))
def legal():
    leaves=[(0,),(1,),(2,)];internal=[(0,1),(0,2),(1,2),(0,1,2)];result=[]
    for bits in range(16):
        chosen=[s for j,s in enumerate(internal) if (bits>>j)&1]
        if all(not set(a)&set(b) or set(a)<=set(b) or set(b)<=set(a) for j,a in enumerate(chosen) for b in chosen[:j]):
            result.append(canonical(leaves+chosen))
    return sorted(set(result),key=lambda f:(len(f),f))
def contract(point,cal):
    keep=[i for i,s in enumerate(point['supports']) if len(s)==1 or cal['a']*point['r'][i]+cal['b']*point['x'][i]+1e-10>=.16*cal['median_root_depth']]
    return dict(supports=[point['supports'][i] for i in keep],r=[point['r'][i] for i in keep],x=[point['x'][i] for i in keep])

def main():
    cfg=read(EXE/'protocol.json');truth=read(GEN/'evaluator_truth.json');families=legal();assert len(families)==8
    unique={};repairs=[]
    for snapshot in cfg['snapshots']:
        sid=snapshot['id'];result=read(EXE/'records'/f'{sid}.json');state=read(result['current_state_path'])
        ref=result['methods']['all_legal_family_reference'];assert ref['status']=='complete'
        lookup={families[p['family_index']]:p for p in ref['points']}
        native_aug=sorted(set(canonical([*tree['family'],(0,1,2)]) for tree in state['native_bank']),key=lambda f:(len(f),f))
        points=[lookup[f] for f in native_aug]
        best=min(p['validation_mae'] for p in points)
        eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
        chosen=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
        raw=points[chosen];final=contract(raw,result['applied_calibration']);labels=result['terminal_labels']
        W=result['methods']['expanded_pool_wzzt']['selected'];R,X=matrices(final,labels);wr,wx=matrices(W,labels)
        unique[sid]=dict(raw=raw,final=final,native_augmented_families=native_aug,
            current_native_families_containing_universal=sum((0,1,2) in canonical(t['family']) for t in state['native_bank']),
            combined_K_difference_vs_W=float(np.max(abs((R+.35*X)-(wr+.35*wx)))))
        for name,path in result['methods']['current_pool_wzzt']['paths'].items():
            for i,(before,after) in enumerate(zip(path['path'],path['path'][1:])):
                bset=set(tuple(s) for s in before['supports']);aset=set(tuple(s) for s in after['supports'])
                repairs.append(dict(snapshot_id=sid,path=name,iteration=i,
                    added=[list(s) for s in sorted(aset-bset)],removed=[list(s) for s in sorted(bset-aset)],
                    gain=before['objective']-after['objective'],threshold=1e-10+.001*before['objective']))
    evaluated=[]
    case_index={c['id']:c for c in truth['cases']}
    for mapping in truth['evaluations']:
        sid=mapping['snapshot_id'];u=unique[sid];case=case_index[mapping['case_id']]
        result=read(EXE/'records'/f'{sid}.json');labels=result['terminal_labels'];target=mapping['target_family']
        paid=mapping['actual_payment']
        W=score(result['methods']['expanded_pool_wzzt']['selected'],labels,target)
        RNJ=score(result['methods']['current_rebuilt_rnj']['selected'],labels,target)
        augmented=score(u['final'],labels,target);raw=score(u['raw'],labels,target)
        evaluated.append(dict(case_id=mapping['case_id'],snapshot_id=sid,noise_id=case['noise_id'],
            b=case['truth']['b'],c=case['truth']['c'],true_pair=case['truth']['pair'],policy=mapping['policy'],
            conditional_state=mapping['conditional_state'],actual_payment=paid,
            root_augmented_final=augmented,root_augmented_raw=raw,W=W,RNJ=RNJ,
            same_final_support_as_W=augmented['predicted']==W['predicted'],
            combined_K_difference_vs_W=u['combined_K_difference_vs_W']))
    fees=[]
    for policy in sorted(set(e['policy'] for e in truth['evaluations'])):
        rows=[e for e in truth['evaluations'] if e['policy']==policy]
        fees.append(dict(policy=policy,mapped_cases=len(rows),unique_actual_payments=sorted(set(e['actual_payment'] for e in rows)),
            conditional_state=rows[0]['conditional_state'],worst_allowed=1.6))
    rank_rows=[]
    histories={h['id']:h for h in cfg['histories']}
    for snapshot in cfg['snapshots']:
        h=pickle.loads(Path(histories[snapshot['history_id']]['path']).read_bytes())
        f=pickle.loads(Path(snapshot['path']).read_bytes());P,Q,_=observed_arrays(h['training']+f['training'])
        rank_rows.append(dict(snapshot_id=snapshot['id'],noise_id=snapshot['history_id'],
            observed_P_rank=int(np.linalg.matrix_rank(P)),observed_PQ_rank=int(np.linalg.matrix_rank(np.column_stack([P,Q]))),
            P_singular_values=np.linalg.svd(P,compute_uv=False).tolist(),
            note='noisy observed full rank is not proof of physical excitation rank'))
    summary=dict(status='complete_saved_fit_counterfactual',evaluations=len(evaluated),snapshots=len(unique),
        same_final_support_as_W=sum(e['same_final_support_as_W'] for e in evaluated),
        exact_mappings=sum(e['noise_id']=='exact' for e in evaluated),
        exact_same_final_support_as_W=sum(e['noise_id']=='exact' and e['same_final_support_as_W'] for e in evaluated),
        noisy_mappings=sum(e['noise_id']=='current_noise' for e in evaluated),
        noisy_same_final_support_as_W=sum(e['noise_id']=='current_noise' and e['same_final_support_as_W'] for e in evaluated),
        augmented_errors=sum(e['root_augmented_final']['errors'] for e in evaluated),
        W_errors=sum(e['W']['errors'] for e in evaluated),RNJ_errors=sum(e['RNJ']['errors'] for e in evaluated),
        augmented_vs_RNJ_better=sum(e['root_augmented_final']['errors']<e['RNJ']['errors'] for e in evaluated),
        max_combined_K_difference_vs_W=max(e['combined_K_difference_vs_W'] for e in evaluated),
        native_bank_universal_block_count_distribution=dict(Counter(u['current_native_families_containing_universal'] for u in unique.values())),
        current_W_added_block_counts=dict(Counter(str(step['added']) for step in repairs)),
        fees_by_policy=fees,
        scope='same fixed-family LP optima already saved by all8 reference; no new LP and not a deployed new method',
        raw_and_contracted_families_separate=True,no_price_parameters_or_data_changed=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_LP=0,new_QP=0,new_AC=0)
    write(OUT/'simple_root_counterfactual.json',dict(summary=summary,evaluations=evaluated,unique_snapshots=unique))
    write(OUT/'acceptance_repairs.json',repairs);write(OUT/'rank_and_fee_audit.json',dict(ranks=rank_rows,fees=fees))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
