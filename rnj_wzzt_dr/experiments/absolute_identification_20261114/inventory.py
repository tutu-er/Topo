"""Enumerate existing frozen candidate libraries without fitting or future scoring.

Eligibility uses existing historical truth grading only. Existing DR outcomes
are never read or used for case selection. Per-input pointers remain private.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def ref(path):
    p=Path(path).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def category(covered,correct,p):
    return 'missing_truth' if not covered else 'wrong_winner_truth_present' if not correct else 'correct_weak' if p<.99 else 'correct_high_weight'

def load_large():
    root=ROOT/'outputs/unified_posterior_dr_20261112_v1'
    requests=read(root/'requests.json');evaluation={e['id']:e for e in read(root/'evaluation.json')}
    seal=read(root/'selection_seal.json')['scores']
    rows=[]
    for request in requests:
        path=root/'scores'/(request['id']+'.json');s=read(path);e=evaluation[request['id']]
        source=ref(path)
        if source['sha256']!=seal[request['id']]:raise AssertionError('Frozen score drift')
        if s['status']!='complete':continue
        target=next(i for i,m in enumerate(s['models']) if m['topology_id']==s['map_topology_id'])
        if len(s['parameters'])!=len(s['models']):raise AssertionError('Representative alignment')
        rows.append(dict(pool='large96',id=request['id'],case=request['case'],method=request['method'],
            score=source,request=request,labels=s['labels'],H_shape=list(np.asarray(s['parameters'][0]['H']).shape),
            historical_input_key=request['observed']['sha256'],training_points=s['training_points'],validation_points=s['validation_points'],
            target_representative_index=target,target_raw_index=s['map_index'],
            truth_covered=e['truth_covered'],native_correct=e['native_correct'],representative_correct=e['map_correct'],
            p_map=s['p_map'],p_true=e['p_true'],category=category(e['truth_covered'],e['native_correct'],s['p_map']),
            profile=request['profile'],level=request['level'],voltage_sigma=request['voltage_sigma'],replicate=request['replicate'],
            candidate_topologies=len(s['models']),raw_points=s['raw_points'],native_bank_in_score='native_bank' in s,
            parameter_H_available=True,complete_day_observations=True,customer_baseline_in_history=False,
            history_partition='development' if request['replicate']%2==0 else 'confirmation',
            native_source_available=Path(request['result']['path']).exists(),observation_available=Path(request['observed']['path']).exists()))
    return rows,[ref(root/p) for p in ('requests.json','evaluation.json','selection_seal.json','protocol.json')]

def load_customer():
    root=ROOT/'outputs/customer_flexibility_sweep_20261113_v1'
    rows=[];failed=[]
    for e in read(root/'qualification.json'):
        if e['status']!='fitted':failed.append(e['id']);continue
        s=read(e['score']['path'])
        if ref(e['score']['path'])['sha256']!=e['score']['sha256']:raise AssertionError('Customer score drift')
        key=hashlib.sha256((e['observed_refs']['train']['sha256']+e['observed_refs']['validation']['sha256']).encode()).hexdigest()
        rows.append(dict(pool='customer',id=e['id'],case='paper15',method='101',score=e['score'],request=e,
            labels=s['labels'],H_shape=list(np.asarray(s['parameters'][0]['H']).shape),historical_input_key=key,
            training_points=s['training_points'],validation_points=s['validation_points'],
            target_representative_index=s['map_index'],target_raw_index=s.get('raw_map_index'),truth_covered=e['truth_present'],
            native_correct=e['original_winner_correct'],representative_correct=e['posterior_MAP_correct'],p_map=s['p_map'],p_true=e['p_true'],
            category=category(e['truth_present'],e['original_winner_correct'],s['p_map']),
            profile=e['profile'],level=None,voltage_sigma=e['voltage_sigma'],replicate=e['history_seed'],
            candidate_topologies=len(s['models']),raw_points=s['raw_points'],native_bank_in_score=True,
            parameter_H_available=True,complete_day_observations=True,customer_baseline_in_history=e['adoption'],
            history_partition='development' if e['history_seed']%2==0 else 'confirmation',
            native_source_available=(Path(e['directory'])/'native_result.json').exists(),
            observation_available=all(Path(v['path']).exists() for v in e['observed_refs'].values())))
    return rows,[ref(root/p) for p in ('qualification.json','protocol.json')],failed

def load_legacy():
    root=ROOT/'outputs/posterior_mse_validation_20261112_v1'
    protocol=read(root/'protocol.json');evaluation={e['id']:e for e in read(root/'evaluation.json')}
    rows=[]
    for request in protocol['requests']:
        e=evaluation[request['id']]
        path=root/'scores'/(request['id']+'.json');s=read(path)
        case=e.get('case',request.get('case'))
        if case!='paper15':continue
        if request.get('observed'):
            key=request['observed']['sha256'];available=Path(request['observed']['path']).exists()
        else:
            key=hashlib.sha256((request['train']['sha256']+request['validation']['sha256']).encode()).hexdigest()
            available=Path(request['train']['path']).exists() and Path(request['validation']['path']).exists()
        rows.append(dict(pool='legacy_full48_and_strict96',id=request['id'],case=case,method=request['method'],
            score=ref(path),request=request,labels=s['labels'],H_shape=None,historical_input_key=key,
            training_points=s['training_points'],validation_points=s['validation_points'],
            target_representative_index=next(i for i,m in enumerate(s['models']) if m['topology_id']==s['map_topology_id']),
            target_raw_index=s['map_index'],truth_covered=e['truth_covered'],native_correct=e['native_correct'],
            representative_correct=e['map_correct'],p_map=s['p_map'],p_true=e['p_true'],
            category=category(e['truth_covered'],e['native_correct'],s['p_map']),profile='reference',level=None,
            voltage_sigma=e.get('voltage_sigma'),replicate=e.get('seed_index'),
            candidate_topologies=len(s['models']),raw_points=s['raw_points'],native_bank_in_score=False,
            parameter_H_available='parameters' in s,complete_day_observations=True,customer_baseline_in_history=False,
            history_partition='legacy_control_only',native_source_available=Path(request['result']['path']).exists(),observation_available=available))
    return rows,[ref(root/p) for p in ('protocol.json','evaluation.json','selection_seal.json')]

def run(out):
    if out.exists():raise ValueError('Preserve prior inventory')
    out.mkdir(parents=True)
    write(out/'protocol.json',dict(source=ref(__file__),scope='Existing large96 800 libraries, customer72-history grid, olderfull48+strict96 paper15 controls.',
        new_fits=0,new_AC_calls=0,prior_future_DR_results_read=False,
        qualification='Existing historical truth coverage, native-winner correctness, fixed p_map<.99 weak threshold.',
        partition='Deterministic history replicate parity for development versus confirmation. This is a proposed split; all historical pools have been studied before. Fresh future paths still required.',
        scope_limit='Current mainline compatible frozen-library pools; not an exhaustive search of all archived pre-2026-11-12 algorithms.'))
    large,refs=load_large();customer,c_refs,failed=load_customer();legacy,l_refs=load_legacy()
    rows=large+customer+legacy;refs+=c_refs+l_refs
    groups={}
    for key in sorted({(r['pool'],r['case'],r['method']) for r in rows}):
        selected=[r for r in rows if (r['pool'],r['case'],r['method'])==key]
        groups['/'.join(key)]=dict(libraries=len(selected),categories=dict(Counter(r['category'] for r in selected)),
            unique_histories=len({r['historical_input_key'] for r in selected}),
            point_counts=sorted({(r['training_points'],r['validation_points']) for r in selected}),
            weak_at={str(beta):sum(r['native_correct'] and r['p_map']<beta for r in selected) for beta in (.8,.9,.95,.99)},
            parameter_H_available=sum(r['parameter_H_available'] for r in selected),
            native_source_available=sum(r['native_source_available'] for r in selected),
            observation_available=sum(r['observation_available'] for r in selected))
    paper=[r for r in rows if r['case']=='paper15']
    large101=[r for r in large if r['method']=='101']
    actionable=[r for r in rows if (r['case']=='paper15' and r['category']=='correct_weak' and r['method']=='101') or (r['category']=='wrong_winner_truth_present' and r['method']=='101')]
    summary=dict(groups=groups,paper15_total_libraries=len(paper),paper15_unique_history_keys=len({r['historical_input_key'] for r in paper}),
        customer_failed_histories=failed,large101_all_cases=dict(libraries=len(large101),categories=dict(Counter(r['category'] for r in large101))),
        actionable_history_only=[{k:r[k] for k in ('pool','id','case','category','p_map','p_true','replicate','profile','level','voltage_sigma','history_partition')} for r in actionable],
        source_schema=dict(large96='Frozen parameters.H present; map_index is raw index; no native_bank. Native library available in original request.result["result"].',
            customer='Frozen parameters.H and native_bank present; map_index already representative ordinal.',
            legacy='Representative scores without H; reconstruct frozen matrices from native_result and source observations, no refit.'),
        duplicated_experiment_families=['linear_ac_population v1/v2, bidirectional_price_DR v1/v2, residual_guard_ED share the same large96 libraries; they are not extra histories.',
            'Customer placement/capacity variants are resource revisions, not additional identifier libraries; intensity/astra_confirmation reuse customer historys0_c013.'],
        recommended_panel='All mainline paper15 101 libraries (100large96+71customer) give171history conditions with4correctweak,48correcthighweight,119missing; add9large96 101 wrong-winner-truth-present conditions fromsoumalas11/pengwah18 if cross-network resource adaptation is authorized. Keep existing111 and25legacy paper15 libraries as separate method/large-history controls.',
        caution='No paper15 wrong-winner-with-truth condition exists in these inspected mainline pools. Do not retarget a true candidate using offline truth or perturb fitted values to manufacture that class.')
    write(out/'source_refs.json',refs);write(out/'inputs.json',rows);write(out/'summary.json',summary)
    return summary

def build_panel(out):
    """Freeze all400 mainline101 records and a history-only development panel."""
    target=out/'mainline400.json'
    if target.exists():raise ValueError('Preserve fixed mainline panel')
    rows=[r for r in read(out/'inputs.json') if r['pool']=='large96' and r['method']=='101']
    original=ROOT/'outputs/unified_posterior_dr_20261112_v1'
    evaluation={r['id']:r for r in read(original/'evaluation.json')}
    physical={r['request']['id']:r for r in read(ROOT/'outputs/linear_ac_population_20261113_v2/inputs.json')}
    rolemap=dict(correct_high_weight='correct_clear',correct_weak='correct_ambiguous',
        wrong_winner_truth_present='wrong_present',missing_truth='missing')
    write(out/'panel_protocol.json',dict(source=ref(__file__),inventory=ref(out/'inputs.json'),
        physical_source=ref(ROOT/'outputs/linear_ac_population_20261113_v2/inputs.json'),
        historical_grading=ref(original/'evaluation.json'),
        role_rule='correct_clear: original winner correct andp_map>=.99; correct_ambiguous: winner correct andp_map<.99; wrong_present: wrongwinner buttruthinpool; missing: notruecandidate.',
        split="sha256('absolute20261114|'+cohort+'|'+case+'|'+str(replicate)) first8hex parity: even development, odd history_holdout. Same latent historical day stays together across noise/profile variants.",
        panel='All development wrong_present and correct_ambiguous;9correct_clear and9missing selected case-balanced round-robin by a fixed hash rank. Maximum32 histories.',
        final='All400mainline101 entries; history_holdout subset reported separately. Fresh future random streams needed; historical pools and earlier outcomes already studied.',
        no_prior_future_results_used=True,new_identifier_fits=0,new_AC_calls=0))
    for r in rows:
        r['role']=rolemap[r['category']]
        key='|'.join([r['request']['cohort'],r['case'],str(r['replicate'])])
        r['physical_history_group']=key
        r['history_partition']='development' if int(hashlib.sha256(('absolute20261114|'+key).encode()).hexdigest()[:8],16)%2==0 else 'history_holdout'
        p=physical[r['id']]
        r['physical_parameters']=p['parameters'];r['physical_branch_sha256']=p['physical_branch_sha256']
        r['physical_cohort']=r['request']['cohort'];r['original_index']=p['index']
        r['truth_id_for_offline_evaluation']=evaluation[r['id']]['truth_id']
        s=read(r['score']['path']);logp=np.array([m['log_probability'] for m in s['models']])
        order=np.argsort(-logp,kind='stable');count=min(len(order),int(np.searchsorted(np.cumsum(np.exp(logp[order])),.99,side='left'))+1)
        threshold=logp[order[count-1]];kept=[int(i) for i in order if logp[i]>=threshold]
        true_index=next((i for i,m in enumerate(s['models']) if m['topology_id']==r['truth_id_for_offline_evaluation']),None)
        r['retained99_representative_indices']=kept;r['retained99_count']=len(kept)
        r['retained99_truth_covered']=true_index is not None and true_index in kept
        r['truth_representative_index_offline']=true_index
    if len(rows)!=400:raise AssertionError('Expected400101 libraries')
    development=[r for r in rows if r['history_partition']=='development']
    holdout=[r for r in rows if r['history_partition']=='history_holdout']
    selected=[r for r in development if r['role'] in ('wrong_present','correct_ambiguous')]
    def rank(r):return hashlib.sha256(('panel20261114|'+r['id']).encode()).hexdigest()
    for role in ('correct_clear','missing'):
        buckets={case:sorted([r for r in development if r['case']==case and r['role']==role],key=rank) for case in sorted({r['case'] for r in rows})}
        picked=0
        while picked<9:
            changed=False
            for case,bucket in buckets.items():
                if bucket and picked<9:
                    selected.append(bucket.pop(0));picked+=1;changed=True
            if not changed:break
    selected=sorted(selected,key=lambda r:(r['role'],r['case'],r['id']))
    if len(selected)>32:raise AssertionError('Development budget exceeded')
    for name,value in [('mainline400.json',rows),('development32.json',selected),('history_holdout221.json',holdout)]:write(out/name,value)
    def describe(records):
        return dict(libraries=len(records),roles=dict(Counter(r['role'] for r in records)),
            cases={case:dict(Counter(r['role'] for r in records if r['case']==case)) for case in sorted({r['case'] for r in records})},
            physical_history_groups=len({r['physical_history_group'] for r in records}),
            truth_covered=sum(r['truth_covered'] for r in records),retained99_truth_covered=sum(r['retained99_truth_covered'] for r in records),
            mean_candidate_count=float(np.mean([r['candidate_topologies'] for r in records])),
            mean_retained99_count=float(np.mean([r['retained99_count'] for r in records])))
    result=dict(all400=describe(rows),development_pool=describe(development),development_panel=describe(selected),history_holdout=describe(holdout),
        development_ids=[r['id'] for r in selected],
        no_group_overlap=not({r['physical_history_group'] for r in development}&{r['physical_history_group'] for r in holdout}))
    write(out/'mainline400_summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--panel',action='store_true');a=p.parse_args()
    s=build_panel(a.out.resolve()) if a.panel else run(a.out.resolve())
    print(json.dumps(s if a.panel else {k:s[k] for k in ('paper15_total_libraries','paper15_unique_history_keys','large101_all_cases','customer_failed_histories')},indent=2))
