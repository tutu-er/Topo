"""Deferred terminal-phase review of saved points, without new fits or data.

This script has no import-time data access. The caller must explicitly provide
--allow-terminal-read after root confirms the phase terminal. It requires both
fit workers ended plus fits_frozen/scored before opening any phase result.
Only saved support sets and scalar scores enter the report; no P/Q/V or
continuous fitted coefficients are written. Saved weights are used solely to
replay the existing contraction rule. It does not import the identifier.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
C=None
truth=None


def family(point,labels,key='supports'):
    return tuple(sorted({tuple(sorted(labels[i] for i in s)) for s in point[key]},key=lambda s:(len(s),s)))

def facts(point,labels,key='supports'):
    fam=set(family(point,labels,key));return dict(support_count=len(fam),target_present=C in fam,
        full_errors=len(fam^truth),missing=[list(s) for s in sorted(truth-fam)],
        extra=[list(s) for s in sorted(fam-truth)],support_labels=[list(s) for s in sorted(fam,key=lambda s:(len(s),s))])

def cut(point,cal):
    keep=[i for i,s in enumerate(point['supports']) if len(s)==1 or cal['a']*point['r'][i]+cal['b']*point['x'][i]+1e-10>=.16*cal['median_root_depth']]
    return dict(supports=[point['supports'][i] for i in keep])

def describe(name,f,refpath,full_details=False):
    labels=f['terminal_labels'];cal=f['calibrations']['raw'];candidates=f['selection_candidates']
    best=min(p['validation_mae'] for p in candidates);threshold=1.03*best+1e-12
    prep=f['start_preparation'];primary=next(p for p in f['bank'] if p['config']=='raw_RX75_0.16')
    actualsignatures={family(p,labels) for p in candidates}
    rank=sorted(range(len(prep)),key=lambda i:(prep[i]['validation_mae'],len(prep[i]['supports']),i))
    seen={family(primary,labels,'native_family')};replayed=[];events=[]
    for i in rank:
        sig=family(prep[i],labels)
        if sig in seen:
            events.append(dict(source_index=i,status='skipped_existing_family'))
            continue
        seen.add(sig);replayed.append(i);events.append(dict(source_index=i,status='selected_extra_start'))
        if len(replayed)==2:break
    assert replayed==f['extra_start_sources']
    prepinfo=[]
    for i,p in enumerate(prep):
        q=dict(source_index=i,source_config=f['bank'][i]['config'],training_objective=p['objective'],validation_mae=p['validation_mae'],
            validation_rank=rank.index(i)+1,already_fitted_and_validation_scored=True,
            selected_as_extra_start=i in f['extra_start_sources'],
            family_already_in_actual_final_candidates=family(p,labels) in actualsignatures,
            eligible_if_compared_under_actual_global_threshold=p['validation_mae']<=threshold,
            **facts(p,labels))
        if full_details:
            q['bank_native']=facts(f['bank'][i],labels,'native_family')
            q['bank_joint_fit']=facts(f['bank'][i],labels)
            q['contracted_bank_before_existing_refit']=facts(cut(f['bank'][i],cal),labels)
        prepinfo.append(q)
    actual=f['models']['multistart_original_rule'];raw=actual['selected_uncontracted']
    record=dict(record=name,result_source=str(refpath),result_sha256=sha(refpath),
        best_actual_validation_mae=best,actual_3percent_threshold_including_tolerance=threshold,
        actual_selected=dict(origin=actual['selected_origin'],path_index=raw.get('path_index'),source_config=raw.get('source_config'),
            raw=facts(raw,labels),final=facts(actual,labels),training_objective=raw['objective'],validation_mae=raw['validation_mae']),
        primary_native=facts(primary,labels,'native_family'),extra_start_sources=f['extra_start_sources'],
        extra_start_replay_events=events,all_start_preparation_points_actually_fitted_and_scored=True,
        start_preparation=prepinfo if full_details else [{k:v for k,v in p.items() if k not in ('missing','extra','support_labels')} for p in prepinfo])
    coarse=[i for i,p in enumerate(prep) if set(family(p,labels))==truth-{C}]
    if coarse:
        i=min(coarse,key=lambda i:(len(prep[i]['supports']),prep[i]['validation_mae'],i));point=prep[i]
        extended=[*candidates,point];emin=min(p['validation_mae'] for p in extended)
        eligible=[i for i,p in enumerate(extended) if p['validation_mae']<=1.03*emin+1e-12]
        chosen=min(eligible,key=lambda i:(len(extended[i]['supports']),extended[i]['validation_mae'],i))
        picked=extended[chosen]
        record['saved_point_counterfactual']=dict(scope='post-hoc final selection on existing fitted/scored points only; no new fit/path/data and actual output unchanged',
            appended_preparation_source_index=i,appended_source_config=f['bank'][i]['config'],appended_point=facts(point,labels),
            appended_validation_mae=point['validation_mae'],actual_global_validation_mae=best,
            actual_global_threshold=threshold,eligibility_margin=point['validation_mae']-threshold,
            appended_point_was_in_actual_final_candidates=family(point,labels) in actualsignatures,
            counterfactual_selected_appended_point=chosen==len(candidates),
            counterfactual_raw=facts(picked,labels),counterfactual_after_existing_contraction=facts(cut(picked,cal),labels))
    return record

def run(root,phase,output,allow):
    if not allow:raise RuntimeError('Root terminal notification and explicit --allow-terminal-read required')
    root=Path(root).resolve()
    if phase not in ('planning','test'):raise ValueError('phase must be planning or test')
    protocol=read(root/'protocol.json');workers=int(protocol['workers'])
    terminal=[root/phase/'fits_frozen.json',root/phase/'scored.json']+[root/phase/f'worker_{i}_ended.json' for i in range(workers)]
    if any(not p.is_file() for p in terminal):raise RuntimeError('phase is not fully frozen/scored/ended')
    if read(terminal[0]).get('status')!='frozen':raise RuntimeError('fit list not frozen')
    phase_data=read(terminal[1])
    if phase_data.get('status')!='complete':raise RuntimeError('score aggregation not terminal')
    if any(read(p).get('status')!='ended' for p in terminal[2:]):raise RuntimeError('a fit worker is not terminal')
    baseline=read(root/'baseline_evaluation.json')
    global C,truth
    C=tuple(sorted(protocol['target_labels']));truth={tuple(sorted(s)) for s in baseline['true_supports']}
    records=[];skipped=[]
    for row in phase_data['rows']:
        if row['status']!='complete':
            skipped.append({k:row.get(k) for k in ('record_id','policy','path','status')});continue
        result_path=Path(row['result']['path'])
        if sha(result_path)!=row['result']['sha256']:raise RuntimeError('saved result hash drift')
        f=read(result_path);value=describe(row['record_id'],f,result_path,True)
        labels=f['terminal_labels']
        target_in_pool=any(tuple(sorted(s))==C for s in f.get('pool_support_labels',[]))
        target_accepted=any(C in family(p,labels) for s in f['searches'].values() for p in s['path'][1:])
        value.update({k:row.get(k) for k in ('policy','module','path','train_state','validation_state','prices','fit_key')})
        value.update(target_in_pool=target_in_pool,target_ever_in_accepted_path=target_accepted,
            other_method_final_errors={k:p['full_errors'] for k,p in row['models'].items()},
            scalar_source_scope='completed saved result only; no physical truth pickle or test P/Q/V opened')
        records.append(value)
    successes=[v for v in records if v['actual_selected']['final']['full_errors']==0]
    counter=[v['saved_point_counterfactual'] for v in successes if 'saved_point_counterfactual' in v]
    def key(v):return v.get('fit_key') or v['result_sha256']
    unique={key(v):v for v in successes}
    summary=dict(completed_requests=len(records),skipped_requests=len(skipped),successful_requests=len(successes),
        successful_unique_results=len(unique),successful_requests_with_target_pool=sum(v['target_in_pool'] for v in successes),
        successful_requests_with_target_accepted=sum(v['target_ever_in_accepted_path'] for v in successes),
        successful_selected_path_index_counts=dict(Counter(str(v['actual_selected']['path_index']) for v in successes)),
        successes_with_saved_exact_truth_minus_target_preparation=len(counter),
        saved_coarse_still_3percent_eligible=sum(c['eligibility_margin']<=0 for c in counter),
        saved_coarse_not_in_actual_candidates=sum(not c['appended_point_was_in_actual_final_candidates'] for c in counter),
        appending_saved_coarse_would_win=sum(c['counterfactual_selected_appended_point'] for c in counter),
        counterfactual_after_contract_error_hist=dict(Counter(str(c['counterfactual_after_existing_contraction']['full_errors']) for c in counter)))
    report=dict(status='read_only_terminal_review_complete',created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,
        script_sha256=sha(__file__),protocol_sha256=sha(root/'protocol.json'),
        terminal_sources={str(p):sha(p) for p in terminal},baseline_evaluation_sha256=sha(root/'baseline_evaluation.json'),
        scope='post-hoc selection audit on already fitted/scored preparation points; not a new fit, new path or alternative algorithm experiment',
        C=list(C),summary=summary,skipped=skipped,records=records,
        exact_coarse_definition='saved support family equals physical effective truth minus target C; no hypothetical refit',
        zero_new_WZZT_LP_AC_calls=True,no_PQV_arrays_read=True,no_continuous_parameters_in_report=True,
        interpretations_to_check=[
            'Do not infer 3percent exclusion if the saved coarse point remains eligible.',
            'A saved-point counterfactual can expose sensitivity to start admission, but is not an actually rerun algorithm.',
            'Path_index0 identifies the fitted preparation family rather than a newly accepted extension.',
            'Shared successful fit keys are not independent trials; denominators are reported separately.'
        ])
    output=Path(output)
    if output.exists():raise FileExistsError('existing review retained; use a new output filename')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=('planning','test'),required=True);p.add_argument('--out',required=True);p.add_argument('--allow-terminal-read',action='store_true')
    a=p.parse_args();r=run(a.root,a.phase,a.out,a.allow_terminal_read)
    print(json.dumps({'status':r['status'],'phase':r['phase'],'summary':r['summary']}))
