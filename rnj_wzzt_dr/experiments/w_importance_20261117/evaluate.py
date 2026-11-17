"""Independent frozen-bank evaluation for the W-importance ablation.

Only saved observations and result receipts are read. No identification,
parameter fitting, native solver, or physical simulator is imported here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..wzzt_long_grid_20261117 import evaluate as base


MODES=('full','pool_only','skip')
SPLITS=('validation','new_day','cumulative')
TERMINAL={'complete','failed','interrupted','budget_unrun','audit_failed','missing_final_result'}


def _mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def validate_treatment(task, native):
    """Audit the assigned intervention rather than trusting the variant label."""
    mode=task['variant'];base.require(mode in MODES,'Unknown assigned W mode')
    cfg=native['config'];base.require(cfg.get('w_mode','full')==mode,'Saved W mode differs from assigned mode')
    base.require(native.get('method')==task.get('method','101'),'Method differs from assigned method')
    base.require(cfg.get('loss','mse')==task.get('loss','mse'),'Training loss differs')
    expected=task.get('config',{})
    for key,value in expected.items():
        base.require(cfg.get(key)==value,'Saved configuration differs: '+key)
    events=native.get('events',[])
    we=[e for e in events if e.get('stage')=='W']
    if mode=='skip':
        base.require(not we,'skip mode contains a W event')
    if mode=='pool_only':
        base.require(not any(e.get('solver',{}).get('solve_started',False) for e in we),
                     'pool_only mode performed a native W solve')
    e_cap=cfg.get('e_seconds')
    for event in (e for e in events if e.get('stage')=='E'):
        if e_cap is not None:
            base.require(event.get('stage_seconds_cap')==e_cap,'E stage cap receipt differs')
            requested=event.get('solver',{}).get('requested_solve_seconds')
            if requested is not None:
                base.require(requested<=e_cap+1e-6,'Native E received more than its assigned allowance')
    return dict(assigned=mode,saved=cfg.get('w_mode','full'),passed=True)


def path_origins(native):
    """Map saved output ordinals; a source label alone is not causal evidence."""
    path=native.get('path',[]);bank=native['frozen_candidates']
    origins=['basis']+[e['stage'] for e in native.get('events',[]) if e.get('accepted',False)]
    if len(origins)!=len(path):
        return dict(available=False,reason='Accepted events do not map one-to-one to the saved path')
    if len(bank)==len(origins)+1:origins.append('singleton_protection')
    if len(bank)!=len(origins):
        return dict(available=False,reason='Output contraction changed or omitted the candidate index map')
    mapping={int(item.get('index',i)):origins[i] for i,item in enumerate(bank)}
    return dict(available=True,by_index=mapping,
                scope='Origin of the accepted path point before output contraction; not independent module attribution')


def _support_set(value,n):
    result={tuple(map(int,c)) for c in value}
    base.require(all(c and tuple(sorted(set(c)))==c and 0<=min(c) and max(c)<n for c in result),
                 'Invalid saved RNJ support pool')
    return result


def attribution(native,bank,geometry,solver=None):
    n=len(geometry['labels']);zero=float(native['config']['zero_tol'])
    truth=set(geometry['groups']);origins=path_origins(native);selected={}
    raw=native.get('uncontracted_output_path')
    raw_by_index=({int(item.get('index',i)):raw[i] for i,item in enumerate(native['frozen_candidates'])}
        if raw is not None and len(raw)==len(native['frozen_candidates']) else {})
    for split in SPLITS:
        choice=bank['selections'][split]['mse'];index=choice['index']
        row=dict(index=index,origin=origins.get('by_index',{}).get(index),
                 full_exact=choice['topology']['full']['exact'],internal_exact=choice['topology']['internal']['exact'])
        if origins['available']:
            accepted=[(i,e) for i,e in enumerate(native.get('events',[])) if e.get('accepted')]
            ancestor=accepted[:index]
            row['W_path_return_sources']=dict(Counter(e.get('returned_point_source','unavailable') for _,e in ancestor if e['stage']=='W'))
            if solver is not None:
                audited={r['event_index']:r for r in solver['events']}
                row['W_native_improvements_before_selected_point']=sum(
                    audited[i]['native_improved_initial'] is True for i,e in ancestor if e['stage']=='W')
        if index in raw_by_index:
            row['before_contraction']=base.stratified_topology(raw_by_index[index],geometry,zero)
            row['contraction_changed_full_supports']=row['before_contraction']['full']['exact']!=row['full_exact']
        selected[split]=row
    prefix={}
    if origins['available']:
        allowed={i for i,s in origins['by_index'].items() if s in ('basis','W','singleton_protection')}
        for split in SPLITS:
            models=[m for m in bank['models'] if m['index'] in allowed]
            choice=base.strict_select([dict(index=m['index'],stored_internal=m['stored_internal'],
                mse=m['scores'][split]['mse']) for m in models],reference=bank['reference_mse'],
                tie=native['config']['validation_tie'])
            winner=next(m for m in models if m['index']==choice['index'])
            prefix[split]=dict(index=choice['index'],topology=winner['topology'],
                complete_true_family_in_prefix=any(m['topology']['full']['exact'] for m in models),
                scope='Saved W-prefix outputs with unchanged contraction and validation; not a rerun with released E budget')
    guidance=native.get('rnj_pool_guidance')
    if guidance is None:
        pool_evidence=dict(available=False,reason='Original RNJ pool was not persisted; cleaned paths do not reconstruct it')
    else:
        pool=_support_set(guidance.get('pool',guidance.get('supports',[])) if isinstance(guidance,dict) else guidance,n)
        basis={(i,) for i in range(n)}|{tuple(range(n))}
        base_e=pool|basis|{(i,j) for i in range(n) for j in range(i+1,n)}
        w_supports=set()
        for e in native.get('events',[]):
            if e.get('stage')=='W' and e.get('accepted'):
                w_supports |= base.active_supports(e['point'],n,zero)
        pool_evidence=dict(available=True,rnj_pool_count=len(pool),base_E_pool_count=len(base_e),
            rnj_pool_supports=[list(c) for c in sorted(pool)],
            W_true_supports_outside_RNJ_pool=[list(c) for c in sorted((w_supports&truth)-pool)],
            W_true_supports_outside_base_E_pool=[list(c) for c in sorted((w_supports&truth)-base_e)],
            final_true_supports_outside_base_E_pool={split:[list(c) for c in sorted(
                (base.active_supports(next(item.get('point',item) for i,item in enumerate(native['frozen_candidates'])
                    if int(item.get('index',i))==bank['selections'][split]['mse']['index']),n,zero)&truth)-base_e)]
                for split in SPLITS},
            scope='Support availability only. Necessity of native W is tested by paired interventions, not these labels')
    return dict(path_origins=origins,selected=selected,W_prefix=prefix,pool=pool_evidence)


def partial_failure_evidence(reader,directory):
    """Retain native FINAL records from failed attempts, separately from successes."""
    directory=Path(directory);events_path=directory/'events.jsonl';checkpoint_path=directory/'checkpoint.json'
    events=[]
    if events_path.is_file():
        lines=reader.checked(base.file_ref(events_path)).read_text(encoding='utf8').splitlines()
        events=[json.loads(line) for line in lines if line.strip()]
    finals=[e for e in events if e.get('callback_event')=='FINAL']
    current=None;counters=Counter();native={}
    for event in events:
        stage=event.get('stage')
        if stage in ('W_started','T_started','E_started'):
            name=stage[0];current=(name,counters[name]);counters[name]+=1
        if event.get('callback_event')=='FINAL':
            key=current if current is not None else ('unknown',len(native))
            native[key]=dict(stage=key[0],ordinal=key[1],channel='scalar_FINAL',
                **{k:event.get(k) for k in ('model_name','scope','status','objective','bound','gap','wall_seconds')})
    stage_records=0;stage_counters=Counter()
    for path in sorted((directory/'stages').glob('*.json')):
        saved=reader.json(base.file_ref(path));event=saved['event'];stage=event['stage']
        key=(stage,stage_counters[stage]);stage_counters[stage]+=1
        receipt=event.get('solver',{})
        if not receipt.get('solve_started'):continue
        stage_records+=1
        native[key]=dict(stage=stage,ordinal=key[1],channel='complete_stage_receipt',
            **{k:receipt.get(k) for k in ('scope','status','objective','bound','gap','seconds',
                'initial_objective','variables','binary_variables','constraints')})
    recovered=list(native.values())
    checkpoint=reader.json(base.file_ref(checkpoint_path)) if checkpoint_path.is_file() else None
    return dict(event_count=len(events),scalar_native_FINAL_records=len(finals),
        complete_stage_native_records=stage_records,native_final_records=len(recovered),
        native_time_limits=sum(e.get('status')==9 for e in recovered),
        native_status_counts=dict(Counter(str(e.get('status')) for e in recovered)),
        observed_wall_seconds=max((e.get('wall_seconds') or 0. for e in events),default=None),
        accepted_path_points=len(checkpoint.get('path') or []) if checkpoint else None,
        checkpoint_available=checkpoint is not None,
        recovered_native_records=recovered,orphan_native_available=(directory/'native.json').is_file(),
        deduplication='Match stage and sequential call ordinal from *_started events and ordered complete stage receipts',
        scope='Incomplete attempt evidence; no frozen candidate or independent success is inferred')


def aggregate(rows):
    result=base.status_summary(rows);complete=[r for r in rows if r['status']=='complete']
    result['metrics']={}
    for split in SPLITS:
        chosen=[r['bank']['selections'][split]['mse'] for r in complete]
        metric=dict(evaluated=len(chosen),internal_correct=sum(c['topology']['internal']['exact'] for c in chosen),
            full_correct=sum(c['topology']['full']['exact'] for c in chosen),
            internal_F1_mean=_mean(c['topology']['internal']['f1'] for c in chosen),
            internal_FP_sum=sum(c['topology']['internal']['fp'] for c in chosen),
            internal_FN_sum=sum(c['topology']['internal']['fn'] for c in chosen),
            R_relative_error_mean=_mean(c['response']['matrix']['R']['relative_frobenius'] for c in chosen),
            X_relative_error_mean=_mean(c['response']['matrix']['X']['relative_frobenius'] for c in chosen))
        for kind in ('internal','full'):
            metric[kind+'_accuracy_completed']=metric[kind+'_correct']/len(chosen) if chosen else None
            metric[kind+'_accuracy_planned']=metric[kind+'_correct']/len(rows) if rows and result['all_tasks_terminal'] else None
        metric['selected_origin_counts']=dict(Counter(r['attribution']['selected'][split]['origin'] or 'unavailable' for r in complete))
        metric['contraction_repaired_full_count']=sum(
            not r['attribution']['selected'][split].get('before_contraction',{}).get('full',{}).get('exact',True)
            and r['attribution']['selected'][split]['full_exact'] for r in complete)
        result['metrics'][split]=metric
    partial=[r['partial'] for r in rows if r.get('partial')]
    result['coverage']={kind:sum(r['bank']['coverage'][kind] for r in complete) for kind in ('internal','full')}
    result['seconds_completed_sum']=sum(r.get('seconds') or 0. for r in complete)
    result['seconds_completed_mean']=_mean(r.get('seconds') for r in complete)
    result['seconds_all_receipts_sum']=sum(r.get('seconds') or 0. for r in rows)
    result['solver']=dict(completed_native_calls=sum(r['solver']['native_calls'] for r in complete),
        completed_native_time_limits=sum(r['solver']['timeouts'] for r in complete),
        partial_native_final_records=sum(p['native_final_records'] for p in partial),
        partial_native_time_limits=sum(p['native_time_limits'] for p in partial),
        bound_conflicts=sum(r['solver']['bound_conflicts'] for r in complete),
        bound_history_conflicts=sum(r['solver']['bound_history_conflicts'] for r in complete),
        callback_write_errors=sum(r.get('callback_write_errors',0) for r in complete),
        global_tree_optimality=False)
    result['solver']['stages']={}
    for stage in ('W','E'):
        events=[v for r in complete for v in r['solver']['events'] if v['stage']==stage]
        result['solver']['stages'][stage]=dict(native_calls=sum(v['solve_started'] for v in events),
            native_time_limits=sum(v['timed_out'] for v in events),
            closed_native_calls=sum(v['native_subproblem_certificate'] for v in events),
            native_seconds=sum(v['seconds'] or 0. for v in events),
            pretrial_seconds=sum(v['pretrial_seconds'] or 0. for v in events))
    return result


def E_state(row):
    events=[v for v in row.get('solver',{}).get('events',[]) if v['stage']=='E']
    if not events or not any(v.get('solve_started') for v in events):return 'not_started'
    if any(v.get('timed_out') or v.get('status')==9 for v in events):return 'time_limit'
    if all(v.get('solve_started') and v.get('native_subproblem_certificate') and v.get('status')==2 for v in events):return 'closed'
    return 'unresolved'


def _pair_outcomes(details):
    result=dict(pairs=len(details),input_ids=[r['input_id'] for r in details])
    for kind in ('internal','full'):
        gain=[r['input_id'] for r in details if not r['left_'+kind+'_correct'] and r['right_'+kind+'_correct']]
        loss=[r['input_id'] for r in details if r['left_'+kind+'_correct'] and not r['right_'+kind+'_correct']]
        result[kind]=dict(gains=len(gain),losses=len(loss),net=len(gain)-len(loss),gain_ids=gain,loss_ids=loss)
    return result


def paired(left,right,split):
    a={r['input_id']:r for r in left};b={r['input_id']:r for r in right}
    base.require(len(a)==len(left) and len(b)==len(right),'Duplicate paired input')
    keys=sorted(set(a)|set(b));complete=[(k,a[k],b[k]) for k in keys if k in a and k in b
        and a[k]['status']==b[k]['status']=='complete']
    result=dict(planned_inputs=len(keys),completed_pairs=len(complete),unpaired_or_incomplete=len(keys)-len(complete),
        left_status_counts=dict(Counter(r['status'] for r in left)),right_status_counts=dict(Counter(r['status'] for r in right)),
        split=split,direction='right minus left; loss means the right variant loses a correct recovery')
    details=[]
    for key,x,y in complete:
        base.close(x['bank']['reference_mse'],y['bank']['reference_mse'],
                   'Paired variants have different MSE calibration',rtol=1e-9,atol=1e-15)
        px=x.get('attribution',{}).get('pool',{});py=y.get('attribution',{}).get('pool',{})
        if px.get('available') and py.get('available'):
            base.require(px['rnj_pool_supports']==py['rnj_pool_supports'],'Paired variants have different RNJ support pools')
        cx=x['bank']['selections'][split]['mse']['topology'];cy=y['bank']['selections'][split]['mse']['topology']
        details.append(dict(input_id=key,left_internal_correct=cx['internal']['exact'],right_internal_correct=cy['internal']['exact'],
            left_full_correct=cx['full']['exact'],right_full_correct=cy['full']['exact'],
            internal_F1_delta=cy['internal']['f1']-cx['internal']['f1'],
            internal_FP_delta=cy['internal']['fp']-cx['internal']['fp'],internal_FN_delta=cy['internal']['fn']-cx['internal']['fn'],
            seconds_left=x.get('seconds'),seconds_right=y.get('seconds'),
            E_state_left=E_state(x),E_state_right=E_state(y),
            right_over_left_time_ratio=y['seconds']/x['seconds'] if (x.get('seconds') or 0)>0 and (y.get('seconds') or 0)>0 else None))
    result.update({k:v for k,v in _pair_outcomes(details).items() if k in ('internal','full')})
    closed=[r for r in details if r['E_state_left']==r['E_state_right']=='closed']
    limited=[r for r in details if 'time_limit' in (r['E_state_left'],r['E_state_right'])]
    other=[r for r in details if r not in closed and r not in limited]
    result['E_quality_strata']=dict(both_closed=_pair_outcomes(closed),
        at_least_one_time_limit=_pair_outcomes(limited),other_unclosed_or_not_started=_pair_outcomes(other))
    result.update(internal_F1_delta_mean=_mean(r['internal_F1_delta'] for r in details),
        right_over_left_time_ratio_median=float(np.median([r['right_over_left_time_ratio'] for r in details
            if r['right_over_left_time_ratio'] is not None])) if any(r['right_over_left_time_ratio'] is not None for r in details) else None,
        paired_seconds_left=sum(r['seconds_left'] or 0. for r in details),
        paired_seconds_right=sum(r['seconds_right'] or 0. for r in details),details=details,
        scope='Same-input descriptive comparisons; grid points and panels share physical days')
    return result


def evaluate_run(run,out):
    began=perf_counter();run,out=Path(run).resolve(),Path(out).resolve()
    base.require(not out.exists(),'Output exists; choose a new immutable evaluation directory');out.mkdir(parents=True)
    reader=base.Reader();protocol_ref=base.file_ref(run/'protocol.json');protocol=reader.json(protocol_ref)
    tasks=reader.json(protocol.get('tasks_ref',base.file_ref(run/'tasks.json')))
    entries=base._input_rows(reader,protocol);base._source_ref_checks(reader,protocol.get('sources',[]))
    base.require(len({t['task_id'] for t in tasks})==len(tasks),'Duplicate task IDs')
    base.require(len({(t['panel'],t['input_id'],t['variant']) for t in tasks})==len(tasks),'Duplicate task coordinates')
    cfg_groups=defaultdict(list)
    for task in tasks:
        cfg_groups[(task['panel'],task['input_id'],task.get('method','101'))].append(task)
    for group in cfg_groups.values():
        comparable=[{k:v for k,v in t.get('config',{}).items() if k!='w_mode'} for t in group]
        base.require(all(c==comparable[0] for c in comparable),'Assigned budgets or model settings differ between W modes')
    cache={};rows=[];errors=[];reference_by_input={};pool_by_input={}
    final_execution=False
    if (run/'execution.json').is_file():
        final_execution=reader.json(base.file_ref(run/'execution.json')).get('status') in ('all_terminal','complete','terminal_with_audit_failure')
    for task in tasks:
        entry=entries.get(task['input_id'],{});directory=(run/task['result_dir']).resolve()
        base.require(directory.is_relative_to(run),'Task directory escapes study root')
        row=dict(task_id=task['task_id'],input_id=task['input_id'],panel=task['panel'],variant=task['variant'],
            method=task.get('method','101'),loss=task.get('loss','mse'),case=entry.get('case'),N=entry.get('N'),
            pq_sd=entry.get('pq_sd'),repeat=entry.get('repeat'),status='pending',started=(directory/'started.json').exists())
        path=directory/'result.json'
        try:
            if path.is_file():
                result=reader.json(base.file_ref(path));base.require(result['task_id']==task['task_id'],'Task receipt identity differs')
                base.require(result.get('source_protocol')==protocol_ref,'Task used a different study protocol')
                status=result['status'];base.require(status in TERMINAL,'Unexpected terminal task status')
                row.update(status=status,result_ref=base.file_ref(path),seconds=base.finite(result.get('seconds')),error=result.get('error'),
                    failure_details_available=bool(result.get('failure_details')),total_cap_exceeded=result.get('total_cap_exceeded'))
                if status=='complete':
                    base.require(bool(entry),'Input not present in sealed manifest')
                    base.require(result.get('truth_used') is False and result.get('new_day_loaded') is False,
                                 'Worker information-boundary receipt is missing or violated')
                    native=reader.json(result['native']);row['treatment']=validate_treatment(task,native)
                    if task['input_id'] not in cache:cache[task['input_id']]=base.load_input(reader,entry)
                    data=cache[task['input_id']]
                    for key in ('train','validation'):
                        base.require(result['input_refs'][key+'.npz']==entry['files'][key+'.npz'],'Different '+key+' observations')
                        base.require(result['input_fingerprints'][key]==base.observation_fingerprint(data[key]),'Observation fingerprint differs')
                        handoff_key='training_sha256' if key=='train' else 'validation_sha256'
                        if handoff_key in native.get('handoff',{}):
                            base.require(native['handoff'][handoff_key]==result['input_fingerprints'][key],'Handoff fingerprint differs')
                    base.close(result['reference_mse'],native['reference_mse'],'MSE reference differs')
                    ref=float(native['reference_mse']);key=task['input_id']
                    if key in reference_by_input:
                        base.close(ref,reference_by_input[key],'Same-input variants use different MSE calibration',rtol=1e-9,atol=1e-15)
                    reference_by_input[key]=ref
                    guidance=native.get('rnj_pool_guidance')
                    if guidance is not None:
                        pool=sorted(_support_set(guidance['pool'],len(entry['labels'])))
                        if key in pool_by_input:base.require(pool==pool_by_input[key],'Same-input RNJ support pool differs')
                        pool_by_input[key]=pool
                    bank=base.evaluate_bank(native,data['train'],data['validation'],data['new_day'],data['geometry'],Hstar=data['Hstar'])
                    evidence=base.solver_evidence(native,data['train'],training_loss=task.get('loss','mse'),
                        training_reference=result.get('training_reference_loss',native.get('reference_loss',native['reference_mse'])))
                    row.update(bank=bank,solver=evidence,
                        attribution=attribution(native,bank,data['geometry'],evidence),native_ref=result['native'],
                        internal_correct=bank['selections']['validation']['mse']['topology']['internal']['exact'],
                        callback_write_errors=sum(len(m.get('progress_errors',[])) for m in native.get('MIQP',[])))
                else:row['partial']=partial_failure_evidence(reader,directory)
            elif final_execution:
                raise ValueError('All-terminal execution has no task receipt')
        except Exception as exc:
            row.update(original_status=row['status'],status='audit_failed',audit_error=repr(exc));errors.append(dict(task_id=task['task_id'],error=repr(exc)))
        rows.append(row)
    grouped=defaultdict(list);panels=defaultdict(list)
    for row in rows:
        grouped[(row['panel'],row['case'],row['N'],row['pq_sd'],row['variant'])].append(row)
        panels[(row['panel'],row['variant'])].append(row)
    groups=[dict(zip(('panel','case','N','pq_sd','variant'),key),**aggregate(value)) for key,value in sorted(grouped.items(),key=lambda x:str(x[0]))]
    panel_rows=[dict(panel=key[0],variant=key[1],**aggregate(value)) for key,value in sorted(panels.items())]
    comparisons=[]
    for panel in sorted({r['panel'] for r in rows}):
        for left,right in (('full','pool_only'),('pool_only','skip'),('full','skip')):
            for split in SPLITS:
                a=panels.get((panel,left),[]);b=panels.get((panel,right),[])
                if a or b:comparisons.append(dict(panel=panel,left=left,right=right,**paired(a,b,split)))
    for ref in list(reader.sources.values()):
        try:reader.checked(ref)
        except Exception as exc:errors.append(dict(scope='source_changed_during_evaluation',error=repr(exc)))
    summary=base.status_summary(rows)
    report=dict(status='failed' if errors else 'passed' if summary['all_tasks_terminal'] else 'incomplete',
        source_protocol=protocol_ref,evaluator=base.file_ref(__file__),sources=list(reader.sources.values()),
        **summary,errors=errors,groups=groups,panels=panel_rows,paired=comparisons,seconds=perf_counter()-began,
        guarantee='Saved native bounds cover the declared subproblems only; no globally optimal tree or truth probability claim',
        comparison='Three fresh interventions under shared inputs/configuration and fixed E cap; no inherited full run is the primary control',
        failure_scope='Completed native records and failed-attempt partial FINAL records are counted separately',
        dependence='Repeated noise, training lengths and long/grid panels share physical networks and days')
    with (out/'records.jsonl').open('w',encoding='utf8') as stream:
        for row in rows:stream.write(json.dumps(base.json_safe(row),ensure_ascii=False,allow_nan=False)+'\n')
    report['records']=base.file_ref(out/'records.jsonl');base._write(out/'report.json',report)
    fields=('panel','case','N','pq_sd','variant','planned','completed','split','internal_correct','full_correct',
        'internal_accuracy_completed','internal_accuracy_planned','full_accuracy_completed','full_accuracy_planned','internal_F1_mean',
        'internal_FP_sum','internal_FN_sum','seconds_completed_sum','seconds_all_receipts_sum')
    with (out/'summary.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group in groups:
            for split,metric in group['metrics'].items():
                writer.writerow({**{k:group[k] for k in fields if k in group},'split':split,**{k:metric[k] for k in fields if k in metric}})
    lines=['# W重要性配对消融','',f"状态：{report['status']}；计划{report['planned']}，完整任务{report['completed']}，审计错误{len(errors)}。",'',
        f"任务状态：{json.dumps(report['status_counts'],ensure_ascii=False)}。",'',
        '| 面板 | W模式 | 完整/计划 | 验证内部正确 | 验证全部支持正确 | 累计全部支持正确 | 完整任务秒数合计 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in panel_rows:
        v=p['metrics']['validation'];c=p['metrics']['cumulative']
        lines.append(f"| {p['panel']} | {p['variant']} | {p['completed']}/{p['planned']} | {v['internal_correct']} | {v['full_correct']} | {c['full_correct']} | {p['seconds_completed_sum']:.2f} |")
    lines+=['','| 面板 | 左→右 | 评分 | 完整配对 | 全支持新增 | 全支持丢失 | 两侧E均闭界 | 至少一侧E限时 | 右/左耗时中位比 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for p in comparisons:
        if p['split'] not in ('validation','cumulative'):continue
        ratio=p['right_over_left_time_ratio_median']
        lines.append(f"| {p['panel']} | {p['left']}→{p['right']} | {p['split']} | {p['completed_pairs']} | {p['full']['gains']} | "
            f"{p['full']['losses']} | {p['E_quality_strata']['both_closed']['pairs']} | {p['E_quality_strata']['at_least_one_time_limit']['pairs']} | "
            f"{ratio if ratio is not None else 'NA'} |")
    lines+=['','full与pool_only比较检验原生W是否增加恢复；pool_only与skip比较检验逐步QP路径是否增加恢复。',
        '配对净增/丢失、失败分母、F1、原生及partial限时保存在report.json；只有两侧完整的输入进入配对数值比较。',
        'E使用相同声明上限；另核实际求解状态与时间。移除W会改变E池、初值和最终竞争库，不能把源标签单独解释为因果。',
        'W前缀复评沿用已经保存的收缩结果，并非给W释放E预算后重新求解。',
        '结果描述这批固定网络和历史窗口，未完成格子没有准确率；全部终结时仍保留失败及未启动分母。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    base._write(out/'complete.json',dict(status=report['status'],report=base.file_ref(out/'report.json'),errors=len(errors)))
    return report


def study(runroot,allow_partial=True):
    root=Path(runroot).resolve();parent=root/'evaluations';parent.mkdir(exist_ok=True)
    out=parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    result=evaluate_run(root,out)
    receipt=dict(status=result['status'],output=str(out),report=base.file_ref(out/'report.json'),
        planned=result['planned'],completed=result['completed'],all_tasks_terminal=result['all_tasks_terminal'])
    base._write(out/'controller_receipt.json',receipt)
    if result['status']=='failed':raise RuntimeError('Independent audit failed; see '+str(out))
    if not allow_partial and not result['all_tasks_terminal']:raise RuntimeError('Final report requested with pending tasks; report retained at '+str(out))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=evaluate_run(args.run,args.out)
    print(json.dumps({k:result[k] for k in ('status','planned','completed','errors')},ensure_ascii=False))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
