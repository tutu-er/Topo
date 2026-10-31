"""Frozen observed-only baseline for two predeclared existing failed sources.

prepare writes protocol before loading observations. run makes exactly one
existing fit_observed call through SharedFitter, lp_method=2. evaluate first
freezes the complete result, then parses static topology for evaluator-only
full-block errors. No future, test load, AC, or DR is generated.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback
import zipfile

REPO=Path(__file__).resolve().parents[3]
TASK=REPO/'research_experiments/wzzt_dr_selection'
DATA_ROOT=Path(r'D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full')
JOB='main__paper15__b2__reference__v0005__N288'
TARGETS={JOB:[103,104,105], 'main__flynn16__b4__reference__v0005__N288':[201,202,203]}

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)
def reference(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def bound(reference):
    if sha(reference['path'])!=reference['sha256']:raise RuntimeError('input drift '+reference['path'])
    return Path(reference['path'])
def setup():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    sys.path.insert(0,str(REPO))
def configuration(root):
    root=Path(root).resolve();p=read(root/'baseline_protocol.json')
    for filename,digest in p['source_sha256'].items():
        if sha(filename)!=digest:raise RuntimeError('baseline source drift '+filename)
    for item in p['inputs'].values():bound(item)
    return root,p

def prepare(root,job_id=JOB):
    setup();import numpy,pandas,scipy,gurobipy
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import load_history,forecast_mean
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'baseline_protocol.json').exists():raise FileExistsError('baseline protocol already exists; no overwrite')
    inventory_path=TASK/'evidence/scenario_review_v1/source_inventory.json'
    row=next(r for r in read(inventory_path)['rows'] if r['job_id']==job_id)
    row={k:v for k,v in row.items() if k not in ('test','existing_result')}
    case=row['case']
    metadata=DATA_ROOT/'networks'/case/'reference/metadata.json';meta=read(metadata)
    source_metadata=dict(path=str(metadata),sha256=sha(metadata),case_name=meta['source']['case_name'])
    old_result=DATA_ROOT/'jobs'/job_id/'result.json'
    inputs=dict(train=reference(row['train']['path']),validation=reference(row['validation']['path']),
        source_metadata=reference(metadata),static_buses=reference(metadata.parent/'buses.csv'),
        static_branches=reference(metadata.parent/'branches.csv'),original_result=reference(old_result),
        historical_inventory=reference(inventory_path))
    if job_id != JOB:
        selection_path=root.parent/'baseline_source_selection_v2.json'
        selection=read(selection_path)
        if selection.get('status')!='frozen' or selection.get('selected_job_id')!=job_id:
            raise RuntimeError('new source must match the pre-DR source selection lock')
        inputs['case_selection_lock']=reference(selection_path)
    assert inputs['train']['sha256']==row['train']['catalog_sha256']
    assert inputs['validation']['sha256']==row['validation']['catalog_sha256']
    algorithm=source_hashes();sources=dict(algorithm)
    for file in [Path(__file__),TASK/'dr_pool_v2/matrix_runner.py',TASK/'dr_competition_bridge/physical_bridge.py',TASK/'dr_balanced_design/budget_reference_diagnostic.py']:
        sources[str(file.resolve())]=sha(file)
    protocol=dict(id='targeted_pricing_current_baseline_v2',status='frozen',created_utc=now(),repository=str(REPO),
        case=case,job_id=job_id,root_label=int(meta['root_bus']),base_kw=1000.*float(meta['base_mva']),
        voltage_sigma=.0005,pq_sigma=.005,history_rows=[288,288],target_support_labels=TARGETS[job_id],
        source_metadata=source_metadata,history_index_row=row,inputs=inputs,source_sha256=sources,
        algorithm_source_sha256=algorithm,identifier='dr_pool_v2.adapter.fit_observed',lp_method=2,search_seconds=25.,
        preserved_rules=dict(complete_RNJ_families=15,max_extra_starts=2,relative_gain=.001,final_validation=.03,contract=.16,zero_weights=True),
        cohort='existing final joint failure, fixed before any new DR result; old label not assumed to remain wrong',
        truth_boundary='static topology bytes hashed now; graph parsed for scoring only after output freeze',
        original_result_semantic_read_in_this_script=False,no_test_or_future_observations=True,no_DR_or_AC=True,
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=numpy.__version__,
            pandas=pandas.__version__,scipy=scipy.__version__,gurobi=list(gurobipy.gurobi.version())))
    write(root/'baseline_protocol.json',protocol)
    with zipfile.ZipFile(root/'baseline_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for index,file in enumerate(sorted(sources)):z.write(file,arcname=f'{index:03d}_'+Path(file).name)
    # Observations are first loaded after this exact source/input/parameter lock.
    history=load_history(row,expected_case=case,base_kw=protocol['base_kw'],source_metadata=source_metadata)
    forecast=forecast_mean(history)
    for key in ('training','validation'):assert sum(len(v['P_terminal']) for v in history[key])==288
    with (root/'history.pkl').open('xb') as f:pickle.dump(dict(history=history,forecast=forecast),f,protocol=4)
    write(root/'baseline_prepared.json',dict(status='prepared',created_utc=now(),protocol=reference(root/'baseline_protocol.json'),
        history_packet=reference(root/'history.pkl'),source_snapshot=reference(root/'baseline_source_snapshot.zip'),
        rows=[288,288],labels=history['labels'],history_receipt=history['receipt'],graph_parsed=False,solver_calls=0,AC_calls=0))
    write(root/'BASELINE_RECOVERY.json',dict(status='prepared',next_command='run',root=str(root),
        script=str(Path(__file__).resolve()),protocol=reference(root/'baseline_protocol.json'),history_packet=reference(root/'history.pkl'),
        fit_folder=str(root/'baseline_fit'),do_not_restart_terminal_attempt=True))
    print(json.dumps(dict(status='prepared',job_id=job_id,protocol_sha256=sha(root/'baseline_protocol.json'),history_sha256=sha(root/'history.pkl'),rows=[288,288])),flush=True)

def run(root):
    setup();root,p=configuration(root)
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter
    prepared=read(root/'baseline_prepared.json');bound(prepared['protocol']);history_path=bound(prepared['history_packet'])
    assert source_hashes()==p['algorithm_source_sha256']
    if (root/'baseline_ended.json').exists():raise FileExistsError('terminal baseline already retained')
    lock=root/'baseline_driver.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.perf_counter();receipt=dict(status='started',created_utc=now(),protocol_sha256=sha(root/'baseline_protocol.json'),
        rows=[288,288],no_DR_or_AC=True,truth_read_by_driver=False)
    try:
        with history_path.open('rb') as f:history=pickle.load(f)['history']
        fitter=SharedFitter(root/'baseline_cache',sources=p['algorithm_source_sha256'],max_new_fits=1)
        result=fitter(history['training'],history['validation'],root/'baseline_fit',root_label=p['root_label'],
            search_seconds=p['search_seconds'],lp_method=p['lp_method'])
        assert result['training_rows']==result['validation_rows']==288
        receipt.update(status='complete',result=reference(root/'baseline_fit/result.json'),
            input_sha256=result['input_sha256'],LP_calls=len(result['LP_solves']),search_stops=result['search_stop_statuses'],
            elapsed_seconds=time.perf_counter()-started)
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
    finally:
        receipt['completed_utc']=now();write(root/'baseline_ended.json',receipt);lock.unlink(missing_ok=True)
    write(root/'BASELINE_RECOVERY.json',dict(status=receipt['status'],next_command='evaluate' if receipt['status']=='complete' else 'inspect_failure_no_retry',
        root=str(root),script=str(Path(__file__).resolve()),protocol=reference(root/'baseline_protocol.json'),
        history_packet=prepared['history_packet'],ended=reference(root/'baseline_ended.json'),do_not_restart_terminal_attempt=True))
    print(json.dumps({k:receipt.get(k) for k in ('status','rows','LP_calls','search_stops','elapsed_seconds','error_type','message')},ensure_ascii=False),flush=True)
    return 0 if receipt['status']=='complete' else 1

def static_truth(root_label,labels,buses_path,branches_path):
    buses=list(csv.DictReader(buses_path.open(encoding='utf-8-sig')))
    observed={int(b['bus_id']) for b in buses if b['bus_type']=='observed_terminal'}
    assert observed==set(labels)
    adjacency={int(b['bus_id']):[] for b in buses};closed=[]
    for edge in csv.DictReader(branches_path.open(encoding='utf-8-sig')):
        status=edge.get('status','True').lower() in ('true','1')
        actual=edge.get('is_true_closed','True').lower() in ('true','1')
        if not status or not actual:continue
        i,j=int(edge['from_bus']),int(edge['to_bus']);adjacency[i].append((j,edge));adjacency[j].append((i,edge));closed.append((i,j))
    seen=set();family=set()
    def visit(node,parent):
        if node in seen:raise ValueError('static truth is not a rooted tree')
        seen.add(node);descendants={node}&observed
        for child,edge in adjacency[node]:
            if child==parent:continue
            block=visit(child,node);descendants.update(block)
            if block and (float(edge['r_ohm'])!=0. or float(edge['x_ohm'])!=0.):family.add(tuple(sorted(block)))
        return descendants
    assert visit(root_label,None)==observed
    assert len(closed)==len(seen)-1
    return [list(s) for s in sorted(family)]

def evaluate(root):
    setup();root,p=configuration(root);ended=read(root/'baseline_ended.json')
    if ended['status']!='complete':raise RuntimeError('failed baseline retained; cannot score a partial result as complete')
    result_path=bound(ended['result'])
    if (root/'baseline_evaluation.json').exists():raise FileExistsError('baseline already evaluated')
    # This receipt is committed to disk before the static truth graph is parsed.
    write(root/'baseline_fits_frozen.json',dict(status='frozen',created_utc=now(),result=ended['result'],
        baseline_ended=reference(root/'baseline_ended.json'),protocol_sha256=sha(root/'baseline_protocol.json'),graph_not_yet_parsed=True))
    result=read(result_path);labels=result['terminal_labels']
    truth=static_truth(p['root_label'],labels,bound(p['inputs']['static_buses']),bound(p['inputs']['static_branches']))
    from research_experiments.wzzt_dr_selection.dr_balanced_design.budget_reference_diagnostic import score_fit,search_audit,_family,_error
    scores=score_fit(result,truth,p['root_label']);target=tuple(p['target_support_labels']);target_indices=tuple(labels.index(v) for v in target)
    targetset={tuple(s) for s in truth};chosen=result['models']['multistart_original_rule'];before=chosen['selected_uncontracted']
    candidates=result['selection_candidates'];best=min(v['validation_mae'] for v in candidates);cut=1.03*best+1e-12
    details=[]
    for i,point in enumerate(candidates):
        family=_family(point,labels)
        details.append(dict(index=i,origin=point.get('origin'),config=point.get('config'),path_index=point.get('path_index'),
            support_count=len(family),target_present=target in family,full_errors=_error(family,targetset)['full_errors'],
            objective=point['objective'],validation_mae=point['validation_mae'],eligible=point['validation_mae']<=cut))
    paths={}
    for name,search in result['searches'].items():
        paths[name]=dict(reason=search['reason'],points=[dict(index=i,target_present=target in _family(v,labels),
            objective=v['objective'],support_count=len(v['supports'])) for i,v in enumerate(search['path'])],
            target_scans=[dict(iteration=i,current=d['current'],target_objective=v['objective'],gain_current_minus_target=d['current']-v['objective'],
                actual_acceptance_threshold=1e-10+.001*d['current'],removed=v.get('removed'))
                for i,d in enumerate(search['decisions']) for v in d['candidates'] if tuple(v['candidate'])==target_indices])
    output=dict(status='evaluated',created_utc=now(),protocol_sha256=sha(root/'baseline_protocol.json'),fits_frozen=reference(root/'baseline_fits_frozen.json'),
        rows=[result['training_rows'],result['validation_rows']],root_label=p['root_label'],labels=labels,target_support_labels=list(target),
        true_supports=truth,full_truth_blocks=len(truth),scores=scores,search_audit=search_audit(result),LP_calls=len(result['LP_solves']),
        current_target_missing=target not in _family(chosen,labels),current_has_any_full_errors=scores['multistart_original_rule']['full_errors']>0,
        old_failure_not_assumed=True,raw_selected_support_labels=[list(s) for s in sorted(_family(before,labels))],
        final_support_labels=chosen['support_labels'],selected_raw_index=before.get('selection_index'),selected_raw_config=before.get('config'),
        selected_raw_origin=before.get('origin'),best_validation_mae=best,validation_cutoff=cut,
        selected_raw_validation=before['validation_mae'],selected_validation_ratio_to_best=before['validation_mae']/best if best else None,
        selected_distance_inside_validation_band=cut-before['validation_mae'],candidates=details,paths=paths,
        truth_definition='all positive R or X physical edges; repeated descendant support aggregated, exact-zero edges inactive; no output threshold on truth',
        solver_or_AC_calls_in_evaluation=0)
    write(root/'baseline_evaluation.json',output)
    prepared=read(root/'baseline_prepared.json')
    write(root/'BASELINE_RECOVERY.json',dict(status='evaluated',root=str(root),protocol=reference(root/'baseline_protocol.json'),
        history_packet=prepared['history_packet'],evaluation=reference(root/'baseline_evaluation.json'),
        fit_result=ended['result'],observed_source_metadata=p['source_metadata'],target=list(target),
        current_target_missing=output['current_target_missing'],next_step='lock targeted DR only after baseline evidence review',
        do_not_rerun_completed_baseline=True))
    print(json.dumps(dict(status='evaluated',rows=output['rows'],LP_calls=output['LP_calls'],current_target_missing=output['current_target_missing'],
        scores={k:v['full_errors'] for k,v in scores.items()},raw_error=scores['multistart_original_rule']['before_contraction_errors'],
        origin=output['selected_raw_origin'],validation_ratio=output['selected_validation_ratio_to_best'],search_stops=result['search_stop_statuses']),ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','evaluate'));parser.add_argument('--root',required=True);parser.add_argument('--job-id',choices=tuple(TARGETS),default=JOB)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root,args.job_id);return 0
    if args.command=='run':return run(args.root)
    evaluate(args.root);return 0
if __name__=='__main__':raise SystemExit(main())
