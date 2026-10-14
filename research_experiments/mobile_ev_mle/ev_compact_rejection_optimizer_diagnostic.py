"""Post-hoc local optimizer stability on four saved rejection conditions.

No new observations, decisions, thresholds or EV choices are produced. The
original feasible state is retained as an objective upper bound. Instrumented
restarts reuse information_fit_recovery; independent branch arithmetic replays
every saved state. The frozen fitter receives the exact saved asset box.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
from . import likelihood as lk
from . import information_fit_recovery as recovery
from .information_fit_recovery import read_json, write_json, independent_replay, exact_candidate
from .information_replay import _obs, _meter
from . import ev_original_review as review

HERE=Path(__file__).resolve().parent
EXPECTED=((989201,'none'),(989205,'charge_fixed'),(989210,'none'),(989211,'none'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejection_targets(risk):
    """Use only the saved individual-policy rejection records to fix targets."""
    found=[]
    for history in risk['individual_policies']['evaluation']['records']:
        for method,record in history['methods'].items():
            if record.get('false_rejection'):
                true=history['true_name']
                found.append(dict(seed=int(history['history_id']),method=method,true_name=true,
                    original_gap=record['gaps'][true],original_threshold=record['thresholds'][true],
                    original_decision=record))
    found.sort(key=lambda r:(r['seed'],r['method']))
    if tuple((r['seed'],r['method']) for r in found)!=EXPECTED:
        raise ValueError('saved rejection target set differs from the four authorized conditions')
    return found


def saved_fit_options(trial):
    c=trial['config']['physical']
    if trial['source_interface']!=1 or trial['domain']!='asset':
        raise ValueError('expected saved common-source interface1 and asset box')
    return dict(r_bounds=c['asset_bounds']['r_bounds'],x_bounds=c['asset_bounds']['x_bounds'],
                p_bounds=(-1.,1.),q_bounds=(-1.,1.),source_bounds=(-.5,.5),voltage_bounds=(.85,1.15))


def box_replay(candidate,row,obs,meter,options):
    result=independent_replay(candidate,row,obs,meter)
    ok=True
    for key in ('r','x'):
        lo,hi=options[key+'_bounds'];value=np.asarray(row[key],float)
        ok=ok and bool(np.all(value>=np.asarray(lo)-1e-12) and np.all(value<=np.asarray(hi)+1e-12))
    result['saved_asset_box_feasible']=ok
    if not ok:result['errors'].append('saved_asset_box')
    result['success']=bool(result['success'] and ok)
    return result


def retain_best(original,run):
    """A failed numerical stop can still be a feasible objective upper bound."""
    options=[('original',original,original['independent_replay'])]
    for start in run['starts']:
        for key in ('initial','returned','last_finite_state'):
            if key in start:
                state=start[key];options.append((f"restart_{start['index']}_{key}",state,state['independent_replay']))
    if run.get('returned_best_fit') is not None:
        options.append(('returned_best_fit',run['returned_best_fit'],run['returned_best_fit_replay']))
    feasible=[item for item in options if item[2]['success']]
    if not feasible:raise ValueError('no independently feasible original or new state')
    label,state,audit=min(feasible,key=lambda item:item[2]['full_SSE'])
    return dict(label=label,state=state,SSE=audit['full_SSE'],loglik=audit['loglik'],
        success=bool(state.get('success',state.get('solver_success',False))),
        status=state.get('status'),optimality=state.get('optimality'),
        feasible_state_count=len(feasible),global_MLE_certified=False)


def run_candidate(candidate,original,obs,meter,options,seed,max_nfev=500,multistart=8):
    # Runtime argument adaptation only; no frozen source file is edited.
    original_fitter=lk.fit_candidate
    def bounded(*args,**kwargs):return original_fitter(*args,**kwargs,**options)
    lk.fit_candidate=bounded
    try:
        result=recovery.run_fit(candidate,obs,meter,initial=original,
            mode='saved_full_joint_state_plus_seeded_line_restarts',seed=seed,
            max_nfev=max_nfev,multistart=multistart)
    finally:lk.fit_candidate=original_fitter
    result['options'].update(options)
    audits=[]
    for start in result['starts']:
        for key in ('initial','returned','last_finite_state'):
            if key in start:
                start[key]['independent_replay']=box_replay(candidate,start[key],obs,meter,options)
                audits.append(start[key]['independent_replay'])
    if result.get('returned_best_fit') is not None:
        result['returned_best_fit_replay']=box_replay(candidate,result['returned_best_fit'],obs,meter,options)
        audits.append(result['returned_best_fit_replay'])
    result['all_replays_passed']=all(a['success'] for a in audits)
    result['replay_count']=len(audits)
    result['best_feasible_attained']=retain_best(original,result)
    if result['starts']:
        first=result['starts'][0]['initial']
        result['first_start_max_coordinate_shift_from_original']={
            k:float(np.max(abs(np.asarray(first[k])-np.asarray(original[k]))))
            for k in ('r','x','p','q','source_p','source_q')}
    return result


def run(parent,risk_path,output,*,max_nfev=500,multistart=8):
    parent,risk_path,output=map(Path,(parent,risk_path,output))
    if output.exists():raise ValueError('use a new diagnostic directory')
    freeze=read_json(parent/'freeze.json');risk=read_json(risk_path)
    targets=rejection_targets(risk)
    ac_review=read_json(parent/'review_full.json')
    if not(ac_review['success'] and ac_review['audited_histories']==12
           and ac_review['policy_id']==freeze['policy_id']):raise ValueError('parent full review gate')
    source=review.source_audit(parent,freeze)
    if not(source['current_source_matches'] and source['source_count']==72):raise ValueError('source72 changed')
    source_files={name:HERE/name for name in ('ev_compact_rejection_optimizer_diagnostic.py',
                  'information_fit_recovery.py','ev_original_review.py','gradient_price_study.py','information_replay.py')}
    provenance={str(parent/'freeze.json'):sha(parent/'freeze.json'),str(parent/'source_snapshot.json.gz'):sha(parent/'source_snapshot.json.gz'),
                str(parent/'review_full.json'):sha(parent/'review_full.json'),str(risk_path):sha(risk_path)}
    for t in targets:
        p=parent/f"trial_{t['seed']}.json.gz";provenance[str(p)]=sha(p)
        if ac_review['trial_sha256'][p.name]!=sha(p):raise ValueError('parent raw review binding')
    protocol=dict(schema='ev_compact_posthoc_optimizer_diagnostic_v1',policy_id=freeze['policy_id'],
        targets=targets,independent_histories=4,method_conditions=4,candidates_per_condition=4,
        source72_sha256=freeze['source_sha256'],auxiliary_source_sha256={k:sha(v) for k,v in source_files.items()},
        input_sha256=provenance,original_decisions_unchanged=True,statistical_reclassification=False,
        new_calibration_or_evaluation_samples=0,new_selector_runs=0,uses_truth_in_fit_or_initialization=False,
        best_feasible_includes_original_upper_bound=True,global_MLE_certified=False,
        seed_rule='saved policy seed+802+candidate bank index, matching original joint noisy seed rule',
        options=dict(multistart=multistart,max_nfev=max_nfev,tol=1e-10,linear_solver='exact'),
        initial_state='saved original full H+future joint physical coordinates; fitter interior clipping recorded',
        restart_rule='seven additional seeded multiplicative lognormal R/X starts; same saved full joint states',
        scope='post-hoc search stability on selected rejection events; not a newly calibrated decision process',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    write_json(output/'protocol.json',protocol)
    table=[];condition_rows=[];started=perf_counter()
    for target in targets:
        path=parent/f"trial_{target['seed']}.json.gz";trial=read_json(path)
        record=next(r for r in trial['rows'] if r['method']==target['method'])
        if record['status']!='complete' or len(record['fits'])!=4:raise ValueError('original fit coverage')
        if record['true_name']!=target['true_name']:raise ValueError('risk/raw true candidate mismatch')
        # Truth and power are deliberately absent from all solver arguments.
        history=_obs(trial['history']);future=_obs(record['observations']);obs=lk.concat_observations(history,future)
        meter=_meter(trial['measurement_models']['jm']);options=saved_fit_options(trial)
        fits=[]
        bank_names=[r['name'] for r in trial['proposal']['bank']]
        for saved in record['fits']:
            index=bank_names.index(saved['name'])
            candidate=exact_candidate(saved,obs.n);original=dict(saved)
            original['independent_replay']=box_replay(candidate,original,obs,meter,options)
            if not original['independent_replay']['success']:raise ValueError('original objective replay')
            seed=trial['seeds']['policy']+802+index
            result=run_candidate(candidate,original,obs,meter,options,seed,max_nfev,multistart)
            best=result['best_feasible_attained'];new=result.get('returned_best_fit')
            row=dict(seed=target['seed'],method=target['method'],candidate=candidate.name,
                original_SSE=original['loss'],new_returned_SSE=None if new is None else new['loss'],
                best_attained_SSE=best['SSE'],improvement=original['loss']-best['SSE'],
                original_loglik=original['loglik'],best_attained_loglik=best['loglik'],
                original_success=original['success'],original_status=original['status'],
                original_optimality=original['optimality'],new_success=None if new is None else new['success'],
                new_status=None if new is None else new['status'],new_optimality=None if new is None else new['optimality'],
                best_label=best['label'],best_status=best['status'],best_optimality=best['optimality'],
                solver_seconds=result['solver_seconds'],AC_calls=result['attempted_AC_calls']['attempted'],
                replay_count=result['replay_count'],restart_attempts=len(result['starts']),all_replays_passed=result['all_replays_passed'],
                original_R=original['r'],original_X=original['x'],best_R=best['state']['r'],best_X=best['state']['x'])
            table.append(row);fits.append(dict(candidate=candidate.name,original=original,recovery=result,summary=row))
            print(f"{target['seed']} {target['method']} {candidate.name}: original={original['loss']:.12g} best={best['SSE']:.12g} improvement={row['improvement']:.6g}",flush=True)
        original_losses={f['candidate']:f['original']['loss'] for f in fits}
        best_losses={f['candidate']:f['recovery']['best_feasible_attained']['SSE'] for f in fits}
        true=target['true_name']
        current=dict(**target,original_true_gap=original_losses[true]-min(original_losses.values()),
            new_best_attained_true_gap=best_losses[true]-min(best_losses.values()),
            original_losses=original_losses,best_attained_losses=best_losses,
            max_candidate_SSE_improvement=max(f['summary']['improvement'] for f in fits),
            original_decision_retained_unchanged=True,new_calibrated_decision=None)
        condition_rows.append(current)
        write_json(output/f"condition_{target['seed']}_{target['method']}.json.gz",dict(
            target=target,input_raw_sha256=sha(path),source_interface=1,options=options,
            future_observations=record['observations'],history_observations=trial['history'],
            joint_meter=trial['measurement_models']['jm'],candidates=fits,summary=current))
        write_json(output/'checkpoint.json',dict(completed_conditions=len(condition_rows),planned_conditions=4))
    after=review.source_audit(parent,freeze)
    unchanged=all(sha(Path(p))==value for p,value in provenance.items()) and all(sha(source_files[n])==v for n,v in protocol['auxiliary_source_sha256'].items())
    summary=dict(policy_id=freeze['policy_id'],conditions=condition_rows,planned_conditions=4,
        actual_conditions=len(condition_rows),candidate_fits=len(table),restart_attempts=sum(f['restart_attempts'] for f in table),
        AC_calls=sum(f['AC_calls'] for f in table),solver_seconds=sum(f['solver_seconds'] for f in table),
        elapsed_seconds=perf_counter()-started,all_replays_passed=all(f['all_replays_passed'] for f in table),
        immutable_inputs_and_auxiliary_sources_unchanged=unchanged,source72_current_matches=after['current_source_matches'],
        original_decisions_unchanged=True,statistical_reclassification=False,global_MLE_certified=False)
    summary['success']=bool(summary['all_replays_passed'] and unchanged and after['current_source_matches'] and len(table)==16)
    write_json(output/'summary.json',summary)
    with (output/'candidate_comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--risk',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-nfev',type=int,default=500);parser.add_argument('--multistart',type=int,default=8)
    args=parser.parse_args();result=run(args.parent,args.risk,args.output,max_nfev=args.max_nfev,multistart=args.multistart)
    print({k:result[k] for k in ('success','actual_conditions','candidate_fits','AC_calls','elapsed_seconds')},flush=True)


if __name__=='__main__':main()
