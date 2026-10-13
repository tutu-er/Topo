"""Small finite-library audit, separate from the continuous EV selector.

The only loaded controller inputs are public historical meters/fits, candidate
names, device conditions and policy seed. Realized future meters and truth are
never accessed. Enumeration here is a comparison on two completed pilots.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .likelihood import Observations,FitResult,RadialCandidate
from .simulation import model,BASE_KW
from .single_site_milp import SingleSiteConstraints,SingleSitePlan,constant_site_plan,check_single_site_plan,solve_site_step
from .gradient_price import GradientPriceEvaluator
from .gradient_price_search import evaluation_record

HERE=Path(__file__).resolve().parent


def safe(value):
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items() if k!='fit_results'}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value


def save(path,value,compressed=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(safe(value),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(b,mtime=0) if compressed else b)


def load_history_inputs(path):
    path=Path(path);data=json.loads(gzip.decompress(path.read_bytes()))
    # Deliberately access no truth or actual future record keys.
    snapshot=dict(history_id=data['history_id'],config=data['config'],source_sha256=data['source_sha256'],
        policy_id=data['policy_id'],policy_seed=data['seeds']['policy'],history_observations=data['history_observations'],
        historical_fits=data['historical_fits'],design_candidates=data['selection']['design_candidates'],
        device_constraints=data['device_constraints'],main_plan=data['selection']['searches']['full']['plan'],
        main_stored_score=data['selection']['searches']['full']['score'])
    history=Observations(**{k:None if v is None else np.asarray(v,float) for k,v in snapshot['history_observations'].items()})
    fits={};candidates=[]
    selected=set(snapshot['design_candidates'])
    for row in snapshot['historical_fits']:
        if row['name'] not in selected:continue
        candidate=RadialCandidate.from_supports(row['name'],history.n,row['supports']);candidates.append(candidate)
        values={k:np.asarray(row[k],float) for k in ('r','x','p','q','source_p','source_q','bias_p','bias_q','bias_v')}
        fit=FitResult(candidate=row['name'],**values,loss=row['loss'],loglik=row['loglik'],success=row['success'],
            status=row['status'],message=row['message'],nfev=row['nfev'],optimality=row['optimality'],
            predicted={k:np.asarray(v,float) for k,v in row['predicted'].items()},diagnostics=row['diagnostics'])
        fits[row['name']]=fit
    if set(fits)!=selected:raise ValueError('necessary historical reference fit missing')
    # Preserve the declaration order used by the saved core evaluator seeds.
    byname={c.name:c for c in candidates};candidates=[byname[name] for name in snapshot['design_candidates']]
    return snapshot,history,candidates,fits


def plan_from_summary(row):
    return SingleSitePlan(row['site'],np.asarray(row['p']),np.asarray(row['q']),np.asarray(row['sites']),np.asarray(row['soc']),row.get('solver',{}))


def run_history(path,*,deadline=None):
    began=perf_counter();snapshot,history,candidates,fits=load_history_inputs(path)
    constraints=SingleSiteConstraints(**snapshot['device_constraints']);t=snapshot['config']['future_points']
    evaluator=GradientPriceEvaluator(history,candidates,fits,model(history.n),horizon=t,base_kva=BASE_KW,
        q_over_p=constraints.q_over_p,fit_options=snapshot['config']['design_fit_options'],seed=snapshot['policy_seed']+100)
    charge,discharge,access,travel,_=constraints.arrays(t,history.n)
    if np.any(discharge>0) or not np.allclose(charge,charge[0,0]):raise ValueError('audit is restricted to uniform charge-only cap')
    cap=float(charge[0,0]);count=constraints.energy/(constraints.dt_hours*cap)
    if abs(count-round(count))>1e-10:raise ValueError('energy task must equal integer full-power slots')
    count=int(round(count));cache={};records=[];failures=[]
    def evaluate(label,plan,kind,operation):
        if deadline is not None and perf_counter()>=deadline:return False
        validation=check_single_site_plan(plan,constraints)
        if not validation['feasible']:
            failures.append(dict(action=label,kind='operation',validation=validation));return True
        key=(plan.site,np.asarray(plan.p,dtype='<f8').tobytes())
        if key not in cache:cache[key]=evaluator.evaluate(plan.site,BASE_KW*plan.p[:,plan.site])
        result=cache[key]
        records.append(dict(action=label,kind=kind,site=plan.site,plan=plan.summary(),operation=operation,
            validation=validation,evaluation=evaluation_record(result)))
        if not result['success']:failures.append(dict(action=label,kind='necessary_AC_comparison',failures=result['failures']))
        return True
    main=plan_from_summary(snapshot['main_plan'])
    budget_stopped=not evaluate('saved_main_design',main,'main',dict(source='saved_controller_output'))
    for site in range(history.n):
        if budget_stopped:break
        constant=constant_site_plan(site,t,history.n,constraints)
        op={k:v for k,v in constant.items() if k!='plan'}
        if constant['success']:
            budget_stopped=not evaluate(f'constant_site_{site}',constant['plan'],'constant',op)
        else:failures.append(dict(action=f'constant_site_{site}',kind='constant_operation',operation=op))
        if budget_stopped:break
        arrival=int(travel[constraints.initial_site,site]);slots=[s for s in range(arrival,t) if access[s,site]]
        for chosen in combinations(slots,count):
            u=np.zeros(t);u[list(chosen)]=cap
            fixed=solve_site_step(site,t,history.n,constraints,base_power=u,trust_radius=0.,
                intercepts=[0.],gradients=np.zeros((1,t)),time_limit=5.)
            operation={k:v for k,v in fixed.items() if k!='plan'}
            label=f'site_{site}_full_slots_'+''.join(map(str,chosen))
            if not fixed['success']:
                failures.append(dict(action=label,kind='fixed_power_operation',operation=operation));continue
            budget_stopped=not evaluate(label,fixed['plan'],'two_full_power_slots',operation)
            if budget_stopped:break
    main_record=next((r for r in records if r['kind']=='main'),None)
    valid_library=[r for r in records if r['kind']!='main' and r['evaluation']['success']]
    main_score=main_record['evaluation']['score'] if main_record and main_record['evaluation']['success'] else None
    best=max((r['evaluation']['score'] for r in valid_library),default=None)
    unique_library={(r['site'],np.asarray(r['plan']['p'],dtype='<f8').tobytes()) for r in records if r['kind']!='main'}
    permutation=[]
    for site in range(history.n):
        scores=[r['evaluation']['score'] for r in valid_library if r['site']==site and r['kind']=='two_full_power_slots']
        permutation.append(dict(site=site,count=len(scores),score_spread=max(scores)-min(scores) if scores else None,
            expected_time_permutation_invariance='historical mean prediction; independent equal-variance future channels'))
    return dict(history_id=snapshot['history_id'],raw_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        input_snapshot=snapshot,input_snapshot_sha256=hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        actions=records,failures=failures,complete=not budget_stopped,budget_stopped=budget_stopped,
        action_records=len(records),unique_library_plans=len(unique_library),unique_evaluations=len(cache),
        main_score_recomputed=main_score,main_score_stored=snapshot['main_stored_score'],
        main_score_replay_error=None if main_score is None else main_score-snapshot['main_stored_score'],
        best_library_score=best,main_minus_best_library=None if best is None or main_score is None else main_score-best,
        main_not_below_finite_library=bool(best is not None and main_score is not None and main_score>=best-1e-6),
        time_permutation_audit=permutation,fit_calls=evaluator.fit_calls,total_ac_calls=evaluator.ac_evaluations+evaluator.response_ac_calls,
        seconds=perf_counter()-began,selection_uses_truth=False,selection_uses_real_future_noise=False,
        enumeration_role='independent finite comparison; not called by main controller',
        continuous_global_optimality_certified=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-base',type=Path,default=HERE/'evidence'/'gradient_price'/'v2_supported')
    parser.add_argument('--output',type=Path,default=HERE/'evidence'/'gradient_price'/'library_audit.json.gz')
    parser.add_argument('--budget-seconds',type=float,default=60.)
    args=parser.parse_args();began=perf_counter();deadline=began+args.budget_seconds
    histories=[]
    for seed in (800002,800005):
        if perf_counter()>=deadline:break
        histories.append(run_history(args.raw_base/'pilot'/f'trial_{seed}.json.gz',deadline=deadline))
        save(args.output,dict(histories=histories,planned_history_count=2,completed_history_count=sum(h['complete'] for h in histories),
                             seconds=perf_counter()-began,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),True)
        print(json.dumps({k:histories[-1][k] for k in ('history_id','complete','unique_library_plans','main_score_recomputed','best_library_score','seconds')},ensure_ascii=False),flush=True)
    summary=dict(planned_history_count=2,completed_history_count=sum(h['complete'] for h in histories),
        histories=[{k:h[k] for k in ('history_id','complete','unique_library_plans','unique_evaluations','main_score_recomputed',
            'main_score_stored','main_score_replay_error','best_library_score','main_minus_best_library','main_not_below_finite_library',
            'time_permutation_audit','fit_calls','total_ac_calls','seconds')} for h in histories],
        failure_count=sum(len(h['failures']) for h in histories),seconds=perf_counter()-began,
        source_archive_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest() if args.output.exists() else None,
        continuous_global_optimality_certified=False)
    save(args.output.with_name('library_audit_summary.json'),summary)


if __name__=='__main__':main()
