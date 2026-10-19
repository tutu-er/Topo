"""Frozen-policy synthetic AC evaluation of optimization-generated EV plans.

Truth and future noises stay in one_trial's evaluator, after select_plans
returns. The optimizer receives history, public devices, and frozen matrices.
Legacy actions are generated only for comparison methods, not for MILP search.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy

from .likelihood import ac_predict,concat_observations
from .candidates import propose,normalize_family
from .simulation import generate,observe,model,BASE_KW,truth_metadata
from .design import EVAction,EVConstraints,enumerate_actions
from .run_study import dump,fit_bank,fit_record,fingerprint
from .adaptive_history import build_history_context,build_frozen_pairs,evaluate_frozen_pair_actual
from .ev_operations_milp import EVConfig,operation_model,decode_plan,check_plan
from .ev_information_milp import FrozenPair,solve_pair,adaptive_search
from .adaptive_statistics import (policy_trial_scores,calibrate_policy_gates,
    apply_candidate_gates,history_event_summary,paired_binary_effects,paired_continuous_effects)

HERE=Path(__file__).resolve().parent
METHODS=('none','constant_fixed','fixed_pulse','random_action','legacy_library_proxy',
         'fixed_mle_milp','refit_milp')
DEFAULT_POLICY=dict(version='adaptive_milp_v3_frozen',max_centers=2,max_attempts=3,
    source_delta=.05,line_trust=.30,history_loss_slack=12.,forecast='mean',
    fit_multistart=2,fit_max_nfev=300,iterations=3,segments=2,max_segments=8,
    solver_seconds=3.,search_seconds=25.,relative_gap=1e-3,absolute_gap=1e-4,
    post_selection_voltage_limits=[.9,1.1],baseline_action_count=11,
    full_audit_bank='all_three_terminal_star_and_cherries',
    gates_influence_shadow_policy=False,Q_over_P=0.,history_points=12,future_points=8)
POPULATION=dict(version='independent_history_ac_v1',topology_mass=[.25]*4,
    strong_probability=.5,independent_edge_R_X_scale=[.8,1.2],
    latent_load_source='existing simulation.generate iid bounded per-time paths',
    root_voltage=1.,calibrated_bias=True,tracking_gain=1.)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_hashes():
    names=['adaptive_milp_study.py','adaptive_history.py','ev_operations_milp.py',
           'ev_information_milp.py','history_qp_acceleration.py','adaptive_statistics.py','likelihood.py',
           'simulation.py','candidates.py','design.py','symmetric_ac_local.py','run_study.py']
    result={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
    for n in ('passive_bridge_v9.py','wzzt_adapter.py','ac_general_v9.py'):
        result['../mobile_ev_probing/'+n]=hashlib.sha256((HERE.parent/'mobile_ev_probing'/n).read_bytes()).hexdigest()
    return result


def configurations(policy=None):
    p={**DEFAULT_POLICY,**(policy or {})}
    access=np.ones((p['future_points'],3),bool);access[:2,2]=False
    ev=EVConfig(n_slots=p['future_points'],n_sites=3,access=access)
    old=EVConstraints(soc_min=.1,p_max=.3,energy=.3,base_kw=BASE_KW,
        availability=access,travel_slots=np.ones((3,3),int)-np.eye(3,dtype=int),
        travel_energy=.01*(np.ones((3,3))-np.eye(3)),initial_site=0)
    return p,ev,old


def _plan_action(plan,config,name):
    check=check_plan(plan,config,tolerance=2e-6)
    if not check['feasible']:
        raise ValueError('cannot execute an invalid EV plan')
    return EVAction(name,plan.u_kw/BASE_KW,np.zeros_like(plan.u_kw),
                    np.asarray(check['sites'],int),'adaptive_milp')


def _pairs(context,u0,*,fixed=False):
    ctx=context
    if fixed:
        # Deliberate fixed-MLE prediction ablation uses one retained centre.
        ctx={**context,'centres':{n:rows[:1] for n,rows in context['centres'].items()}}
    raw=build_frozen_pairs(ctx,u0,source_mode='fixed' if fixed else 'local_refit',
        line_trust_fraction=0. if fixed else None)
    return ctx,raw,[FrozenPair(**row) for row in raw]


def _physical_admission(context,raw,u,*,limits):
    rows=[];failure=[]
    seen=set()
    for pair in raw:
        for name,index in zip(pair['metadata']['candidate_names'],pair['metadata']['centre_indices']):
            key=(name,index,pair['metadata']['scenario_index'])
            if key in seen:continue
            seen.add(key)
            centre=context['centres'][name][index];fit=centre['fit'];scenario=context['scenarios'][key[2]]
            try:
                gain=scenario['tracking_gain'];p=scenario['p0']+gain*np.asarray(u)/BASE_KW
                q=scenario['q0'];sp=np.full(len(p),np.mean(fit.source_p))+scenario['source_shift_p']
                sq=np.full(len(p),np.mean(fit.source_q))+scenario['source_shift_q']
                # Current frozen policy is mean; explicit alternate trend modes
                # are evaluated through the context's exact replay helper.
                from .adaptive_history import _future_for_center
                physical=_future_for_center(context,centre,scenario,np.asarray(u))['geometry']['physical']
                voltage=abs(physical['voltage']);lo=float(voltage.min());hi=float(voltage.max())
                ok=lo>=limits[0] and hi<=limits[1]
                rows.append(dict(candidate=name,centre=index,scenario=key[2],vmin=lo,vmax=hi,feasible=ok))
                if not ok:failure.append('predicted_voltage_domain')
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                failure.append(str(error))
    return dict(feasible=not failure,checks=rows,failures=failure)


def _search_record(result):
    return {**{k:v for k,v in result.items() if k!='plan'},
            'proxy_stop_reason':result.get('status'),
            'proxy_gap_met':result.get('status')=='proxy_gap_met'}


def _select_plans_impl(history,candidates,meter,proposal,*,policy=None,seed=0,
                 historical_fits=None,known_history_ev_kw=None,_call_counter=None):
    """History-only controller; has no generator/truth/future-data argument."""
    p,ev,old=configurations(policy)
    library,rejected=enumerate_actions(ev.n_slots,3,old,max_actions=p['baseline_action_count'])
    zero=EVAction('none',np.zeros((ev.n_slots,3)),np.zeros((ev.n_slots,3)),np.zeros(ev.n_slots,int),'none')
    constant=next(a for a in library if a.name=='constant_site_0')
    pulse=next(a for a in library if a.name=='early_site_0')
    random=library[int(np.random.default_rng(seed+90000).integers(len(library)))]
    choices={'none':zero,'constant_fixed':constant,'fixed_pulse':pulse,'random_action':random}
    context=build_history_context(history,candidates,meter,horizon=ev.n_slots,
        historical_fits=historical_fits,max_centers=p['max_centers'],max_attempts_per_candidate=p['max_attempts'],
        center_loss_slack=p['history_loss_slack'],line_trust_fraction=p['line_trust'],
        source_delta_bound=p['source_delta'],max_nfev=p['fit_max_nfev'],multistart=p['fit_multistart'],
        seed=seed+100,passive_metadata=proposal,known_history_ev_kw=known_history_ev_kw,
        forecast_support=[dict(name=p['forecast'],baseline=p['forecast'],source_baseline=p['forecast'])])
    record=dict(history=context['summary'],methods={},library_rejections=rejected,
        selector_uses_truth=False,selector_uses_future_noise=False,device_config=asdict(ev),policy=p)
    u0=constant.p*BASE_KW
    raw=[];pairs=[];library_rows=[]
    if context['status']=='ready':
        try:
            _,raw,pairs=_pairs(context,u0)
            # Identical frozen score on the old library isolates action-search
            # restriction from a change in discrimination objective.
            library_rows=[]
            for action in library:
                solved=[solve_pair(pair,action.p*BASE_KW) for pair in pairs]
                valid=all(s['success'] for s in solved)
                library_rows.append(dict(action=action.name,success=valid,
                    q_lower=min(s['q_lower'] for s in solved),q_upper=min(s['q_upper'] for s in solved),
                    qp_acceleration_counts={key:sum(s['acceleration']==key for s in solved) for key in ('history_QR','full_BVLS')},
                    max_qp_gap=max(s['numeric_gap'] for s in solved)))
            if not all(row['success'] for row in library_rows):
                raise RuntimeError('legacy comparison nuisance failure retained')
            chosen=min(library_rows,key=lambda row:(-row['q_lower'],row['action']))['action']
            choices['legacy_library_proxy']=next(a for a in library if a.name==chosen)
            record['methods']['legacy_library_proxy']=dict(status='complete',selected=chosen,
                actions=library_rows,criterion='same frozen soft-history proxy as refit MILP')
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            record['methods']['legacy_library_proxy']=dict(status='selection_failure_fallback',error=str(error),actions=library_rows)
    else:
        record['methods']['legacy_library_proxy']=dict(status='selection_failure_fallback',
            error='historical context failure',actions=library_rows)
    for name,fixed in [('fixed_mle_milp',True),('refit_milp',False)]:
        method={};method_started=perf_counter();calls_before=_call_counter['count']
        try:
            ctx,raw_here,pairs_here=_pairs(context,u0,fixed=fixed)
            result=adaptive_search(ev,pairs_here,initial_us=[u0,np.zeros_like(u0)],
                max_iterations=p['iterations'],segments=p['segments'],max_segments=p['max_segments'],
                solver_seconds=p['solver_seconds'],total_seconds=p['search_seconds'],
                relative_gap=p['relative_gap'],absolute_gap=p['absolute_gap'],seed=seed)
            method=_search_record(result)
            plan=result['plan']
            if not result['success'] or plan is None:
                raise RuntimeError('MILP/inner failure: '+str(result.get('failures')))
            admission=_physical_admission(ctx,raw_here,plan.u_kw,limits=p['post_selection_voltage_limits'])
            method['prediction_admission']=admission
            if not admission['feasible']:
                raise RuntimeError('selected prediction failed voltage admission')
            choices[name]=_plan_action(plan,ev,name)
            none_sol=[solve_pair(pair,np.zeros_like(u0)) for pair in pairs_here]
            method['none_proxy_lower']=min(s['q_lower'] for s in none_sol)
            method['none_proxy_upper']=min(s['q_upper'] for s in none_sol)
            method['proxy_gain_lower_vs_none']=result['proxy_lower']-method['none_proxy_upper']
            rollback=[]
            for row,pair in zip(raw_here,pairs_here):
                solved=solve_pair(pair,plan.u_kw)
                try:
                    check=evaluate_frozen_pair_actual(ctx,row,plan.u_kw,solved['z'])
                    rollback.append(dict(pair_name=pair.name,proxy_q_lower=solved['q_lower'],proxy_q_upper=solved['q_upper'],**{k:v for k,v in check.items() if k not in ['actual_residual','affine_residual']}))
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                    rollback.append(dict(physically_feasible=False,error=str(error)))
            method.update(status='complete',actual_rollback=rollback,operational=check_plan(plan,ev),
                selected_u_kw=plan.u_kw,actual_plan=plan.as_dict())
            record['methods'][name]=method
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            method.update(status='selection_failure_fallback',error=str(error))
            record['methods'][name]=method
        record['methods'][name]['total_selection_seconds']=perf_counter()-method_started
        record['methods'][name]['selection_ac_predict_calls']=_call_counter['count']-calls_before
    for name in METHODS:
        if name not in choices:
            choices[name]=constant
    # An explicit operational fallback remains part of the frozen policy.
    # Algorithm failures are counted separately from actual joint-fit failures.
    return choices,record,context


def select_plans(history,candidates,meter,proposal,*,policy=None,seed=0,
                 historical_fits=None,known_history_ev_kw=None):
    """H-only controller with restored, process-local physical-call counters."""
    from . import likelihood,symmetric_ac_local
    original=likelihood.ac_predict;original_symmetric=symmetric_ac_local.ac_predict
    counter={'count':0};started=perf_counter()
    def counted(*args,**kwargs):
        counter['count']+=1
        return original(*args,**kwargs)
    likelihood.ac_predict=counted;symmetric_ac_local.ac_predict=counted
    try:
        choices,record,context=_select_plans_impl(history,candidates,meter,proposal,
            policy=policy,seed=seed,historical_fits=historical_fits,
            known_history_ev_kw=known_history_ev_kw,_call_counter=counter)
        record['selector_ac_predict_calls']=counter['count']
        record['selector_total_seconds']=perf_counter()-started
        record['ac_call_definition']='one ac_predict call, including failed calls; not flow iterations'
        return choices,record,context
    finally:
        likelihood.ac_predict=original;symmetric_ac_local.ac_predict=original_symmetric


def _evaluation_path(seed,*,stress=None):
    streams=np.random.SeedSequence(seed).spawn(3)
    generator=int(streams[0].generate_state(1)[0]);policy_seed=int(streams[1].generate_state(1)[0])
    rng=np.random.default_rng(streams[2]);topology=int(rng.integers(4))
    family=((),((0,1),),((0,2),),((1,2),))[topology]
    case='strong' if rng.random()<POPULATION['strong_probability'] else 'weak'
    path=generate(generator,case,truth_family=family)
    path.r*=rng.uniform(.8,1.2,len(path.r));path.x*=rng.uniform(.8,1.2,len(path.x))
    if stress=='bias':path.bias['v']+=np.array([-.0006,0,.0006])
    if stress=='source_drift':path.source_p[12:]+=.10;path.source_q[12:]+=.035
    if stress=='load_trend':path.p[12:]+=np.linspace(0,.05,8)[:,None]
    return path,policy_seed,topology,case


def counted_joint_fit_bank(*args,**kwargs):
    """Count every joint AC call, including work before a fit raises."""
    from . import likelihood
    original=likelihood.ac_predict;counter={'count':0}
    def counted(*a,**k):
        counter['count']+=1
        return original(*a,**k)
    likelihood.ac_predict=counted
    try:
        fitted,failed=fit_bank(*args,**kwargs)
        return fitted,failed,counter['count']
    finally:
        likelihood.ac_predict=original


def one_trial(seed,*,policy=None,stress=None,output=None):
    began=perf_counter();p,ev,old=configurations(policy)
    path,policy_seed,topology,case=_evaluation_path(seed,stress=stress)
    history,_=observe(path,slice(0,12));meter=model(3)
    candidates,proposal=propose(history,run_wzzt=True,wzzt_time_limit=.3)
    if stress=='missing_truth':
        # Predeclared defect removes clade01 regardless of which topology is true.
        candidates=[c for c in candidates if normalize_family(c.supports,3)!=((0,1),)]
    choices,selection,context=select_plans(history,candidates,meter,proposal,policy=p,seed=policy_seed)
    # No truth-derived initial values; every initializer was obtained from H.
    initial={name:centres[0]['fit'] for name,centres in context['centres'].items() if centres}
    true_family=normalize_family(path.candidate.supports,3)
    true_index=next((i for i,c in enumerate(candidates) if normalize_family(c.supports,3)==true_family),None)
    records=[];loss_rows=[];success_rows=[]
    observed_arrays={f'history_{key}':getattr(history,key) for key in ('p','q','v','master_p','master_q','root_v')}
    for method in METHODS:
        method_started=perf_counter()
        action=choices[method];future,physical=observe(path,slice(12,None),action)
        joint=concat_observations(history,future)
        fitted,failed,joint_ac_calls=counted_joint_fit_bank(candidates,joint,meter,initial=initial,seed=policy_seed+2000,
            multistart=p['fit_multistart'],max_nfev=p['fit_max_nfev'])
        losses=[float(fitted[c.name].loss) if c.name in fitted else np.inf for c in candidates]
        success=[bool(c.name in fitted and fitted[c.name].success) for c in candidates]
        complete=all(success);loss_rows.append(losses);success_rows.append(success)
        row=dict(method=method,status='complete' if complete else 'joint_fit_failure',
            action=action.metadata(old),fits=[fit_record(c,fitted[c.name]) for c in candidates if c.name in fitted],
            losses=losses,success=success,fit_failures=failed,
            selection_status=selection['methods'].get(method,{}).get('status','predeclared_baseline'),
            true_covered=true_index is not None,true_index=true_index,
            observation_hash=fingerprint(joint.p,joint.q,joint.v,joint.master_p,joint.master_q),
            voltage_min=float(physical['v'].min()),voltage_max=float(physical['v'].max()),
            operational_voltage_ok=bool(physical['v'].min()>=.9 and physical['v'].max()<=1.1),
            ac_calls=joint_ac_calls,returned_fit_ac_calls=sum(f.diagnostics.get('ac_evaluations',0) for f in fitted.values()),
            is_global_MLE=False,seconds=perf_counter()-method_started)
        for name,fit in fitted.items():
            observed_arrays.update({f'{method}__{name}__fit_{key}':getattr(fit,key) for key in ('p','q','source_p','source_q')})
        observed_arrays.update({f'{method}_{key}':getattr(future,key) for key in ('p','q','v','master_p','master_q','root_v')})
        if complete:
            best=int(np.argmin(losses));row['winner']=candidates[best].name
            row['gaps']=(np.asarray(losses)-min(losses)).tolist()
            if true_index is not None:
                fit=fitted[candidates[true_index].name]
                row.update(true_loss=losses[true_index],true_gap=row['gaps'][true_index],
                    wrong_winner=best!=true_index,
                    impedance_relative_rmse=float(np.sqrt(np.mean(np.r_[((fit.r-path.r)/path.r)**2,((fit.x-path.x)/path.x)**2]))),
                    impedance_absolute_rmse_pu=float(np.sqrt(np.mean(np.r_[(fit.r-path.r)**2,(fit.x-path.x)**2]))))
        records.append(row)
    score=policy_trial_scores(loss_rows,true_index,success=success_rows,coverage_ok=true_index is not None)
    result=dict(history_id=str(seed),seed=seed,policy_seed=policy_seed,case=case,truth_topology=topology,
        stress=stress,candidate_names=[c.name for c in candidates],true_index=true_index,
        selection=selection,records=records,score_record=asdict(score),seconds=perf_counter()-began,
        selection_uses_truth=False,selection_uses_future_noise=False,truth=truth_metadata(path),
        independent_history_unit=True)
    if output is not None:
        dump(Path(output)/f'trial_{seed}.json',result)
        np.savez_compressed(Path(output)/f'trial_{seed}_observations.npz',**observed_arrays)
    return result


def run_batch(output,*,phase,count,start_seed,policy=None,stress=None,deadline_utc=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    p,_,_=configurations(policy);hashes=source_hashes()
    config=dict(phase=phase,count=count,start_seed=start_seed,policy=p,population=POPULATION,
        policy_id=_hash(dict(policy=p,source_hashes=hashes)),population_id=_hash(POPULATION),
        source_sha256=hashes,stress=stress,methods=METHODS,history_unit='one independent generator path')
    config['runtime']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    if (output/'config.json').exists():
        old=json.loads((output/'config.json').read_text())
        if old!=json.loads(json.dumps(config)):
            raise ValueError('refusing to mix different policy/source configurations in existing batch')
    dump(output/'config.json',config)
    rows=[];stop_reason="count_complete"
    for i in range(count):
        if deadline_utc is not None and datetime.now(timezone.utc)>=datetime.fromisoformat(deadline_utc):
            stop_reason="wall_clock_budget_checkpoint";break
        seed=start_seed+i;file=output/f'trial_{seed}.json'
        if file.exists():
            row=json.loads(file.read_text())
        else:row=one_trial(seed,policy=p,stress=stress,output=output)
        rows.append(row)
        print(json.dumps(dict(phase=phase,trial=i+1,count=count,seed=seed,case=row['case'],
            truth_topology=row['truth_topology'],seconds=row['seconds'],
            designs={k:v['status'] for k,v in row['selection']['methods'].items()},
            complete=row['score_record']['complete']),ensure_ascii=False),flush=True)
        dump(output/'progress.json',dict(config=config,completed=len(rows),last_seed=seed))
    dump(output/'summary.json',dict(config=config,history_count=len(rows),stop_reason=stop_reason,
        seconds=sum(row['seconds'] for row in rows),complete_history_count=sum(row['score_record']['complete'] for row in rows),
        selection_failure_counts={m:sum(row['selection']['methods'].get(m,{}).get('status')=='selection_failure_fallback' for row in rows) for m in METHODS},
        joint_fit_failure_counts={m:sum(next(x for x in row['records'] if x['method']==m)['status']!='complete' for row in rows) for m in METHODS},
        truth_topology_counts={str(i):sum(row['truth_topology']==i for row in rows) for i in range(4)},
        expansion_justified=False))
    return config,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('pilot','calibration','evaluation','stress'),default='pilot')
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--start-seed',type=int,default=710001)
    parser.add_argument('--output',type=Path,default=HERE/'outputs'/'adaptive_milp'/'pilot_v1')
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--deadline-utc',help='ISO UTC hard stop checked before each independent history; running trial completes')
    parser.add_argument('--stress',choices=('bias','source_drift','load_trend','missing_truth'))
    args=parser.parse_args();p=None if args.policy is None else json.loads(args.policy.read_text())
    run_batch(args.output,phase=args.phase,count=args.count,start_seed=args.start_seed,policy=p,stress=args.stress,deadline_utc=args.deadline_utc)


if __name__=='__main__':main()
