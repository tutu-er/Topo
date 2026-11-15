"""Quarter-hour closed-loop development driver with pre-dispatch plan seals."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import shutil
import sys
import traceback
import numpy as np

from ... import io
from ...observations import Observation
from ..history288_pq_noise_20261115 import run as history
from ..distribution_gate_20261114.run import planning_context
from .config import Config
from .domain import Fleet
from .state import FrozenBank,ScoreState
from .feedback import FeedbackController,seal_monitor
from .evidence import RollingEvidence,Ledger,EpochResetRequired
from .execution import Evaluator,DeliveryEstimator,CostLedger
from . import solve


ROOT=Path(__file__).resolve().parents[3]
DEFAULT_SOURCE=ROOT/'outputs/history288_pq_noise_execution_20261115_v1'


def utc():return datetime.now(timezone.utc).isoformat()


def checked(ref):
    if io.ref(ref['path'])!=ref:raise ValueError('Source hash changed: '+ref['path'])
    return Path(ref['path'])


def sources(out):
    paths={Path(__file__).with_name('PROTOCOL.md')}
    for m in list(sys.modules.values()):
        filename=getattr(m,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT) and 'outputs' not in p.parts and not p.name.startswith('test_'):
                paths.add(p)
    paths.update(Path(__file__).parent.glob('*.py'))
    result=[]
    for p in sorted(paths):
        if p.name in ('audit.py','test_audit.py') or p.name.startswith('test_'):continue
        result.append(io.ref(p))
        dest=out/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    return result


def load_planning(entry):
    context=planning_context(entry,entry['base_kw'])
    with np.load(checked(entry['forecast_ref']),allow_pickle=False) as z:
        forecast=z[entry.get('forecast_key','forecast00')].copy()
    metadata=io.read(checked(entry['node_time_bandwidth']))
    with np.load(checked(metadata['arrays']),allow_pickle=False) as z:
        bands={key:z[key].copy() for key in z.files}
    spec=io.read(checked(entry['equipment_spec']))
    fleet=Fleet(spec,context['labels'],context['base'],band_arrays=bands)
    bank=FrozenBank.from_score(context['score'],context['val'])
    return context,forecast,fleet,bank


def prepare(out,*,source=DEFAULT_SOURCE,config=None,repeats=1,input_ids=None):
    out,source=Path(out).resolve(),Path(source).resolve()
    if out.exists():raise FileExistsError('Preserve existing complete or partial run: '+str(out))
    config=Config() if config is None else config
    prior=io.read(source/'complete.json');p=io.read(source/'protocol.json')
    if prior.get('status')!='complete' or prior.get('source_unchanged') is not True:
        raise ValueError('Completed unchanged dense-history source required')
    entries=io.read(checked(p['inputs']))
    if input_ids:
        selected=[e for e in entries if e['id'] in set(input_ids)]
        if len(selected)!=len(set(input_ids)):raise ValueError('Unknown or duplicate requested input')
    else:
        # Fixed development labels, not selected for new-day success.
        selected=[e for e in entries if history.pq_sd(e)==.005 and
                  (history.history_repeat(e)==0 or (e['case']=='soumalas11' and history.history_repeat(e)==3))]
        if len(selected)!=3:raise ValueError('Expected the three predeclared development histories')
    if not 1<=repeats<=16:raise ValueError('Use 1..16 already sealed evaluation paths')
    for entry in selected:
        context,forecast,fleet,bank=load_planning(entry)
        if forecast.shape!=(96,2*fleet.N):raise ValueError('Expected unchanged 96-point forecast')
        for ref in (entry['score'],entry['files']['validation.npz'],entry['equipment_spec'],entry['evaluation_noise']):
            checked(ref)
    out.mkdir(parents=True)
    io.write(out/'inputs.json',selected)
    policies=['no_dfs',*('rolling_'+m for m in config.modes)]
    protocol=dict(created_utc=utc(),phase='quarter_hour_feedback_development',config=config.record(),
        source_run=str(source),source_completion=io.ref(source/'complete.json'),inputs=io.ref(out/'inputs.json'),
        source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),sources=sources(out),
        repeats=repeats,policies=policies,expected_trajectories=len(selected)*repeats*len(policies),
        expected_rounds=len(selected)*repeats*len(policies)*96,
        environment=io.environment(),historical_refits=0,new_physical_days=0,
        interpretation='Known two physical backgrounds; quarter-hour integration development, not independent confirmation',
        firm_service='Guaranteed local normal-service tracking; eta applies only to extra identification requests; original device ratings unchanged',
        scoring='Full-native cumulative validation+observed SSE, then family minimum; no per-step parameter refitting',
        probability='Per-round sealed remaining-plan conditional bound only; no whole-day closed-loop .9 guarantee',
        optimizer_objective='Requested activation plus nonrefundable new daily capacity; actual delivered bill separately settled',
        capacity_contract='Whole remaining requested peak purchased before dispatch, no refund when later plans shrink',
        latency='Measured simulated wall time; real hardware deadline not validated; no future observations enter a decision')
    io.write(out/'protocol.json',protocol)
    return protocol


def evidence(case,bank,snapshot,forecast,mode,config,ledger,forecast_sd=None):
    return RollingEvidence(case,bank,snapshot,forecast,mode=mode,beta=config.beta,eta=config.eta,
                           lambda_start=config.lambda_start,progress=ledger,forecast_sd=forecast_sd)


def run_trajectory(entry,policy,repeat,root,config,env):
    directory=root/'trajectories'/entry['id']/f'path{repeat:02d}'/policy
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if directory.exists():
        raise RuntimeError('Partial trajectory preserved; inspect checkpoints instead of resetting its budget: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(created_utc=utc(),pid=os.getpid(),policy=policy,repeat=repeat))
    began=perf_counter()
    context,forecast,fleet,bank=load_planning(entry)
    io.write(directory/'bank.json',dict(H=bank.H,means=bank.means,initial_sse=bank.initial_sse,
        topology_ids=bank.topology_ids,internal=bank.internal,sigma2=bank.sigma2,
        initial_cells=bank.initial_cells,reference_mse=bank.reference_mse,tie=bank.tie,
        fingerprint=bank.fingerprint,labels=bank.labels))
    state=ScoreState(bank)
    controller=FeedbackController(dt=.25,confirmation_hours=config.confirmation_hours,
        recovery_hours=config.recovery_hours,caution_fraction=config.caution_fraction)
    delivery=DeliveryEstimator(fleet,config.initial_delivery_gain,config.minimum_delivery_gain)
    billing=CostLedger(fleet,config.kappa_p,config.kappa_q)
    prefix=np.empty((0,fleet.R));rows=[];observations=[];data=[]
    evaluator=None;previous_seal=None;ledger=None;epoch=0;warm_total=None
    shared_prep=perf_counter()-began
    elapsed=shared_prep;mode=None if policy=='no_dfs' else policy.removeprefix('rolling_')
    status='complete';failure=None
    try:
        for k in range(96):
            tick=perf_counter();folder=directory/f'step{k:03d}';folder.mkdir()
            before=state.snapshot();before['physical_epoch']=epoch
            pre_feedback=controller.snapshot()
            rounds_left=96-k
            available=max(0.,config.day_solver_seconds-elapsed)
            budget=min(config.round_solver_seconds,available/rounds_left)
            # Service may be required even when identification compute budget is exhausted.
            service,sr=fleet.service_plan(prefix,env=env,seconds=max(.05,min(5.,budget)),warm_tail=warm_total)
            if service is None:
                raise RuntimeError('Firm device service unavailable: '+sr['status'])
            fraction=0. if controller.state=='pause' else config.caution_fraction if controller.state=='caution' else 1.
            case=fleet.remaining(prefix,service,delivery.gains,caution_fraction=fraction,paid_capacity=billing.paid_capacity)
            shifted=forecast[k:]+service@fleet.B.T/fleet.base_kw
            forecast_sd=.05*np.abs(forecast[k:])
            u=np.zeros(case.shape);quote=dict(status='no_identification_procurement',feasible=None,optimizer_invoked=False)
            ev=None;progress_event=None
            if mode is not None and controller.state!='pause' and available>0:
                try:
                    try:
                        ev=evidence(case,bank,before,shifted,mode,config,ledger,forecast_sd)
                    except EpochResetRequired as exc:
                        epoch+=1;before['physical_epoch']=epoch;ledger=None
                        progress_event=dict(event='explicit_epoch_reset',reason=str(exc),epoch_id=epoch)
                        ev=evidence(case,bank,before,shifted,mode,config,None,forecast_sd)
                    remaining=budget-(perf_counter()-tick)
                    warm=None if warm_total is None else (warm_total-service)/delivery.gains
                    if remaining>0:
                        point,quote=solve.plan(case,ev,env=env,seconds=remaining,
                            paid_capacity=billing.paid_capacity,capacity_prices=billing.prices,warm=warm)
                    else:
                        point=None;quote=dict(status='round_budget_in_preprocessing',feasible=False,optimizer_invoked=False)
                    if point is not None and config.progress and ledger is None:
                        ledger=Ledger.from_plan(ev,point,epoch_id=epoch)
                        ev_with_progress=evidence(case,bank,before,shifted,mode,config,ledger,forecast_sd)
                        if solve.check(case,ev_with_progress,point)['passed']:
                            ev=ev_with_progress
                            quote['progress_initialized']=True
                            quote['audit']=solve.check(case,ev,point)
                        else:
                            remaining=budget-(perf_counter()-tick)
                            if remaining>0:
                                point,second=solve.plan(case,ev_with_progress,env=env,seconds=remaining,
                                    paid_capacity=billing.paid_capacity,capacity_prices=billing.prices,warm=point)
                                quote=dict(second,first_quote=quote,progress_initialized=True)
                            else:
                                point=None;quote=dict(quote,status='progress_unverified_budget',feasible=False)
                            ev=ev_with_progress
                    if point is not None:u=point
                    else:quote['executed_service_fallback']=True
                except Exception as exc:
                    # Preserve a quote implementation/numerical exception, but
                    # do not cancel an independently verified physical service.
                    quote=dict(status='quote_exception_service_fallback',feasible=False,
                        optimizer_invoked=None,error=repr(exc),traceback=traceback.format_exc(),
                        executed_service_fallback=True)
                    ledger=None;epoch+=1;before['physical_epoch']=epoch
                    u=np.zeros(case.shape)
            elif mode is not None:
                quote.update(status='pause_firm_service' if controller.state=='pause' else 'daily_budget_exhausted',
                             executed_service_fallback=True)
            monitor=seal_monitor(bank,before,step=k)
            request=case.request_delta(u)
            planfile=folder/'plan.npz'
            np.savez_compressed(planfile,service_tail=service,extra_tail=u,request_tail=request,
                estimated_gains=delivery.gains,forecast_remaining=forecast[k:],shifted_forecast=shifted)
            round_planning_seconds=perf_counter()-tick
            elapsed+=round_planning_seconds
            seal=dict(created_utc=utc(),step=k,completed_steps=k,information_cutoff_step=k-1,
                observation_source='Only previously received quarter-hour telemetry',
                previous_seal=previous_seal,previous_receipt=None if k==0 else io.ref(directory/f'step{k-1:03d}'/'receipt.json'),
                bank=io.ref(directory/'bank.json'),snapshot=before,feedback=pre_feedback,
                mode=mode,monitor=monitor,service_receipt=sr,plan=io.ref(planfile),quote=quote,
                extra_action_fraction=fraction,round_budget_seconds=budget,
                day_compute_before_seconds=elapsed-round_planning_seconds,
                round_planning_seconds=round_planning_seconds,paid_capacity_before=billing.paid_capacity,
                progress_event=progress_event,progress_ledger=None if ledger is None else ledger.snapshot())
            io.write(folder/'seal.json',seal);previous_seal=io.ref(folder/'seal.json')
            # First future/true-network read happens strictly after a sealed request.
            if evaluator is None:
                evaluator=Evaluator(entry,fleet,repeat)
                io.write(directory/'randomness.json',evaluator.save_randomness(directory/'randomness.npz'))
            outcome=evaluator.step(k,service[0],u[0])
            obs=outcome['observation'];after=state.update(obs,step=k)
            feed=controller.observe(bank,monitor,np.c_[obs.p,obs.q][0],obs.y[0],
                previous_target=before['native_target'],current_target=after['native_target'])
            prefix=np.vstack((prefix,outcome['total_actual_delta']))
            physical=fleet.audit_prefix(prefix)
            if not physical['passed']:
                raise ArithmeticError('Executed physical state or firm-service reachability violated')
            gain_after=delivery.update(u[0],outcome['extra_actual'])
            cost=billing.settle(k,request,outcome['total_actual_delta'][None,:],outcome['root_kw'],service_tail=service)
            obsfile=folder/'observation.npz'
            np.savez_compressed(obsfile,P=obs.p,Q=obs.q,Y=obs.y,V=obs.voltage,root=obs.root,
                clean_P=outcome['clean_P'],clean_Q=outcome['clean_Q'],clean_V=outcome['clean_V'],
                clean_root=outcome['clean_root'],base_P=outcome['base_P'],base_Q=outcome['base_Q'],
                total_actual_delta=outcome['total_actual_delta'],extra_delivered=outcome['extra_actual'],
                **{f'ac_{key}':value for key,value in outcome['ac_state'].items()})
            row=dict(step=k,seal=previous_seal,observation=io.ref(obsfile),snapshot_after=after,
                feedback_after=controller.snapshot(),feedback_report=feed,gain_after=gain_after,
                service_first=service[0],extra_request_first=u[0],extra_delivered_first=outcome['extra_actual'],
                request_first=request[0],total_actual_first=outcome['total_actual_delta'],
                physical=physical,cost=cost,ac=outcome['ac'],
                minimum_voltage=outcome['minimum_voltage'],maximum_voltage=outcome['maximum_voltage'],
                quote_status=quote['status'],new_gate_feasible=quote.get('feasible'),
                quote_exception=quote.get('error'),
                round_wall_seconds=perf_counter()-tick,cumulative_compute_seconds=elapsed,
                round_planning_seconds=round_planning_seconds,
                compute_budget_overrun=elapsed>config.day_solver_seconds,
                receipt_utc=utc())
            io.write(folder/'receipt.json',row)
            rows.append(row);observations.append(obs);data.append(outcome)
            with (directory/'records.jsonl').open('a',encoding='utf-8') as stream:
                import json
                stream.write(json.dumps(dict(step=k,receipt=io.ref(folder/'receipt.json')),ensure_ascii=False)+'\n')
            warm_total=case.total_delta(u)[1:].copy()
        observed=Observation(np.vstack([o.p for o in observations]),np.vstack([o.q for o in observations]),
            np.vstack([o.y for o in observations]),bank.labels,np.vstack([o.voltage for o in observations]),
            np.concatenate([o.root for o in observations]))
        legacy=evaluator.final_evaluation(observed)
        final_snapshot=state.snapshot()
        if legacy['native_selected_index']!=final_snapshot['native_target']:
            raise ArithmeticError('Independent original native selector differs from rolling cumulative SSE')
        truth_id=legacy['truth_family_id']
        profile_true_weight=final_snapshot['weights'].get(truth_id,0.)
        evaluation={k:v for k,v in legacy.items() if k.startswith('native_')}
        evaluation.update(truth_covered=truth_id in bank.topology_ids,
            profile_true_weight=profile_true_weight,
            profile_true_pass_beta=profile_true_weight>=config.beta,
            profile_selected_family=final_snapshot['target_family'],
            historical_target_correct=legacy['historical_target_correct'])
        io.write(directory/'legacy_representative_evaluation.json',legacy)
        np.savez_compressed(directory/'arrays.npz',P=observed.p,Q=observed.q,Y=observed.y,V=observed.voltage,
            root=observed.root,actual_delta=prefix,service=np.asarray([r['service_first'] for r in rows]),
            extra_request=np.asarray([r['extra_request_first'] for r in rows]),
            extra_delivered=np.asarray([r['extra_delivered_first'] for r in rows]))
    except Exception as exc:
        status='failed';failure=dict(error=repr(exc),traceback=traceback.format_exc(),completed_steps=len(rows))
        evaluation=None
    result=dict(id=entry['id'],case=entry['case'],history_repeat=entry['history_repeat'],pq_relative_sd=history.pq_sd(entry),
        policy=policy,repeat=repeat,status=status,completed_steps=len(rows),expected_steps=96,failure=failure,
        final_snapshot=state.snapshot(),evaluation=evaluation,
        activation_yuan=billing.activation,availability_yuan=billing.capacity,
        service_yuan=billing.activation+billing.capacity,root_energy_yuan=billing.grid,
        nonzero_extra_rounds=sum(np.max(np.abs(r['extra_request_first']),initial=0.)>1e-10 for r in rows),
        service_recovery_rounds=sum(np.max(np.abs(r['service_first']),initial=0.)>1e-10 for r in rows),
        target_changes=sum(r['feedback_report']['target_changed'] for r in rows),
        pause_rounds=sum(r['feedback_after']['state']=='pause' for r in rows),
        caution_rounds=sum(r['feedback_after']['state']=='caution' for r in rows),
        quote_status_counts={s:sum(r['quote_status']==s for r in rows) for s in sorted({r['quote_status'] for r in rows})},
        quote_exceptions=sum(r.get('quote_exception') is not None for r in rows),
        new_ac_rounds=sum(r['ac'].get('status')!='reused_sealed_zero_row' for r in rows),
        minimum_voltage=min((r['minimum_voltage'] for r in rows),default=None),
        maximum_voltage=max((r['maximum_voltage'] for r in rows),default=None),
        compute_seconds=elapsed,shared_preparation_seconds=shared_prep,
        wall_seconds=perf_counter()-began,records=io.ref(directory/'records.jsonl') if rows else None,
        arrays=io.ref(directory/'arrays.npz') if status=='complete' else None)
    io.write(directory/'complete.json',result)
    return result


def run(out):
    out=Path(out).resolve();protocol=io.read(out/'protocol.json')
    for ref in protocol['sources']:checked(ref)
    entries=io.read(checked(protocol['inputs']));config=Config(**protocol['config'])
    import gurobipy as gp
    results=[]
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        for entry in entries:
            for repeat in range(protocol['repeats']):
                for policy in protocol['policies']:
                    result=run_trajectory(entry,policy,repeat,out,config,env)
                    results.append(result)
                    print(entry['id'],policy,result['status'],result['completed_steps'],flush=True)
    unchanged=all(io.ref(ref['path'])==ref for ref in protocol['sources'])
    result=dict(status='complete' if all(r['status']=='complete' and not r['quote_exceptions'] for r in results) and unchanged else 'completed_with_failures',
        expected_trajectories=protocol['expected_trajectories'],trajectories=len(results),
        expected_rounds=protocol['expected_rounds'],completed_rounds=sum(r['completed_steps'] for r in results),
        failures=[dict(id=r['id'],policy=r['policy'],repeat=r['repeat'],failure=r['failure']) for r in results if r['status']!='complete'],
        quote_exceptions=sum(r['quote_exceptions'] for r in results),
        source_unchanged=unchanged,results=results,protocol=io.ref(out/'protocol.json'))
    io.write(out/'complete.json',result)
    print({k:v for k,v in result.items() if k not in ('results','failures')},flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--input-id',action='append')
    parser.add_argument('--modes',nargs='+',choices=Config().modes,default=list(Config().modes))
    parser.add_argument('--day-seconds',type=float,default=1200.)
    parser.add_argument('--round-seconds',type=float,default=30.)
    args=parser.parse_args()
    if not args.resume:
        cfg=replace(Config(),modes=tuple(args.modes),day_solver_seconds=args.day_seconds,round_solver_seconds=args.round_seconds)
        p=prepare(args.out,source=args.source,config=cfg,repeats=args.repeats,input_ids=args.input_id)
        print(dict(prepared=True,expected_trajectories=p['expected_trajectories'],expected_rounds=p['expected_rounds']),flush=True)
    if not args.prepare_only:run(args.out)


if __name__=='__main__':main()
