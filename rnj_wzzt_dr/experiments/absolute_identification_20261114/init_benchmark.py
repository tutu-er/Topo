"""Same symmetric QCQP, three initialization recipes, fixed old forecasts."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import argparse,atexit,shutil,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence,solve as old_solve
from ..flexibility_sweep_20261113.fitting import calibrated_means
from .commonseeds import build_portfolio
from .solver import SymmetricRecovery,plan
from .data import load_entry
from .driver import resources

ROOT=Path(__file__).resolve().parents[3]
PANEL=ROOT/'outputs/astra_inventory_20261114/development32.json'
OLD=ROOT/'outputs/linear_ac_population_20261113_v2/population/jobs'
BUDGETS=(3.,12.,60.)
WORK={}


def prepare(out):
    if out.exists():raise FileExistsError('Preserve previous initializer benchmarks')
    out.mkdir(parents=True)
    scope=dict(created_utc=datetime.now(timezone.utc).isoformat(),panel=io.ref(PANEL),budgets=list(BUDGETS),workers=4,
        selection='Two per network from fixed development32; prefer positive-action requirement under symmetric_guard at99% historical target mass; then correct_ambiguous/wrong_present/missing/correct_clear, p_map,id. No AC outcomes read.',
        forecast='Old sealed day000 forecast, rotated08:00 and augmented by fixed known device baselines; same transformed point for every method/budget.',
        objective='Same symmetric_guard QCQP at target_mass0.99; existing-ray, common-portfolio and no-start differ only in initialization.',
        generation='Two-second budget for both nonempty seed recipes; exact actual generation time charged to each total3/12/60second budget despite caching. Native solver also includes seed auditing/model assembly.',
        existing_recipe='One worst-pair probability-margin gradient, resource LP extreme,41fractions; retain first feasible or maximum-margin warm point, adapting prior Astra recipe to this same symmetric objective.',
        common_recipe='Shared fixedrotation and ±prediction-distance LPs for up to3historically strongest pairs plus old gamma2supportingLP; quarter fractions; original resource domain only.',
        no_start='No external resource points; native solver may run its own internal heuristics.',
        old_reference='One12second quote of oldgamma2+oldguards/oldinitialrecipe on same input; different optimization problem, descriptive reference only.',
        zero_rule='Zero-sufficient inputs retained as controls; solve once to verify and reuse analyticalzero receipt across initialization labels, no repeated timed native solves.',
        new_AC_days=0,actual_future_observations_read=False,new_identifier_fits=0,environment=io.environment())
    io.write(out/'selection_protocol.json',scope)
    screened=[]
    for entry in io.read(PANEL):
        score,train,validation,truth_unused=load_entry(entry)
        case=resources(entry)
        d=OLD/entry['id']/'day000'
        seal=io.read(d/'policy_seal.json')
        if io.ref(d/'planner.npz')['sha256']!=seal['planner']['sha256']:raise ValueError('Old forecast seal drift')
        with np.load(d/'planner.npz') as z:forecast=np.roll(z['forecast'],-32,axis=0)
        bp,bq=case.device_baseline_pq_kw()
        forecast=forecast+np.c_[bp,bq]/200.
        model=SymmetricRecovery(case,score,validation,forecast,200.,'symmetric_guard',.99)
        zero=model.verify_expansion(np.zeros(case.shape))
        screened.append(dict(entry=entry,score=score,forecast=forecast,case=case.config,zero=zero,
            old_forecast=io.ref(d/'planner.npz'),old_seal=io.ref(d/'policy_seal.json')))
    rank={'correct_ambiguous':0,'wrong_present':1,'missing':2,'correct_clear':3}
    selected=[]
    for network in sorted({r['entry']['case'] for r in screened}):
        options=sorted([r for r in screened if r['entry']['case']==network],key=lambda r:(r['zero']['feasible'],rank[r['entry']['role']],r['entry']['p_map'],r['entry']['id']))
        selected.extend(options[:2])
    io.write(out/'zero_screen.json',[dict(id=r['entry']['id'],case=r['entry']['case'],role=r['entry']['role'],zero=r['zero']) for r in screened])
    records=[]
    for r in selected:
        d=out/'inputs'/r['entry']['id'];d.mkdir(parents=True)
        io.write(d/'score.json',r['score']);io.write(d/'case.json',r['case'])
        io.write(d/'entry.json',r['entry']);np.savez_compressed(d/'forecast.npz',forecast=r['forecast'])
        records.append(dict(id=r['entry']['id'],case=r['entry']['case'],role=r['entry']['role'],zero=r['zero'],
            entry=io.ref(d/'entry.json'),score=io.ref(d/'score.json'),resource=io.ref(d/'case.json'),
            forecast=io.ref(d/'forecast.npz'),old_forecast=r['old_forecast'],old_seal=r['old_seal']))
    sources={}
    for name in ('init_benchmark.py','commonseeds.py','solver.py','data.py','driver.py'):
        source=Path(__file__).with_name(name);shutil.copyfile(source,out/name)
        sources[name]=io.ref(source)
    protocol=dict(scope,selection_protocol=io.ref(out/'selection_protocol.json'),screen=io.ref(out/'zero_screen.json'),
        sources=sources,selected=records,maximum_symmetric_quotes=72,old_reference_quotes=8,
        actual_nonzero_inputs=sum(not r['zero']['feasible'] for r in records))
    io.write(out/'protocol.json',protocol);io.write(out/'lock.json',dict(protocol=io.ref(out/'protocol.json')))
    return protocol


def existing_seed(case,evidence):
    began=perf_counter();zero=np.zeros(case.shape);baseline=evidence.direct(zero)
    if not evidence.pairs:return {'zero':zero},dict(seconds=perf_counter()-began,source='zero')
    margins=evidence.h+baseline['J']-evidence.z*np.sqrt(2*baseline['J'])
    index=int(np.argmin(margins));J=max(evidence.c[index],1e-15)
    gradient=2*evidence.a[index]-evidence.z*np.sqrt(2)*evidence.a[index]/np.sqrt(J)
    direction,_=case.linear_optimum(-gradient)
    best=None;best_margin=-np.inf;feasible=None
    for fraction in np.linspace(0,1,41):
        u=fraction*direction;info=evidence.verify_expansion(u)
        margin=info['minimum_pair_margin']
        if margin>best_margin:best,best_margin=u.copy(),margin
        if info['feasible'] and case.audit(u)['passed']:feasible=u.copy();break
    point=feasible if feasible is not None else best
    return {'existing_ray':point},dict(seconds=perf_counter()-began,source='worst_pair_LP_and41fractions',
        feasible=feasible is not None,cost_yuan=case.fee(point),budget_exceeded=perf_counter()-began>2.05)


def initialize(out):
    import gurobipy as gp
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();atexit.register(env.dispose)
    WORK.update(out=Path(out),env=env,protocol=io.read(Path(out)/'protocol.json'))


def run_input(record):
    out=WORK['out'];d=out/'quotes'/record['id'];d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():return io.read(d/'complete.json')
    if (d/'started.json').exists():raise ValueError('Unfinished initializer benchmark lock')
    io.write(d/'started.json',dict(pid=os.getpid(),record=record))
    entry=io.read(record['entry']['path']);score,training,validation,_=load_entry(entry)
    case=CustomerCase(io.read(record['resource']['path']),score['labels'])
    with np.load(record['forecast']['path']) as z:forecast=z['forecast']
    model=SymmetricRecovery(case,score,validation,forecast,200.,'symmetric_guard',.99)
    rows=[]
    pools={'none':{}};generation={'none':dict(seconds=0.)}
    if not record['zero']['feasible']:
        tick=perf_counter()
        try:pools['existing'],generation['existing']=existing_seed(case,model)
        except Exception as exc:pools['existing'],generation['existing']={},dict(seconds=perf_counter()-tick,error=repr(exc))
        pools['common'],generation['common']=build_portfolio(case,score,forecast,200.,env=WORK['env'],seconds=2.)
    else:
        pools.update(existing={},common={});generation.update(existing=dict(seconds=0.),common=dict(seconds=0.))
    io.write(d/'seed_generation.json',generation)
    for kind,pool in pools.items():np.savez_compressed(d/('private_seeds_'+kind+'.npz'),**pool)
    analytical=None
    for seconds in BUDGETS:
        for kind in ('existing','common','none'):
            charged=generation[kind]['seconds'];began=perf_counter()
            try:
                if analytical is not None:
                    receipt,u=analytical[0].copy(),analytical[1].copy();reused=True
                else:
                    receipt,u=plan(case,score,validation,forecast,200.,mode='symmetric_guard',target_mass=.99,
                        seconds=max(.001,seconds-charged),env=WORK['env'],seed_points=pools[kind]);reused=False
                if record['zero']['feasible'] and analytical is None:analytical=(receipt.copy(),u.copy())
                verification=None
                if u is not None:
                    direct=model.verify_expansion(u)
                    if not direct['feasible'] or not case.audit(u)['passed']:raise ArithmeticError('Returned point failed original-model verification')
                    verification=dict(resource_feasible=True,direct_probability_feasible=True,
                        fee_error=abs(case.fee(u)-receipt['cost_yuan']),minimum_pair_margin=direct['minimum_pair_margin'],
                        minimum_guard_gain=direct['minimum_guard_gain'])
                    np.savez_compressed(d/f'private_{kind}_{seconds:g}.npz',u=u)
                row=dict(id=record['id'],case=record['case'],seconds_requested=seconds,kind=kind,status='complete',
                    receipt=receipt,verification=verification,external_seed_seconds=charged,
                    charged_total_seconds=charged+receipt.get('seconds',0.),wall_seconds=perf_counter()-began,
                    analytical_zero_reused=reused,zero_sufficient=record['zero']['feasible'])
            except Exception as exc:
                row=dict(id=record['id'],case=record['case'],seconds_requested=seconds,kind=kind,status='failed',
                    error=repr(exc),traceback=traceback.format_exc(),external_seed_seconds=charged)
            rows.append(row);io.write(d/'private_results.json',rows)
            print('init',record['id'],seconds,kind,row['status'],row.get('receipt',{}).get('status'),flush=True)
    H=np.asarray([p['H'] for p in score['parameters']]);target=score['map_index'];others=[j for j in range(len(H)) if j!=target]
    tick=perf_counter()
    try:
        evidence=CandidateEvidence(case,H,target,others,forecast,score['sigma2'],200.,calibrated_means(score,validation))
        receipt,u=old_solve(evidence,2.,seconds=max(.001,12.-(perf_counter()-tick)),env=WORK['env'])
        old=dict(status='complete',receipt=receipt,charged_total_seconds=perf_counter()-tick)
        if u is not None:np.savez_compressed(d/'private_old_reference12.npz',u=u)
    except Exception as exc:old=dict(status='failed',error=repr(exc),traceback=traceback.format_exc())
    io.write(d/'old_reference12.json',old)
    result=dict(id=record['id'],case=record['case'],quotes=len(rows),failed=sum(r['status']!='complete' for r in rows),
        zero_sufficient=record['zero']['feasible'],results=io.ref(d/'private_results.json'))
    io.write(d/'complete.json',result);return result


def summarize(out):
    p=io.read(out/'protocol.json')
    rows=[r for f in sorted((out/'quotes').glob('*/private_results.json')) for r in io.read(f)]
    summary=dict(protocol=io.ref(out/'protocol.json'),selected_inputs=len(p['selected']),nonzero_inputs=p['actual_nonzero_inputs'],
        completed_inputs=len(list((out/'quotes').glob('*/complete.json'))),failed_rows=sum(r['status']!='complete' for r in rows),groups={})
    for seconds in BUDGETS:
        for kind in ('existing','common','none'):
            selected=[r for r in rows if r['seconds_requested']==seconds and r['kind']==kind and not r.get('zero_sufficient',False)]
            valid=[r for r in selected if r['status']=='complete'];feasible=[r for r in valid if r['receipt']['feasible']]
            bounds=[r['receipt'].get('lower_bound_yuan') for r in valid if r['receipt'].get('lower_bound_yuan') is not None]
            summary['groups'][f'{seconds:g}_{kind}']=dict(inputs=len(selected),feasible=len(feasible),
                unresolved=sum(not r['receipt']['feasible'] and not r['receipt'].get('infeasibility_proved',False) for r in valid),
                infeasible_proved=sum(r['receipt'].get('infeasibility_proved',False) for r in valid),
                mean_feasible_cost=None if not feasible else float(np.mean([r['receipt']['cost_yuan'] for r in feasible])),
                mean_available_bound=None if not bounds else float(np.mean(bounds)),
                global_certificates=sum(r['receipt'].get('global_optimality_verified',False) for r in feasible),
                initial_retained=sum(r['receipt'].get('source')=='verified_initial_retained' for r in feasible),
                native_improved_initial=sum(r['receipt'].get('source')=='native_improved_initial' for r in feasible),
                native_found=sum(r['receipt'].get('source')=='native_found' for r in feasible),
                mean_charged_total_seconds=None if not valid else float(np.mean([r['charged_total_seconds'] for r in valid])),
                mean_external_seed_seconds=None if not valid else float(np.mean([r['external_seed_seconds'] for r in valid])))
    io.write(out/'summary.json',summary);return summary


def run(out):
    p=io.read(out/'protocol.json');results=[]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(str(out),)) as pool:
        futures={pool.submit(run_input,e):e for e in p['selected']}
        for future in as_completed(futures):
            try:result=future.result()
            except Exception as exc:result=dict(id=futures[future]['id'],status='worker_failed',error=repr(exc),traceback=traceback.format_exc())
            results.append(result);io.write(out/'progress.json',results)
    summary=summarize(out)
    io.write(out/'complete.json',dict(summary=io.ref(out/'summary.json'),results=results))
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','summarize']);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();out=args.out.resolve()
    result=prepare(out) if args.action=='prepare' else run(out) if args.action=='run' else summarize(out)
    print('done',args.action,result.get('actual_nonzero_inputs',result.get('nonzero_inputs')),flush=True)
