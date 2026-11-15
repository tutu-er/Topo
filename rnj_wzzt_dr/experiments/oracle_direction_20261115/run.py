"""Two oracle QCQPs on frozen forecasts, followed by linear-only delivery replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np
from scipy.special import logsumexp

from ... import io
from ..frozen_rx_guard_20261115 import run as previous
from ..measured_pq_flex_20261114.analyze import physical_response
from ..flexibility_stability_20261114 import solver as prices
from . import geometry,solver

MODES=('distance_only','protected')


def prepare(source,out,*,smoke=False,seconds=600.,preflight=None):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    prior=io.read(source/'protocol.json');done=io.read(source/'complete.json')
    if done['status']!='complete' or not done['source_unchanged']:raise ValueError('Completed frozen input run required')
    previous.check_refs([prior['inputs'],*prior['sources']])
    audit_path=Path('outputs/frozen_rx_guard_audit_20261115_v1/report.json').resolve()
    audit=io.read(audit_path)
    if audit['status']!='passed' or audit['source_protocol']!=io.ref(source/'protocol.json'):raise ValueError('Matching audit required')
    if not smoke:
        if preflight is None or io.read(preflight)['status']!='passed':raise ValueError('Independent smoke audit required')
    coverage={r['id']:r for r in audit['input_coverage']};entries=io.read(prior['inputs']['path'])
    for entry in entries:
        score=io.read(previous.old.checked(entry['score']));true=coverage[entry['id']]['true_native_indices']
        target=min(true,key=lambda k:(score['native_bank'][k]['history_sse'],k))
        family=score['native_bank'][target]['topology_id']
        entry.update(oracle_target=target,oracle_true_indices=true,
            oracle_competitors=[k for k,r in enumerate(score['native_bank']) if r['topology_id']!=family])
        previous.context_planning(entry)
    out.mkdir(parents=True);io.write(out/'inputs.json',entries)
    refs=previous.panel.snapshot(out,entries)
    for path in (Path(__file__),Path(geometry.__file__),Path(solver.__file__),Path(__file__).with_name('PROTOCOL.md')):
        path=path.resolve();destination=out/'source_snapshot'/path.relative_to(previous.old.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        if not any(r['path']==str(path) for r in refs):refs.append(io.ref(path))
    p=dict(created_utc=previous.old.utc(),phase='oracle_direction_diagnostic',oracle=True,smoke=smoke,
        inputs=io.ref(out/'inputs.json'),source_run_protocol=io.ref(source/'protocol.json'),source_audit=io.ref(audit_path),
        modes=list(MODES),solver_seconds=2. if smoke else float(seconds),workers=3,repeats=1 if smoke else 16,
        expected_quotes=6,expected_delivery_rows=6*(1 if smoke else 16),sources=refs,environment=io.environment(),
        preflight=None if preflight is None else io.ref(preflight),
        information='Truth network only for oracle Hstar/target and offline model evaluation; frozen forecast and original resource domain',
        prohibitions='No AC, refitting, actual future P/Q/V, new noise streams or deployable policy training',
        expansion='sum_h u_hT Q_h u_h + a_hT u_h + c; a already includes factor2',
        delivery='Original16 eta only; forecast P is load baseline; linear Y=Hstar X; no new measurement noise',
        scope='Three known development settings on two physical days; linear oracle model only',
        bound_scope='Maximization: verified feasible z lower bound, valid native bound upper bound; finite resource-domain QCQP only')
    p.update(previous.old._git_receipt([Path(r['path']) for r in refs]));io.write(out/'protocol.json',p)
    return p


def linear_metrics(case,score,g,u):
    X=g['forecast']+case.shift_pq_pu(u,g['base_kw']);Y=X@g['Hstar'].T
    bank=score['native_bank'];H=np.array([p['H'] for p in bank]);hist=np.array([p['history_sse'] for p in bank])
    total=hist+np.square(Y[None]-np.einsum('tc,knc->ktn',X,H)).sum((1,2))
    scaled=total/(score['cells']+Y.size)/score['reference_mse']
    eligible=np.flatnonzero(scaled<=scaled.min()+score['validation_tie'])
    chosen=int(min(eligible,key=lambda k:(bank[k]['internal'],scaled[k],int(k))))
    target=g['target_native'];ids=g['competitor_native_indices'];sigma2=g['sigma2']
    L=(total[ids]-total[target])/(2*sigma2)
    Dresponse=np.einsum('tc,knc->ktn',X,g['D']);error=X@(g['Hstar']-g['Hc']).T
    J=np.sum(Dresponse**2,axis=(1,2))/(2*sigma2)
    bias=np.einsum('tn,ktn->k',error,Dresponse)/sigma2
    h=(hist[ids]-hist[target])/(2*sigma2)
    if not np.allclose(L,h+J+bias,rtol=1e-9,atol=1e-8):raise ArithmeticError('Oracle score identity failed')
    e=float(np.sum(error*error)/(2*sigma2));weights=-total/(2*sigma2);weights=np.exp(weights-logsumexp(weights))
    true=[k for k,r in enumerate(bank) if r['topology_id']==bank[target]['topology_id']]
    return dict(J=J.tolist(),e=e,h=h.tolist(),bias=bias.tolist(),logodds=L.tolist(),
        minimum_logodds=float(L.min()),strongest_foreign_native=int(ids[int(np.argmin(L))]),
        native_total_sse=total.tolist(),native_selected_index=chosen,internal_correct=chosen in true,
        true_native_weight=float(weights[true].sum()),response_error_to_distance_ratio=[float(np.sqrt(e/j)) if j>0 else None for j in J])


def run_case(entry):
    p,out=previous.old.WORK['p'],previous.old.WORK['out'];directory=out/'cases'/entry['id']
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);began=perf_counter();io.write(directory/'started.json',dict(pid=os.getpid(),utc=previous.old.utc()))
    planning,forecast,case,seeds=previous.context_planning(entry);score=planning['score']
    Hstar=physical_response(entry['case'],score['labels'],entry['physical_cache']['network_hash'])
    g=geometry.build(case,score,forecast,Hstar,entry['oracle_target'],entry['oracle_competitors'],entry['base_kw'])
    np.savez_compressed(directory/'geometry.npz',**g)
    io.write(directory/'resource_case.json',case.config)
    shutil.copyfile(entry['initial_points']['path'],directory/'initial_points.npz')
    io.write(directory/'geometry.json',dict(arrays=io.ref(directory/'geometry.npz'),score=entry['score'],
        target=entry['oracle_target'],competitors=entry['oracle_competitors'],initial_points=entry['initial_points'],
        physical_network_hash=entry['physical_cache']['network_hash'],base_kw=entry['base_kw'],expansion=p['expansion']))
    seeds=dict(seeds,explicit_zero=np.zeros(case.shape));controls={};plans={}
    baseline=linear_metrics(case,score,g,np.zeros(case.shape));io.write(directory/'zero.json',baseline)
    for mode in p['modes']:
        try:
            receipt,u,z=solver.plan(case,dict(g,mode=mode),seeds,seconds=p['solver_seconds'],
                env=previous.old.WORK['env'],record_directory=directory/(mode+'_diagnostics'))
            if u is None:u=np.zeros(case.shape)
            check=geometry.direct(case,g,u,mode=mode)
            if not case.audit(u)['passed'] or check['direct_z'] < -1e-7:raise ValueError('Returned action fails direct oracle constraints')
            receipt.update(direct=check,linear=linear_metrics(case,score,g,u),cost=prices.cost_breakdown(case,u),quantities=case.amounts(u),
                positive_direction_found=check['direct_z']>1e-6)
        except Exception as exc:
            u=np.zeros(case.shape);receipt=dict(status='exception',error=repr(exc),traceback=traceback.format_exc(),
                positive_direction_found=False,direct=geometry.direct(case,g,u,mode=mode),linear=baseline,
                cost=prices.cost_breakdown(case,u),quantities=case.amounts(u))
        controls[mode]=u;plans[mode]=receipt;np.savez_compressed(directory/(mode+'_action.npz'),action=u)
        io.write(directory/(mode+'_plan.json'),receipt)
        print(entry['id'],mode,receipt['status'],'z',receipt['direct']['direct_z'],'e/e0',receipt['direct']['e']/max(g['e_c'],1e-30),flush=True)
    np.savez_compressed(directory/'planner.npz',forecast=forecast,**controls);io.write(directory/'plans.json',plans)
    # Oracle input and both actions are sealed before reading any execution eta.
    io.write(directory/'seal.json',dict(protocol=io.ref(out/'protocol.json'),planner=io.ref(directory/'planner.npz'),plans=io.ref(directory/'plans.json'),
        geometry=io.ref(directory/'geometry.json'),resource_case=io.ref(directory/'resource_case.json'),initial_points=io.ref(directory/'initial_points.npz'),
        score=entry['score'],train=entry['files']['train.npz'],validation=entry['files']['validation.npz'],actual_new_day_opened=False))
    rows=[]
    for repeat in range(p['repeats']):
        source=Path(entry['prior_directory'])/f'execution{repeat:02d}'/'paired_randomness.npz';ref=io.ref(source)
        with np.load(source,allow_pickle=False) as z:eta=z['eta'].copy()
        rd=directory/f'execution{repeat:02d}';rd.mkdir();np.savez_compressed(rd/'eta.npz',eta=eta)
        for mode,u in controls.items():
            try:
                delivered,execution=case.execute(u,forecast[:,:case.N]*entry['base_kw'],eta)
                row=dict(status='complete',mode=mode,repeat=repeat,eta_source=ref,delivery=execution,
                    direct=geometry.direct(case,g,delivered,mode=mode),linear=linear_metrics(case,score,g,delivered),
                    cost=prices.cost_breakdown(case,u,delivered),quantities=case.amounts(delivered))
                np.savez_compressed(rd/(mode+'.npz'),delivered=delivered)
            except Exception as exc:row=dict(status='execution_failed',mode=mode,repeat=repeat,error=repr(exc),traceback=traceback.format_exc())
            rows.append(row)
        io.write(directory/'deliveries.json',rows)
    original=previous.arrays(io.read(entry['prior_seal']['path'])['planner']['path'])
    io.write(directory/'old_action_linear_references.json',{name:dict(linear=linear_metrics(case,score,g,original[name]),
        cost=prices.cost_breakdown(case,original[name]),direct=geometry.direct(case,g,original[name],mode='distance_only'))
        for name in previous.panel.POLICIES})
    complete=dict(status='complete',id=entry['id'],seconds=perf_counter()-began,quotes=len(plans),
        positive_directions=sum(r['positive_direction_found'] for r in plans.values()),
        quote_exceptions=sum(r['status']=='exception' for r in plans.values()),delivery_rows=len(rows),
        failed_delivery_rows=sum(r['status']!='complete' for r in rows),plans=io.ref(directory/'plans.json'),deliveries=io.ref(directory/'deliveries.json'))
    io.write(directory/'complete.json',complete);return complete


def run(out):
    out=Path(out).resolve();p=io.read(out/'protocol.json');previous.check_refs([p['inputs'],*p['sources']])
    if (out/'lock.json').exists() or (out/'complete.json').exists():raise RuntimeError('Preserve existing run and locks')
    io.write(out/'lock.json',dict(pid=os.getpid(),utc=previous.old.utc()));receipts=[];failures=[]
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=previous.old.initialize,initargs=(str(out),)) as pool:
            jobs={pool.submit(run_case,e):e for e in io.read(p['inputs']['path'])}
            for task in as_completed(jobs):
                try:receipts.append(task.result())
                except Exception as exc:failures.append(dict(id=jobs[task]['id'],error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(receipts=receipts,failed_tasks=failures))
        result=dict(status='complete' if not failures else 'incomplete',receipts=receipts,failed_tasks=failures,
            quotes=sum(r['quotes'] for r in receipts),positive_directions=sum(r['positive_directions'] for r in receipts),
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),delivery_rows=sum(r['delivery_rows'] for r in receipts),
            failed_delivery_rows=sum(r['failed_delivery_rows'] for r in receipts),
            source_unchanged=all(io.ref(r['path'])==r for r in p['sources']))
        io.write(out/'complete.json',result);return result
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('outputs/frozen_rx_guard_execution_20261115_v1'))
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--preflight',type=Path);a=parser.parse_args()
    prepare(a.source,a.out,smoke=a.smoke,seconds=a.seconds,preflight=a.preflight);print(run(a.out))


if __name__=='__main__':main()
