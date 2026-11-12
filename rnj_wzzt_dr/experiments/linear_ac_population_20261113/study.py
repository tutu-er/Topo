"""Locked AC population study: true load -> noisy forecast -> QCQP -> AC -> meters."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,shutil,subprocess,traceback,atexit
import numpy as np
from scipy.special import logsumexp
from ... import io
from ...observations import load
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..unified_posterior_dr_20261112.physical import open_runtime,generate_background,measured_channels
from ..unified_posterior_dr_20261112.posterior import retained
from ..expected_logodds_admm_20261113.run import REFERENCE
from .solver import Model,solve


def prepare(out,study_seed=2026111305,development=None):
    require(not out.exists(),'Use a new output directory; old runs are preserved')
    out.mkdir(parents=True)
    requests=[r for r in io.read(REFERENCE/'requests.json') if r['method']=='101']
    require(len(requests)==400,'Expected 400 frozen history inputs')
    seals=io.read(REFERENCE/'selection_seal.json')['scores'];generation={};inputs=[]
    for index,r in enumerate(requests):
        scorefile=REFERENCE/'scores'/(r['id']+'.json');require(sha(scorefile)==seals[r['id']],'Historical library changed')
        s=io.read(scorefile);target=int(s['map_index']);prob=np.array([m['probability'] for m in s['models']])
        order=np.argsort(-prob,kind='stable');others=[int(j) for j in order if j!=target]
        competitors=[j for j in others if prob[j]>=.01]
        if others and others[0] not in competitors:competitors.insert(0,others[0])
        require(sha(r['observed']['path'])==r['observed']['sha256'],'History observations changed')
        observed=load(r['observed']['path']);scale=np.max(abs(observed.p),axis=0)
        batch=r['batch']
        if batch not in generation:generation[batch]=io.read(Path(batch)/'generation.json')
        metadata=generation[batch][r['job']]['metadata']
        parameters={k:metadata[k] for k in ('case','replicate','voltage_sigma','pq_sigma','root_sigma','study_seed','profile','level')}
        require(parameters['pq_sigma']<.05,'Prediction SD must exceed P/Q meter SD')
        inputs.append(dict(request=r,index=index,scores=io.ref(scorefile),target=target,competitors=competitors,
            scale=scale,parameters=parameters,physical_branch_sha256=metadata['physical_branch_sha256']))
    io.write(out/'requests.json',requests);io.write(out/'inputs.json',inputs)
    # Deterministic history-only pilot: four posterior regimes in each network.
    pilot=[]
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        available=[x for x in inputs if x['request']['case']==case]
        for value in (.6,.8,.95,.999):
            chosen=min(available,key=lambda x:(abs(io.read(x['scores']['path'])['p_map']-value),x['index']))
            pilot.append(chosen['request']['id']);available.remove(chosen)
    sources={}
    for path in sorted((ROOT/'rnj_wzzt_dr').rglob('*.py')):
        if '__pycache__' in path.parts:continue
        # Independent audit/report programs record their own versions; neither
        # is imported by planning or physical execution.
        if path.parent==Path(__file__).parent and path.name in ('audit.py','analyze.py'):continue
        rel=path.relative_to(ROOT);dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);sources[rel.as_posix()]=io.ref(path)
    for cohort in ('main','pressure'):open_runtime(out,cohort)
    protocol=dict(created_utc=utc(),commit_at_lock=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=io.environment(),sources=sources,inputs=io.ref(out/'inputs.json'),
        gamma=[float(np.log(2)),2.,4.],prediction_relative_sd=.05,qratio=.35,beta=.9,study_seed=study_seed,
        competitors='all other historical posterior weights >=1%; always include strongest runner-up; full library scores all observations',
        target='historical MAP only; actual truth never selects the planning target',
        response='deltaP=historical_node_peak*x_hour, deltaQ=.35*deltaP, exact deterministic linear delivery; no physical power, SOC, energy, availability, clipping or integer constraints',
        fee='mean squared signed hourly request; declared synthetic dimensionless fee, not calibrated currency',
        forecast='Phat=Ptrue*(1+.05*Zp), Qhat=Qtrue*(1+.05*Zq), independent standard normals per node/slot, independent of original meter streams',
        voltage='only original radial AC network generates evaluation voltage; never Htarget*P+e',
        solver='analytic single-competitor primal/dual QCQP; joint feasibility verification; warm-started native joint QCQP only when the valid gap is not closed',
        solver_seconds_per_plan=.15,global_gap_tolerance=1e-5,workers=4,
        pilot=dict(ids=pilot,days=1,replicate_base=700000),
        population=dict(ids=[r['id'] for r in requests],days=4,replicate_base=100000),
        confirmation=dict(days=64,replicate_base=500000,selection='after population: up to eight largest gamma=4 paired true-posterior gains with true candidate present, at most two per network; plus four weakest/negative controls, one per network; all choices sealed before confirmation'),
        primary='population all 400 inputs, all fixed gamma levels, paired AC/meter paths; confirmation selected cases separately',
        failures='retain all planning and AC failures; planning failure applies explicit zero-action fallback, not a qualified zero-cost quote; AC failures retain missing posterior and count as nonrecovery',
        truth='offline coverage, scoring and development-case selection only; no physical line parameters in solver inputs',
        development_failure=None if development is None else io.ref(development),
        new_identifier_fits=0,new_parameter_fits=0,private='raw forecast, P/Q/V, fitted matrices, physical state and actions remain in ignored outputs')
    io.write(out/'protocol.json',protocol);io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=utc(),protocol_sha256=sha(out/'protocol.json')))
    return protocol


def forecast(p,q,rng,sd):
    power=np.column_stack((p,q));noise=rng.normal(size=power.shape)
    return power*(1+sd*noise),noise


def evaluate(score,observed,truth,target):
    H=np.asarray([p['H'] for p in score['parameters']]);history=np.array([p['history_sse'] for p in score['parameters']])
    power=np.column_stack((observed.p,observed.q))
    residual=observed.y[None]-np.einsum('tc,knc->ktn',power,H)
    losses=history+np.sum(residual*residual,axis=(1,2));ll=-(losses-losses.min())/(2*score['sigma2']);ll-=logsumexp(ll)
    best=int(np.argmax(ll));ids=[m['topology_id'] for m in score['models']]
    ti=next((i for i,t in enumerate(ids) if t==truth['truth_id']),None);kept=retained(ll,.9)
    selected=set(map(tuple,score['models'][best]['active_clades']));true=set(map(tuple,truth['truth_clades']))
    return dict(p_true=None if ti is None else float(np.exp(ll[ti])),p_target=float(np.exp(ll[target])),
        p_max=float(np.exp(ll[best])),truth_covered=ti is not None,historical_target_correct=ids[target]==truth['truth_id'],
        map_correct=ids[best]==truth['truth_id'],correct_unique=len(kept)==1 and ti in kept,
        wrong_unique=len(kept)==1 and ti not in kept,retained_count=len(kept),truth_retained=ti in kept,
        true_pass90=bool(ti is not None and ll[ti]>=np.log(.9)),missing=len(true-selected),extra=len(selected-true),
        selected_index=best,log_probability=ll)


_WORK={}
def initialize(out):
    import gurobipy as gp
    out=Path(out);_WORK.update(out=out,protocol=io.read(out/'protocol.json'),inputs={x['request']['id']:x for x in io.read(out/'inputs.json')},
        truth={x['id']:x for x in io.read(REFERENCE/'evaluation.json')},modules={})
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();_WORK['env']=env;atexit.register(env.dispose)


def run_input(identity,phase,days,replicate_base):
    out=_WORK['out'];protocol=_WORK['protocol'];entry=_WORK['inputs'][identity];r=entry['request']
    directory=out/phase/'jobs'/identity;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return io.read(directory/'complete.json')
    io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=utc(),input_sha256=entry['scores']['sha256']))
    score=io.read(entry['scores']['path']);require(sha(entry['scores']['path'])==entry['scores']['sha256'],'Candidate library changed')
    H=np.asarray([p['H'] for p in score['parameters']]);target=entry['target'];competitors=entry['competitors']
    model=Model(H,target,competitors,entry['scale'],score['sigma2'],protocol['qratio']) if competitors else None
    cohort=r['cohort']
    if cohort not in _WORK['modules']:_WORK['modules'][cohort]=open_runtime(out,cohort)[0]
    module=_WORK['modules'][cohort];rows=[];started=perf_counter()
    for day in range(days):
        d=directory/f'day{day:03d}';d.mkdir(exist_ok=True);replicate=replicate_base+128*entry['index']+day
        parameters=dict(entry['parameters'],replicate=replicate,study_seed=protocol['study_seed'])
        common=dict(id=identity,case=r['case'],cohort=cohort,profile=r['profile'],level=r['level'],voltage_sigma=r['voltage_sigma'],
                    phase=phase,day=day,replicate=replicate,target=target,competitors=competitors)
        controls={'no_dr':np.zeros((24,H.shape[1]))};plans={}
        try:
            background=generate_background(module,parameters)
            require(background['metadata']['physical_branch_sha256']==entry['physical_branch_sha256'],'Actual network version changed')
            require(tuple(background['labels'])==tuple(score['labels']),'Terminal order changed')
            rng=np.random.default_rng(np.random.SeedSequence([protocol['study_seed'],replicate,991]))
            prediction,forecast_noise=forecast(background['p'],background['q'],rng,protocol['prediction_relative_sd'])
            if model is not None:model.prepare(prediction)
            # Planner receives only frozen estimated matrices, historical scale,
            # and noisy forecast. All policies sealed before evaluation voltages.
            for gamma in protocol['gamma']:
                label=f'g{gamma:.6g}'
                plan_start=perf_counter()
                try:
                    result=solve(model,gamma,seconds=protocol['solver_seconds_per_plan'],env=_WORK['env']) if model is not None else dict(qualified=True,x=controls['no_dr'].copy(),cost=0.,lower_bound=0.,relative_gap=0.,global_optimality_verified=True,status='single_candidate_zero',source='no_competitor',seconds=0.,gains=[])
                except Exception as exc:
                    result=dict(qualified=False,cost=None,lower_bound=None,relative_gap=None,global_optimality_verified=False,
                        status='solver_exception_zero_fallback',seconds=perf_counter()-plan_start,error=repr(exc),traceback=traceback.format_exc())
                x=result.pop('x',None);plans[label]=result
                controls['optimized_'+label]=x if result['qualified'] else np.zeros_like(controls['no_dr'])
                pattern=np.where((np.arange(24)[:,None]//2+np.arange(H.shape[1])[None,:])%2==0,1.,-1.)
                controls['rotation_'+label]=pattern*np.sqrt(result['cost'] or 0.) if result['qualified'] else np.zeros_like(pattern)
            np.savez_compressed(d/'planner.npz',forecast=prediction,**controls)
            io.write(d/'plans.json',plans)
            io.write(d/'policy_seal.json',dict(created_utc=utc(),planner=io.ref(d/'planner.npz'),plans=io.ref(d/'plans.json'),
                planner_inputs=['forecast_PQ','frozen_candidate_H','history_power_scale','frozen_sigma2','historical_target_and_competitors'],
                true_voltage_or_line_parameters_in_planner=False,physical_seed=replicate))
        except Exception as exc:
            for strategy in ['no_dr']+[f'{s}_g{g:.6g}' for g in protocol['gamma'] for s in ('optimized','rotation')]:
                rows.append(dict(common,strategy=strategy,status='background_or_planning_failed',error=repr(exc),traceback=traceback.format_exc()))
            io.write(directory/'results.json',rows);continue
        observed_records=[];valid_names=[];physical_p=[];physical_q=[];voltages=[];roots=[];weight_vectors=[]
        for strategy,x in controls.items():
            plan=None if strategy=='no_dr' else plans[strategy.split('_',1)[1]]
            cost=float(np.mean(x*x));delta=np.repeat(x,4,axis=0)*np.asarray(entry['scale'])
            p=background['p']+delta;q=background['q']+protocol['qratio']*delta
            info=dict(common,strategy=strategy,cost=cost,planning_failed=bool(plan is not None and not plan['qualified']),
                planned_gain=None if plan is None else plan.get('gains'),planning_seconds=None if plan is None else plan.get('seconds'),
                global_optimality_verified=None if plan is None else plan.get('global_optimality_verified'),
                absolute_DR_kwh=float(.25*background['network']['base_kw']*np.abs(delta).sum()),
                request_rms=float(np.sqrt(cost)),request_max=float(np.max(abs(x))),
                negative_P_fraction=float(np.mean(p<0)),linear_delivery_exact=True)
            try:
                voltage,ac=(background['voltage'],background['ac']) if np.all(x==0) else module._ac_voltages(background['network'],background['labels'],p,q,background['root'])
                observed=measured_channels(background,p,q,voltage)
                calculated=evaluate(score,observed,_WORK['truth'][identity],target)
                ll=calculated.pop('log_probability')
                info.update(status='complete',**calculated,voltage_violation_cells=int(((voltage<.9)|(voltage>1.1)).sum()),
                    minimum_voltage=float(voltage.min()),maximum_voltage=float(voltage.max()),ac=ac)
                valid_names.append(strategy);physical_p.append(p);physical_q.append(q);voltages.append(voltage);roots.append(observed.root);weight_vectors.append(ll)
                observed_records.append((observed.p,observed.q,observed.voltage,observed.root))
            except Exception as exc:info.update(status='AC_or_scoring_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(info)
        if valid_names:
            np.savez_compressed(d/'observed.npz',strategy=np.asarray(valid_names),labels=np.asarray(score['labels']),
                P=np.stack([v[0] for v in observed_records]),Q=np.stack([v[1] for v in observed_records]),
                V=np.stack([v[2] for v in observed_records]),root=np.stack([v[3] for v in observed_records]),log_probability=np.stack(weight_vectors))
            np.savez_compressed(d/'physical.npz',strategy=np.asarray(valid_names),P=np.stack(physical_p),Q=np.stack(physical_q),V=np.stack(voltages),
                baseline_P=background['p'],baseline_Q=background['q'],true_root=background['root'],
                standardized_meter_P=background['noise'][0],standardized_meter_Q=background['noise'][1],
                standardized_meter_V=background['noise'][2],standardized_meter_root=background['noise'][3],forecast_noise=forecast_noise)
        io.write(d/'generation.json',background['metadata']);io.write(directory/'results.json',rows)
    receipt=dict(id=identity,status='complete',phase=phase,days=days,rows=len(rows),seconds=perf_counter()-started,
        failures=sum(x['status']!='complete' for x in rows),result=io.ref(directory/'results.json'))
    io.write(directory/'complete.json',receipt);return receipt


def run_phase(out,phase):
    protocol=io.read(out/'protocol.json')
    require(not (out/phase/'complete.json').exists(),'Completed phase must not be rerun')
    for rel,ref in protocol['sources'].items():require(sha(ROOT/rel)==ref['sha256'],'Source drift: '+rel)
    phase_protocol=protocol[phase] if phase!='confirmation' else io.read(out/'confirmation_selection.json')
    ids=phase_protocol['ids'];days=phase_protocol['days'];base=phase_protocol['replicate_base']
    directory=out/phase;directory.mkdir(exist_ok=True)
    require(not (directory/'started.json').exists(),'Existing phase has a start lock; inspect processes and partial outputs before resuming')
    io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=utc(),protocol=io.ref(out/'protocol.json'),ids=ids,days=days))
    receipts=[];failures=[];began=perf_counter()
    with ProcessPoolExecutor(max_workers=protocol['workers'],initializer=initialize,initargs=(str(out),)) as pool:
        futures={pool.submit(run_input,i,phase,days,base):i for i in ids}
        for f in as_completed(futures):
            try:receipts.append(f.result())
            except Exception as exc:failures.append(dict(id=futures[f],error=repr(exc),traceback=traceback.format_exc()))
            if (len(receipts)+len(failures))%8==0 or len(receipts)+len(failures)==len(ids):
                io.write(directory/'checkpoint.json',dict(completed=len(receipts),worker_failures=failures,planned=len(ids),seconds=perf_counter()-began))
                print(phase,len(receipts)+len(failures),'/',len(ids),'seconds',round(perf_counter()-began,1),'row_failures',sum(x['failures'] for x in receipts),flush=True)
    io.write(directory/'complete.json',dict(status='complete' if not failures else 'complete_with_worker_failures',receipts=receipts,
        worker_failures=failures,seconds=perf_counter()-began,contexts=len(ids)*days,rows=sum(r['rows'] for r in receipts)))


def select(out):
    require((out/'population/complete.json').exists(),'Finish entire population before selecting examples')
    require(not (out/'confirmation_selection.json').exists(),'Selection already locked')
    inputs=io.read(out/'inputs.json');ranked=[]
    for entry in inputs:
        identity=entry['request']['id'];rows=io.read(out/'population/jobs'/identity/'results.json')
        base={r['day']:r for r in rows if r['strategy']=='no_dr' and r['status']=='complete'}
        pair=[(r,base[r['day']]) for r in rows if r['strategy']=='optimized_g4' and r['status']=='complete' and r['day'] in base and r['p_true'] is not None]
        if len(pair)!=4:continue
        ranked.append(dict(id=identity,case=entry['request']['case'],gain=float(np.mean([r['p_true']-b['p_true'] for r,b in pair])),
            recovery_change=sum(r['map_correct']-b['map_correct'] for r,b in pair),history_target_correct=pair[0][0]['historical_target_correct']))
    best=[];worst=[]
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        group=sorted([r for r in ranked if r['case']==case],key=lambda r:(-r['gain'],-r['recovery_change'],r['id']))
        best.extend(group[:2]);remaining=[r for r in group if r['id'] not in {x['id'] for x in best}]
        if remaining:worst.append(remaining[-1])
    parent=io.read(out/'protocol.json')['confirmation']
    io.write(out/'confirmation_selection.json',dict(created_utc=utc(),ids=[r['id'] for r in best+worst],days=parent['days'],
        replicate_base=parent['replicate_base'],best=best,weak_controls=worst,selection_basis='four population days only; no confirmation future opened'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['prepare','pilot','population','select','confirmation'],required=True)
    p.add_argument('--study-seed',type=int,default=2026111305);p.add_argument('--development',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.stage=='prepare':prepare(out,a.study_seed,a.development)
    elif a.stage=='select':select(out)
    else:run_phase(out,a.stage)


if __name__=='__main__':main()
