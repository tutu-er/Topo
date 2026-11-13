"""Paired capacity scenarios using the previously audited true-AC execution loop."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import argparse,shutil,subprocess,atexit,traceback
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..unified_posterior_dr_20261112.physical import open_runtime
from ..linear_ac_population_20261113 import study as engine
from .solver import solve

OLD=ROOT/'outputs/linear_ac_population_20261113_v2'
SCENARIOS={'unbounded':None,'cap10':.1,'cap30':.3,'cap50':.5}


def prepare(out):
    require(not out.exists(),'Keep existing run');out.mkdir(parents=True)
    require(io.read(OLD/'confirmation/independent_audit.json')['status']=='passed','Require audited prior study')
    prior=io.read(OLD/'protocol.json');inputs=io.read(OLD/'inputs.json')
    for entry in inputs:require(sha(entry['scores']['path'])==entry['scores']['sha256'],'Frozen library changed')
    sources={}
    for p in sorted((ROOT/'rnj_wzzt_dr').rglob('*.py')):
        if '__pycache__' in p.parts:continue
        if p.parent==Path(__file__).parent and p.name in ('audit.py','report.py','diagnose.py'):continue
        relative=p.relative_to(ROOT);target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);sources[relative.as_posix()]=io.ref(p)
    cfg=dict(prior,created_utc=utc(),sources=sources,study_seed=2026111308,
        commit_at_lock=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        response='bidirectional aggregate net power around the prior/day-ahead baseline; normalized signed incentive x gives deltaP=m*x, deltaQ=.35*deltaP',
        bounds='|x_ih|<=alpha, alpha=.1/.3/.5 of historical nodal peak; capacities are declared scenarios, not measured fleet capacity; net export allowed',
        price='normalized convex opportunity cost C=mean(x^2); marginal incentive linear in x; no claim of calibrated monetary settlement',
        solver='unbounded single-pair dual bounds retained; original box-constrained joint QCQP for binding capacities, with independently verified warm points',
        solver_seconds_per_plan=.4,previous_completed_study=io.ref(OLD/'protocol.json'),
        phase_gamma={'pilot':[float(__import__('numpy').log(2)),2.,4.],'population':[2.],'confirmation':[float(__import__('numpy').log(2)),2.,4.]},
        pilot=dict(ids=prior['pilot']['ids'],days=1,replicate_base=3000000),
        population=dict(ids=prior['population']['ids'],days=4,replicate_base=1000000),
        confirmation=dict(ids=io.read(OLD/'confirmation_selection.json')['ids'],days=32,replicate_base=2000000,
            selection='exactly the 12 previously locked positive/weak examples; retained before any new capacity-test paths'),
        development_failure=None,
        failures='infeasible/no-quote applies explicit zero fallback and is counted; infeasibility proven by valid box bound or native global status only; no seed replacement',
        allowed_net_injection=True,new_identifier_fits=0,new_parameter_fits=0)
    io.write(out/'protocol.json',cfg);io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=utc(),protocol_sha256=sha(out/'protocol.json')))
    for name,alpha in SCENARIOS.items():
        root=out/name;root.mkdir();shutil.copyfile(OLD/'inputs.json',root/'inputs.json');shutil.copyfile(OLD/'requests.json',root/'requests.json')
        io.write(root/'protocol.json',dict(cfg,alpha=alpha,scenario=name))
        for cohort in ('main','pressure'):open_runtime(root,cohort)
    return cfg


_CACHE={};_ENV=None;_ROOT=None;_PHASE=None
def initialize(out,phase):
    import gurobipy as gp
    global _ENV,_ROOT,_PHASE
    _ROOT=Path(out);_PHASE=phase
    _ENV=gp.Env(empty=True);_ENV.setParam('OutputFlag',0);_ENV.start();atexit.register(_ENV.dispose)


def price_solver(model,gamma,seconds=.4,env=None):
    return solve(model,gamma,engine._WORK['protocol']['alpha'],seconds=seconds,env=env)


def work(task):
    scenario,identity=task;root=_ROOT/scenario
    if scenario not in _CACHE:
        cfg=io.read(root/'protocol.json');cfg['gamma']=cfg['phase_gamma'][_PHASE]
        _CACHE[scenario]=dict(out=root,protocol=cfg,inputs={e['request']['id']:e for e in io.read(root/'inputs.json')},
            truth={e['id']:e for e in io.read(ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json')},modules={},env=_ENV)
    # Reuse the exact audited forecast -> frozen policies -> AC -> meters loop.
    # Only the planning function changes; it receives no hidden physical state.
    engine._WORK=_CACHE[scenario];engine.solve=price_solver
    phase_cfg=engine._WORK['protocol'][_PHASE]
    return scenario,engine.run_input(identity,_PHASE,phase_cfg['days'],phase_cfg['replicate_base'])


def execute(out,phase):
    cfg=io.read(out/'protocol.json');phase_cfg=cfg[phase]
    require(not (out/f'{phase}_complete.json').exists(),'Completed phase is preserved')
    require(not (out/f'{phase}_started.json').exists(),'Existing start lock: inspect actual process before resuming')
    for relative,ref in cfg['sources'].items():require(sha(ROOT/relative)==ref['sha256'],'Execution source changed')
    io.write(out/f'{phase}_started.json',dict(pid=os.getpid(),created_utc=utc(),protocol=io.ref(out/'protocol.json')))
    tasks=[(scenario,identity) for scenario in SCENARIOS for identity in phase_cfg['ids']]
    receipts={name:[] for name in SCENARIOS};failed={name:[] for name in SCENARIOS};began=perf_counter();done=0
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(str(out),phase)) as pool:
        futures={pool.submit(work,t):t for t in tasks}
        for f in as_completed(futures):
            try:name,value=f.result();receipts[name].append(value)
            except Exception as exc:
                name,identity=futures[f];failed[name].append(dict(id=identity,error=repr(exc),traceback=traceback.format_exc()))
            done+=1
            if done%32==0 or done==len(tasks):
                io.write(out/f'{phase}_checkpoint.json',dict(done=done,total=len(tasks),seconds=perf_counter()-began,worker_failures=failed))
                print(phase,done,'/',len(tasks),'seconds',round(perf_counter()-began,1),'execution_failures',sum(r['failures'] for values in receipts.values() for r in values),flush=True)
    for name in SCENARIOS:
        io.write(out/name/phase/'complete.json',dict(status='complete' if not failed[name] else 'complete_with_worker_failures',
            receipts=receipts[name],worker_failures=failed[name],contexts=len(phase_cfg['ids'])*phase_cfg['days'],
            rows=sum(r['rows'] for r in receipts[name]),seconds=perf_counter()-began))
    io.write(out/f'{phase}_complete.json',dict(status='complete',unique_new_days=len(phase_cfg['ids'])*phase_cfg['days'],
        scenarios=len(SCENARIOS),paired_capacity_backgrounds=True,rows=sum(r['rows'] for values in receipts.values() for r in values),
        seconds=perf_counter()-began,worker_failures=failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','pilot','population','confirmation'],required=True)
    a=p.parse_args();prepare(a.run.resolve()) if a.stage=='prepare' else execute(a.run.resolve(),a.stage)
