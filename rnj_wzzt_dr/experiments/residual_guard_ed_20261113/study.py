"""Locked paired AC study; original identifiers and score matrices stay frozen."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import argparse,atexit,shutil,subprocess,traceback
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..unified_posterior_dr_20261112.physical import open_runtime
from ..linear_ac_population_20261113 import study as engine
from .model import calibrate
from .solver import solve

OLD=ROOT/'outputs/bidirectional_price_dr_20261113_v2'
SCENARIOS={'nominal30':(.3,False),'guard30':(.3,True),'nominal50':(.5,False),'guard50':(.5,True)}
CONFIRM=('main__input121__101','main__input195__101','pressure__input026__101','main__input039__101')
EXCLUDED=('audit.py','report.py','prices.py','longrun.py','effects.py','diagnose.py')


def prepare(out):
    require(not out.exists(),'Preserve earlier runs');out.mkdir(parents=True)
    require(io.read(OLD/'complete.json')['status']=='complete','Require completed prior bounded study')
    prior=io.read(OLD/'cap30/protocol.json');entries=io.read(OLD/'cap30/inputs.json')
    truth={r['id']:r for r in io.read(ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json')}
    groups={'evidence_gap':[],'already_clear':[],'missing_truth_control':[]};qualification={};scores={}
    for e in entries:
        identity=e['request']['id'];s=io.read(e['scores']['path']);scores[identity]=s
        ti=next((i for i,m in enumerate(s['models']) if m['topology_id']==truth[identity]['truth_id']),None)
        p=None if ti is None else s['models'][ti]['probability']
        qualification[identity]=dict(truth_index=ti,history_true_weight=p,history_MAP_correct=ti==e['target'])
        if p is not None and p<.99:groups['evidence_gap'].append(identity)
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        available=[e for e in entries if e['request']['case']==case]
        clear=next(e for e in available if (qualification[e['request']['id']]['history_true_weight'] or 0)>=.99)
        missing=min((e for e in available if qualification[e['request']['id']]['truth_index'] is None),
            key=lambda e:(abs(scores[e['request']['id']]['p_map']-.8),e['index']))
        groups['already_clear'].append(clear['request']['id']);groups['missing_truth_control'].append(missing['request']['id'])
    identities=set(i for values in groups.values() for i in values);selected=[e for e in entries if e['request']['id'] in identities]
    require(len(groups['evidence_gap'])==28 and len(selected)==36,'Qualification differs from audited library')
    sources={}
    for p in sorted((ROOT/'rnj_wzzt_dr').rglob('*.py')):
        if '__pycache__' in p.parts or (p.parent==Path(__file__).parent and p.name in EXCLUDED):continue
        relative=p.relative_to(ROOT);target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);sources[relative.as_posix()]=io.ref(p)
    cfg=dict(prior,created_utc=utc(),sources=sources,study_seed=2026111313,
        commit_at_lock=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        alpha=None,scenario=None,gamma=[2.,4.],solver_seconds_per_plan=.8,
        residual_guard='history 32 validation points; one ridge-shrunk output scale per candidate/meter; require nonnegative corrected evidence increments for BOTH endpoints of every planned comparison; nominal gain still >=gamma',
        scoring='all original frozen H, history SSE and sigma2 unchanged; correction is a planning guard only',
        qualification='offline truth-containing + historical true weight <99%, plus four easy and four missing-truth controls; historical MAP remains the operating target, no true matrix/label enters solver',
        phase_gamma={'pilot':[2.],'population':[2.],'confirmation':[4.]},
        pilot=dict(ids=list(CONFIRM),days=1,replicate_base=7000000),
        population=dict(ids=[e['request']['id'] for e in selected],days=16,replicate_base=8000000),
        confirmation=dict(ids=list(CONFIRM),days=32,replicate_base=9000000),
        previous_completed_study=io.ref(OLD/'protocol.json'),development_failure=None,
        native_relative_gap_tolerance=1e-5,new_identifier_fits=0,new_parameter_fits=0,
        price_scope='separate fixed-commitment economic-dispatch feasibility and illustrative yuan/MWh accounting; not used to tune physical-study cases')
    io.write(out/'protocol.json',cfg);io.write(out/'qualification.json',dict(groups=groups,items={i:qualification[i] for i in identities}))
    for scenario,(alpha,guard) in SCENARIOS.items():
        root=out/scenario;root.mkdir();io.write(root/'inputs.json',selected);io.write(root/'requests.json',[e['request'] for e in selected])
        io.write(root/'protocol.json',dict(cfg,alpha=alpha,guard=guard,scenario=scenario))
        for e in selected:
            require(sha(e['scores']['path'])==e['scores']['sha256'],'Frozen score changed')
            require(sha(e['request']['observed']['path'])==e['request']['observed']['sha256'],'Historical observations changed')
            c,means=calibrate(scores[e['request']['id']],e['request']['observed']['path'])
            file=root/'calibration'/(e['request']['id']+'.npz');file.parent.mkdir(exist_ok=True)
            np.savez_compressed(file,scale=c,mean=means)
        for cohort in ('main','pressure'):open_runtime(root,cohort)
    io.write(out/'lock.json',dict(pid=os.getpid(),protocol_sha256=sha(out/'protocol.json'),created_utc=utc()))


_ROOT=None;_PHASE=None;_ENV=None;_CACHE={}
def initialize(out,phase):
    import gurobipy as gp
    global _ROOT,_PHASE,_ENV
    _ROOT=Path(out);_PHASE=phase;_ENV=gp.Env(empty=True);_ENV.setParam('OutputFlag',0);_ENV.start();atexit.register(_ENV.dispose)


def planner(base,gamma,seconds=.8,env=None):
    context=engine._WORK;means=context.get('active_means') if context['protocol']['guard'] else None
    return solve(base,gamma,context['protocol']['alpha'],means=means,seconds=seconds,env=env)


def work(task):
    scenario,identity=task;root=_ROOT/scenario
    if scenario not in _CACHE:
        cfg=io.read(root/'protocol.json');cfg['gamma']=cfg['phase_gamma'][_PHASE]
        _CACHE[scenario]=dict(out=root,protocol=cfg,inputs={e['request']['id']:e for e in io.read(root/'inputs.json')},
            truth={e['id']:e for e in io.read(ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json')},modules={},env=_ENV)
    engine._WORK=_CACHE[scenario]
    with np.load(root/'calibration'/(identity+'.npz'),allow_pickle=False) as p:engine._WORK['active_means']=p['mean'].copy()
    engine.solve=planner;cfg=engine._WORK['protocol'][_PHASE]
    return scenario,engine.run_input(identity,_PHASE,cfg['days'],cfg['replicate_base'])


def execute(out,phase):
    cfg=io.read(out/'protocol.json');pc=cfg[phase]
    require(not (out/f'{phase}_started.json').exists(),'Inspect existing process/lock before resuming')
    for relative,ref in cfg['sources'].items():require(sha(ROOT/relative)==ref['sha256'],'Execution source changed '+relative)
    io.write(out/f'{phase}_started.json',dict(pid=os.getpid(),created_utc=utc(),protocol=io.ref(out/'protocol.json')))
    receipts={s:[] for s in SCENARIOS};failed={s:[] for s in SCENARIOS};began=perf_counter();done=0
    tasks=[(s,i) for s in SCENARIOS for i in pc['ids']]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(str(out),phase)) as pool:
        futures={pool.submit(work,t):t for t in tasks}
        for f in as_completed(futures):
            try:s,value=f.result();receipts[s].append(value)
            except Exception as exc:
                s,i=futures[f];failed[s].append(dict(id=i,error=repr(exc),traceback=traceback.format_exc()))
            done+=1
            if done%8==0 or done==len(tasks):
                result=dict(done=done,total=len(tasks),seconds=perf_counter()-began,worker_failures=failed)
                io.write(out/f'{phase}_checkpoint.json',result);print(phase,done,'/',len(tasks),'seconds',round(result['seconds'],1),flush=True)
    for s in SCENARIOS:io.write(out/s/phase/'complete.json',dict(status='complete' if not failed[s] else 'complete_with_worker_failures',
        receipts=receipts[s],worker_failures=failed[s],contexts=len(pc['ids'])*pc['days'],rows=sum(r['rows'] for r in receipts[s]),seconds=perf_counter()-began))
    io.write(out/f'{phase}_complete.json',dict(status='complete' if not any(failed.values()) else 'complete_with_worker_failures',
        unique_new_days=len(pc['ids'])*pc['days'],scenarios=4,paired=True,worker_failures=failed,seconds=perf_counter()-began,
        rows=sum(r['rows'] for values in receipts.values() for r in values)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','pilot','population','confirmation'],required=True)
    a=p.parse_args();prepare(a.run.resolve()) if a.stage=='prepare' else execute(a.run.resolve(),a.stage)
