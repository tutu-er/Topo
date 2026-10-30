"""Freeze, tune on development validation, then run the held-out matrix."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter,sleep
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rnj_wzzt_core'))
import numpy as np

CASES=('paper15','soumalas11','flynn16','pengwah18')
METHODS=('original_leaf','original_RNJ','pool_RNJ','rebuild_RNJ')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(30):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==29:raise
            sleep(.02)


def ident(panel,case,bundle,condition,sigma,N):
    return f'{panel}__{case}__b{bundle}__{condition}__v{round(sigma*10000):04d}__N{N}'


def catalog():
    dev=[ident('main',c,b,'reference',v,48) for c in CASES for b in (0,1) for v in (.0002,.001)]
    test=[ident('main',c,b,'reference',v,N) for c in CASES for b in (2,3,4)
          for v in (.0002,.0005,.001) for N in (24,48,96)]
    test += [ident('stress',c,2,condition,.0005,48) for c in CASES
             for condition in ('weak_excitation','root_noise','outliers')]
    test += [ident('topology',f'random{n}_t{t}',2,'reference',v,48)
             for n in (8,16,24,32) for t in (1,2) for v in (.0002,.001)]
    test += [ident('case33','case33_terminal',2,'reference',v,N) for v in (.0002,.001) for N in (48,288)]
    return dev,test


def freeze(output):
    import gurobipy as gp,scipy,platform
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    dev,test=catalog();parent=ROOT/'outputs/wzzt_paper_20261029_full/jobs'
    metadata={}
    for job in [*dev,*test]:
        payload=read(parent/job/'payload.json')
        for name in ('train','validation','test'):assert Path(payload[name]).is_file(),(job,name)
        metadata[job]=payload
        prepare_payload=dict(job=job,N=payload['N'],train=payload['train'],validation=payload['validation'],
            train_sha256=sha(payload['train']),validation_sha256=sha(payload['validation']),
            destination=str(output/'common'/job))
        write(output/'common'/job/'payload.json',prepare_payload)
    folders=['rnj_wzzt_core/rnj_wzzt','research_experiments/rebuild_benchmark','research_experiments/local_rebuild',
        'research_experiments/wzzt_retune','research_experiments/wzzt_refine','research_experiments/wzzt_systematic',
        'research_experiments/wzzt_fine','research_experiments/wzzt_paper','research_experiments/wzzt_solver_tuning',
        'research_experiments/rnj']
    snapshot=output/'source'
    for folder in folders:
        for file in (ROOT/folder).rglob('*.py'):
            target=snapshot/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),development=dev,evaluation=test,
        development_inputs=len(dev),evaluation_inputs=len(test),methods=METHODS,workers=4,solver_threads=1,
        search_seconds=20.,per_MIP_seconds=2.,LP_seconds=5.,worker_hard_seconds=180.,
        common_preprocessing='fresh training-only ordered R/X, raw/demean NJ/RNJ and ridge .001/.01 pool; raw RNJ bank shared',
        initialization='original_leaf uses singleton leaves; all RNJ controls use raw RX75 0.16 native family',
        original_model='core unrestricted one-support extension; all accepted supports including zero remain structural decisions',
        budgets='equal search wall caps including LP warm starts/build; output-selection and common frontend costs recorded separately',
        development_search_configs=[dict(b=b,relative_gain=g) for b in (4,5) for g in (.001,.005)],
        selection_grid=dict(resolution=[.08,.16,.24],validation_fraction=[0.,.01,.03]),
        hyperparameter_rule='min mean log validation MAE relative to initial RNJ; within 0.1% choose fewer internal supports, lower runtime, b4, higher gain, fixed lexicographic order',
        information='no topology truth or test observations used for development tuning; all test metrics read only after freeze',
        numerical=dict(scale=1000.,MIPGap=.001,MIPGapAbs_physical=1e-10,feasibility=1e-9,seed=20261030),
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version(),platform=platform.platform()),
        source_hashes={p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*.py')},
        scope='108 main matched inputs + 12 stress + 16 random topology + 4 case33; synthesis only, not field measurements')
    assert len(dev)==16 and len(test)==140
    write(output/'protocol.json',protocol);write(output/'evaluation_catalog.json',metadata)
    print('FROZEN',len(dev),'development',len(test),'evaluation',flush=True)


def prepare_worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.wzzt_retune.engine import build_bank
    from research_experiments.wzzt_refine.core import regularized_pool
    from .engine import Bank,prepare,canonical
    p=read(payload);destination=Path(p['destination']);then=perf_counter()
    assert sha(p['train'])==p['train_sha256'] and sha(p['validation'])==p['validation_sha256']
    obs=observed(p['train'],p['N']);data=prepare(obs)
    bank,pool,fronts,calibrations=build_bank(obs);aug=regularized_pool(obs,pool)
    initial=next(q['family'] for q in bank if q['method']=='RNJ' and q['config']=='raw_RX75_0.16')
    calibration=calibrations['raw'];R=np.asarray(fronts['raw']['R']);X=np.asarray(fronts['raw']['X'])
    kernel=.75*R/calibration['mu_r']+.25*X/calibration['mu_x']
    fits=Bank(data,aug['pool'])
    try:
        points=[]
        for member in bank:
            if member['method']=='RNJ' and member['preprocess']=='raw':points.append(fits.fit(member['family']))
        result=dict(job=p['job'],labels=list(obs[0].columns),pool=aug['pool'],initial=canonical(initial),
            calibration=calibration,kernel=kernel,bank=points,seconds=perf_counter()-then,status='completed',
            LP=fits.lp.solves,LP_failures=fits.failures,source_input_hashes=dict(train=p['train_sha256'],validation=p['validation_sha256']))
        write(destination/'result.json',result)
    finally:fits.close()


def validation_candidates(bank,points,validation,calibration,resolution):
    from .engine import loss
    contracted=[]
    for p in points:
        keep=[s for s,r,x in zip(p['supports'],p['r'],p['x']) if len(s)==1 or
            calibration['a']*r+calibration['b']*x+1e-10>=resolution*max(calibration['median_root_depth'],1e-12)]
        q=bank.fit(keep)
        contracted.append(dict(point=q,validation=loss(validation,q),internal=sum(1<len(s)<bank.data.n for s in q['supports'])))
    return contracted


def choose(candidates,fraction):
    best=min(q['validation'] for q in candidates)
    eligible=[(i,q) for i,q in enumerate(candidates) if q['validation']<=best*(1+fraction)+1e-12]
    i,q=min(eligible,key=lambda t:(len(t[1]['point']['supports']),t[1]['validation'],t[0]))
    return dict(q,index=i)


def search_worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from .engine import Bank,prepare,search,pool_repair,canonical,loss
    p=read(payload);common=read(p['common']);then=perf_counter()
    train=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']))
    bank=Bank(train,common['pool']);initialized=perf_counter()-then
    try:
        if p['method']=='original_leaf':initial=[(i,) for i in range(train.n)];kind='forward'
        elif p['method']=='original_RNJ':initial=common['initial'];kind='forward'
        elif p['method']=='pool_RNJ':initial=common['initial'];kind='pool'
        else:initial=common['initial'];kind='rebuild'
        config=p['config']
        if kind=='pool':trajectory=pool_repair(bank,initial,seconds=20.,relative=config['relative_gain'])
        else:trajectory=search(bank,initial,np.asarray(common['kernel']),kind=kind,b=config['b'],
                              relative=config['relative_gain'],seconds=20.,per_MIP=2.)
        select_start=perf_counter()
        # All methods receive the same RNJ bank at final selection. A path-only
        # view is retained to separate initialization from search effects.
        points=[*common['bank'],*trajectory['path']]
        bank.extend(s for q in points for s in q['supports'])
        resolutions=(.08,.16,.24)
        fractions=(0.,.01,.03)
        selected={};selected_path={}
        for resolution in resolutions:
            candidates=validation_candidates(bank,points,validation,common['calibration'],resolution)
            for fraction in fractions:
                selected[f'{resolution:g}_{fraction:g}']=choose(candidates,fraction)
                selected_path[f'{resolution:g}_{fraction:g}']=choose(candidates[len(common['bank']):],fraction)
        initial_point=bank.fit(common['initial'])
        legacy_scores=[loss(validation,q) for q in points]
        oldbest=min(legacy_scores);oldeligible=[i for i,v in enumerate(legacy_scores) if v<=oldbest*1.03+1e-12]
        oldindex=min(oldeligible,key=lambda i:(len(points[i]['supports']),legacy_scores[i],i));q=points[oldindex]
        active=[i for i,(s,r,x) in enumerate(zip(q['supports'],q['r'],q['x'])) if len(s)==1 or
            common['calibration']['a']*r+common['calibration']['b']*x+1e-10>=.16*common['calibration']['median_root_depth']]
        legacy=dict(supports=[q['supports'][i] for i in active],r=[q['r'][i] for i in active],x=[q['x'][i] for i in active])
        legacy['objective']=loss(train,legacy)
        write(p['result'],dict(job=p['job'],method=p['method'],phase=p['phase'],config=config,
            trajectory=trajectory,selected=selected,selected_path=selected_path,legacy=legacy,initial=initial_point,
            initial_validation=loss(validation,initial_point),selection_seconds=perf_counter()-select_start,
            initialized_seconds=initialized,elapsed_seconds=perf_counter()-then,common_seconds=common['seconds'],
            LP=bank.lp.solves,LP_failures=bank.failures,status='completed',truth_used=False,test_opened=False,
            common_sha256=sha(p['common']),input_hashes=common['source_input_hashes']))
    finally:bank.close()


def dispatch(output,phase,tasks):
    output=Path(output).resolve();protocol=read(output/'protocol.json');snapshot=output/'source'
    env=os.environ.copy();env['PYTHONPATH']=str(snapshot)+os.pathsep+str(snapshot/'rnj_wzzt_core')
    pending=list(tasks);active={};done=0;started=perf_counter()
    while pending or active:
        while pending and len(active)<protocol['workers']:
            p=pending.pop(0);target=Path(p['result']);payload=target.with_name('payload.json');write(payload,p)
            log=target.with_name('worker.log').open('w',encoding='utf-8')
            kind='prepare' if phase.startswith('prepare') else 'search'
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.rebuild_benchmark.run','worker',kind,str(payload)],
                cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[str(target)]=(proc,log,perf_counter(),p)
        for key,(proc,log,start,p) in list(active.items()):
            target=Path(p['result'])
            if proc.poll() is None and perf_counter()-start>protocol['worker_hard_seconds']:
                proc.kill();proc.wait();write(target.with_name('timeout.json'),dict(reason='180s worker hard limit'))
            if proc.poll() is not None:
                log.close()
                if not target.exists():write(target.with_name('error.json'),dict(exit_code=proc.returncode,job=p['job']))
                done+=1;del active[key]
                print(phase,done,'/',len(tasks),p['job'],p.get('method',''),target.exists(),flush=True)
        write(output/(phase+'_progress.json'),dict(planned=len(tasks),finished=done,active=[p['job'] for _,_,_,p in active.values()],
                                                  pending=len(pending),seconds=perf_counter()-started))
        if active:sleep(.5)
    write(output/(phase+'_completion.json'),dict(planned=len(tasks),returned=sum(Path(p['result']).exists() for p in tasks),seconds=perf_counter()-started))


def prepare_phase(output,split):
    output=Path(output).resolve();protocol=read(output/'protocol.json')
    jobs=protocol['development' if split=='development' else 'evaluation']
    tasks=[]
    for job in jobs:
        p=read(output/'common'/job/'payload.json');result=output/'common'/job/'result.json'
        if result.exists():continue
        tasks.append(dict(p,result=str(result)))
    dispatch(output,'prepare_'+split,tasks)


def run_development(output):
    output=Path(output).resolve();protocol=read(output/'protocol.json');tasks=[]
    for job in protocol['development']:
        source=read(output/'common'/job/'payload.json')
        for i,config in enumerate(protocol['development_search_configs']):
            result=output/'development'/job/f'c{i}'/'result.json'
            tasks.append(dict(source,phase='development',method='rebuild_RNJ',config=config,
                common=str(output/'common'/job/'result.json'),result=str(result)))
    dispatch(output,'development',tasks)


def freeze_parameters(output):
    output=Path(output).resolve();protocol=read(output/'protocol.json');records=[]
    for i,config in enumerate(protocol['development_search_configs']):
        for resolution in protocol['selection_grid']['resolution']:
            for fraction in protocol['selection_grid']['validation_fraction']:
                values=[];complexities=[];runtimes=[];by_case={};hashes={}
                for job in protocol['development']:
                    file=output/'development'/job/f'c{i}'/'result.json'
                    if not file.exists():continue
                    r=read(file);q=r['selected'][f'{resolution:g}_{fraction:g}']
                    value=float(np.log(q['validation']/max(r['initial_validation'],1e-12)))
                    values.append(value);complexities.append(q['internal']);runtimes.append(r['elapsed_seconds'])
                    case=job.split('__')[1];by_case.setdefault(case,[]).append(value);hashes[str(file)]=sha(file)
                records.append(dict(config_id=i,**config,resolution=resolution,validation_fraction=fraction,
                    available=len(values),mean_log_validation_ratio=float(np.mean(values)) if values else None,
                    mean_internal=float(np.mean(complexities)) if values else None,
                    mean_seconds=float(np.mean(runtimes)) if values else None,by_case=by_case,development_hashes=hashes))
    complete=[r for r in records if r['available']==len(protocol['development'])]
    if not complete:raise RuntimeError('no complete development configuration; inspect every failure')
    best=min(r['mean_log_validation_ratio'] for r in complete)
    eligible=[r for r in complete if r['mean_log_validation_ratio']<=best+np.log(1.001)]
    selected=min(eligible,key=lambda r:(r['mean_internal'],r['mean_seconds'],r['b'],-r['relative_gain'],r['resolution'],r['validation_fraction']))
    folds={}
    for excluded in CASES:
        ranked=[(np.mean([v for c,a in r['by_case'].items() if c!=excluded for v in a]),r) for r in complete]
        score,r=min(ranked,key=lambda t:(t[0],t[1]['mean_internal']))
        folds[excluded]=dict(config={k:r[k] for k in ('b','relative_gain','resolution','validation_fraction')},
                             held_network_log_ratio=float(np.mean(r['by_case'][excluded])))
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),parameters={k:selected[k] for k in
        ('b','relative_gain','resolution','validation_fraction')},selected=selected,grid=records,
        leave_one_network_out=folds,selection_rule=protocol['hyperparameter_rule'],
        test_opened=False,topology_truth_used=False,protocol_sha256=sha(output/'protocol.json'))
    path=output/'frozen_parameters.json'
    if path.exists():raise FileExistsError('parameters already frozen')
    write(path,value);print('PARAMETERS FROZEN',value['parameters'],flush=True)


def run_evaluation(output):
    output=Path(output).resolve();protocol=read(output/'protocol.json');frozen=read(output/'frozen_parameters.json');tasks=[]
    if not (output/'frozen_method_parameters.json').is_file():
        raise RuntimeError('freeze every baseline output policy before evaluation')
    for job in protocol['evaluation']:
        source=read(output/'common'/job/'payload.json')
        for method in METHODS:
            result=output/'evaluation'/job/method/'result.json'
            tasks.append(dict(source,phase='evaluation',method=method,config=frozen['parameters'],
                common=str(output/'common'/job/'result.json'),result=str(result),frozen_sha256=sha(output/'frozen_parameters.json'),
                frozen_method_sha256=sha(output/'frozen_method_parameters.json')))
    dispatch(output,'evaluation',tasks)


def controls_development(output):
    output=Path(output).resolve();protocol=read(output/'protocol.json');frozen=read(output/'frozen_parameters.json');tasks=[]
    for job in protocol['development']:
        source=read(output/'common'/job/'payload.json')
        for method in METHODS[:-1]:
            result=output/'development_controls'/job/method/'result.json'
            tasks.append(dict(source,phase='development',method=method,config=frozen['parameters'],
                common=str(output/'common'/job/'result.json'),result=str(result)))
    dispatch(output,'development_controls',tasks)


def freeze_method_parameters(output):
    output=Path(output).resolve();protocol=read(output/'protocol.json');frozen=read(output/'frozen_parameters.json')
    search_config={k:frozen['parameters'][k] for k in ('b','relative_gain')}
    config_id=frozen['selected']['config_id'];methods={}
    for method in METHODS:
        methods[method]={}
        for view in ('selected','selected_path'):
            records=[]
            for resolution in (.08,.16,.24):
                for fraction in (0.,.01,.03):
                    values=[];internal=[]
                    for job in protocol['development']:
                        file=(output/'development'/job/f'c{config_id}'/'result.json') if method=='rebuild_RNJ' else output/'development_controls'/job/method/'result.json'
                        if not file.exists():raise RuntimeError('incomplete baseline tuning: '+str(file))
                        r=read(file);choice=r[view][f'{resolution:g}_{fraction:g}']
                        values.append(np.log(choice['validation']/max(r['initial_validation'],1e-12)));internal.append(choice['internal'])
                    records.append(dict(resolution=resolution,validation_fraction=fraction,mean_log_validation_ratio=float(np.mean(values)),mean_internal=float(np.mean(internal))))
            best=min(r['mean_log_validation_ratio'] for r in records)
            eligible=[r for r in records if r['mean_log_validation_ratio']<=best+np.log(1.001)]
            selected=min(eligible,key=lambda r:(r['mean_internal'],r['mean_log_validation_ratio'],r['resolution'],r['validation_fraction']))
            methods[method][view]=dict(parameters={**search_config,**{k:selected[k] for k in ('resolution','validation_fraction')}},grid=records,selected=selected)
    file=output/'frozen_method_parameters.json'
    if file.exists():raise FileExistsError('method parameters already frozen')
    write(file,dict(created_utc=datetime.now(timezone.utc).isoformat(),methods=methods,search_config=search_config,
        development_only=True,test_opened=False,topology_truth_used=False,source_hyperparameters_sha256=sha(output/'frozen_parameters.json')))
    print('BASELINE AND REBUILD OUTPUT PARAMETERS FROZEN', {k:v['selected']['parameters'] for k,v in methods.items()},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','prepare','development','select','controls','select_methods','evaluation','worker'))
    parser.add_argument('output');parser.add_argument('extra',nargs='?');a=parser.parse_args()
    if a.command=='freeze':freeze(a.output)
    elif a.command=='prepare':prepare_phase(a.output,a.extra or 'development')
    elif a.command=='development':run_development(a.output)
    elif a.command=='select':freeze_parameters(a.output)
    elif a.command=='controls':controls_development(a.output)
    elif a.command=='select_methods':freeze_method_parameters(a.output)
    elif a.command=='evaluation':run_evaluation(a.output)
    else:
        payload=a.extra
        try:(prepare_worker if a.output=='prepare' else search_worker)(payload)
        except Exception as exc:
            p=read(payload);target=Path(p['result']).with_name('error.json')
            write(target,dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
