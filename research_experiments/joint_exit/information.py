"""Offline complete small-region profiles under oracle exterior structure."""
from __future__ import annotations
import argparse,os,subprocess,sys,traceback
from pathlib import Path
from time import perf_counter,sleep
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from .run import ROOT,read,write,sha
from .selection import choose
from research_experiments.local_rebuild.model import canonical
from research_experiments.local_rebuild.matched import all_hierarchies
from research_experiments.rebuild_benchmark.engine import prepare,loss
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from rnj_wzzt.estimation import laminar_l1_milp as core


def local_score(predicted,true):
    predicted=set(predicted);true=set(true)
    return dict(F1=2*len(predicted&true)/(len(predicted)+len(true)) if predicted or true else 1.,
        exact=predicted==true,TP=len(predicted&true),FP=len(predicted-true),FN=len(true-predicted))


def block_errors(data,point):
    R,X=core.build_matrices_from_atoms(data.n,point['supports'],point['r'],point['x'])
    v=np.concatenate([np.mean(abs(y-p@R.T-q@X.T),axis=1) for p,q,y in zip(data.p,data.q,data.target)])
    return v.reshape(-1,12).mean(axis=1)


def exact_rank_one_example():
    p=np.arange(1.,6.)[:,None]*np.ones((1,3));z=np.array([1.,1.,0.])
    true_R=.1*np.eye(3)+.2*np.outer(z,z);alternative=np.diag([.5,.5,.1])
    assert np.max(abs(p@true_R.T-p@alternative.T))<1e-12
    assert np.max(abs(np.eye(3)@true_R.T-np.eye(3)@alternative.T))>.1
    return dict(P_rank=int(np.linalg.matrix_rank(p)),true_support=[0,1],true_r=.2,
        alternative_singletons=[.5,.5,.1],maximum_prediction_difference=float(np.max(abs(p@true_R.T-p@alternative.T))),
        rich_excitation_prediction_difference=float(np.max(abs(true_R-alternative))),
        claim='exact nonidentifiability for this rank-one input design and unknown singleton impedances, not for arbitrary excitation')


def worker(payload):
    p=read(payload);meta=p['meta'];common=read(p['common']);n=len(common['labels']);labels=common['labels'];pos={v:i for i,v in enumerate(labels)}
    physical=physical_truth(Path(meta['truth_network']),labels)
    true=canonical(tuple(sorted(pos[v] for v in s)) for s in physical['clades'])
    small=[s for s in true if 2<=len(s)<=5]
    regions=canonical(s for s in small if not any(set(s)<set(t) for t in small))
    basis=canonical([*((i,) for i in range(n)),tuple(range(n))]);full=canonical([*basis,*true])
    z=np.array([[i in s for i in range(n)] for s in full],float)
    geometry=np.column_stack([np.outer(a,a).ravel() for a in z])
    rr=np.linalg.lstsq(geometry,physical['R'].ravel(),rcond=None)[0]
    xx=np.linalg.lstsq(geometry,physical['X'].ravel(),rcond=None)[0]
    assert max(np.max(abs(geometry@rr-physical['R'].ravel())),np.max(abs(geometry@xx-physical['X'].ravel())))<1e-9
    coefficients={s:(rr[i],xx[i]) for i,s in enumerate(full)}
    reference=read(p['baseline'])['trajectory']['reference_MAE'];penalty=.003*reference
    data={}
    for split in ('train','validation','test'):
        path=Path(meta[split]);N=meta['N'] if split=='train' else None
        noisy=prepare(observed(path,N));clean=prepare(observed(path.with_name(f'{split}_truth.npz'),N))
        linear=core._PreparedScenarios(clean.labels,clean.p,clean.q,
            tuple(pp@physical['R'].T+qq@physical['X'].T for pp,qq in zip(clean.p,clean.q)))
        data[split]=dict(noisy=noisy,clean_ac=clean,linear=linear)
    all_candidates=set(full)
    region_families={}
    for V in regions:
        b=len(V);families=[]
        for h in all_hierarchies(b):
            local=canonical(tuple(V[i] for i in s) for s in h)
            families.extend([local,canonical([*local,V])])
        region_families[V]=families
        all_candidates.update(s for h in families for s in h)
    rows=[];profiles=[];signals=[];checks=0;lp_solves=0
    for regime in ('linear','clean_ac','noisy'):
        train=data['train'][regime];val=data['validation'][regime];test=data['test'][regime]
        bank=FloorBank(train,all_candidates,common['calibration'],seconds=10.)
        try:
            true_point=bank.fit(full,preserve=True)
            pp,qq,yy=train.p[0],train.q[0],train.target[0]
            features=np.column_stack([((a@zz)[:,None]*zz).ravel() for a in (pp,qq) for zz in z])
            noise=float(np.sqrt(np.mean((yy-pp@physical['R'].T-qq@physical['X'].T)**2)))
            for C in small:
                index=full.index(C);signal=features[:,index]*rr[index]+features[:,len(full)+index]*xx[index]
                other=np.delete(features,[index,len(full)+index],axis=1)
                residual=signal-other@np.linalg.lstsq(other,signal,rcond=1e-10)[0]
                contrast=float(np.sqrt(np.mean(residual**2)))
                a,b=coefficients[C];cal=common['calibration'];strength=(cal['a']*a+cal['b']*b)/cal['median_root_depth']
                signals.append(dict(job=p['job'],case=meta['case'],N=meta['N'],voltage_sigma=meta['voltage_sigma'],regime=regime,
                    support=C,size=len(C),contrast_RMS=contrast,residual_RMS=noise,contrast_to_residual=contrast/noise if noise>1e-12 else None,
                    physical_normalized_strength=float(strength),below_016=bool(strength<.16),below_008=bool(strength<.08),below_024=bool(strength<.24)))
            for V,families in region_families.items():
                centered=pp[:,V]-pp[:,V].mean(axis=0)
                singular=np.linalg.svd(centered,compute_uv=False)
                design_rank=int(np.linalg.matrix_rank(centered))
                singular_ratio=float(singular[-1]/singular[0]) if singular[0]>1e-15 else None
                actual=canonical(s for s in true if set(s)<=set(V));outside=canonical(s for s in full if len(s)==1 or not set(s)<=set(V))
                actual_profile=bank.fit([*outside,*actual],preserve=True);candidates={}
                for h in families:
                    point=bank.fit([*outside,*h],preserve=True)
                    effective=canonical(s for s,r,x in zip(point['supports'],point['r'],point['x']) if 1<len(s)<n and
                        set(s)<=set(V) and max(r,x)>1e-9)
                    if effective in candidates:continue
                    blocks=block_errors(val,point)
                    candidates[effective]=dict(point=point,train=point['objective'],validation=float(blocks.mean()),blocks=blocks.tolist(),
                        internal=len(effective),effective=effective,index=len(candidates))
                items=list(candidates.values());best=min(items,key=lambda q:(q['train'],q['internal']))
                wrong=[q for q in items if q['effective']!=actual]
                wrong_loss=min((q['train'] for q in wrong),default=None)
                trained=[q for q in items if q['train']<=best['train']+1e-10]
                choices={'train':min(trained,key=lambda q:(q['internal'],q['train'])),
                    'regularized':min(items,key=lambda q:(q['train']+penalty*q['internal'],q['internal'])),
                    'validation':choose(items,'strict'),'paired':choose(items,'paired')}
                margin=None if wrong_loss is None else wrong_loss-actual_profile['objective']
                tol=1e-10+.001*actual_profile['objective']
                for method,q in choices.items():
                    rows.append(dict(job=p['job'],case=meta['case'],panel=meta['panel'],bundle=meta['bundle'],N=meta['N'],
                        voltage_sigma=meta['voltage_sigma'],condition=meta['condition'],regime=regime,region=V,size=len(V),method=method,
                        **local_score(q['effective'],actual),train_MAE=q['point']['objective'],validation_MAE=q['validation'],test_MAE=loss(test,q['point']),
                        true_profile_MAE=actual_profile['objective'],wrong_minus_true_profile=margin,comparison_tolerance=tol,
                        candidates=len(families),effective_candidates=len(items),
                        P_rank=design_rank,P_singular_ratio=singular_ratio,
                        wrong_beats_true=margin is not None and margin < -tol,
                        near_zero_linear_gap=regime=='linear' and margin is not None and margin<=1e-9))
                profiles.extend(dict(job=p['job'],case=meta['case'],regime=regime,region=V,effective=q['effective'],
                    train=q['train'],validation=q['validation'],internal=q['internal'],true_local=q['effective']==actual) for q in items)
            checks+=sum(r['status']==2 for r in bank.lp.solves);lp_solves+=len(bank.lp.solves)
            assert all(r['status']==2 and max(r['max_constraint_violation'],r['max_bound_violation'])<=1e-7 for r in bank.lp.solves)
        finally:bank.close()
    write(p['result'],dict(job=p['job'],truth_used=True,scope='oracle known exterior structure; exhaustive all local hierarchies including optional region root',
        region_count=len(regions),rows=rows,signals=signals,profiles=profiles,LP_solves=lp_solves,checks=checks,
        information_protocol_sha256=p['information_protocol_sha256'],source_hash=sha(Path(__file__))))


def run(output):
    import pandas as pd
    protocol=read(output/'protocol.json');catalog=read(output/'catalog.json')
    assert (output/'frozen.json').exists() and read(output/'evaluation_completion.json')['returned']==120
    info=output/'information_protocol.json'
    if not info.exists():write(info,dict(jobs=protocol['evaluation'],regimes=['linear','clean_ac','noisy'],
        region_rule='maximal true terminal clades of size 2..5; oracle exterior only for diagnosis',
        parameter_tuning_uses_truth=False,source_hash=sha(Path(__file__)),exact_counterexample=exact_rank_one_example(),
        frozen_sha256=sha(output/'frozen.json')))
    pending=[]
    for job in protocol['evaluation']:
        result=output/'information'/job/'result.json'
        if result.exists() or result.with_name('error.json').exists():continue
        pending.append(dict(job=job,meta=catalog[job],common=str(output/'common'/job/'result.json'),
            baseline=str(output/'evaluation'/job/'baseline20.json'),result=str(result),information_protocol_sha256=sha(info)))
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+str(ROOT/'rnj_wzzt_core');active={};done=0;total=len(pending);then=perf_counter()
    while pending or active:
        while pending and len(active)<8:
            p=pending.pop(0);result=Path(p['result']);payload=result.with_name('payload.json');write(payload,p)
            log=result.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.joint_exit.information','worker',str(payload)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[p['job']]=proc,log,perf_counter(),p
        for job,(proc,log,start,p) in list(active.items()):
            result=Path(p['result'])
            if proc.poll() is None and perf_counter()-start>900:
                proc.kill();proc.wait();write(result.with_name('error.json'),dict(reason='information worker hard timeout'))
            if proc.poll() is not None:
                log.close();done+=1;del active[job]
                if not result.exists() and not result.with_name('error.json').exists():write(result.with_name('error.json'),dict(exit_code=proc.returncode))
                print('information',done,'/',total,job,result.exists(),flush=True)
        write(output/'information_progress.json',dict(planned=len(protocol['evaluation']),returned=sum((output/'information'/j/'result.json').exists() for j in protocol['evaluation']),
            active=len(active),pending=len(pending),seconds=perf_counter()-then))
        if active:sleep(.5)
    rows=[];signals=[];profiles=[];failures=[]
    for job in protocol['evaluation']:
        result=output/'information'/job/'result.json'
        if not result.exists():failures.append(dict(job=job,error=read(result.with_name('error.json'))));continue
        r=read(result);assert r['information_protocol_sha256']==sha(info);rows+=r['rows'];signals+=r['signals'];profiles+=r['profiles']
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'terminal_regions.csv',index=False)
    pd.DataFrame(signals).to_csv(dest/'terminal_signals.csv',index=False)
    pd.DataFrame(profiles).to_csv(dest/'terminal_profiles.csv',index=False)
    frame=pd.DataFrame(rows)
    frame.groupby(['panel','regime','method']).agg(regions=('job','size'),F1=('F1','mean'),exact=('exact','sum'),
        wrong_beats_true=('wrong_beats_true','sum'),near_zero_linear_gap=('near_zero_linear_gap','sum')).to_csv(dest/'terminal_summary.csv')
    frame.groupby(['panel','regime','method','size','N','voltage_sigma']).agg(regions=('job','size'),F1=('F1','mean'),exact=('exact','sum')).to_csv(dest/'terminal_conditions.csv')
    write(output/'information_completion.json',dict(planned=120,returned=120-len(failures),failures=failures,source_hash=sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('path');a=parser.parse_args()
    if a.command=='worker':
        try:worker(Path(a.path))
        except Exception:
            write(Path(read(a.path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()));raise
    else:run(Path(a.path).resolve())
