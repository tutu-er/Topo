"""Frozen paired 3/7-day rooted local inference and noisy holdout tests."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import zipfile
import numpy as np
import scipy
from .. import likelihood as lk
from .model import prepare_case,line_map,project,public_domain
from .response import fit_qp,fit_profile,score_fixed
from .ac_fit import fit

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(obj,default=convert,ensure_ascii=False,allow_nan=False)
    if path.suffix=='.gz':
        with gzip.open(path,'wt',encoding='utf-8') as f:f.write(body)
    else:path.write_text(body+'\n',encoding='utf-8')


def load(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def rms(z):return float(np.sqrt(np.mean(np.asarray(z)**2)))


def relative(a,b):return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-15))


def diagnostics(candidate,training,holdout,truth,trained,held):
    r,x=np.asarray(trained['r']),np.asarray(trained['x']);n=training['history_rows'];focus=truth['focus_positions']
    output={}
    for label,sl in (('train',slice(0,n)),('holdout',slice(672,768))):
        response=lk.ac_predict(candidate,r,x,truth['p'][sl],truth['q'][sl],root_v=truth['root_v'][sl])['v']
        delta=response-truth['v'][sl]
        output[label+'_clean_V_RMSE_pu']=rms(delta)
        output[label+'_clean_focus_V_RMSE_pu']=rms(delta[:,focus])
        contrast=delta[:,focus]-delta[:,focus].mean(axis=1,keepdims=True)
        output[label+'_clean_focus_contrast_RMSE_pu']=rms(contrast)
    if held is not None and 'v' in held:
        output['holdout_latent_V_RMSE_pu']=rms(np.asarray(held['v'])-truth['v'][672:768])
    if candidate.name!='base':return output
    rt,xt=np.asarray(truth['r']),np.asarray(truth['x']);physical=np.hypot(rt,xt)>0
    leaf=np.array([i for i,s in enumerate(candidate.supports) if len(s)==1 and s[0] in focus])
    groups=dict(all=np.flatnonzero(physical),path=np.asarray(truth['path_edges'],int),leaf=leaf)
    bp=truth['p'][:n]@candidate.M;bq=truth['q'][:n]@candidate.M
    true_combo=bp*rt+bq*xt;estimated_combo=bp*r+bq*x
    for label,ix in groups.items():
        output[label+'_R_relative']=relative(r[ix],rt[ix]);output[label+'_X_relative']=relative(x[ix],xt[ix])
        output[label+'_RPXQ_relative']=relative(estimated_combo[:,ix],true_combo[:,ix])
    if 'p' in trained:
        estimated_state=np.asarray(trained['p'])@candidate.M*r+np.asarray(trained['q'])@candidate.M*x
        output['leaf_RPXQ_with_latent_relative']=relative(estimated_state[:,leaf],true_combo[:,leaf])
    pobs=training['observed']['p']@candidate.M;qobs=training['observed']['q']@candidate.M
    cosine=np.sum(pobs*qobs,axis=0)/np.maximum(np.linalg.norm(pobs,axis=0)*np.linalg.norm(qobs,axis=0),1e-20)
    output['leaf_PQ_cosine_mean']=float(cosine[leaf].mean())
    output['leaf_XR_true']=xt[leaf]/rt[leaf];output['leaf_XR_fitted']=x[leaf]/np.maximum(r[leaf],1e-20)
    if training['context']=='CUT':
        truep=lk.ac_predict(candidate,rt,xt,truth['p'],truth['q'],root_v=truth['root_v'])
        rr=r.copy();xx=x.copy();rr[0]=0;xx[0]=0
        localized=lk.ac_predict(candidate,rr,xx,truth['p'][672:768],truth['q'][672:768],root_v=abs(truep['voltage'][672:768,0]))['v']
        output['holdout_local_gauge_aligned_V_RMSE_pu']=rms(localized-truth['v'][672:768])
    return output


def run(spec,folder,pilot=False):
    started=perf_counter();folder=Path(folder);key=f"{spec['name']}_{spec['seed']}_{spec['N']}_{spec['context']}"
    train,hold,truth=prepare_case(spec['name'],spec['seed'],spec['N'],spec['context'])
    observed_path=folder/'inputs'/f'{key}.npz';observed_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(observed_path,**{k:v for k,v in train['observed'].items()},
        root=np.asarray(train['root_meter'][0]) if train['root_meter'] is not None else np.empty(0),
        sigma_p=train['sigma']['p'],sigma_q=train['sigma']['q'],sigma_v=train['sigma']['v'])
    main=spec['context'] in ('FULL','CUT');bank=train['candidates'] if main and not pilot else train['candidates'][:1]
    fits={};initializers={}
    for j,candidate in enumerate(bank):
        qp=fit_qp(candidate,train);initializers[candidate.name]=qp
        if main:fits[candidate.name+'/QP']=qp
        ac=fit(candidate,train,qp['theta'],seed=spec['seed']+j*100,multistart=4,max_iter=200)
        fits[candidate.name+'/AC']=ac
        if main:fits[candidate.name+'/RESP']=fit_profile(candidate,train,qp['theta'],seed=spec['seed']+j*100,multistart=4)
    inference=folder/'inference'/f'{key}.json.gz'
    save(inference,dict(spec=spec,fits=fits,initializers=initializers,input_sha256=sha(observed_path),truth_used=False))
    seal=sha(inference);held_fits={};rows=[];cross={};common_ac={}
    for label,trained in fits.items():
        cname,method=label.split('/');candidate=next(c for c in bank if c.name==cname)
        row=dict(spec,candidate=cname,method=method,train_qualified=trained.get('qualified',False),
                 train_J=trained.get('J'),seconds=trained.get('seconds'),ac_calls=trained.get('ac_calls',0),
                 input_sha256=sha(observed_path),inference_sha256=seal)
        try:
            if 'r' not in trained:raise RuntimeError('no attained training lines')
            if method=='AC':
                held=fit(candidate,hold,np.asarray(trained['theta']),fixed_lines=(trained['r'],trained['x']),max_iter=120)
                cross[label]=score_fixed(candidate,hold,trained['r'],trained['x'])
            else:
                held=score_fixed(candidate,hold,trained['r'],trained['x'],covariance=trained.get('covariance'))
            backsub=held if method=='AC' else fit(candidate,hold,np.asarray(trained['theta']),fixed_lines=(trained['r'],trained['x']),max_iter=120)
            common_ac[label]=backsub
            held_fits[label]=held
            row.update(holdout_J=held['J'],holdout_qualified=held.get('qualified',True),
                       holdout_ac_calls=held.get('ac_calls',0),status='completed',
                       projected_gradient=trained.get('projected_gradient'),qp_gap=trained.get('absolute_gap'),
                       common_AC_holdout_J=backsub['J'],common_AC_holdout_qualified=backsub['qualified'],
                       common_AC_holdout_calls=0 if method=='AC' else backsub['ac_calls'],
                       information_rank=trained.get('information',{}).get('rank') if trained.get('information') else None,
                       information_width=trained.get('information',{}).get('width') if trained.get('information') else None)
            row.update(diagnostics(candidate,train,hold,truth,trained,held if method=='AC' else None))
        except Exception as error:
            row.update(status='evaluation_failed',error=repr(error),traceback=traceback.format_exc())
        rows.append(row)
    result=dict(spec=spec,rows=rows,held_fits=held_fits,ac_lines_response_holdout=cross,common_ac_holdout=common_ac,
        inference_sha256=seal,inference_unchanged=sha(inference)==seal,
        truth_evaluated_after_training_seal=True,seconds=perf_counter()-started)
    save(folder/'records'/f'{key}.json.gz',result)
    return dict(spec=spec,rows=rows,seconds=result['seconds'],inference_sha256=seal)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();folder=Path(args.output)
    if folder.exists():raise FileExistsError('fresh output folder required')
    folder.mkdir(parents=True)
    specs=[dict(name=name,seed=seed,N=n,context=context) for name in ('paper15','paper11')
        for seed in ([35991] if args.pilot else range(36001,36005))
        for n in ([288] if args.pilot else [288,672]) for context in ('FULL','CUT','OLD-RATIO','FOCUS-V')]
    sources={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT):sources[str(path.relative_to(ROOT))]=sha(path)
    for relative in ('research_experiments/mobile_ev_mle/physical_local_v1/catalog.json',
        'rnj_wzzt_core/rnj_wzzt/data/paper_style_terminal_lv.py','rnj_wzzt_core/rnj_wzzt/data/paper_style_case_bank.py',
        'research_experiments/mobile_ev_mle/rooted_local_fit_v1/protocol.md','research_experiments/mobile_ev_mle/rooted_local_fit_v1/math.md'):
        sources[relative]=sha(ROOT/relative)
    import gurobipy as gp
    config=dict(specifications=specs,source_sha256=sources,pilot=args.pilot,workers=args.workers,
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version(),platform=platform.platform(),
        multistart=4,max_iter=200,gtol=1e-5,holdout=[672,768],domain=public_domain(),alpha=None,
        methods=['AC','RESP','QP'],truth_in_estimator=False,full_and_cut_same_meter_budget=False)
    save(folder/'configuration.json',config)
    with zipfile.ZipFile(folder/'scientific_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for relative in sources:z.writestr(relative.replace('\\','/'),(ROOT/relative).read_bytes())
    results=[];started=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,spec,str(folder),args.pilot):spec for spec in specs}
        for future in as_completed(futures):
            spec=futures[future]
            try:result=future.result()
            except Exception as error:result=dict(spec=spec,status='worker_failed',error=repr(error),traceback=traceback.format_exc(),rows=[])
            results.append(result);save(folder/'progress.json',dict(done=len(results),total=len(specs),last_spec=spec,elapsed=perf_counter()-started))
            print(len(results),'/',len(specs),spec,result.get('status','returned'),round(result.get('seconds',0),2),flush=True)
    save(folder/'results.json',dict(results=results,seconds=perf_counter()-started,configuration_sha256=sha(folder/'configuration.json')))
    save(folder/'completed.json',dict(conditions=len(results),seconds=perf_counter()-started,worker_failures=sum(z.get('status')=='worker_failed' for z in results),results_sha256=sha(folder/'results.json')))


if __name__=='__main__':main()
