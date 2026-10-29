"""Bounded development diagnostic for existing fixed-tree AC P/Q/V fitting.

No new dates, DR, true impedances or true supports enter candidate fitting.
Frozen observed-amplitude sigmas approximate the relative-noise generator.
Validation is fixed-impedance latent-input profiling, NOT pure prediction.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research_experiments.wzzt_dr_selection.dr_paper15_update.study import inputs,read,sha,evaluate
from research_experiments.wzzt_dr_selection.dr_paper15_update.candidates_update import write,plain
from research_experiments.wzzt_dr_selection.dr_paper15_update.ac_family_refit import family_network,predict_ac
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize
from rnj_wzzt.estimation.ac_measurement_fit import ACMeasurementSigma,fit_ac_pqv


def core_observed(obs):
    p,q,v,root=obs
    return dict(P_terminal=p,Q_terminal=q,V_terminal=v,root_voltage=root,root_observation='noisy')


def fixed_sigmas(obs,voltage_relative):
    p,q,v,root=obs
    # Exactly positive values required; report zeros rather than add sigma floors.
    if min(float(np.min(np.abs(a.to_numpy()))) for a in obs)<=0:
        raise ValueError('zero observed amplitude: relative sigma unavailable without model change')
    return ACMeasurementSigma(p=.005*p.abs(),q=.005*q.abs(),v=voltage_relative*v.abs(),root_v=.0002*root.abs())


def raw_initial_loss(point,obs,sigma):
    pred=predict_ac(point,obs,base_kv=.4,base_mva=.2)
    return float(np.sum(((pred['voltage'].to_numpy()-obs[2].to_numpy())/sigma.v.to_numpy())**2))


def point_from_net(net,template):
    factor=.5*net.base_kv**2/net.base_mva
    lookup={tuple(row.support_indices):(float(row.r_ohm)/factor,float(row.x_ohm)/factor)
            for row in net.branches.itertuples()}
    supports=template['supports']
    return dict(supports=supports,r=[lookup[tuple(s)][0] for s in supports],x=[lookup[tuple(s)][1] for s in supports])


def _source_hashes(parent,helper):
    out=dict(parent['source_sha256'])
    import rnj_wzzt.estimation.ac_measurement_fit as core
    for p in (Path(__file__),Path(helper),Path(core.__file__)):
        out[str(p.resolve())]=sha(p)
    for p,h in out.items():
        if sha(p)!=h:raise RuntimeError('source closure drift '+p)
    return out


def prepare(parent_root,root,helper,scale_variant='default'):
    parent_root,root=Path(parent_root),Path(root);root.mkdir(parents=True,exist_ok=True)
    parent=read(parent_root/'development_protocol.json');target=root/'protocol.json'
    if scale_variant not in ('default','jac'):raise ValueError('unknown numerical scale variant')
    sources=_source_hashes(parent,helper)
    if target.exists():
        protocol=read(target)
        if protocol['settings'].get('scale_variant','default')!=scale_variant:raise RuntimeError('frozen scale variant changed')
        if protocol['source_sha256']!=sources:raise RuntimeError('frozen EIV sources changed')
        for p,h in protocol['input_sha256'].items():
            if sha(p)!=h:raise RuntimeError('frozen EIV input changed '+p)
        return protocol
    case_ids=['reference__N24','high_voltage_noise__N96'];cases=[];hashes={}
    for cid in case_ids:
        c=next(x for x in parent['cases'] if x['id']==cid)
        files=[Path(c['original_fit']),parent_root/'development'/cid/'ac.json']
        original=read(files[0]);ac=read(files[1]);alternatives=[];seen={}
        for origin,point in [('original_RNJ',original['models']['rnj_bank']),('original_WZZT',original['models']['multistart_original_rule']),
                             ('AC_parent_RNJ',ac['models']['rnj_refit']),('AC_parent_WZZT',ac['models']['wzzt_refit'])]:
            point={key:point[key] for key in ('supports','r','x')}
            identity=hashlib.sha256(json.dumps(point,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if identity in seen:alternatives[seen[identity]]['origins'].append(origin)
            else:seen[identity]=len(alternatives);alternatives.append(dict(index=len(alternatives),identity=identity,origins=[origin],point=point))
        if len(alternatives)>4:raise RuntimeError('candidate cap')
        for f in files:hashes[str(f)]=sha(f)
        train,val=inputs(c);tobs=normalize(train,'training')[0];vobs=normalize(val,'validation')[0]
        cases.append(dict(id=cid,parent_case=c,training_rows=len(tobs[0]),validation_rows=len(vobs[0]),
                          voltage_sigma_relative=.002 if cid.startswith('high') else .0002,candidates=alternatives))
    for p,h in parent['input_sha256'].items():hashes[p]=h
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),phase='fixed-tree AC EIV diagnostic on already-seen development cases',
        cases=cases,source_sha256=sources,input_sha256=hashes,parent_protocol_sha256=sha(parent_root/'development_protocol.json'),
        settings=dict(max_nfev=20,tol=1e-5,worker_timeout_seconds=600,zero_lower_impedance=0.,scale_variant=scale_variant,
            upper_impedance='candidate-specific safe loss-sublevel bound from observations and feasible incumbent; no fallback',
            sigma='frozen .005|Pobs|/.005|Qobs|, voltage .0002/.002|Vobs|, root .0002|rootobs|',
            validation='fixed-impedance latent P/Q/root profile; also direct observed-input AC prediction',
            output_contraction='none; retain zero/weak weights; evaluate effective at original 1e-9 threshold',
            price_design=False,new_future_data=False,claims='approximate EIV local diagnostic, not exact MLE or topology guarantee'),
        stop='exactly two fixed inputs, four unique initial points per input at most; retain all failure/timeout/unavailable',
        acceptance='report direct prediction, true-kernel error and full structural errors independently; no promotion on fitted loss alone')
    write(target,protocol);return protocol


def worker(protocol_path,case_index,candidate_index,folder,helper):
    folder=Path(folder);protocol=read(protocol_path);case=protocol['cases'][case_index];cand=case['candidates'][candidate_index]
    result=dict(case=case['id'],candidate=cand['index'],origins=cand['origins'],status='running')
    try:
        for path,h in protocol['source_sha256'].items():
            if sha(path)!=h:raise RuntimeError('worker frozen source drift '+path)
        import importlib.util
        spec=importlib.util.spec_from_file_location('eiv_levelset',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        train,val=inputs(case['parent_case']);tobs=normalize(train,'training')[0];vobs=normalize(val,'validation')[0]
        sigma=fixed_sigmas(tobs,case['voltage_sigma_relative']);vsigma=fixed_sigmas(vobs,case['voltage_sigma_relative'])
        point=cand['point'];net=family_network(point,list(tobs[0].columns),base_kv=.4,base_mva=.2)
        incumbent=raw_initial_loss(point,tobs,sigma)
        # The bound helper receives observed-only arrays and candidate downstream sets.
        bound=module.derive_eiv_levelset_bounds(net,core_observed(tobs),sigma,bound_kind='triangle')
        if abs(bound.get('incumbent_loss_L',incumbent)-incumbent)>1e-8*max(1.,incumbent):
            raise RuntimeError('independent incumbent loss mismatch')
        result.update(initial_standardized_loss=incumbent,bound=bound)
        write(folder/'preflight.json',result)
        if bound['status']!='certified' or not bound.get('api_usable',False):
            result['status']='bound_unavailable';write(folder/'result.json',result);return
        upper=float(bound['global_upper_ohm'])
        max_initial=float(net.branches[['r_ohm','x_ohm']].to_numpy().max())
        if max_initial>upper*(1+1e-9):raise RuntimeError('derived bound excludes incumbent')
        import rnj_wzzt.estimation.ac_measurement_fit as core
        forward=core.solve_ac_power_flow_timeseries;original_optimizer=core.least_squares
        calls=0;phase='training';scale_receipts=[]
        def scaled_optimizer(*args,**kwargs):
            supplied=kwargs['x_scale'];variant=protocol['settings'].get('scale_variant','default')
            scale_receipts.append(dict(phase=phase,variant=variant,original_scale_min=float(np.min(supplied)),
                                       original_scale_max=float(np.max(supplied)),variable_count=int(np.size(supplied))))
            if variant=='jac':kwargs['x_scale']='jac'
            return original_optimizer(*args,**kwargs)
        def counted(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls%20==0:write(folder/'progress.json',dict(pid=os.getpid(),AC_calls=calls,phase=phase,updated_at=datetime.now(timezone.utc).isoformat()))
            return forward(*args,**kwargs)
        core.solve_ac_power_flow_timeseries=counted
        core.least_squares=scaled_optimizer
        started=time.perf_counter();settings=protocol['settings']
        fit=fit_ac_pqv(net,core_observed(tobs),sigma=sigma,fit_impedances=True,r_bounds_ohm=(0.,upper),x_bounds_ohm=(0.,upper),
                       max_nfev=settings['max_nfev'],tol=settings['tol'])
        fitted=point_from_net(fit.fitted_net,point)
        private=folder/'latent_training.pkl'
        private.write_bytes(pickle.dumps(dict(fitted=fit.fitted,standardized_residuals=fit.standardized_residuals),protocol=pickle.HIGHEST_PROTOCOL))
        fitdata=dict(success=fit.success,status=fit.status,message=fit.message,loss=fit.loss,initial_loss=fit.initial_loss,
            nfev=fit.nfev,njev=fit.njev,optimality=fit.optimality,diagnostics=fit.diagnostics,point=fitted,
            elapsed_seconds=time.perf_counter()-started,numerical_scaling=list(scale_receipts),
            private_latent_sha256=sha(private),minimum_latent_P=float(fit.fitted['P_terminal'].to_numpy().min()),
            minimum_latent_Q=float(fit.fitted['Q_terminal'].to_numpy().min()))
        # Do not retain household/latent trajectories in public evidence.
        write(folder/'training_fit.json',fitdata);result['training']=fitdata
        if not fit.success:
            result['status']='training_unresolved';result['AC_calls']=calls;write(folder/'result.json',result);return
        phase='validation_profile';started=time.perf_counter()
        vf=fit_ac_pqv(fit.fitted_net,core_observed(vobs),sigma=vsigma,fit_impedances=False,
                      max_nfev=settings['max_nfev'],tol=settings['tol'])
        vprivate=folder/'latent_validation.pkl'
        vprivate.write_bytes(pickle.dumps(dict(fitted=vf.fitted,standardized_residuals=vf.standardized_residuals),protocol=pickle.HIGHEST_PROTOCOL))
        result['validation_profile']=dict(success=vf.success,status=vf.status,message=vf.message,loss=vf.loss,
            nfev=vf.nfev,njev=vf.njev,diagnostics=vf.diagnostics,elapsed_seconds=time.perf_counter()-started,
            numerical_scaling=list(scale_receipts),private_latent_sha256=sha(vprivate))
        # Physical truth loaded only after parameter fitting and profile diagnostics.
        result['evaluation']=evaluate(dict(initial=point,EIV=fitted),list(tobs[0].columns),train,val)
        result.update(status='complete' if vf.success else 'validation_unresolved',AC_calls=calls)
    except Exception as exc:result.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    write(folder/'result.json',result)


def run(parent_root,root,helper,max_workers,scale_variant='default'):
    root=Path(root);p=prepare(parent_root,root,helper,scale_variant);lock=root/'run.lock'
    if lock.exists():raise RuntimeError('existing EIV diagnostic lock; inspect process')
    write(lock,dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()));started=0
    try:
        for ci,c in enumerate(p['cases']):
            for ki,k in enumerate(c['candidates']):
                folder=root/c['id']/str(ki);folder.mkdir(parents=True,exist_ok=True)
                if (folder/'result.json').exists():continue
                if started>=max_workers:continue
                if (folder/'worker_started.json').exists():raise RuntimeError('partial worker retained; no implicit restart')
                started+=1;write(folder/'worker_started.json',dict(started_at=datetime.now(timezone.utc).isoformat()))
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--parent',str(parent_root),
                         '--helper',str(helper),'--worker',str(ci),str(ki),'--folder',str(folder)]
                began=time.perf_counter()
                with (folder/'worker.log').open('w',encoding='utf-8') as log:
                    try:
                        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=p['settings']['worker_timeout_seconds'],
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        write(folder/'worker_exit.json',dict(exit_code=completed.returncode,elapsed_seconds=time.perf_counter()-began))
                        if not (folder/'result.json').exists():
                            write(folder/'result.json',dict(case=c['id'],candidate=ki,status='worker_failed',exit_code=completed.returncode))
                    except subprocess.TimeoutExpired:
                        progress=read(folder/'progress.json') if (folder/'progress.json').exists() else None
                        write(folder/'result.json',dict(case=c['id'],candidate=ki,status='timeout',progress=progress,
                                                      elapsed_seconds=time.perf_counter()-began))
                print(json.dumps(dict(case=c['id'],candidate=ki,status=read(folder/'result.json')['status'])),flush=True)
    finally:lock.unlink(missing_ok=True)
    records=[]
    for c in p['cases']:
        for k in c['candidates']:
            f=root/c['id']/str(k['index'])/'result.json'
            if f.exists():records.append(read(f))
    planned=sum(len(c['candidates']) for c in p['cases'])
    write(root/'summary.json',dict(planned=planned,finished=len(records),status='complete' if len(records)==planned else 'partial',records=records))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--parent',type=Path,required=True)
    a.add_argument('--helper',type=Path,required=True);a.add_argument('--prepare-only',action='store_true');a.add_argument('--max-workers',type=int,default=1)
    a.add_argument('--scale-variant',choices=['default','jac'],default='default')
    a.add_argument('--worker',type=int,nargs=2);a.add_argument('--folder',type=Path);args=a.parse_args()
    if args.worker is not None:worker(args.root/'protocol.json',*args.worker,args.folder,args.helper)
    elif args.prepare_only:prepare(args.parent,args.root,args.helper,args.scale_variant)
    else:run(args.parent,args.root,args.helper,args.max_workers,args.scale_variant)
