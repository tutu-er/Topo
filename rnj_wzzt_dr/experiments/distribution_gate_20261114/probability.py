"""Independent declared-model probabilities from frozen native squared errors.

No identifier, production evidence evaluator, optimizer or AC solver is used.
The Gaussian calculation uses the complete joint (power, voltage-error) form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from statistics import NormalDist
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import beta, binom

ROOT=Path(__file__).resolve().parents[3]
POLICIES=('no_dfs','anchor','normal_fixed','mgf_fixed','mgf_forecast','mgf_forecast_rotation')
MODES=('normal_fixed','mgf_fixed','mgf_forecast')
LAWS=('fixed','forecast')


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(item):
    path=Path(item['path']);actual=ref(path)
    if actual['sha256']!=item['sha256']:raise ValueError('Changed source: '+str(path))
    return path


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def target_set(history,sigma2,mass=.99):
    logw=-(np.asarray(history)-min(history))/(2*sigma2)
    weights=np.exp(logw);weights/=weights.sum();order=np.argsort(-weights,kind='stable')
    stop=min(int(np.searchsorted(np.cumsum(weights[order]),mass)),len(order)-1)
    threshold=logw[order[stop]]
    return [int(i) for i in order if logw[i]>=threshold-1e-12],weights


def build_model(score,validation,native=None):
    """Recover K and the two historical voltage-noise variances from inputs."""
    labels=list(score['labels']);n=len(labels);H=np.asarray([x['H'] for x in score['native_bank']],float)
    history=np.asarray([x['history_sse'] for x in score['native_bank']],float)
    representatives=[int(x['index']) for x in score['models']]
    if native is not None:
        reconstructed=[]
        for item in native['frozen_candidates']:
            point=item['point'];mat=np.zeros((n,2*n))
            for support,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
                ix=np.ix_(support,support);mat[:,:n][ix]+=r;mat[:,n:][ix]+=x
            reconstructed.append(mat)
        if not np.allclose(H,reconstructed,atol=1e-14,rtol=1e-14):raise ValueError('Native support reconstruction differs')
    power=np.c_[validation['P'],validation['Q']];y=validation['Y'];sigma2=float(score['sigma2'])
    new_history=np.asarray([np.square(y-power@h.T).sum() for h in H])
    if not np.allclose(history,new_history,atol=1e-13,rtol=1e-12):raise ValueError('Frozen native history SSE differs')
    repH=H[representatives]
    predictions=np.einsum('tc,knc->ktn',power,repH)
    residual=y[None]-predictions
    factor=1+(predictions*residual).sum(axis=1)/(np.square(predictions).sum(axis=1)+len(power)*sigma2)
    means=factor[:,:,None]*repH
    root_sd=2*.0005*float(np.median(validation['root']))**2
    independent_sd=2*.0005*float(np.median(validation['V']))**2
    covariance=root_sd**2*np.ones((n,n))+independent_sd**2*np.eye(n)
    targets,weights=target_set(history[representatives],sigma2)
    families=[x['topology_id'] for x in score['native_bank']]
    return dict(H=H,history=history,representatives=representatives,means=means,amplitude=factor,
        targets=targets,weights=weights,families=families,labels=labels,n=n,sigma2=sigma2,
        root_sd=root_sd,independent_sd=independent_sd,covariance=covariance,
        internal=np.asarray([x['internal'] for x in score['native_bank']]),score=score,
        validation_cells=y.size)


def shift(config,u,labels,base_kw):
    u=np.asarray(u,float);n=len(labels);answer=np.zeros((u.shape[0]*4,2*n));lookup={v:k for k,v in enumerate(labels)}
    if u.shape[0]!=24 or u.shape[1]!=len(config['resources']):raise ValueError('Action dimensions differ')
    for col,row in enumerate(config['resources']):
        node=lookup[row['node_label']];a=np.repeat(u[:,col],4)/base_kw
        if row['kind']=='reactive':answer[:,n+node]+=a
        else:
            answer[:,node]+=a
            answer[:,n+node]+=a*float(row.get('q_over_p',0.))
    return answer


class PairIntegral:
    """L=h+z'A z, z=(power,error), evaluated through joint whitening."""
    def __init__(self,Hi,Hj,K,sigma2,covariance,power_sd,h):
        self.h=float(h);self.D=Hi-Hj;self.K=K;self.Hi=Hi;self.Hj=Hj
        self.V=self.D.T@covariance@self.D/sigma2**2
        self.sigma2=sigma2;T,p=power_sd.shape;n=Hi.shape[0];self.T=T
        E=K-Hi;Q=(.5*self.D.T@self.D+.5*(E.T@self.D+self.D.T@E))/sigma2
        A=np.zeros((p+n,p+n));A[:p,:p]=Q;A[:p,p:]=self.D.T/(2*sigma2);A[p:,:p]=self.D/(2*sigma2)
        L=np.zeros((T,p+n,p+n));L[:,:p,:p]=np.eye(p)[None]*power_sd[:,None,:]
        L[:,p:,p:]=np.linalg.cholesky(covariance)[None]
        B=np.einsum('tji,jk,tkl->til',L,A,L,optimize=True)
        self.Q=Q;self.B=(B+B.transpose(0,2,1))/2
        self.G=np.einsum('tji,jk->tik',L,A[:,:p],optimize=True)
        self.eigenvalues,self.eigenvectors=np.linalg.eigh(self.B)
        self.trace=float(np.trace(self.B,axis1=1,axis2=2).sum())
        self.constant_variance=float(2*np.square(self.B).sum())
        self.projected_G=np.einsum('tji,tjk->tik',self.eigenvectors,self.G,optimize=True)
        self.selected_lambda=.5;self.halvings=0
        while np.min(1+2*self.selected_lambda*self.eigenvalues)<=0:
            self.selected_lambda*=.5;self.halvings+=1
            if self.halvings>79:raise ArithmeticError('No positive integral domain')

    def moments(self,power):
        square=float(np.einsum('ti,ij,tj->',power,self.Q,power,optimize=True))
        a=np.einsum('tij,tj->ti',self.G,power,optimize=True)
        return self.h+square+self.trace,self.constant_variance+4*float(np.square(a).sum())

    def log_mgf(self,power,lam=None):
        lam=self.selected_lambda if lam is None else float(lam)
        eigen=1+2*lam*self.eigenvalues
        if np.min(eigen)<=0:return None
        a=np.einsum('tij,tj->ti',self.projected_G,power,optimize=True)
        square=float(np.einsum('ti,ij,tj->',power,self.Q,power,optimize=True))
        return float(-lam*(self.h+square)-.5*np.log(eigen).sum()+2*lam*lam*np.sum(a*a/eigen))

    def receipt(self,power,tie,alpha,mode):
        mu,var=self.moments(power);lam=self.selected_lambda
        logbound=lam*tie+self.log_mgf(power)
        if mode=='normal_fixed':
            failure=NormalDist().cdf((tie-mu)/np.sqrt(var)) if var>0 else float(mu<=tie)
            margin=mu-NormalDist().inv_cdf(1-alpha)*np.sqrt(max(0,var))-tie
        else:
            failure=min(1.,float(np.exp(min(logbound,0.))))
            margin=np.log(alpha)-logbound
        return dict(mean=mu,variance=var,lambda_selected=lam,lambda_halvings=self.halvings,
            fixed_input_mean=float(self.h+np.einsum('ti,ij,tj->',power,self.Q,power,optimize=True)),
            fixed_input_variance=float(np.einsum('ti,ij,tj->',power,self.V,power,optimize=True)),
            minimum_integral_eigenvalue=float(np.min(1+2*lam*self.eigenvalues)),
            log_failure_bound=logbound,mgf_failure_bound=min(1.,float(np.exp(min(logbound,0.)))),
            pair_failure=failure,margin=float(margin),alpha=alpha,mode=mode)


def exact_gates(model,forecast,power,mode):
    """Independent all-native foreign-family constraints (no old J gate)."""
    T,n=power.shape[0],model['n'];score=model['score'];sigma2=model['sigma2']
    tie=max(1e-7,(score['cells']+T*n)*score['reference_mse']*score['validation_tie']/(2*sigma2))
    sd=.05*abs(forecast) if mode=='mgf_forecast' else np.zeros_like(forecast)
    cache=model.setdefault('_integral_cache',{})
    law_key=('forecast' if mode=='mgf_forecast' else 'fixed',hashlib.sha256(np.ascontiguousarray(forecast).tobytes()).hexdigest())
    rows=[];totals={str(t):0. for t in model['targets']}
    for target in model['targets']:
        i=model['representatives'][target];competitors=[j for j,f in enumerate(model['families']) if f!=model['families'][i]]
        alpha=.1/len(competitors)
        for j in competitors:
            h=(model['history'][j]-model['history'][i])/(2*sigma2)
            key=(*law_key,target,j)
            if key not in cache:cache[key]=PairIntegral(model['H'][i],model['H'][j],model['means'][target],sigma2,model['covariance'],sd,h)
            integral=cache[key]
            row=integral.receipt(power,tie,alpha,mode);row.update(target=target,target_native=i,competitor_native=j,h=h)
            rows.append(row);totals[str(target)]+=row['pair_failure']
    guards=[];selected=set(model['targets']);representatives=model['representatives']
    for left in range(len(representatives)):
        for right in range(left+1,len(representatives)):
            if left not in selected and right not in selected:continue
            for i,j in ((left,right),(right,left)):
                Hi,Hj=model['H'][representatives[i]],model['H'][representatives[j]];K=model['means'][i]
                def contrast(x):
                    y=x@K.T
                    return (np.square(y-x@Hj.T).sum()-np.square(y-x@Hi.T).sum())/(2*sigma2)
                guards.append(dict(target=i,competitor=j,gain=float(contrast(power)-contrast(forecast))))
    feasible=all(r['margin']>=-1e-7 and r['pair_failure']<=r['alpha']+1e-9 for r in rows) and all(g['gain']>=-1e-7 for g in guards)
    return dict(rows=rows,target_failure_sums=totals,worst_recovery_lower=max(0.,1-max(totals.values())),tie=tie,
        guards=guards,feasible=feasible,minimum_guard=min((g['gain'] for g in guards),default=None),
        minimum_margin=min(r['margin'] for r in rows),probability_scope='Conditional on historical K mean, stated Gaussian power and shared-root/independent-node voltage residual law; frozen five-native SSE. No actual-AC claim.')


def interval(success,n,alpha=.05):
    return [0. if success==0 else float(beta.ppf(alpha/2,success,n-success+1)),
        1. if success==n else float(beta.ppf(1-alpha/2,success+1,n-success))]


def select_native(model,total,cells):
    scaled=total/(model['validation_cells']+cells)/model['score']['reference_mse']
    eligible=scaled<=scaled.min(axis=1,keepdims=True)+model['score']['validation_tie']
    # Deterministic original tie: fewest internal edges, scaled loss, native index.
    order=np.lexsort((np.broadcast_to(np.arange(total.shape[1]),total.shape),scaled,
        np.broadcast_to(model['internal'],total.shape)),axis=1)
    pos=np.take_along_axis(eligible,order,axis=1).argmax(axis=1)
    return np.take_along_axis(order,pos[:,None],axis=1)[:,0]


def forecast_probability(folder,entry,protocol,out,draws,seed,batch_size):
    seal=read(folder/'seal.json')
    for key in ('planner','plans','score','protocol'):checked(seal[key])
    planner=arrays(checked(seal['planner']));plans=read(checked(seal['plans']))
    if set(planner)!={'forecast',*POLICIES}:raise ValueError('Unexpected policies')
    score=read(checked(entry['score']));val=arrays(checked(entry['files']['validation.npz']))
    native=read(checked(entry['files']['101_native.json']));model=build_model(score,val,native)
    config=read(folder/'resource_case.json');base=float(config.get('base_kw',200.))
    forecast=planner['forecast'];power={p:forecast+shift(config,planner[p],model['labels'],base) for p in POLICIES}
    draw=int(folder.name.removeprefix('forecast'));T,n=forecast.shape[0],model['n'];N=len(model['H'])
    total=np.empty((len(LAWS),len(model['targets']),len(POLICIES),draws,N))
    for start in range(0,draws,batch_size):
        count=min(batch_size,draws-start);block=start//batch_size
        def rng(part):return np.random.default_rng(np.random.SeedSequence([seed,int(entry['index']),draw,block,part]))
        xi=rng(0).normal(size=(count,T,2*n))*.05*abs(forecast)[None]
        e=rng(1).normal(size=(count,T,n))*model['independent_sd']+rng(2).normal(size=(count,T,1))*model['root_sd']
        for law_index,law in enumerate(LAWS):
            error=xi if law=='forecast' else 0.
            for policy_index,policy in enumerate(POLICIES):
                X=power[policy][None]+error
                if X.shape[0]==1:X=np.broadcast_to(X,(count,T,2*n))
                predictions=np.einsum('btc,knc->kbtn',X,model['H'],optimize=True)
                for target_index,target in enumerate(model['targets']):
                    Y=np.einsum('btc,nc->btn',X,model['means'][target],optimize=True)+e
                    losses=np.square(predictions-Y[None]).sum(axis=(2,3)).T
                    total[law_index,target_index,policy_index,start:start+count]=losses+model['history']
    saved=out/f'forecast{draw:02d}_samples.npz'
    np.savez_compressed(saved,total_sse=total,targets=np.asarray(model['targets']),representative_native_indices=np.asarray(model['representatives']),
        policies=np.asarray(POLICIES),laws=np.asarray(LAWS))
    rows=[];sample_events={};theory_cache={}
    tie=max(1e-7,(score['cells']+T*n)*score['reference_mse']*score['validation_tie']/(2*model['sigma2']))
    for law_index,law in enumerate(LAWS):
        for target_index,target in enumerate(model['targets']):
            i=model['representatives'][target];foreign=[j for j,f in enumerate(model['families']) if f!=model['families'][i]]
            for policy_index,policy in enumerate(POLICIES):
                allS=total[law_index,target_index,policy_index]
                winner=select_native(model,allS,T*n)
                family=np.asarray([model['families'][j]==model['families'][i] for j in winner])
                margins=(allS[:,foreign]-allS[:,i,None])/(2*model['sigma2'])
                representative=np.all(margins>=tie,axis=1)
                # A fixed-MGF policy retains its actual conservative gate bound;
                # fixed-P normal tails are an additional exact-model reference.
                theory_mode=('mgf_fixed' if policy=='mgf_fixed' else 'normal_fixed') if law=='fixed' else 'mgf_forecast'
                cachekey=(policy,theory_mode)
                if cachekey not in theory_cache:theory_cache[cachekey]=exact_gates(model,forecast,power[policy],theory_mode)
                gates=theory_cache[cachekey];pairs=[r for r in gates['rows'] if r['target']==target]
                union=min(1.,sum(r['pair_failure'] for r in pairs));failures=int((~representative).sum())
                observed_mu=margins.mean(axis=0);observed_var=margins.var(axis=0,ddof=1)
                pairrows=[]
                for k,pair in enumerate(pairs):
                    count=int((margins[:,k]<tie).sum());pred=pair['pair_failure']
                    pairrows.append(dict(pair,observed_failure_count=count,observed_failure_rate=count/draws,
                        observed_failure_95_CI=interval(count,draws),observed_logodds_mean=float(observed_mu[k]),observed_logodds_variance=float(observed_var[k]),
                        mean_error_in_standard_errors=None if pair['variance']<=0 else float((observed_mu[k]-pair['mean'])/np.sqrt(pair['variance']/draws)),
                        one_sided_bound_exceedance_tail=float(binom.sf(count-1,draws,pred))))
                selected_mode=policy if policy in MODES else 'mgf_forecast' if policy.endswith('rotation') else None
                q=plans.get(selected_mode,{})
                row=dict(forecast=draw,law=law,target_representative=target,target_native=i,target_topology_id=model['families'][i],policy=policy,
                    n=draws,target_family_winner_count=int(family.sum()),target_family_winner_rate=float(family.mean()),target_family_winner_95_CI=interval(int(family.sum()),draws),
                    representative_beats_all_foreign_count=int(representative.sum()),representative_success_rate=float(representative.mean()),representative_success_95_CI=interval(int(representative.sum()),draws),
                    comparison_count=len(foreign),theory_mode=theory_mode,theoretical_union_failure_upper=union,theoretical_representative_success_lower=1-union,
                    one_sided_union_bound_exceedance_tail=float(binom.sf(failures-1,draws,union)),pairs=pairrows,
                    new_gate_feasible=q.get('new_gate_feasible',q.get('new_constraints_feasible')),anchor_fallback=q.get('anchor_fallback',False),
                    declared_law_matched=(selected_mode=='mgf_forecast')==(law=='forecast') if selected_mode else None)
                # Also retain MGF's conservative fixed-input bound instead of equating it with the normal tail.
                if law=='fixed':
                    cachekey=(policy,'mgf_fixed')
                    if cachekey not in theory_cache:theory_cache[cachekey]=exact_gates(model,forecast,power[policy],'mgf_fixed')
                    row['fixed_MGF_union_failure_upper']=min(1.,theory_cache[cachekey]['target_failure_sums'][str(target)])
                    normal_key=(policy,'normal_fixed')
                    if normal_key not in theory_cache:theory_cache[normal_key]=exact_gates(model,forecast,power[policy],'normal_fixed')
                    row['fixed_exact_normal_union_failure_upper']=min(1.,theory_cache[normal_key]['target_failure_sums'][str(target)])
                rows.append(row);sample_events[law,target,policy]=(family,representative)
    comparisons=[]
    for law in LAWS:
        for target in model['targets']:
            for policy in POLICIES[1:]:
                for baseline in ('no_dfs','anchor'):
                    if policy==baseline:continue
                    a=sample_events[law,target,policy][0];b=sample_events[law,target,baseline][0]
                    comparisons.append(dict(law=law,target_representative=target,policy=policy,baseline=baseline,n=draws,
                        gained=int(np.sum(a&~b)),lost=int(np.sum(~a&b)),mean_difference=float(np.mean(a.astype(float)-b))))
    return dict(forecast=draw,source_seal=ref(folder/'seal.json'),sample_file=ref(saved),rows=rows,comparisons=comparisons,
        root_voltage_error_sd=model['root_sd'],independent_voltage_error_sd=model['independent_sd'],targets=model['targets'],
        conditioning='Each target is assumed to have its historical amplitude-corrected K mean; this is a declared-model Monte Carlo check, not AC recovery.')


def run(source,out,draws=2048,seed=2026111418,batch_size=128):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not(source/'complete.json').exists():raise ValueError('Completed sealed run required')
    if draws<=0 or batch_size<=0:raise ValueError('Positive sample sizes required')
    protocol=read(source/'protocol.json');entries=read(checked(protocol['inputs']))
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'probability.py')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),source_run=ref(source/'protocol.json'),draws=draws,
        seed=seed,batch_size=batch_size,stream='PCG64 SeedSequence[seed,entry.index,forecast,block,component0=power1=node2=root]; common arrays across policies and targets',
        environment=dict(python=sys.version,numpy=np.__version__),
        law='Power error 0 or 5% diagonal frozen forecast scale; shared-root plus independent-node squared-voltage Gaussian error, independent of power and across times. Planned nominal delivery.',
        reporting='Exact binomial intervals assume independent declared-model simulation paths. Exceedance tails are diagnostic, many comparisons unadjusted; no sampling deviation alone proves a formula defect.',
        intended_primary_draws=2048)
    write(out/'protocol.json',own);results=[];failures=[]
    for entry in entries:
        for draw in range(protocol['forecasts']):
            folder=source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            try:results.append(forecast_probability(folder,entry,protocol,out,draws,seed,batch_size))
            except Exception as exc:
                import traceback
                failures.append(dict(id=entry['id'],forecast=draw,error=repr(exc),traceback=traceback.format_exc()))
            write(out/'progress.json',dict(completed=len(results),failed=len(failures),last_forecast=draw))
    result=dict(status='complete' if not failures else 'failed',protocol=ref(out/'protocol.json'),results=results,failures=failures)
    write(out/'report.json',result);write(out/'complete.json',dict(status=result['status'],report=ref(out/'report.json'),source_unchanged=ref(__file__)['sha256']==own['source']['sha256']))
    return result


def self_test():
    H=np.array([[1.,.2]]);J=np.array([[.3,-.1]]);K=np.array([[.9,.25]]);m=np.array([[1.2,.3],[.7,-.2]])
    cov=np.array([[.04]]);s2=.5;h=.17;obj=PairIntegral(H,J,K,s2,cov,np.zeros_like(m),h)
    mu,var=obj.moments(m);D=H-J;y=m@K.T
    direct=h+float((np.square(y-m@J.T).sum()-np.square(y-m@H.T).sum())/(2*s2))
    directvar=float(np.sum((m@D.T)**2)*.04/s2**2)
    assert np.isclose(mu,direct) and np.isclose(var,directvar)
    for lam in (.5,.1,1.):assert np.isclose(obj.log_mgf(m,lam),-lam*mu+.5*lam*lam*var,atol=1e-12)
    # Scalar Gaussian-square known integral, with independent error set tiny.
    scalar=PairIntegral(np.array([[1.]]),np.array([[0.]]),np.array([[1.]]),1.,np.array([[1e-20]]),np.array([[.2]]),0.)
    known=-.5*np.log(1+.5*.2**2)-.5*.5*1.2**2/(1+.5*.2**2)
    assert np.isclose(scalar.log_mgf(np.array([[1.2]]),.5),known,atol=1e-12)
    return dict(status='passed',fixed_Gaussian_mean_variance=True,fixed_Gaussian_MGF=True,random_power_square_integral=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--draws',type=int,default=2048);p.add_argument('--seed',type=int,default=2026111418);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(self_test())
    else:
        if args.run is None or args.out is None:p.error('--run and --out required')
        answer=run(args.run,args.out,args.draws,args.seed,args.batch_size);print(dict(status=answer['status'],forecasts=len(answer['results']),failures=len(answer['failures'])))
        raise SystemExit(0 if answer['status']=='complete' else 1)
