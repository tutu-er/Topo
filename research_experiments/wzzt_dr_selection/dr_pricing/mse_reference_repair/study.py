"""Historical reference repair, fixed-candidate QCQP, and independent AC test.

The original device graph, physics and candidate parameters are reused. A raw
historical model is a voltage predictor, not a declaration of topology truth.
All numeric trajectories and fitted parameters stay in --out outside Git.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
import zipfile
import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse import csr_matrix
from quadratic import build_quadratic_gate
from local_candidates import proposals, canonical

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OLDER=HERE.parent/'mse_existing_cases'
sys.path.insert(0,str(OLDER))
spec=importlib.util.spec_from_file_location('_existing_case_driver',OLDER/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io,core=old.io,old.core
from vendor.solver import build_device_graph,_values,_attribute
from selection_mse import choose_mse


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def root_data(folder):
    return old.packet(folder)


def enrich(data,extra):
    """Only add historical QP outputs; no new-day parameter estimation."""
    d=dict(data);models=list(data['models']);seen={m.get('canonical_parameter_sha256',m['parameter_sha256']) for m in models}
    for row in extra:
        point,H,key=core._point(row['point'],len(d['labels']))
        if key in seen:continue
        seen.add(key);r=d['history_drop']-d['history_power']@H.T
        models.append(dict(id=row['id'],point=point,H=H,internal=sum(1<len(c)<len(d['labels']) for c in point['supports']),
            history_sse=float(np.sum(r*r)),history_mse=float(np.mean(r*r)),parameter_sha256=key,origin=row['origin']))
    d['models']=models;d['H']=np.stack([m['H'] for m in models]);d['history_sse']=np.array([m['history_sse'] for m in models])
    return d


def prepare(out):
    old.prepare(out)
    protocol=io.read(out/'protocol.json')
    protocol.update(id='general_reference_QCQP_v1',status='preparing',
        working_law='fixed Gaussian measured-power forecast error with historical RMS, shared same-node P/Q factors and exact AR1 .7; separate historical raw voltage predictor',
        fallback='no_purchase when no probability-feasible quote is found; no paid unqualified fallback',
        new_fits=0,new_test_fits=0,
        reference='existing raw path model, minimum first192 historical validation MSE; no parameter refit',
        reference_audit='last96 historical validation, already seen development data; not a new global holdout',
        primary_planning_pool='original output library; local additions are a separate same-observation diagnostic',
        power_covariance='four historical daily forecast residual RMS per channel; same-node P/Q factor; preserve original AR1 .7 and independent nodes',
        voltage_covariance='same fixed diagonal SD as previous H1 benchmark; changing the reference does not reduce uncertainty; no duplicate PQ meter term',
        lambda_grid=[.25,.125,.0625],beta=.9,
        optimizer=dict(seconds_per_lambda=90,mip_gap=.02,threads=1),
        no_feasible_quote='return zero with no_purchase status; do not pay an unqualified fallback',
        paths=[dict(day=i,profile=i%3,replicate=1140000+173*i,response_seed=2026111400+i) for i in range(6)])
    # Lock the design and fitting choices before any new candidate fit or actual day.
    io.write(out/'preparation_protocol.json',protocol)
    summaries=[];history_QP_calls=0
    backend_path=old.PARENT.parent/'miqp_loss_benchmark_20261111/source/research_experiments/miqp_loss_benchmark_20261111/squared_backend.py'
    backend=load_file('_reference_native_QP',backend_path)
    for name,bundle,noise in protocol['cases']:
        folder=out/f'day96__{name}__b{bundle}__reference__{noise}__N96';config,d=root_data(folder)
        native=io.read(io.checked(config['result']));byid={m['id']:m for m in native['models']}
        raw=[byid[k] for k in native['variants']['tau_0.001']['path_model_ids']]
        scores=[]
        for m in raw:
            H=np.column_stack((m['R'],m['X']));r=d['history_drop'][:192]-d['history_power'][:192]@H.T
            scores.append(float(np.mean(r*r)))
        j=min(range(len(raw)),key=lambda i:(scores[i],i));Hbar=np.column_stack((raw[j]['R'],raw[j]['X']))
        io.write(folder/'reference_selection_frozen.json',dict(model_id=raw[j]['id'],selection_mse=scores[j],
            selection_rows=192,source=io.ref(io.checked(config['result'])),actual_used=False,truth_used=False))
        residual=d['history_drop']-d['history_power']@Hbar.T
        reference_audit=dict(model_id=raw[j]['id'],selection_mse=scores[j],audit_mse=float(np.mean(residual[192:]**2)),
            baseline_audit_mse=float(np.mean((d['history_drop'][192:]-d['history_power'][192:]@d['H'][d['reference']].T)**2)))
        np.savez_compressed(folder/'reference.npz',Hbar=Hbar,bias=np.zeros(d['p'].N))
        # Uniform finite family completion, never using true supports to select a region.
        families=proposals(d['models'][d['current']]['point'],d['models'][d['reference']]['point'],d['p'].N)
        common_path=Path(native['provenance']['parent_common_file']);common=io.read(common_path);cal=common['calibration']
        pool=canonical([c for row in families for c in row['family']]);extra=[];fits=[]
        if pool:
            qp=backend.PoolQP((d['train_power'][:,:d['p'].N]*1000,d['train_power'][:,d['p'].N:]*1000,d['train_drop']*1000),
                pool,loss_scale=d['K'],bound=2.,seconds=300.)
            try:
                for k,row in enumerate(families):
                    try:
                        rawpoint=qp.fit(row['family'])
                        family=[c for c,r,x in zip(rawpoint['supports'],rawpoint['r'],rawpoint['x']) if len(c)==1 or
                            cal['a']*r+cal['b']*x+1e-10>=.24*cal['median_root_depth']]
                        pt=qp.fit(family);extra.append(dict(id=f'local_{k:03}',point=pt,origin=row['origin'],regions=row['regions']))
                        fits.append(dict(origin=row['origin'],status='complete',raw_supports=len(rawpoint['supports']),output_supports=len(pt['supports'])))
                    except Exception as exc:fits.append(dict(origin=row['origin'],status='failed',error=repr(exc)))
                io.write(folder/'QP_audit.json',dict(source=io.ref(backend_path),solves=qp.solves,proposals=fits,common=io.ref(common_path)))
                history_QP_calls+=len(qp.solves)
            finally:qp.close()
        io.write(folder/'extra_models.json',extra)
        expanded=enrich(d,extra);sel=choose_mse(expanded['history_sse']/d['history_drop'].size,
            [m['internal'] for m in expanded['models']],K=d['K'],candidate_ids=[m['id'] for m in expanded['models']])
        summaries.append(dict(case=name,reference=reference_audit,proposals=fits,
            original_models=len(d['models']),expanded_models=len(expanded['models']),historical_augmented_selected=sel['selected']))
    protocol['status']='frozen';protocol['source_sha256'].update({str(p.resolve()):io.sha(p) for p in HERE.glob('*.py')})
    protocol['new_fits']=protocol['new_historical_QP_calls']=history_QP_calls
    protocol['source_sha256'][str(backend_path.resolve())]=io.sha(backend_path)
    protocol['prepared_inputs']=[io.ref(p) for p in out.glob('day96__*/reference*.json')]
    protocol['prepared_inputs'] += [io.ref(p) for p in out.glob('day96__*/reference.npz')]
    protocol['prepared_inputs'] += [io.ref(p) for p in out.glob('day96__*/extra_models.json')]
    io.write(out/'protocol.json',protocol);io.write(out/'reference_summary.json',summaries)
    with zipfile.ZipFile(out/'repair_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in HERE.glob('*.py'):z.write(p,p.name)
    return summaries


def psd_factor(C):
    vals,vecs=np.linalg.eigh((C+C.T)/2)
    if vals.min(initial=0)<-1e-10*max(1.,abs(vals).max(initial=0)):raise ValueError('Covariance not PSD')
    keep=vals>np.finfo(float).eps*max(1.,vals.max(initial=0))
    return vecs[:,keep]*np.sqrt(np.maximum(vals[keep],0))


class BlockGate:
    """The full correlated Gaussian integral, represented by sparse precision.

    v=Pi^-1 W mu is an auxiliary linear equality, not an optimization freedom.
    The resulting score is exactly a quadratic in node kW controls z, including
    all original AR1 correlations. Pi's block tridiagonal form avoids forming
    a dense control Hessian; the model is still a QCQP.
    """
    def __init__(self,K,L,Pi,chol,forecast,base,c,h,lam,delta):
        self.K,self.L,self.Pi,self.chol=K,L,Pi,chol
        self.forecast,self.base,self.c,self.h,self.lam,self.delta=forecast,base,c,h,lam,delta
        self.W=np.kron(np.eye(len(forecast)),L.T@K)
    def latent(self,z):
        mu=self.forecast+z/self.base
        return cho_solve((self.chol,True),self.W@mu.ravel())
    def value(self,z):
        mu=self.forecast+z/self.base;v=self.latent(z)
        return float(np.einsum('ti,ij,tj->',mu,self.K,mu)-2*v@self.Pi@v+self.c)
    def gradient(self,z):
        mu=self.forecast+z/self.base;v=self.latent(z).reshape(len(mu),-1)
        return (2*mu@self.K-4*v@self.L.T@self.K)/self.base
    def bound(self,z):
        v=self.value(z);lf=self.lam*self.delta-v
        return dict(score=v,h=self.h,margin=v-self.h,success_lower=float(-np.expm1(min(0.,lf))),sufficient=v>=self.h)


def correlated_gate(H0,H1,Hbar,forecast,sigma,sd,phi,base,lam,delta):
    """Exact fixed-AR Gaussian integral. delta is already in normalized SSE."""
    T=forecast.shape[0];N=H0.shape[0];s=float(sd.max())
    D0,D1=(Hbar-H0)/s,(Hbar-H1)/s
    A=D0.T@D0-D1.T@D1
    D=D0-D1;K=lam*A-2*lam*lam*(D.T*(sd/s)**2)@D;K=(K+K.T)/2
    L=np.vstack((np.diag(sigma[:N]),np.diag(sigma[N:])))
    precision=np.eye(T)*(1+phi*phi)/(1-phi*phi)
    precision[0,0]=precision[-1,-1]=1/(1-phi*phi) if T>1 else 1.
    precision+=np.diag(np.full(T-1,-phi/(1-phi*phi)),1)+np.diag(np.full(T-1,-phi/(1-phi*phi)),-1)
    Pi=np.kron(precision,np.eye(N))+2*np.kron(np.eye(T),L.T@K@L)
    Pi=(Pi+Pi.T)/2
    try:chol=np.linalg.cholesky((Pi+Pi.T)/2)
    except np.linalg.LinAlgError as exc:raise ValueError('Correlated Gaussian integration matrix is not positive definite') from exc
    logdetJ=float(2*np.log(np.diag(chol)).sum()+N*(T-1)*np.log1p(-phi*phi))
    return BlockGate(K,L,Pi,chol,forecast,base,.5*logdetJ,float(lam*delta+np.log(10)),lam,delta)


def gate_for(d,Hbar,lam):
    N,T=d['p'].N,96
    H0,H1=d['H'][[d['current'],d['reference']]]
    residual=d['history_drop']-d['history_power']@H1.T
    sd=np.maximum(np.sqrt(np.mean(residual**2,axis=0)),2*np.sqrt(2)*d['voltage_sigma']);s=float(sd.max())
    hist=np.concatenate((d['train_power'].reshape(1,T,2*N),d['history_power'].reshape(3,T,2*N)))
    sigma=np.sqrt(np.mean((hist-d['forecast'][None])**2,axis=(0,1)))
    delta=float(((d['history_drop'].size+T*N)*1e-12/d['K']-
        d['history_sse'][d['current']]+d['history_sse'][d['reference']])/(s*s))
    return correlated_gate(H0,H1,Hbar,d['forecast'],sigma,sd,.7,d['base_kw'],lam,delta)


def aggregate(d,u):
    dp,dq=d['original_device'].aggregate(u)
    return np.column_stack((dp,dq))


def solve_one(d,g,out,seconds):
    """Solve the original indefinite QCQP with the unchanged causal device graph."""
    import gurobipy as gp
    p=d['p'];zero=p.nominal(p.zero);node=d['original_device'].node;ratio=d['original_device'].qratio
    # Deterministic full signed score-gradient seeds are warm starts only.
    grad=g.gradient(np.zeros((96,2*p.N)));gu=np.array([grad[:,n]+ratio[r]*grad[:,p.N+n] for r,n in enumerate(node)])
    gx=p.device.hourly_vjp(gu,zero['cache']).ravel()[p.active]
    seeds=[zero,p.nominal(np.where(gx>=0,p.upper,p.lower)),p.nominal(np.where(gx>=0,p.lower,p.upper))]
    good=[v for v in seeds if g.bound(aggregate(d,v['mean_u']))['sufficient']]
    warm=min(good,key=lambda v:v['fee']) if good else max(seeds,key=lambda v:g.value(aggregate(d,v['mean_u'])))
    out.mkdir(parents=True,exist_ok=False);began=time.perf_counter();best=None
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model('reference_signed_SSE_QCQP',env=env) as m:
            for k,v in dict(OutputFlag=1,LogToConsole=0,LogFile=str(out/'gurobi.log'),TimeLimit=seconds,
                Threads=1,MIPGap=.02,NonConvex=2,Seed=0,MIPFocus=1,NumericFocus=2,
                FeasibilityTol=1e-8,IntFeasTol=1e-8,DualReductions=0).items():m.setParam(k,v)
            x,u,cost=build_device_graph(m,p,warm['x'])
            lo=np.zeros((96,2*p.N));hi=lo.copy()
            for r,n in enumerate(node):
                lo[:,n]+=p.device.delta_lower_kw[r];hi[:,n]+=p.device.delta_upper_kw[r]
                lo[:,p.N+n]+=ratio[r]*p.device.delta_lower_kw[r];hi[:,p.N+n]+=ratio[r]*p.device.delta_upper_kw[r]
            z=m.addMVar((96,2*p.N),lb=lo,ub=hi,name='node_delta_kw')
            for t in range(96):
                for n in range(p.N):
                    ids=np.flatnonzero(node==n)
                    m.addConstr(z[t,n]==gp.quicksum(u[r,t].expr for r in ids))
                    m.addConstr(z[t,p.N+n]==gp.quicksum(float(ratio[r])*u[r,t].expr for r in ids))
            z.Start=aggregate(d,warm['mean_u']);score=gp.QuadExpr(float(g.c))
            mapping=cho_solve((g.chol,True),g.W)/g.base
            center=cho_solve((g.chol,True),g.W@g.forecast.ravel())
            positive=np.maximum(mapping,0);negative=np.minimum(mapping,0)
            vlo=center+positive@lo.ravel()+negative@hi.ravel()
            vhi=center+positive@hi.ravel()+negative@lo.ravel()
            v=m.addMVar(len(center),lb=vlo,ub=vhi,name='Gaussian_integral_aux')
            m.addConstr(csr_matrix(g.Pi)@v==g.W@g.forecast.ravel()+csr_matrix(g.W/g.base)@z.reshape(-1))
            v.Start=g.latent(aggregate(d,warm['mean_u']))
            for t in range(96):
                mean=g.forecast[t]
                score+=(z[t]@(g.K/g.base**2)@z[t]+(2*mean@g.K/g.base)@z[t]).item()+float(mean@g.K@mean)
            score-=2*(v@csr_matrix(g.Pi)@v).item()
            m.addQConstr(score>=g.h+1e-6*max(1.,abs(g.h)));m.setObjective(cost,gp.GRB.MINIMIZE)
            m.update();m.write(str(out/'model.mps.gz'));m.optimize()
            result=dict(status=int(m.Status),has_solution=bool(m.SolCount),lower_bound=_attribute(m,'ObjBound'),
                reported_gap=_attribute(m,'MIPGap') if m.SolCount else None,seconds=time.perf_counter()-began,
                variables=m.NumVars,binaries=m.NumBinVars)
            points=list(good)
            if m.SolCount:
                raw=_values(x);raw=p.expand(np.clip(raw.ravel()[p.active],p.lower,p.upper))
                points.append(p.nominal(p.requests_to_values(raw)))
            feasible=[v for v in points if p.audit_delivery(v['mean_u'])['passed'] and g.bound(aggregate(d,v['mean_u']))['sufficient']]
            if feasible:best=min(feasible,key=lambda v:v['fee'])
    result['qualified']=best is not None
    if best is not None:
        result.update(fee=best['fee'],bound=g.bound(aggregate(d,best['mean_u'])))
        np.savez_compressed(out/'candidate.npz',x=best['x'],u=best['mean_u'])
    io.write(out/'summary.json',result)
    return best,result


def plan(out):
    if (out/'all_prices_frozen.json').exists():
        frozen=io.read(out/'all_prices_frozen.json')
        return io.read(io.checked(frozen['plans']))
    protocol=old.verify(out);summaries=[]
    for item in protocol['prepared_inputs']:io.checked(item)
    for name,bundle,noise in protocol['cases']:
        folder=out/f'day96__{name}__b{bundle}__reference__{noise}__N96'
        if (folder/'policy_freeze.json').exists():summaries.append(io.read(folder/'plan_summary.json'));continue
        _,d=root_data(folder);p=d['p'];reference=io.arrays(folder/'reference.npz')
        if np.any(reference['bias']!=0):
            raise ValueError('This sparse case runner assumes zero intercept; use the generic quadratic gate for a nonzero bias')
        Hbar=reference['Hbar'];zero=p.nominal(p.zero)
        candidates=[];traces=[]
        for j,lam in enumerate(protocol['lambda_grid']):
            try:g=gate_for(d,Hbar,lam)
            except ValueError as exc:traces.append(dict(lambda_=lam,status='integration_not_positive',error=str(exc)));continue
            bound=g.bound(np.zeros((96,2*p.N)))
            if bound['sufficient']:
                candidates.append((zero,g));traces.append(dict(lambda_=lam,status='certified_zero',bound=bound));break
            best,result=solve_one(d,g,folder/f'lambda_{j}',protocol['optimizer']['seconds_per_lambda'])
            traces.append(dict(lambda_=lam,zero_bound=bound,result=result))
            if best is not None:candidates.append((best,g))
        if candidates:
            best,g=min(candidates,key=lambda v:v[0]['fee']);status='qualified';bound=g.bound(aggregate(d,best['mean_u']))
        else:best=zero;g=None;status='no_purchase_no_feasible_certificate';bound=None
        fee=best['fee'];rotate,rotation=io.rotation(p,fee)
        np.savez_compressed(folder/'quotes.npz',zero=zero['mean_u'],optimized=best['mean_u'],rotation=rotate,hourly_requests=best['x'])
        summary=dict(case=name,status=status,fee=fee,certificate=bound,trace=traces,rotation=rotation,
            no_unqualified_payment=not bool(fee and not candidates),selected_reference=io.read(folder/'reference_selection_frozen.json')['model_id'])
        io.write(folder/'plan_summary.json',summary)
        io.write(folder/'policy_freeze.json',dict(status='frozen',quote=io.ref(folder/'quotes.npz'),
            case=io.ref(folder/'case.json'),source=io.ref(out/'protocol.json'),reference=io.ref(folder/'reference.npz'),actual_future_used=False))
        summaries.append(summary);print('plan',name,status,fee,flush=True)
    io.write(out/'plans.json',summaries)
    io.write(out/'all_prices_frozen.json',dict(status='frozen',plans=io.ref(out/'plans.json'),protocol=io.ref(out/'protocol.json')))
    return summaries


def actual(out):
    if (out/'complete.json').exists():
        return io.read(out/'summary.json')
    protocol=old.verify(out);io.checked(io.read(out/'all_prices_frozen.json')['plans']);rows=[];truth={};calls=0
    for item in protocol['prepared_inputs']:io.checked(item)
    for ci,(name,bundle,noise) in enumerate(protocol['cases']):
        folder=out/f'day96__{name}__b{bundle}__reference__{noise}__N96';cfg,d=root_data(folder)
        expanded=enrich(d,io.read(folder/'extra_models.json'));quotes=io.arrays(io.checked(io.read(folder/'policy_freeze.json')['quote']))
        meta=io.read(io.checked(cfg['metadata']))
        rt=old.open_runtime(old.DATA/'source/rnj_wzzt_core',dict(cfg['metadata'],case_name=meta['source']['case_name']),
            repo_root=REPO,protocol_ref=io.ref(out/'protocol.json'),labels=d['labels'],case=name,v_sigma=d['voltage_sigma'],pq_sigma=d['pq_sigma'])
        old.audit_network(rt,folder,name);execution=old._load_execution();dev=d['original_device']
        for path in protocol['paths']:
            bg=old.build_background(rt,path['profile'],path['replicate']+ci*10000)
            multiplier=dev.draw(path['response_seed']+100*ci,1,stress=False)[0];cache={}
            for policy in protocol['policies']:
                target=folder/'actual'/f"day{path['day']}_{policy}";target.mkdir(parents=True,exist_ok=True);receipt=target/'receipt.json'
                if receipt.exists():rows.append(io.read(receipt));continue
                row=dict(case=name,job=cfg['job'],day=path['day'],policy=policy,fee=dev.fees(quotes[policy]))
                key=io.contract_hash(quotes[policy])
                try:
                    if key in cache:
                        prior=cache[key];row.update({k:v for k,v in prior.items() if k not in ('policy','fee')},alias_of=prior['policy'])
                    else:
                        res=execution.execute(rt,bg,dev,quotes[policy],multiplier,f"reference_repair_{name}_{path['day']}_{policy}")
                        old.dump(target/'result.pkl',res);calls+=1;row.update(status=res['summary']['status'],physical=res['summary'])
                        if row['status']=='complete':
                            scene=res['scene'];P=np.column_stack((np.asarray(scene['P_terminal']),np.asarray(scene['Q_terminal'])));Y=np.asarray(scene['drop_target'])
                            np.savez_compressed(target/'observed.npz',power=P,drop=Y,delivered=res['private']['delivered'])
                            row['observation']=io.ref(target/'observed.npz');row['views']={}
                            for view,dt in [('original',d),('local_completion',expanded)]:
                                sse=np.array([np.sum((Y-P@H.T)**2) for H in dt['H']]);loss=(dt['history_sse']+sse)/(dt['history_drop'].size+Y.size)
                                row['views'][view]=choose_mse(loss,[m['internal'] for m in dt['models']],K=dt['K'],candidate_ids=[m['id'] for m in dt['models']])
                        cache[key]=dict(row)
                except Exception as exc:row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                io.write(receipt,row);rows.append(row);io.write(out/'checkpoint.json',dict(records=len(rows),AC_calls=calls))
            print('actual',name,path['day'],len(rows),calls,flush=True)
        truth[name]=execution.effective_truth_supports(rt['context'].net)
    io.write(out/'predictions_frozen.json',dict(rows=rows,plans=io.ref(out/'all_prices_frozen.json')))
    for row in rows:
        if row['status']!='complete':continue
        _,d=root_data(out/row['job']);expanded=enrich(d,io.read(out/row['job']/'extra_models.json'));target={tuple(sorted(s)) for s in truth[row['case']]}
        for view,dt in [('original',d),('local_completion',expanded)]:
            picked=next(m for m in dt['models'] if m['id']==row['views'][view]['selected'])
            pred={tuple(sorted(dt['labels'][i] for i in s)) for s in picked['point']['supports']}
            row['views'][view]['errors']=dict(missing=len(target-pred),extra=len(pred-target),total=len(target^pred))
    io.write(out/'rows.json',rows);tables=[]
    for name,_,_ in protocol['cases']:
        for policy in protocol['policies']:
            group=[r for r in rows if r['case']==name and r['policy']==policy];ok=[r for r in group if r['status']=='complete']
            tables.append(dict(case=name,policy=policy,planned=len(group),completed=len(ok),
                fee=float(np.mean([r['fee'] for r in group])),views={v:dict(total_errors=sum(r['views'][v]['errors']['total'] for r in ok),
                recoveries=sum(r['views'][v]['errors']['total']==0 for r in ok),selected=[r['views'][v]['selected'] for r in ok]) for v in ['original','local_completion']}))
    summary=dict(status='complete',tables=tables,records=len(rows),unique_AC_calls=calls,independent_new_days=12,
        failures=sum(r['status']=='failed' for r in rows),unresolved=sum(r['status'] not in ['complete','failed'] for r in rows))
    io.write(out/'summary.json',summary);io.write(out/'complete.json',dict(status='complete'))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','plan','actual']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(globals()[args.action](args.out.resolve()),default=lambda x:x.tolist(),allow_nan=False))
