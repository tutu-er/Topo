"""A portable paper15 experiment: plan -> freeze -> Gaussian check -> actual AC.

No topology fitting is performed. Historical H0/H1 and their output supports
are supplied by the topology interface. The original device/fee, exponential
gate and global solver are verbatim copies under vendor/. See README.md for
units, input arrays, information boundaries, commands and statistical scope.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import zipfile

import numpy as np
from vendor.exponential_dr import build_gate
from vendor.problem import load_problem
from vendor.solver import solve

HERE = Path(__file__).resolve().parent
POLICIES = ('optimized','zero','rotation')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def plain(x):
        if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [plain(v) for v in x]
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        return x
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def checked(reference, relative_to=None):
    path=Path(reference['path'])
    if not path.is_absolute(): path=Path(relative_to)/path
    if sha(path)!=reference['sha256']: raise ValueError('Changed frozen input: '+str(path))
    return path


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def contract_hash(value):
    """Exact contract equality, treating the IEEE +0/-0 representations alike."""
    canonical=np.where(np.asarray(value)==0,0.,value).astype(np.float64)
    return hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()


def load(case):
    """Planning-only reader: neither physical.npz nor network data are opened."""
    case=Path(case).resolve()
    cfg=read(case/'case.json')
    p=load_problem(case/cfg['pricing_case'])
    a=arrays(case/cfg['model_arrays'])
    T,N=p.T,p.N
    if (a['H0'].shape!=(N,2*N) or a['H1'].shape!=(N,2*N)
            or a['forecast_power'].shape!=(T,2*N) or a['likelihood_sd'].shape!=(T,N)
            or a['history_training_power'].shape!=(cfg['history_training_slots'],2*N)
            or a['history_power'].shape!=(int(p.history_count),2*N)
            or a['history_drop'].shape!=(int(p.history_count),N)
            or np.any(a['likelihood_sd']<=0)):
        raise ValueError('Incompatible complete-history/candidate/forecast interface')
    if cfg['history_training_slots']!=288 or int(p.history_count)!=288:
        raise ValueError('This locked paper15 protocol retains all288/288 history rows')
    hist_sd=np.tile(a['likelihood_sd'],(len(a['history_power'])//T,1))
    residuals=[a['history_drop']-a['history_power']@a[key].T for key in ('H0','H1')]
    S0,S1=[float(np.sum((r/hist_sd)**2)) for r in residuals]
    hist_mae=np.array([np.mean(np.abs(r)) for r in residuals])
    np.testing.assert_allclose(hist_mae,p.history_losses,atol=1e-6,rtol=1e-12)
    plan=cfg['planning']
    delta=float(plan['absolute_squared_tolerance']-(S0-plan['kappa']*S1))
    mean=(a['forecast_power']@(a['H1']-a['H0']).T)/a['likelihood_sd']
    # These are full covariance/factors of the STANDARDIZED model difference,
    # not the covariance of raw P/Q. Normalization is an exact linear transform.
    factor=a['difference_spatial_factor']
    phi=float(a['power_time_ar'])
    if factor.shape[:2]!=(T,N) or not 0<=phi<1:
        raise ValueError('Invalid standardized-difference Gaussian factor')
    corr=phi**np.abs(np.arange(T)[:,None]-np.arange(T)[None,:])
    C=np.einsum('tnc,smc,ts->tnsm',factor,factor,corr).reshape(T*N,T*N)
    C=(C+C.T)/2
    return case,cfg,p,a,dict(S0=S0,S1=S1,delta=delta,mean=mean,C=C)


def make_gate(p,a,data,planning,*,full_controls):
    """Apply the unchanged gate to normalized voltage-difference coordinates.

    New V'=(V-H0 P)/sd, P'=(H1-H0)P/sd gives H0'=0,H1'=I,
    SigmaV'=I, SigmaP'=C, and preserves both standardized squared losses.
    A one-zero-column B screens zero without materializing5760² coefficients.
    """
    size=p.T*p.N
    if full_controls:
        node=a['resource_node_index'].astype(int);ratio=a['resource_q_over_p']
        dh=a['H1']-a['H0']
        effect=(dh[:,node]/float(a['base_power_kw'])
                +dh[:,p.N+node]*ratio/float(a['base_reactive_kvar']))
        B=np.zeros((size,p.device.R*p.T))
        for t in range(p.T):
            B[t*p.N:(t+1)*p.N,np.arange(p.device.R)*p.T+t]=effect/a['likelihood_sd'][t,:,None]
    else:
        B=np.zeros((size,1))
    return build_gate(np.zeros((size,size)),np.eye(size),data['mean'].ravel(),B,
                      data['C'],np.eye(size),kappa=planning['kappa'],
                      beta=planning['beta'],delta=data['delta'])


def rotation(p, fee):
    """Fixed hourly alternation; choose only its scalar size to match the fee."""
    r=np.arange(p.device.R)
    x=np.where((np.arange(p.device.H)[None,:]//2+(r%4)[:,None])%2==0,1.,-1.)
    x=np.minimum(np.maximum(x,p.expand(p.lower)),p.expand(p.upper))
    if fee==0: return np.zeros((p.device.R,p.T)),dict(scale=0.,cost=0.)
    def delivery(alpha): return p.device.nominal(alpha*x)
    if p.device.fees(delivery(1.))<fee:
        raise ValueError('Rotation cannot match this fee without enlarging original resources')
    lo,hi=0.,1.
    for _ in range(70):
        mid=(lo+hi)/2
        if p.device.fees(delivery(mid))<fee:lo=mid
        else:hi=mid
    u=delivery((lo+hi)/2)
    return u,dict(scale=(lo+hi)/2,cost=p.device.fees(u))


def plan(case,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists(): raise ValueError('Existing attempt preserved; use a new plan directory')
    start=time.perf_counter()
    case,cfg,p,a,data=load(case)
    inputs=[case/'case.json',p.case_path,p.arrays_path,case/cfg['model_arrays'],case/cfg['candidate_file']]
    request=dict(protocol=ref(case/'PROTOCOL.json'),inputs=[ref(x) for x in inputs],
                 sources={str(x.relative_to(HERE)):sha(x) for x in [HERE/'run.py',*sorted((HERE/'vendor').glob('*.py'))]},
                 history_slots=[cfg['history_training_slots'],int(p.history_count)],new_slots=p.T,
                 resources=p.device.R,active_hourly_requests=len(p.active),
                 history_squared=dict(S0=data['S0'],S1=data['S1'],remaining_delta=data['delta']),
                 forecast='issued power forecast; no test observations or reference residual conditioning',
                 python=platform.python_version(),numpy=np.__version__,truth_or_physical_data_read=False)
    write(out/'request.json',request)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in request['sources']:z.write(HERE/name,name)
    gate=make_gate(p,a,data,cfg['planning'],full_controls=False)
    zero=p.nominal(p.zero);bound=gate.probability_bound([0.])
    if bound['sufficient_condition_met'] and zero['fee']==0 and p.audit_delivery(zero['mean_u'])['passed']:
        solution=dict(status='CERTIFIED_COST_ZERO',objective=0.,lower_bound=0.,gap=0.,
                      qualified=True,global_gap_target_met=True,solver_performed=False,
                      certificate='All original fees nonnegative; zero delivery feasible in the full60-resource domain')
        u=zero['mean_u'];x=zero['x']
    else:
        gate=make_gate(p,a,data,cfg['planning'],full_controls=True)
        result=solve(p,gate,gate.h,out/'solve',**cfg['planning']['solver'])
        solution={key:value for key,value in result.items() if not isinstance(value,np.ndarray)}
        if not result['qualified']:
            write(out/'summary.json',dict(status='unresolved',solver=solution,zero_bound=bound))
            write(out/'complete.json',dict(status='complete_unresolved'))
            return read(out/'summary.json')
        u,x=result['mean_u'],result['x']
        bound=gate.probability_bound(u)
    fee=p.device.fees(u)
    rotate,rotation_info=rotation(p,fee)
    contracts=dict(optimized=u,zero=np.zeros_like(u),rotation=rotate)
    audits={name:p.audit_delivery(value) for name,value in contracts.items()}
    if not all(row['passed'] for row in audits.values()):raise ValueError('Original device audit failed')
    np.savez_compressed(out/'quotes.npz',**contracts,optimized_hourly_requests=x)
    freeze=dict(status='frozen',quote=dict(path='quotes.npz',sha256=sha(out/'quotes.npz')),
                request=dict(path='request.json',sha256=sha(out/'request.json')),
                no_actual_future_or_truth_used=True)
    write(out/'policy_freeze.json',freeze)
    m=p.T*p.N;kappa=cfg['planning']['kappa'];beta=cfg['planning']['beta']
    h_upper=data['delta']/(4*kappa)-np.log1p(-beta)-m/2*np.log((kappa+1)/(2*kappa))
    summary=dict(status='complete',qualified=True,solver=solution,planning_bound=bound,
                 h_upper_without_covariance_determinant=h_upper,history_squared=request['history_squared'],
                 fees={name:p.device.fees(value) for name,value in contracts.items()},rotation=rotation_info,
                 device_audits=audits,resource_count=p.device.R,seconds=time.perf_counter()-start,
                 unique_contracts=len({contract_hash(value) for value in contracts.values()}),
                 probability_scope='conditional on frozen model1 and declared fixed Gaussian errors; sufficient lower bound',
                 original_final_selection='MAE common3percent, new day weight1, frozen original outputs')
    write(out/'summary.json',summary);write(out/'complete.json',dict(status='complete'))
    return summary


def frozen(case,folder):
    folder=Path(folder).resolve();f=read(folder/'policy_freeze.json')
    if f['status']!='frozen':raise ValueError('Unfrozen plan')
    quote=checked(f['quote'],folder);request=read(checked(f['request'],folder))
    case=Path(case).resolve();cfg=read(case/'case.json')
    expected={case/'case.json',case/cfg['pricing_case'],case/cfg['model_arrays'],case/cfg['candidate_file']}
    pricing=read(case/cfg['pricing_case'])
    expected.add((case/cfg['pricing_case']).parent/pricing['arrays_file'])
    actual={checked(r).resolve() for r in request['inputs']}
    if actual!={p.resolve() for p in expected}:raise ValueError('Supplied case differs from frozen planning case')
    if checked(request['protocol']).resolve()!=(case/'PROTOCOL.json').resolve():
        raise ValueError('Frozen protocol differs from supplied case')
    for name,digest in request['sources'].items():
        if sha(HERE/name)!=digest:raise ValueError('Planning source changed after freeze: '+name)
    return folder,arrays(quote)


def working_sample(case,folder,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return read(out/'summary.json')
    case,cfg,p,a,data=load(case);folder,quotes=frozen(case,folder)
    ev=cfg['evaluation'];count=ev['working_model_paths'];rng=np.random.default_rng(ev['working_model_seed'])
    node=a['resource_node_index'].astype(int);ratio=a['resource_q_over_p'];dh=a['H1']-a['H0']
    effect=dh[:,node]/float(a['base_power_kw'])+dh[:,p.N+node]*ratio/float(a['base_reactive_kvar'])
    means={name:data['mean']+np.einsum('nr,rt->tn',effect,quotes[name])/a['likelihood_sd'] for name in POLICIES}
    write(out/'request.json',dict(plan=ref(folder/'policy_freeze.json'),paths=count,seed=ev['working_model_seed'],
        scope='declared Gaussian background plus common voltage error; nominal deterministic DR, no AC'))
    values={name:[] for name in POLICIES};phi=float(a['power_time_ar'])
    for first in range(0,count,128):
        batch=min(128,count-first)
        colored=rng.standard_normal((batch,p.T,a['difference_spatial_factor'].shape[2]))
        for t in range(1,p.T):colored[:,t]=phi*colored[:,t-1]+np.sqrt(1-phi**2)*colored[:,t]
        eta=np.einsum('tnc,btc->btn',a['difference_spatial_factor'],colored)
        e=rng.standard_normal((batch,p.T,p.N));e2=np.sum(e*e,axis=(1,2))
        for name in POLICIES:
            z=means[name][None]+eta
            L=np.sum(z*z+2*e*z,axis=(1,2))-(cfg['planning']['kappa']-1)*e2
            values[name].append(L)
    rows={}
    for name in POLICIES:
        L=np.concatenate(values[name]);np.savez_compressed(out/(name+'.npz'),L=L)
        rows[name]=dict(planned=count,completed=count,successes=int(np.sum(L>data['delta'])),
                        fraction=float(np.mean(L>data['delta'])),mean_L=float(L.mean()),min_L=float(L.min()))
    summary=dict(status='complete',independent_random_paths=count,independent_natural_days=0,rows=rows)
    write(out/'summary.json',summary);write(out/'complete.json',dict(status='complete'))
    return summary


def actual(case,folder,out):
    """Freeze all selections first; consult true supports only after that file."""
    import ac  # lazy: planning never imports/opens the physical network
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return read(out/'summary.json')
    lock=out/'running.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        case,cfg,p,a,data=load(case);folder,quotes=frozen(case,folder)
        ev=cfg['evaluation'];count=ev['actual_paths']
        write(out/'request.json',dict(plan=ref(folder/'policy_freeze.json'),planned_paths=count,
              policies=POLICIES,response_seed=ev['response_seed'],meter_seed=ev['meter_seed'],
              source=ref(__file__),ac_source=ref(HERE/'ac.py'),
              selection=p.config['selection'],truth_gate='after predictions_freeze.json'))
        runtime=ac.open_runtime(case,folder/'policy_freeze.json')
        responses=np.random.SeedSequence(ev['response_seed']).spawn(count)
        meters=np.random.SeedSequence(ev['meter_seed']).spawn(count)
        groups={};alias={}
        for name in POLICIES:
            key=contract_hash(quotes[name])
            if key not in groups:groups[key]=name
            alias[name]=groups[key]
        rows=read(out/'rows.json') if (out/'rows.json').exists() else []
        done={(r['path'],r['policy']) for r in rows};observations=out/'observations';observations.mkdir(exist_ok=True)
        for path in range(count):
            response_seed=int(np.random.default_rng(responses[path]).integers(1,2**62))
            meter_seed=int(np.random.default_rng(meters[path]).integers(1,2**62))
            multiplier=runtime['device'].draw(response_seed,1,stress=False)[0]
            for name in groups.values():
                if (path,name) in done:continue
                row=dict(path=path,policy=name,fee=p.device.fees(quotes[name]),status='unresolved')
                try:
                    result=ac.evaluate(runtime,quotes[name],multiplier,meter_seed,f'p{path}_{name}')
                    row['physical']=result['summary']
                    if result['summary']['status']!='complete':raise ArithmeticError('Original AC delivery/solve failed')
                    power,drop=result['power'],result['drop_micro']
                    selection=p.score_observations(power,drop,a['H0'],a['H1'])
                    r0=(drop-power@a['H0'].T)/a['likelihood_sd']
                    r1=(drop-power@a['H1'].T)/a['likelihood_sd']
                    S0,S1=float(np.sum(r0*r0)),float(np.sum(r1*r1));L=S0-cfg['planning']['kappa']*S1
                    saved=observations/f'{path:03d}_{name}.npz'
                    np.savez_compressed(saved,power=power,drop_micro=drop,delivered=result['delivered'])
                    row.update(status='predicted',observation=ref(saved),selection=selection,
                               S0=S0,S1=S1,L=L,squared_condition=L>data['delta'],squared_margin=L-data['delta'])
                except Exception as exc:
                    row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                rows.append(row);write(out/'rows.json',rows)
            print(f'AC path {path+1}/{count}, unique contracts {len(groups)}',flush=True)
        write(out/'predictions.json',dict(rows=rows,aliases=alias,truth_read=False))
        write(out/'predictions_freeze.json',dict(status='frozen',predictions=ref(out/'predictions.json')))
        truth=ac.finalize(runtime,out/'predictions_freeze.json')
        candidates=read(case/cfg['candidate_file'])
        truth_set={tuple(sorted(x)) for x in truth['supports']}
        errors={}
        for model in candidates['models']:
            supports={tuple(sorted(x)) for x in model['frozen_output_supports']}
            errors[model['id']]=dict(missing=len(truth_set-supports),extra=len(supports-truth_set),
                                    errors=len(truth_set^supports))
        for row in rows:
            if row['status']=='predicted':row.update(status='complete',**errors[row['selection']['selected']])
        write(out/'scored_rows.json',rows)
        metrics={}
        for name in POLICIES:
            subset=[r for r in rows if r['policy']==alias[name]];ok=[r for r in subset if r['status']=='complete']
            failed=sum(r['status']=='failed' for r in subset)
            metrics[name]=dict(planned=count,completed=len(ok),failed=failed,
                unresolved=count-len(ok)-failed,squared_successes=sum(r['squared_condition'] for r in ok),
                original_MAE_exclusions=sum(r['selection']['coarse_excluded'] for r in ok),
                complete_recoveries=sum(r['errors']==0 for r in ok),missing=sum(r['missing'] for r in ok) if ok else None,
                extra=sum(r['extra'] for r in ok) if ok else None,total_errors=sum(r['errors'] for r in ok) if ok else None,
                prepaid_fee=p.device.fees(quotes[name]),
                mean_actual_absolute_DR_kwh=float(np.mean([r['physical']['resource_audit']['absolute_DR_kwh'] for r in ok])) if ok else None,
                mean_squared_L=float(np.mean([r['L'] for r in ok])) if ok else None,
                mean_MAE_margin=float(np.mean([r['selection']['selection_margin'] for r in ok])) if ok else None,
                aliases_observations_of=alias[name])
        summary=dict(status='complete',unique_AC_records=len(rows),planned_unique_AC=count*len(groups),
             labelled_policy_outputs=count*len(POLICIES),independent_random_paths=count,independent_natural_days=1,
             rows=metrics,candidate_errors=errors,new_fit_calls=0,new_W_search_calls=0,
             evaluation='original frozen historical RNJ+WZZT pair final validation and output map; not a new search')
        write(out/'summary.json',summary);write(out/'complete.json',dict(status='complete'))
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','sample','ac','all'])
    parser.add_argument('--case',type=Path,default=HERE)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    folder=args.out/'plan' if args.action=='all' else args.plan
    if args.action in ('plan','all'):
        result=plan(args.case,args.out if args.action=='plan' else folder)
        print(json.dumps(result,default=lambda x:x.tolist(),allow_nan=False))
        if not result.get('qualified'):return
    if args.action in ('sample','all'):
        if folder is None:parser.error('--plan is required')
        print(json.dumps(working_sample(args.case,folder,args.out/'sample' if args.action=='all' else args.out)))
    if args.action in ('ac','all'):
        if folder is None:parser.error('--plan is required')
        print(json.dumps(actual(args.case,folder,args.out/'ac' if args.action=='all' else args.out)))


if __name__=='__main__':main()
