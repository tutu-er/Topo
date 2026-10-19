"""Predeclared continuous-curve and internal-impedance development diagnostic.

Truth/noise enter run_condition only; select_curves accepts H and public inputs.
See ev_profile_milp_protocol.md. No prior rejection threshold is reused.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import LinearConstraint
from . import likelihood as lk
from . import information_study as old
from . import gradient_price_study as io
from .simulation import model, observe, truth_metadata, BASE_KW
from .candidates import propose, normalize_family
from .information_context import quality_models, meter_to_record
from .ev_flexibility_study import device
from .ev_service_milp import canonical_service_plan, check_service_plan
from .single_site_milp import SingleSitePlan
from .symmetric_ac_local import baseline_from_history, mean_and_jacobian, whiten_observations
from .ev_profile_ac_proxy import build_pair_models, bounded_audit
from .ev_profile_chord_milp import FrozenQuadratic, solve_profile_chord_milp

HERE=Path(__file__).resolve().parent
CONFIG=dict(version='profile_chord_development_v1',seeds=[986101,986102,986103],
    truth_internal_r=[.0003,.003,.012],fit_internal_r_floors=[1e-6,.001,.004],
    x_r_ratio=2/3,horizon=4,history=12,site=0,grid_energy_kwh=3.,power_kw=6.,
    quality='temporal',segments=4,milp_seconds=5.,history_half_sse_gate=6.,
    fit_options=dict(multistart=2,max_nfev=200,tol=1e-10,linear_solver='exact'),
    no_formal_risk_threshold=True,independent_base_histories=3)

def hashes():
    names=['ev_profile_milp_study.py','ev_profile_ac_proxy.py','ev_profile_chord_milp.py',
        'ev_profile_milp_protocol.md','likelihood.py','simulation.py','candidates.py',
        'symmetric_ac_local.py','information_study.py','information_context.py',
        'gradient_price_study.py','ev_service_milp.py','ev_flexibility_study.py',
        'single_site_milp.py']
    result=old.source_hashes()
    result.update({n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names})
    return result

def bounds(candidate,floor):
    internal=np.array([1<len(s)<candidate.n for s in candidate.supports])
    return dict(r_bounds=(np.where(internal,floor,1e-6),np.full(candidate.m,.1)),
        x_bounds=(np.where(internal,max(1e-6,2*floor/3),1e-6),np.full(candidate.m,.1)))

def fit_all(candidates,obs,meter,floor,seed,initial=None):
    fits={};failed={};cost=dict(ac_calls=0,fit_calls=0,seconds=0.)
    for k,c in enumerate(candidates):
        f,e,t=io.fit_all_candidates([c],obs,meter,initial=initial,seed=seed+k,
            fit_options={**CONFIG['fit_options'],**bounds(c,floor)})
        fits.update(f);failed.update(e)
        for key in cost:cost[key]+=t[key]
    return fits,failed,cost

def plan(u):
    u=np.asarray(u,float)
    if np.all(u==0):
        return SingleSitePlan(0,np.zeros((4,3)),np.zeros((4,3)),np.zeros(4,int),
            np.full(5,.4),dict(no_EV_diagnostic=True))
    return canonical_service_plan(0,u/BASE_KW,3,device())

def score(m,u):
    e=m['d']+m['T']@np.asarray(u)
    return .5*(float(e@e)-float(m['offset']))

def select_curves(history,candidates,fits,hmodel,fmodel,floor,*,refine=False):
    began=perf_counter();constant=np.full(4,3.)
    choices={'none':np.zeros(4),'constant':constant,'profile_milp':constant,'fixed_mle_milp':constant}
    record=dict(uses_truth=False,uses_future_noise=False,site=0,source_same_interface=True,
        operation='continuous0to6kW;sum12;four15min;terminal26.85kWh',solves={},fallback=False)
    if len(fits)!=len(candidates) or any(not f.success for f in fits.values()):
        record.update(fallback=True,reason='history_fit_failure',seconds=perf_counter()-began)
        return choices,record,None
    ordered=sorted(candidates,key=lambda c:(fits[c.name].loss,c.name))
    gap=.5*(fits[ordered[1].name].loss-fits[ordered[0].name].loss)
    record['history_half_sse_gap']=gap
    record['activation_would_occur']=gap<=CONFIG['history_half_sse_gate']
    # Diagnostic runs also inspect already-separated H, separately labelled.
    try:
        context=build_pair_models(history,candidates,fits,hmodel,fmodel,site=0,horizon=4,
            reference_kw=constant,floor_r=floor,base_kw=BASE_KW)
        if not context['success']:raise ValueError(str(context.get('diagnostics',context)))
        record['context_diagnostics']=context['diagnostics']
        record['pairs']=[dict(a=m['a'],b=m['b']) for m in context['models']]
        eq=LinearConstraint(np.ones((1,4)),12.,12.)
        for method,key in [('profile_milp','models'),('fixed_mle_milp','fixed_models')]:
            qs=[FrozenQuadratic(m.get('name',str(i)),np.sqrt(.5)*m['d'],np.sqrt(.5)*m['T'],.5*m['offset'])
                for i,m in enumerate(context[key])]
            out=solve_profile_chord_milp(qs,np.zeros(4),np.full(4,6.),eq,
                segments=CONFIG['segments'],time_limit=CONFIG['milp_seconds'],feasible_incumbents=[constant])
            record['solves'][method]=out
            if out.get('u') is not None and out.get('physical_validation',{}).get('feasible',False):
                choices[method]=np.asarray(out['u'])
            else:record.setdefault('failures',{})[method]='MILP_no_valid_incumbent'
            if refine and method=='profile_milp':
                record['refinement_segments8']=solve_profile_chord_milp(qs,np.zeros(4),np.full(4,6.),eq,
                    segments=8,time_limit=CONFIG['milp_seconds'],feasible_incumbents=[constant,choices[method]])
        record['curve_scores']={name:dict(profile=min(score(m,u) for m in context['models']),
            fixed=min(score(m,u) for m in context['fixed_models'])) for name,u in choices.items()}
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
        record.update(fallback=True,reason=type(e).__name__+':'+str(e));context=None
    record['seconds']=perf_counter()-began
    return choices,record,context

def mean_observation(history,fit,candidate,hmodel,fmodel,u):
    baseline=baseline_from_history(history,hmodel,4)
    p=baseline['p'].copy();p[:,0]+=np.asarray(u)/BASE_KW
    phy=lk.ac_predict(candidate,fit.r,fit.x,p,baseline['q'],
        np.full(4,np.mean(fit.source_p)),np.full(4,np.mean(fit.source_q)),baseline['root_v'])
    values={k:np.asarray(phy[k])+ (np.asarray(getattr(fmodel,'bias_'+k)) if k in ('p','q','v') else 0.)
            for k in ('p','q','v','master_p','master_q')}
    return lk.Observations(root_v=baseline['root_v'],**values)

def local_audit(context,choices,history,candidates,fits,hm,fm,jm,floor,seed,full):
    rows=[];byname={c.name:c for c in candidates};calls=0
    for method in ('constant','profile_milp','fixed_mle_milp'):
        u=choices[method]
        for k,m in enumerate(context['models']):
            row=dict(method=method,a=m['a'],b=m['b'],free_score=score(m,u))
            try:
                audit=bounded_audit(m,u);own=bounded_audit(m['self_model'],u)
                row['bounded']=audit;row['bounded_self']=own
                row['bounded_score_upper']=.5*(audit['primal_upper']-own['tangent_lower_numeric'])
                row['bounded_score_lower']=.5*(audit['tangent_lower_numeric']-own['primal_upper'])
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
                row['bounded_failure']=str(e)
            # Independent nonlinear forward mean versus the frozen affine source response.
            predicted=mean_observation(history,fits[m['a']],byname[m['a']],hm,fm,u);calls+=1
            ga=m['geometry_a']['future'];jb=ga['state_jacobian']
            G=jb[:,np.arange(4)*(2*history.n+2)]/BASE_KW
            exact=whiten_observations(predicted,fm)
            affine=ga['mean']+G@(u-np.full(4,3.))
            response=np.linalg.norm(exact-ga['mean'])
            row['source_forward_linear_error_norm']=float(np.linalg.norm(exact-affine))
            row['source_forward_response_norm']=float(response)
            row['source_forward_relative_linear_error']=float(np.linalg.norm(exact-affine)/response) if response>1e-10 else None
            if full:
                joint=lk.concat_observations(history,predicted)
                cf,failed,cost=fit_all([byname[m['a']],byname[m['b']]],joint,jm,floor,seed+500+k,initial=fits)
                calls+=cost['ac_calls']
                row.update(predicted_joint_fits=[io.fit_record(byname[n],f) for n,f in cf.items()],
                    predicted_joint_failures=failed,predicted_joint_cost=cost,
                    predicted_observations=io.observations_record(predicted))
                if not failed and len(cf)==2:
                    row['nonlinear_predicted_margin_half_sse']=.5*(cf[m['b']].loss-cf[m['a']].loss)
            rows.append(row)
    return rows,calls

def run_condition(seed,family,rlevel,floor):
    began=perf_counter();gen=old.configuration()
    path,seeds,_,_,permutation,white,amplitude=old.make_path(seed,gen,
        case='star' if not family else 'moderate',truth_family=family,quality='temporal')
    internal=np.array([1<len(s)<3 for s in path.candidate.supports])
    path.r[internal]=rlevel;path.x[internal]=2*rlevel/3
    history,_=observe(path,slice(0,12));hm=model(3)
    candidates,proposal=propose(history,run_wzzt=True,wzzt_time_limit=.3)
    fits,failed,hcost=fit_all(candidates,history,hm,floor,seeds['policy']+10)
    fm,jm,quality=quality_models(history,hm,kind='temporal',row_permutation=permutation)
    choices,selection,context=select_curves(history,candidates,fits,hm,fm,floor,
        refine=seed==986101 and floor==1e-6 and rlevel==.0003)
    truth_family=normalize_family(path.candidate.supports,3)
    true_name=next(c.name for c in candidates if normalize_family(c.supports,3)==truth_family)
    tb=bounds(path.candidate,floor)
    covered=bool(np.all(path.r>=tb['r_bounds'][0]) and np.all(path.x>=tb['x_bounds'][0]))
    rows=[];cache={};evalcalls=0
    for method,u in choices.items():
        ev=plan(u);valid=dict(feasible=True,no_EV_diagnostic=True) if method=='none' else check_service_plan(ev,device())
        row=dict(method=method,u_kw=u,validation=valid,truth_in_parameter_domain=covered,true_name=true_name)
        if not valid['feasible']:
            row['status']='operation_failure';rows.append(row);continue
        try:
            obs,phy=old.future_observe(path,ev,fm,white)
            row['voltage_range']=[float(np.min(abs(phy['voltage']))),float(np.max(abs(phy['voltage'])))]
            if row['voltage_range'][0]<.9 or row['voltage_range'][1]>1.1:
                row['status']='physical_failure';rows.append(row);continue
            key=np.asarray(u).tobytes();hit=key in cache
            if not hit:
                cache[key]=fit_all(candidates,lk.concat_observations(history,obs),jm,floor,seeds['policy']+200,initial=fits)
            jf,jfail,jcost=cache[key];evalcalls+=0 if hit else jcost['ac_calls']
            row.update(status='fit_failure' if jfail or len(jf)!=len(candidates) else 'complete',
                fits=[io.fit_record(c,jf[c.name]) for c in candidates if c.name in jf],
                fit_failures=jfail,cost=jcost,cache_hit=hit,observations=io.observations_record(obs),
                physical={k:phy[k] for k in ('p','q','v','master_p','master_q')})
            if row['status']=='complete':
                losses={n:float(f.loss) for n,f in jf.items()}
                row.update(losses=losses,winner=min(losses,key=losses.get),
                    margin=min(v for n,v in losses.items() if n!=true_name)-losses[true_name],
                    impedance_rmse=float(np.sqrt(np.mean(np.r_[(jf[true_name].r-path.r)**2,(jf[true_name].x-path.x)**2]))))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
            row.update(status='physical_or_fit_exception',error=type(e).__name__+':'+str(e))
        rows.append(row)
    audit=[];auditcalls=0
    if context:
        audit,auditcalls=local_audit(context,choices,history,candidates,fits,hm,fm,jm,floor,seeds['policy'],seed==986101)
    return dict(seed=seed,family=family,truth_r=rlevel,fit_floor=floor,config=CONFIG,source_sha256=hashes(),
        history=io.observations_record(history),historical_fits=[io.fit_record(c,fits[c.name]) for c in candidates if c.name in fits],
        historical_failures=failed,historical_cost=hcost,truth=truth_metadata(path),truth_in_parameter_domain=covered,
        seeds=seeds,white_future=white,proposal=proposal,quality=quality,
        measurement_models=dict(history=meter_to_record(hm),future=meter_to_record(fm),joint=meter_to_record(jm)),
        selection=selection,proxy_models=context,rows=rows,audit=audit,seconds=perf_counter()-began,
        counted_history_ac=hcost['ac_calls'],counted_evaluation_ac=evalcalls,counted_audit_ac=auditcalls,
        proxy_ac=context.get('diagnostics',{}).get('ac_calls') if context else None,
        stage='development_no_risk_calibration',old_threshold_applied=False,
        independent_unit='base_seed;scale_and_floor_repeated_conditions')

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=21);p.add_argument('--output',type=Path,default=HERE/'evidence'/'ev_profile_milp'/'v1');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=CONFIG,hashes=hashes(),stage='declared_before_runs')
    freeze=args.output/'freeze.json'
    if freeze.exists() and io.read_json(freeze)!=io.serial(manifest):raise ValueError('freeze mismatch; preserve old run')
    io.write_json(freeze,manifest)
    conditions=[(s,f,r,b) for s,f in [(986101,((0,1),)),(986102,((0,2),)),(986103,())]
        for r in (CONFIG['truth_internal_r'] if f else [0.]) for b in CONFIG['fit_internal_r_floors']]
    started=perf_counter();done=[]
    for s,f,r,b in conditions[:args.limit]:
        out=args.output/f'trial_{s}_r{r:g}_floor{b:g}.json.gz'
        if out.exists():record=io.read_json(out)
        else:
            record=run_condition(s,f,r,b);io.write_json(out,record,compressed=True)
        done.append(dict(file=out.name,seed=s,truth_r=r,floor=b,covered=record['truth_in_parameter_domain'],
            fallback=record['selection']['fallback'],seconds=record['seconds'],
            methods=[{k:x.get(k) for k in ('method','status','margin','winner')} for x in record['rows']]))
        io.write_json(args.output/'checkpoint.json',dict(completed=done,planned_conditions=21,base_seeds=3,
            source_sha256=hashes(),formal_calibration=0,formal_evaluation=0,whole_research_complete=False))
        print(json.dumps(io.serial(done[-1]),ensure_ascii=False),flush=True)
        if perf_counter()-started>1800:break

if __name__=='__main__':main()
