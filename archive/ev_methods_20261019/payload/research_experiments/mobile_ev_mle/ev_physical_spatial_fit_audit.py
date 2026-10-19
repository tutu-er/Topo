"""Two-history standard-parameter electrical-position AC-fit diagnostic.

All four predeclared scenarios are evaluated. No site/curve selector exists.
Noise is shared within an independent history for paired comparisons, and
truth enters only generation and offline diagnostics. Operations and mobility
costs are examined separately in ev_ev_ess_spatial_audit.py.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from .ev_physical_cases import case33_local_standard,case33_local_record
from . import likelihood as lk
from .simulation import model
from .candidates import RadialCandidate
from .gradient_price_study import fit_all_candidates,observations_record
from .information_context import meter_to_record

HERE=Path(__file__).resolve().parent
CONFIG=dict(version='standard_case33_spatial_fit_diagnostic_v1',seeds=[987201,987202],
    history_count=12,future_count=4,base_kw=1000.,base_mva=1.,base_kv=12.66,
    p_q_common_relative_halfwidth=.01,p_q_independent_relative_halfwidth=.001,
    source_p_kw_center=10.,source_p_kw_halfwidth=2.,source_q_kvar_center=3.,source_q_kvar_halfwidth=1.,
    curve_kw=[6.,0.,6.,0.],grid_energy_kwh=3.,dt_hours=.25,
    p_sigma_pu=.0006,q_sigma_pu=.0004,v_sigma_pu=.00012,master_p_sigma_pu=.001,master_q_sigma_pu=.001,
    fit=dict(multistart=2,max_nfev=200,tol=1e-10,linear_solver='exact'),
    scenarios=['none','site0','site1','site2'],candidate_bank=['star','clade_01','clade_02','clade_12'],
    truth_family=[[0,2]],source_interface='same physical node1 for generation and every fit',
    field_PQV_data=False,formal_calibration=0,formal_evaluation=0,selector_calls=0,
    operation_scope='electrical response only; equal 3kWh grid energy, no claim of mobile terminal-SOC feasibility')


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def hashes():
    names=('ev_physical_spatial_fit_audit.py','ev_physical_spatial_fit_protocol.md','ev_physical_cases.py',
        'likelihood.py','simulation.py','candidates.py','gradient_price_study.py','information_context.py')
    result={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
    root=HERE.parents[1]
    path=root/'terminal_case33'/'data'/'case33bw_raw.py'
    result['../../terminal_case33/data/case33bw_raw.py']=hashlib.sha256(path.read_bytes()).hexdigest()
    legacy=HERE.parent/'mobile_ev_probing'/'ac_general_v9.py'
    result['../mobile_ev_probing/ac_general_v9.py']=hashlib.sha256(legacy.read_bytes()).hexdigest()
    return result


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(serial(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)


def read(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def candidates():
    return tuple(RadialCandidate.from_supports(name,3,family) for name,family in (
        ('star',()),('clade_01',((0,1),)),('clade_02',((0,2),)),('clade_12',((1,2),))))


def generated_history(seed):
    case=case33_local_standard(); total=CONFIG['history_count']+CONFIG['future_count']
    streams=np.random.SeedSequence(seed).spawn(4)
    seeds={name:int(stream.generate_state(1)[0]) for name,stream in zip(('background','source','noise','optimizer'),streams)}
    rng=np.random.default_rng(seeds['background']);common=rng.uniform(-.01,.01,(total,1))
    p=case.p0[None,:]*(1+common+rng.uniform(-.001,.001,(total,3)))
    q=case.q0[None,:]*(1+common+rng.uniform(-.001,.001,(total,3)))
    source_rng=np.random.default_rng(seeds['source'])
    sp=(10+source_rng.uniform(-2,2,total))/case.base_kw
    sq=(3+source_rng.uniform(-1,1,total))/case.base_kw
    root=np.ones(total); meter=model(3); noise_rng=np.random.default_rng(seeds['noise'])
    noise={channel:noise_rng.standard_normal((total,3) if channel in ('p','q','v') else total)
        for channel in ('p','q','v','master_p','master_q')}
    physical=lk.ac_predict(case.candidate,case.r,case.x,p,q,sp,sq,root)
    history=observed(physical,meter,noise,slice(0,CONFIG['history_count']))
    return dict(case=case,meter=meter,history=history,p=p,q=q,source_p=sp,source_q=sq,root_v=root,
        noise=noise,seeds=seeds,ac_simulation_calls=1,
        h_correlation_p=np.corrcoef(p[:CONFIG['history_count']].T),
        h_correlation_q=np.corrcoef(q[:CONFIG['history_count']].T))


def observed(physical,meter,noise,time_slice):
    fields={}
    for channel in ('p','q','v','master_p','master_q'):
        value=np.asarray(physical[channel])[time_slice]
        sigma=getattr(meter,channel+'_sigma');bias=getattr(meter,'bias_'+channel) if channel in ('p','q','v') else 0.
        fields[channel]=value+bias+sigma*np.asarray(noise[channel])[time_slice]
    return lk.Observations(**fields,root_v=np.asarray(physical['root_v'])[time_slice])


def fit_record(candidate,fit,case):
    result=dict(name=candidate.name,supports=[list(s) for s in candidate.supports],**fit.summary(),
        half_loss=.5*float(fit.loss),r_ohm=fit.r*case.zbase_ohm,x_ohm=fit.x*case.zbase_ohm,
        p=fit.p,q=fit.q,source_p=fit.source_p,source_q=fit.source_q,
        predicted={key:fit.predicted[key] for key in ('p','q','v','master_p','master_q') if key in fit.predicted},
        global_MLE_certified=False)
    return result


def fit_metrics(fits,bank,generated,truth_p,truth_q,truth_sp,truth_sq):
    valid={name:f for name,f in fits.items() if f.success and np.isfinite(f.loss)}
    truth=valid.get('clade_02'); wrong=[f for name,f in valid.items() if name!='clade_02']
    complete=len(valid)==len(bank)
    ranking=sorted(valid,key=lambda name:(valid[name].loss,name))
    answer=dict(all_required_fits_success=complete,successful_candidates=len(valid),candidate_denominator=len(bank),
        candidate_ranking=ranking,best_candidate=None if not ranking else ranking[0],
        truth_in_predeclared_bank=True,full_SSE_signed_margin=None,half_SSE_loglik_margin=None,
        source_error_by_candidate={})
    if complete and truth is not None:
        closest=min(wrong,key=lambda f:f.loss)
        margin=float(closest.loss-truth.loss)
        answer.update(full_SSE_signed_margin=margin,half_SSE_loglik_margin=.5*margin,
            actual_loglik_margin=float(truth.loglik-closest.loglik),
            loglik_margin_identity_error=abs(float(truth.loglik-closest.loglik)-.5*margin))
    case=generated['case']; h=CONFIG['history_count']
    for name,fit in fits.items():
        if not np.isfinite(fit.source_p).all():continue
        sp=(fit.source_p-truth_sp)*case.base_kw;sq=(fit.source_q-truth_sq)*case.base_kw
        answer['source_error_by_candidate'][name]=dict(p_rmse_kw=float(np.sqrt(np.mean(sp*sp))),
            q_rmse_kvar=float(np.sqrt(np.mean(sq*sq))),
            H_p_rmse_kw=float(np.sqrt(np.mean(sp[:h]**2))),H_q_rmse_kvar=float(np.sqrt(np.mean(sq[:h]**2))),
            N_p_rmse_kw=None if len(sp)<=h else float(np.sqrt(np.mean(sp[h:]**2))),
            N_q_rmse_kvar=None if len(sq)<=h else float(np.sqrt(np.mean(sq[h:]**2))))
    if truth is not None:
        # Truth supports and line order coincide only for this candidate.
        dr=truth.r-case.r;dx=truth.x-case.x
        answer['true_candidate_impedance_error']=dict(r_rmse_pu=float(np.sqrt(np.mean(dr**2))),
            x_rmse_pu=float(np.sqrt(np.mean(dx**2))),r_rmse_ohm=float(np.sqrt(np.mean(dr**2))*case.zbase_ohm),
            x_rmse_ohm=float(np.sqrt(np.mean(dx**2))*case.zbase_ohm),
            r_relative_L2=float(np.linalg.norm(dr)/np.linalg.norm(case.r)),
            x_relative_L2=float(np.linalg.norm(dx)/np.linalg.norm(case.x)),
            edge_R_error_ohm=dr*case.zbase_ohm,edge_X_error_ohm=dx*case.zbase_ohm,
            terminal_p_rmse_kw=float(np.sqrt(np.mean((truth.p-truth_p)**2))*case.base_kw),
            terminal_q_rmse_kvar=float(np.sqrt(np.mean((truth.q-truth_q)**2))*case.base_kw))
    return answer


def run_history(seed):
    began=perf_counter();before=hashes(); generated=generated_history(seed);bank=candidates()
    case=generated['case'];meter=generated['meter'];h=CONFIG['history_count'];t=CONFIG['future_count']
    fits,failed,cost=fit_all_candidates(bank,generated['history'],meter,
        seed=generated['seeds']['optimizer'],fit_options=CONFIG['fit'])
    historical_metrics=fit_metrics(fits,bank,generated,generated['p'][:h],generated['q'][:h],
        generated['source_p'][:h],generated['source_q'][:h])
    rows=[];simulation_calls=1
    for index,scenario in enumerate(CONFIG['scenarios']):
        site=None if scenario=='none' else int(scenario[-1]);u=np.zeros(t) if site is None else np.asarray(CONFIG['curve_kw'])
        npower=generated['p'][h:].copy()
        if site is not None:npower[:,site]+=u/case.base_kw
        physical=lk.ac_predict(case.candidate,case.r,case.x,npower,generated['q'][h:],
            generated['source_p'][h:],generated['source_q'][h:],generated['root_v'][h:]);simulation_calls+=1
        # observed() indexes a full-time noise array; this future physical block
        # already has four rows and needs matching tail errors separately.
        noise_tail={key:value[h:] for key,value in generated['noise'].items()}
        future=observed(physical,meter,noise_tail,slice(None)); joint=lk.concat_observations(generated['history'],future)
        jointfits,joint_failed,joint_cost=fit_all_candidates(bank,joint,meter,initial=fits,
            seed=generated['seeds']['optimizer']+100,fit_options=CONFIG['fit'])
        truth_p=np.vstack((generated['p'][:h],npower))
        metrics=fit_metrics(jointfits,bank,generated,truth_p,generated['q'],generated['source_p'],generated['source_q'])
        row=dict(scenario=scenario,site=site,u_kw=u,grid_energy_kwh=.25*float(u.sum()),
            future=observations_record(future),joint_observations=observations_record(joint),
            fits=[fit_record(c,jointfits[c.name],case) for c in bank if c.name in jointfits],
            failures=joint_failed,cost=joint_cost,metrics=metrics,
            deterministic_future_v=physical['v'],deterministic_root_p_kw=physical['master_p']*case.base_kw,
            simulation_AC_equation_residual=physical['ac_equation_residual'],
            simulation_AC_power_balance_residual=physical['power_balance_residual'],
            power_includes_EV_once=True,mobile_service_feasibility_not_evaluated=True)
        rows.append(row)
    if hashes()!=before:raise ValueError('frozen diagnostic source changed during run')
    return dict(schema='two_H_physical_scale_position_diagnostic_v1',seed=seed,config=CONFIG,
        source_sha256=before,case=case33_local_record(),history=observations_record(generated['history']),
        historical_fits=[fit_record(c,fits[c.name],case) for c in bank if c.name in fits],
        historical_failures=failed,historical_metrics=historical_metrics,historical_cost=cost,
        meter=meter_to_record(meter),seeds=generated['seeds'],
        generated_truth=dict(p=generated['p'],q=generated['q'],source_p=generated['source_p'],source_q=generated['source_q']),
        historical_correlation_p=generated['h_correlation_p'],historical_correlation_q=generated['h_correlation_q'],
        common_standardized_noise=generated['noise'],rows=rows,
        total_candidate_fit_calls=cost['fit_calls']+sum(r['cost']['fit_calls'] for r in rows),
        counted_fit_AC_calls=cost['ac_calls']+sum(r['cost']['ac_calls'] for r in rows),
        counted_simulation_AC_calls=simulation_calls,seconds=perf_counter()-began,
        independent_unit='one independently generated history; four scenario futures are paired repeats',
        global_MLE_certified=False,formal_error_rates_estimated=False,selector_calls=0,
        impedance_lower_bound='existing numerical positivity only, not new physical asset floor')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=CONFIG['seeds']);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);source=hashes()
    freeze=args.output/'freeze.json'
    if freeze.exists():
        old=read(freeze)
        if old['source_sha256']!=source or old['config']!=CONFIG:raise ValueError('existing freeze differs')
    else:write(freeze,dict(config=CONFIG,source_sha256=source,seed_plan=CONFIG['seeds'],scope='two-history electrical mechanism diagnostic'))
    for seed in args.seeds:
        if seed not in CONFIG['seeds']:raise ValueError('seed not predeclared')
        target=args.output/f'history_{seed}.json.gz'
        if target.exists():raise ValueError('completed frozen history already exists; do not overwrite')
        result=run_history(seed);write(target,result)
        print(json.dumps(dict(seed=seed,seconds=result['seconds'],fit_calls=result['total_candidate_fit_calls'],
            AC_calls=result['counted_fit_AC_calls'],historical_margin=result['historical_metrics']['full_SSE_signed_margin'],
            scenario_margins={r['scenario']:r['metrics']['full_SSE_signed_margin'] for r in result['rows']},
            failures={'H':result['historical_failures'],**{r['scenario']:r['failures'] for r in result['rows']}}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
