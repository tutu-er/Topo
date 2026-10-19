"""H-only continuous all-candidate EV design; independent AC development.

This stage preserves old frozen modules and data. Truth/noise occur only in
prepare/evaluate, never select_curves. Proxy bounds are for a common outer
bilinear relaxation; original action-specific boxes and full AC are separate.
"""
from __future__ import annotations
import argparse, base64, gzip, hashlib, json
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from . import likelihood as lk
from . import information_study as frozen
from . import gradient_price_study as io
from .simulation import model, observe, truth_metadata, BASE_KW
from .candidates import propose, normalize_family
from .information_context import quality_models, meter_to_record
from .ev_flexibility_study import device
from .ev_service_milp import canonical_service_plan, check_service_plan
from .single_site_milp import SingleSitePlan
from .ev_profile_ac_proxy import build_pair_models
from .ev_ac_moving_cross import build_from_saved, fixed_action_qp
from .ev_bilinear_support_milp import BilinearPair
from .ev_bilinear_score_exchange import exchange_bilinear_scores, charging_service_domain
from .ev_profile_chord_milp import FrozenQuadratic, solve_profile_chord_milp
from .ev_ac_design_guard import guard_predicted_actions

HERE=Path(__file__).resolve().parent
CASES=(((),True),((),False),(((0,1),),True),(((0,1),),False),
       (((0,2),),True),(((0,2),),False),(((1,2),),True),(((1,2),),False))
CONFIG=dict(version='moving_cross_all_pairs_development_v1',seeds=list(range(987101,987109)),
    quality=['uniform','temporal'],history=12,horizon=4,site=0,base_kw=20.,
    high_correlation_innovation=.00025,low_correlation_innovation=.012,
    internal_r=.003,internal_x=.002,line_lognormal_sigma=.10,numeric_positive_floor=1e-6,
    future_correlation_regime='same as historical generator; observed EV can break it',
    impedance_floor_claim='numerical positivity only; no field engineering minimum',
    power_kw=6.,grid_energy_kwh=3.,terminal_battery_kwh=26.85,eta=.95,
    all_ordered_pairs=12,score='half of proxy joint cost minus attained competitor H cost',
    proxy_domain='common outer box; original action-domain separately audited',
    exchange=dict(max_iterations=4,segments=4,max_segments=16,solver_seconds=2.,
                  total_seconds=10.,absolute_gap=.01,relative_gap=.01),
    guard='two H-only full-AC predicted actions; raw MILP retained as diagnostic',
    fit=dict(multistart=2,max_nfev=200,tol=1e-10,linear_solver='exact'),
    development_only=True,formal_calibration=0,formal_evaluation=0,
    old_threshold_applied=False,correction_strict_gate_passed='record separately, not assumed')
METHODS=('none','constant','quality_lp','random_price_lp','fixed_mle_milp','cross_raw_milp','cross_guarded_milp')

def source_hashes():
    names=['ev_cross_all_study.py','ev_ac_moving_cross.py','ev_bilinear_score_exchange.py',
        'ev_bilinear_support_milp.py','ev_bilinear_exchange.py','ev_profile_chord_milp.py',
        'ev_profile_ac_proxy.py','ev_service_milp.py','ev_flexibility_study.py',
        'symmetric_ac_local.py','information_context.py','ev_cross_execution_protocol.md',
        'ev_ac_design_guard.py','ev_cross_pilot_protocol.md']
    result=frozen.source_hashes()
    result.update({n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names})
    return result

def make_development_path(seed,family,high_correlation):
    cfg=frozen.configuration()
    cfg['line_lognormal_sigma']=CONFIG['line_lognormal_sigma']
    cfg['population']['internal_r_pu']={**cfg['population']['internal_r_pu'],'moderate':CONFIG['internal_r']}
    cfg['population']['internal_x_pu']={**cfg['population']['internal_x_pu'],'moderate':CONFIG['internal_x']}
    path,seeds,case,quality,order,white,amp=frozen.make_path(seed,cfg,
        case='star' if not family else 'moderate',truth_family=family,quality='uniform')
    rng=np.random.default_rng(seeds['generator']^0x5EC3A)
    amplitude=CONFIG['high_correlation_innovation' if high_correlation else 'low_correlation_innovation']
    times=len(path.p);common=rng.uniform(-.012,.012,(times,1))
    path.p[:]=.12+common+rng.uniform(-amplitude,amplitude,(times,3))
    path.q[:]=.04+.25*common+rng.uniform(-.5*amplitude,.5*amplitude,(times,3))
    return path,seeds,order,white,dict(high_correlation=high_correlation,
        innovation_amplitude_pu=amplitude,p_correlation=np.corrcoef(path.p[:12].T),
        q_correlation=np.corrcoef(path.q[:12].T),future_background_correlation_p=np.corrcoef(path.p[12:].T),
        future_background_correlation_q=np.corrcoef(path.q[12:].T),
        population_correlation_p=.012**2/(.012**2+amplitude**2),
        population_correlation_q=.012**2/(.012**2+4*amplitude**2),
        historical_and_future_same_regime=True,standard_or_measured_data=False)

def prepare(seed,family,correlation):
    path,seeds,order,white,correcord=make_development_path(seed,family,correlation)
    history,_=observe(path,slice(0,12));hm=model(3)
    candidates,proposal=propose(history,run_wzzt=True,wzzt_time_limit=.3)
    fits,failed,cost=io.fit_all_candidates(candidates,history,hm,seed=seeds['policy']+10,
        fit_options=CONFIG['fit'])
    return dict(path=path,seeds=seeds,order=order,white=white,correlation=correcord,
        history=history,hm=hm,candidates=candidates,fits=fits,failed=failed,
        historical_cost=cost,proposal=proposal)

def linear_curve(prices,lower,upper,domain):
    # All SOC lower/upper rows preserved, including fixed service equality.
    A=np.asarray(domain.A);lo=np.asarray(domain.lb);hi=np.asarray(domain.ub)
    exact=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
    up=np.isfinite(hi)&~exact;down=np.isfinite(lo)&~exact
    result=linprog(-np.asarray(prices),A_ub=np.vstack((A[up],-A[down])),
        b_ub=np.r_[hi[up],-lo[down]],A_eq=A[exact],b_eq=hi[exact],
        bounds=list(zip(lower,upper)),method='highs')
    return dict(success=bool(result.success),u=None if result.x is None else result.x,
        status=int(result.status),message=str(result.message),
        objective=None if result.fun is None else -float(result.fun))

def build_all_context(history,candidates,fits,hm,fm,*,site,lower,upper,base_kw=BASE_KW):
    old_models=[];fixed=[];moving=[];pairs=[];calls=0
    byname={c.name:c for c in candidates}
    for a,b in combinations(sorted(candidates,key=lambda c:c.name),2):
        part=build_pair_models(history,[a,b],fits,hm,fm,site=site,base_kw=base_kw)
        if not part['success']:raise ValueError('required pair context:'+str(part['diagnostics']))
        calls+=part['diagnostics']['ac_calls']
        old_models.extend(part['models']);fixed.extend(part['fixed_models'])
        for old in part['models']:
            converted=build_from_saved(old,byname[old['b']],fm,action_lower=lower,action_upper=upper)
            calls+=converted['mixed_derivatives']['ac_derivative_calls']
            converted['name']=old['name'];moving.append(converted)
            d=converted['domain']
            pairs.append(BilinearPair(old['name'],converted['c'],converted['B'],converted['A0'],
                converted['At'],d['outer_lower'],d['outer_upper']))
    required=len(candidates)*(len(candidates)-1)
    if len(pairs)!=required or len({p.name for p in pairs})!=required:
        raise ValueError('all ordered competitors required')
    return dict(old_models=old_models,fixed_models=fixed,moving_models=moving,pairs=pairs,ac_calls=calls)

def select_curves(history,candidates,fits,hm,fm,quality,*,policy_seed,joint_model,site=0,service=None):
    """No simulated path, true candidate or future noise accepted."""
    began=perf_counter();service=device() if service is None else service
    domain=charging_service_domain(site,4,history.n,service,power_scale=BASE_KW)
    constant=domain.get('constant_u'); choices={m:constant for m in METHODS};choices['none']=np.zeros(4)
    record=dict(uses_truth=False,uses_future_noise=False,site=site,all_pairs_required=12,
        score_mode='added',numeric_floor_only=True,domain={k:v for k,v in domain.items()
        if k not in ('constraints','constant_plan')},context_ac_calls=0,failures={},fallback=False)
    if not domain['success'] or constant is None:
        record.update(fallback=True,reason='operation_domain_or_constant_failure',seconds=perf_counter()-began)
        return choices,record,None
    lo,hi,equality=domain['lower'],domain['upper'],domain['constraints']
    for method,prices in [('quality_lp',np.asarray(quality['v_repeat_counts'])[:,site]),
        ('random_price_lp',np.random.default_rng(policy_seed+908).standard_normal(4))]:
        solved=linear_curve(prices,lo,hi,equality);record[method]=solved
        if solved['success']:choices[method]=solved['u']
        else:record['failures'][method]='LP_failure'
    if len(fits)!=len(candidates) or any(not f.success for f in fits.values()):
        record.update(fallback=True,reason='history_fit_failure',seconds=perf_counter()-began)
        return choices,record,None
    ranking=sorted(candidates,key=lambda c:(fits[c.name].loss,c.name))
    record.update(history_ranking=[dict(name=c.name,loss=float(fits[c.name].loss)) for c in ranking],
        activation_would_occur=.5*(fits[ranking[1].name].loss-fits[ranking[0].name].loss)<=6.)
    context=None
    try:
        context=build_all_context(history,candidates,fits,hm,fm,site=site,lower=lo,upper=hi)
        record['context_ac_calls']=context['ac_calls'];record['pairs']=[dict(a=m['a'],b=m['b']) for m in context['moving_models']]
        fixedqs=[FrozenQuadratic(m['name'],np.sqrt(.5)*m['d'],np.sqrt(.5)*m['T'],.5*fits[m['b']].loss)
            for m in context['fixed_models']]
        fixedout=solve_profile_chord_milp(fixedqs,lo,hi,equality,segments=4,
            time_limit=2.,feasible_incumbents=[constant]);record['fixed_solver']=fixedout
        if fixedout['u'] is not None:choices['fixed_mle_milp']=fixedout['u']
        else:record['failures']['fixed_mle_milp']='MILP_no_valid_curve'
        offsets={m['name']:float(fits[m['b']].loss) for m in context['moving_models']}
        out=exchange_bilinear_scores(context['pairs'],lo,hi,equality,offsets=offsets,
            initial_u=constant,**CONFIG['exchange']);record['cross_solver']=out
        if out['u'] is not None:choices['cross_raw_milp']=out['u']
        else:record['failures']['cross_raw_milp']='MILP_no_valid_curve'
        checked=guard_predicted_actions(history,candidates,fits,hm,fm,joint_model,
            baseline_u=constant,candidate_u=choices['cross_raw_milp'],site=site,base_kw=BASE_KW,
            fit_options=CONFIG['fit'],seed=policy_seed+801)
        record['ac_guard']=checked;choices['cross_guarded_milp']=checked['selected_u']
        record['curves_audit']={}
        for method in ('constant','quality_lp','fixed_mle_milp','cross_raw_milp'):
            pairrows=[];u=choices[method]
            for m in context['moving_models']:
                original=fixed_action_qp(m,u,domain='original')
                outer=fixed_action_qp(m,u,domain='outer')
                pairrows.append(dict(a=m['a'],b=m['b'],original=original,outer=outer,
                    added_score_interval=[.5*(original['tangent_lower_numeric']-fits[m['b']].loss),
                                          .5*(original['primal_upper']-fits[m['b']].loss)]))
            record['curves_audit'][method]=pairrows
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        record.update(fallback=True,reason=type(error).__name__+':'+str(error))
    record['seconds']=perf_counter()-began
    return choices,record,context

def evaluate(prepared,choices,fm,jm,*,site=0,service=None):
    service=device() if service is None else service
    history,path=prepared['history'],prepared['path'];candidates=prepared['candidates'];fits=prepared['fits']
    family=normalize_family(path.candidate.supports,3)
    truth_names=[c.name for c in candidates if normalize_family(c.supports,3)==family]
    rows=[];cache={};calls=0
    for method,u in choices.items():
        row=dict(method=method,u_kw=u,site=site,candidate_miss=len(truth_names)!=1)
        if u is None:row.update(status='operation_failure');rows.append(row);continue
        ev=canonical_service_plan(site,np.asarray(u)/BASE_KW,3,service)
        valid=dict(feasible=True,no_EV_diagnostic=True) if method=='none' else check_service_plan(ev,service)
        row['validation']=valid
        if not valid['feasible']:row.update(status='operation_failure');rows.append(row);continue
        try:
            obs,phy=frozen.future_observe(path,ev,fm,prepared['white'])
            row['voltage_range']=[float(np.min(abs(phy['voltage']))),float(np.max(abs(phy['voltage'])))]
            row.update(observations=io.observations_record(obs),
                physical={k:phy[k] for k in ('p','q','v','master_p','master_q')})
            if row['voltage_range'][0]<.9 or row['voltage_range'][1]>1.1:
                row.update(status='physical_voltage_failure');rows.append(row);continue
            key=ev.p.astype('<f8').tobytes()+ev.q.astype('<f8').tobytes();hit=key in cache
            if not hit:cache[key]=io.fit_all_candidates(candidates,lk.concat_observations(history,obs),jm,
                initial=fits,seed=prepared['seeds']['policy']+201,fit_options=CONFIG['fit'])
            jf,failed,cost=cache[key];calls+=0 if hit else cost['ac_calls']
            row.update(status='fit_failure' if failed or len(jf)!=len(candidates) else 'complete',
                cache_hit=hit,failures=failed,cost=cost,observations=io.observations_record(obs),
                physical={k:phy[k] for k in ('p','q','v','master_p','master_q')},
                fits=[io.fit_record(c,jf[c.name]) for c in candidates if c.name in jf])
            if row['status']=='complete':
                losses={n:float(f.loss) for n,f in jf.items()};row.update(losses=losses,winner=min(losses,key=losses.get))
                if len(truth_names)==1:
                    true=truth_names[0];row.update(true_name=true,
                        margin=min(v for n,v in losses.items() if n!=true)-losses[true],
                        loglik_margin=.5*(min(v for n,v in losses.items() if n!=true)-losses[true]),
                        impedance_rmse=float(np.sqrt(np.mean(np.r_[(jf[true].r-path.r)**2,(jf[true].x-path.x)**2]))),
                        source_state_rmse=float(np.sqrt(np.mean(np.r_[
                            (jf[true].source_p-path.source_p)**2,(jf[true].source_q-path.source_q)**2]))),
                        source_error_scope='offline evaluation of inferred source states, not independent source telemetry')
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            row.update(status='physical_or_fit_exception',error=str(error))
        rows.append(row)
    return rows,calls

def run_condition(prepared,quality):
    began=perf_counter();h,hm=prepared['history'],prepared['hm']
    fm,jm,qr=quality_models(h,hm,kind=quality,row_permutation=prepared['order'])
    choices,selection,context=select_curves(h,prepared['candidates'],prepared['fits'],hm,fm,qr,
        policy_seed=prepared['seeds']['policy'],joint_model=jm)
    rows,calls=evaluate(prepared,choices,fm,jm)
    # Save all moving matrices for exact proxy replay, but not redundant old pair arrays.
    return dict(base_seed=prepared['path'].seed,seeds=prepared['seeds'],correlation=prepared['correlation'],
        history=io.observations_record(h),historical_fits=[io.fit_record(c,prepared['fits'][c.name])
            for c in prepared['candidates'] if c.name in prepared['fits']],historical_failures=prepared['failed'],
        historical_cost=prepared['historical_cost'],proposal=prepared['proposal'],
        quality=qr,measurement_models=dict(history=meter_to_record(hm),future=meter_to_record(fm),joint=meter_to_record(jm)),
        selection=selection,moving_models=None if context is None else context['moving_models'],
        fixed_models=None if context is None else [{k:m[k] for k in ('name','a','b','d','T')} for m in context['fixed_models']],
        truth=truth_metadata(prepared['path']),white_future=prepared['white'],rows=rows,
        evaluation_ac_calls=calls,selection_ac_calls=selection['context_ac_calls']+
            selection.get('ac_guard',{}).get('totals',{}).get('ac_calls',0),
        selection_forecast_calls=selection.get('ac_guard',{}).get('totals',{}).get('forecast_calls',0),
        seconds=perf_counter()-began,development_only=True,independent_unit='base seed; quality repeats paired')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=16)
    parser.add_argument('--output',type=Path,default=HERE/'evidence/ev_cross_all/v1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();freeze=dict(config=CONFIG,source_sha256=hashes,stage='declared_before_new_trials')
    fpath=args.output/'freeze.json'
    if fpath.exists():
        if io.read_json(fpath)!=io.serial(freeze):raise ValueError('frozen version differs; choose new output')
    else:
        io.write_json(fpath,freeze)
        snapshot={n:dict(sha256=h,bytes_base64=base64.b64encode((HERE/n).read_bytes()).decode('ascii')) for n,h in hashes.items()}
        io.write_json(args.output/'source_snapshot.json.gz',dict(schema='source_bytes_base64',sources=snapshot),compressed=True)
    tasks=[(seed,*case,q) for seed,case in zip(CONFIG['seeds'],CASES) for q in CONFIG['quality']]
    prepared_cache={};count=0
    for seed,family,corr,q in tasks[:args.limit]:
        path=args.output/f'trial_{seed}_{q}.json.gz'
        if path.exists():
            saved=io.read_json(path)
            if saved['source_sha256']!=hashes or saved['config']!=io.serial(CONFIG):raise ValueError('resume source/config mismatch')
            count+=1;continue
        if source_hashes()!=hashes:raise ValueError('source changed during study')
        if seed not in prepared_cache:prepared_cache[seed]=prepare(seed,family,corr)
        result=run_condition(prepared_cache[seed],q)
        result.update(base_seed=seed,config=CONFIG,source_sha256=hashes)
        io.write_json(path,result,compressed=True);count+=1
        io.write_json(args.output/'checkpoint.json',dict(completed=count,planned=16,last=path.name,
            completed_independent_seeds=len({int(p.stem.split('_')[1]) for p in args.output.glob('trial_*.json.gz')}),
            research_complete=False,stage='development'))
        print(json.dumps(io.serial(dict(seed=seed,quality=q,seconds=result['seconds'],
            selection_seconds=result['selection']['seconds'],fallback=result['selection']['fallback'],
            margins={r['method']:r.get('margin') for r in result['rows']}))),flush=True)

if __name__=='__main__':main()
