"""New battery-service-fair development study; frozen studies remain unchanged.

Selectors receive H, H fits and public operations only. Full AC evaluation
has separate truth/noise inputs. No old calibration threshold is transferred.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from . import information_study as frozen
from . import gradient_price_study as io
from . import likelihood
from .simulation import observe, model, BASE_KW, truth_metadata
from .likelihood import concat_observations
from .candidates import propose, normalize_family
from .information_context import quality_models, meter_to_record
from .information_context_counter import build_centres
from .information_targeting import history_target_weights
from .structured_information import StructuredInformationEvaluator
from .single_site_milp import SingleSitePlan
from .ev_service_milp import (EVServiceConstraints, constant_site_plan,
    solve_site_step, check_service_plan, canonical_service_plan, optimize_service_curves, SearchSettings,
    search_record)

HERE=Path(__file__).resolve().parent
METHODS=('none','constant_fixed','quality_charge','charge_fixed','charge_random',
         'charge_mobile','v2g_fixed','v2g_mobile','charge_staff')
CASES=(('star',()),('moderate',((0,1),)),('strong',((0,2),)),
       ('moderate',((1,2),)),('weak',((0,1),)),('weak',((1,2),)))
CONFIG=dict(version='battery_service_development_v1',history_points=12,
    future_points=4,dt_hours=.25,base_kva=20.,service_battery_energy_pu_h=.1425,
    p_charge_max=.3,p_discharge_max=.3,eta_charge=.95,eta_discharge=.95,
    terminal_energy_pu_h=1.3425,setup_slots=0,gate_half_sse=6.,
    search=dict(max_iterations=4,max_starts=3,initial_trust_pu=.15,
        minimum_trust_pu=.003,gain_tolerance=1e-6,milp_time_limit=5.,gradient_mode='full'),
    evaluation_fit_options=dict(multistart=2,max_nfev=200,tol=1e-10,linear_solver='exact'),
    controller_seconds_gate=15.,condition_seconds_gate=90.,
    calibration_available=False,old_threshold_applied=False)

def save(path, value):
    io.write_json(path,value,compressed=Path(path).suffix=='.gz')

def source_hashes():
    hashes=frozen.source_hashes()
    for name in ('ev_flexibility_study.py','ev_service_milp.py'):
        hashes[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return hashes

def device(*,bidirectional=False,staff=False):
    access=np.ones((4,3),bool)
    if not staff: access[:2,2]=False
    return EVServiceConstraints(terminal_energy=CONFIG['terminal_energy_pu_h'],
        p_charge_max=.3,p_discharge_max=.3 if bidirectional else 0.,
        availability=access,travel_slots=np.ones((3,3),int)-np.eye(3,dtype=int),
        travel_energy=.01*(np.ones((3,3))-np.eye(3)),initial_site=0,
        setup_slots=0,eta_charge=.95,eta_discharge=.95)

def _plan_key(plan):
    return plan.p.astype('<f8').tobytes(),plan.q.astype('<f8').tobytes()

def _zero(history):
    return SingleSitePlan(0,np.zeros((4,history.n)),np.zeros((4,history.n)),
        np.zeros(4,int),np.full(5,.4),dict(method='no_EV_diagnostic'))

def select(history,candidates,fits,history_model,future_model,quality,*,seed):
    """Pure planning interface: no future physical path/truth/noise arguments."""
    began=perf_counter();charge=device();v2g=device(bidirectional=True);staff=device(staff=True)
    constraints={m:charge for m in METHODS};constraints['v2g_fixed']=constraints['v2g_mobile']=v2g
    constraints['charge_staff']=staff
    constant=constant_site_plan(0,4,history.n,charge)
    baseline=constant.get('plan');choices={m:baseline for m in METHODS};choices['none']=_zero(history)
    method={m:dict(status='constant_fallback',predicted_score=None) for m in METHODS}
    method['none']['status']='diagnostic';method['constant_fixed']['status']=constant['status']
    result=dict(methods=method,searches={},devices={k:asdict(v) for k,v in
        dict(charge=charge,v2g=v2g,staff=staff).items()},context={},centres=[],
        public_quality=quality,selector_uses_truth=False,selector_uses_future_noise=False,
        selector_ac_calls=0,selector_fit_calls=0,global_design_optimality_certified=False,
        staff_model='external_connection_windows;setup0 ideal;no_joint_crew_route')
    cheap=[]
    counts=np.asarray(quality['v_repeat_counts'],float)
    for j in range(history.n):
        # A deliberately cheap public-quality rule, identical battery service.
        price=counts.sum(axis=1)
        op=solve_site_step(j,4,history.n,charge,intercepts=[0.],
            gradients=price[None,:],time_limit=5.)
        cheap.append(dict(site=j,operation={k:v for k,v in op.items() if k!='plan'}))
        normalized=(op['incumbent']*.25/op['validation']['grid_net_pu_hours'] if op['success'] else None)
        cheap[-1]['average_public_quality_per_grid_energy']=normalized
        if op['success'] and (method['quality_charge']['predicted_score'] is None or
            normalized>method['quality_charge']['predicted_score']):
            choices['quality_charge']=op['plan'];method['quality_charge']=dict(
                status='public_quality_optimized',predicted_score=normalized,
                score_kind='public_average_repeat_count_per_charge_energy;not_AC_information',selected_site=j)
    result['cheap_operations']=cheap
    ranking=sorted([c for c in candidates if c.name in fits and fits[c.name].success
        and np.isfinite(fits[c.name].loss)],key=lambda c:(fits[c.name].loss,c.name))
    complete=len(ranking)==len(candidates) and len(ranking)>=2
    gap=.5*(fits[ranking[1].name].loss-fits[ranking[0].name].loss) if complete else None
    activated=bool(complete and gap<=CONFIG['gate_half_sse'])
    result.update(activated=activated,history_gap=gap)
    if not activated:
        result.update(inactive_reason='H_fit_failure' if not complete else 'H_already_separated',
            seconds=perf_counter()-began)
        return choices,constraints,result
    try:
        centres,context=build_centres(history,candidates,fits,history_model,horizon=4,
            fit_options=CONFIG['evaluation_fit_options'],gate_half_sse=6.,
            max_anchors=2,perturb_scale_pu=.003,seed=seed+400,retain_perturbed_points=True)
        result['context']=context;result['centres']=frozen.centres_record(centres)
        result['selector_fit_calls']+=context.get('fit_calls',0)
        result['selector_ac_calls']+=context.get('ac_calls',0)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        centres=[];result['context']=dict(success=False,error=str(error))
        result['selector_fit_calls']+=int(getattr(error,'fit_calls',0))
        result['selector_ac_calls']+=int(getattr(error,'ac_calls',0))
    weights=history_target_weights(candidates,fits,epsilon=.15,tau=1.)
    result['target_weights']=io.serial({k:v for k,v in weights.items() if k!='weights'})
    if not centres or not weights['success']:
        result.update(inactive_reason='no_H_supported_context_or_weights',seconds=perf_counter()-began)
        return choices,constraints,result
    evaluator=None
    try:
        evaluator=StructuredInformationEvaluator(history,centres,history_model,
            future_model=future_model,base_kva=BASE_KW,q_over_p=0.,
            weight=.5*np.diag([.003**2,.003**2]),fd_step_kw=.01,rcond=1e-10,
            target_weights=weights['weights'])
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        result['evaluator_failure']=str(error)
        result['selector_ac_calls']+=int(getattr(error,'ac_calls',0))
        result['seconds']=perf_counter()-began
        return choices,constraints,result
    result['selector_ac_calls']+=evaluator.initialization_ac_calls
    settings=SearchSettings(**CONFIG['search']);runs={}
    for tag,dev in (('charge',charge),('v2g',v2g),('staff',staff)):
        start=perf_counter();before=evaluator._ac_calls
        try:
            run=optimize_service_curves(evaluator,dev,settings=settings)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            run=dict(success=False,error=str(error),plan=None,site_plans={},
                site_evaluations={},total_ac_calls=evaluator._ac_calls-before,
                seconds=perf_counter()-start)
        runs[tag]=run;result['searches'][tag]=search_record(run)
        result['selector_ac_calls']+=int(run.get('total_ac_calls',0))
    random_site=int(np.random.default_rng(seed+903).integers(history.n))
    result['random_site']=random_site
    mapping={'charge_fixed':('charge',0),'charge_random':('charge',random_site),
        'charge_mobile':('charge',runs['charge'].get('selected_site')),
        'v2g_fixed':('v2g',0),'v2g_mobile':('v2g',runs['v2g'].get('selected_site')),
        'charge_staff':('staff',runs['staff'].get('selected_site'))}
    for name,(tag,j) in mapping.items():
        plan=runs[tag].get('site_plans',{}).get(j)
        evaluation=runs[tag].get('site_evaluations',{}).get(j)
        if plan is not None and evaluation is not None:
            choices[name]=plan;method[name]=dict(status='information_optimized',
                selected_site=int(j),predicted_score=evaluation['score'],
                criterion=evaluation['criterion'])
        else:method[name].update(status='selection_failure_fallback',reason='required_site_search_failed')
    # Preserve a known feasible incumbent when the new hardware/worker set
    # contains the old one. This requires no extra search or source truth.
    for expanded,base in (('v2g_fixed','charge_fixed'),('v2g_mobile','charge_mobile'),
                          ('charge_staff','charge_mobile')):
        original=choices[base];value=method[base].get('predicted_score')
        old_validation=check_service_plan(original,constraints[base]) if original is not None else dict(feasible=False)
        candidate=canonical_service_plan(original.site,original.p[:,original.site],history.n,constraints[expanded],solver=original.solver) if old_validation['feasible'] else None
        validation=check_service_plan(candidate,constraints[expanded]) if candidate is not None else dict(feasible=False)
        own=method[expanded].get('predicted_score')
        kept=bool(validation['feasible'] and value is not None and (own is None or value>own))
        result.setdefault('expanded_domain_guards',[]).append(dict(method=expanded,from_method=base,
            candidate_feasible=validation['feasible'],retained=kept,candidate_score=value,old_score=own))
        if kept:
            choices[expanded]=candidate;method[expanded]=dict(status='expanded_domain_incumbent_guard',
                selected_site=candidate.site,predicted_score=value,from_method=base)
    scores=[]
    for name in ('constant_fixed','quality_charge'):
        if choices[name] is not None:
            before=evaluator._ac_calls
            try:
                e=evaluator.evaluate(choices[name].site,BASE_KW*choices[name].p[:,choices[name].site],with_gradient=False)
                scores.append(dict(method=name,evaluation=e))
                method[name]['information_score']=e.get('score')
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                scores.append(dict(method=name,error=str(error)))
            result['selector_ac_calls']+=evaluator._ac_calls-before
    result['baseline_information_checks']=scores
    result['seconds']=perf_counter()-began
    return choices,constraints,result

def evaluate_actions(history,path,white,candidates,hfits,future_model,joint_model,
                     choices,constraints,selection,*,seed):
    """Evaluation alone sees future truth/noise; complete AC H+N refits."""
    initial={name:f for name,f in hfits.items() if f.success};cache={};rows=[]
    family=normalize_family(path.candidate.supports,3)
    true_name=next((c.name for c in candidates if normalize_family(c.supports,3)==family),None)
    for name in METHODS:
        began=perf_counter();plan=choices[name]
        row=dict(method=name,true_name=true_name,true_covered=true_name is not None,
            selection=selection['methods'][name],action=None if plan is None else plan.summary())
        if plan is None:
            row.update(status='operational_failure',seconds=perf_counter()-began);rows.append(row);continue
        check=dict(feasible=True,diagnostic_no_charge_task=True) if name=='none' else check_service_plan(plan,constraints[name])
        row['validation']=check
        if not check['feasible']:
            row.update(status='operational_failure',seconds=perf_counter()-began);rows.append(row);continue
        try:
            future,physical=frozen.future_observe(path,plan,future_model,white)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            row.update(status='physical_failure',error=str(error),seconds=perf_counter()-began);rows.append(row);continue
        voltage=abs(physical['voltage']);row.update(voltage_min=float(voltage.min()),voltage_max=float(voltage.max()))
        if voltage.min()<.9 or voltage.max()>1.1:
            row.update(status='physical_failure',error='voltage_domain',seconds=perf_counter()-began);rows.append(row);continue
        joint=concat_observations(history,future);key=_plan_key(plan);hit=key in cache
        if not hit:
            cache[key]=io.fit_all_candidates(candidates,joint,joint_model,initial=initial,
                seed=seed+200,fit_options=CONFIG['evaluation_fit_options'])
        fits,failed,cost=cache[key]
        losses={c.name:float(fits[c.name].loss) for c in candidates
            if c.name in fits and fits[c.name].success and np.isfinite(fits[c.name].loss)}
        complete=len(losses)==len(candidates)
        row.update(status='complete' if complete else 'joint_fit_failure',losses=losses,
            fit_failures=failed,fits=[io.fit_record(c,fits[c.name]) for c in candidates if c.name in fits],
            observations=io.observations_record(future),physical={k:physical[k] for k in ("p","q","v","master_p","master_q")},cache_hit=hit,
            ac_calls=0 if hit else cost['ac_calls'],fit_calls=0 if hit else cost['fit_calls'])
        if complete and true_name is not None:
            row.update(winner=min(losses,key=lambda n:(losses[n],n)),
                signed_margin_full_SSE=min(v for n,v in losses.items() if n!=true_name)-losses[true_name],
                true_gap=losses[true_name]-min(losses.values()),
                impedance_absolute_rmse_pu=float(np.sqrt(np.mean(np.r_[
                    (fits[true_name].r-path.r)**2,(fits[true_name].x-path.x)**2]))))
        row['seconds']=perf_counter()-began;rows.append(row)
    return rows

def run_condition(seed,case,family,quality,*,prepared=None):
    began=perf_counter();config=frozen.configuration();hashes=source_hashes();new_history_cost=prepared is None
    if prepared is None:
        path,seeds,label,_,permutation,white,amplitude=frozen.make_path(seed,config,
            case=case,truth_family=family,quality='uniform')
        history,_=observe(path,slice(0,12));hmodel=model(3)
        candidates,proposal=propose(history,run_wzzt=True,wzzt_time_limit=.3)
        fits,hfail,hcost=io.fit_all_candidates(candidates,history,hmodel,
            seed=seeds['policy']+10,fit_options=CONFIG['evaluation_fit_options'])
        prepared=(path,seeds,label,permutation,white,amplitude,history,hmodel,candidates,proposal,fits,hfail,hcost)
    path,seeds,label,permutation,white,amplitude,history,hmodel,candidates,proposal,fits,hfail,hcost=prepared
    future,joint,quality_record=quality_models(history,hmodel,kind=quality,row_permutation=permutation)
    choices,devices,selection=select(history,candidates,fits,hmodel,future,quality_record,seed=seeds['policy'])
    rows=evaluate_actions(history,path,white,candidates,fits,future,joint,choices,devices,selection,seed=seeds['policy'])
    record=dict(history_id=str(seed),run_id=str(seed)+'_'+quality,case=case,quality=quality,
        stage='development;not_risk_calibrated',config=CONFIG,generation_config=config,seeds=seeds,
        source_sha256=hashes,history_hash=io.fingerprint(history.p,history.q,history.v,history.master_p,history.master_q),
        history_observations=io.observations_record(history),historical_fits=[io.fit_record(c,fits[c.name]) for c in candidates if c.name in fits],
        historical_failures=hfail,historical_cost=hcost,historical_cost_cache_hit=not new_history_cost,candidates=[dict(name=c.name,supports=c.supports) for c in candidates],
        proposal=proposal,measurement_models=dict(history=meter_to_record(hmodel),future=meter_to_record(future),joint=meter_to_record(joint)),
        selection=selection,records=rows,truth=truth_metadata(path),white_future_noise=white,
        selector_uses_truth=False,selector_uses_future_noise=False,old_threshold_applied=False,
        ESS_fixed_response_alias='same injected curve/device as fixed bidirectional EV',
        independent_history_unit=True,multiple_quality_rows_are_not_independent_histories=True,
        seconds=perf_counter()-began,total_ac_calls=(hcost['ac_calls'] if new_history_cost else 0)+selection['selector_ac_calls']+sum(r.get('ac_calls',0) for r in rows),
        gate_passed=bool(not hfail and selection['seconds']<=15. and perf_counter()-began<=90.
            and all(r['status']=='complete' for r in rows)))
    return record,prepared

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=2)
    parser.add_argument('--output',type=Path,default=HERE/'evidence/ev_flexibility/v1_development')
    args=parser.parse_args()
    if not 1<=args.count<=6: raise ValueError('one to six declared independent H')
    args.output.mkdir(parents=True,exist_ok=True);hashes=source_hashes()
    frozen_data=dict(config=CONFIG,source_sha256=hashes,seeds=list(range(981001,981007)),
        cases=CASES,methods=METHODS,quality=('uniform','temporal','spatial'),
        protocol_sha256=hashlib.sha256((HERE/'ev_flexibility_protocol.md').read_bytes()).hexdigest())
    freeze=args.output/'freeze.json'
    if freeze.exists() and io.read_json(freeze)!=io.serial(frozen_data):
        raise ValueError('freeze differs; preserve old outputs and use a new version')
    save(freeze,frozen_data)
    snapshot={name:gzip.compress((HERE/name).read_bytes(),mtime=0).hex() for name in hashes}
    save(args.output/'source_snapshot.json.gz',dict(encoding='gzip_hex',sources=snapshot,hashes=hashes))
    done=[];began=perf_counter()
    for index in range(args.count):
        seed=981001+index;case,family=CASES[index];prepared=None;history_signature=None
        for quality in ('uniform','temporal','spatial'):
            target=args.output/('trial_'+str(seed)+'_'+quality+'.json.gz')
            if target.exists():
                record=io.read_json(target)
                if record['run_id']!=str(seed)+'_'+quality or record['config']!=io.serial(CONFIG) or record['source_sha256']!=hashes:
                    raise ValueError('existing trial identity/config/source differs; preserve and stop')
            else:
                record,prepared=run_condition(seed,case,family,quality,prepared=prepared)
                save(target,record)
            signature=(record['history_hash'],record['candidates'],[{k:f[k] for k in ('name','loss','r','x','p','q','source_p','source_q')} for f in record['historical_fits']])
            if history_signature is None: history_signature=signature
            elif history_signature!=signature: raise ValueError('same-H quality conditions have different H fits/candidates')
            done.append(dict(run_id=record['run_id'],history_id=record['history_id'],case=case,
                quality=quality,gate_passed=record['gate_passed'],seconds=record['seconds'],
                selection_seconds=record['selection']['seconds'],
                complete_methods=sum(r['status']=='complete' for r in record['records'])))
            save(args.output/'progress.json',dict(requested_histories=args.count,planned_histories=6,
                completed_conditions=len(done),independent_histories=len(set(d['history_id'] for d in done)),
                conditions=done,all_stage_gates_passed=all(d['gate_passed'] for d in done),
                elapsed_current_invocation_seconds=perf_counter()-began,whole_research_complete=False))
            print(json.dumps(done[-1]),flush=True)
            if not record['gate_passed']:
                print('Stage gate failed; preserved record; no automatic expansion.',flush=True);return

if __name__=='__main__': main()
