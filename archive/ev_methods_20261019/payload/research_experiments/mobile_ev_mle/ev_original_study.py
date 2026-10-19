"""Physical LV case generator and independent evaluation for original-domain EV design.

The selector is deliberately passed historical meters/fits and public device
limits only. Simulator truth and held-out errors stay in make_case/evaluate.
"""
from __future__ import annotations
import argparse, ast, base64, gzip, hashlib, json, os, platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from . import likelihood as lk
from . import gradient_price_study as io
from .candidates import RadialCandidate, propose, normalize_family
from .simulation import model
from .information_context import meter_to_record
from .gradient_price import _models_for_future
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
from .ev_original_ac_check import (build_multisite_context,evaluate_multisite_predicted_action,
    make_ac_context,evaluate_ac_witness,recenter_at_absolute)
from scipy.optimize import milp,Bounds,LinearConstraint

HERE=Path(__file__).resolve().parent
BASE_KW=20.; BASE_KV=.4; ZBASE=8.
FAMILIES=((),((0,1),),((0,2),),((1,2),))
CHANNELS=('p','q','v','master_p','master_q')
ASSET=dict(source='https://pandapower.readthedocs.io/en/v3.2.0/std_types/basic.html',
    accessed='2026-10-14',read_locations='basic_line_std_types, LV NAYY4x50/120/150SE',
    R_ohm_km=[.642,.225,.208],X_ohm_km=[.083,.080,.080],
    declared_length_km=[.025,.150],length_is_synthetic_family=True,
    source_scope='official 3.2.0 library, 50Hz; balanced three-phase series-line simulation, synthetic lengths')
ASSET_BOUNDS=dict(r_bounds=(.208*.025/ZBASE,.642*.150/ZBASE),
                  x_bounds=(.080*.025/ZBASE,.083*.150/ZBASE))
CONFIG=dict(version='original_domain_mean_mobile_v2',history=12,horizon=6,dt_hours=.25,
    base_kw=BASE_KW,base_kv=BASE_KV,zbase_ohm=ZBASE,asset=ASSET,asset_bounds=ASSET_BOUNDS,
    historical_p_kw=[2.4]*3,historical_q_kvar=[.8]*3,
    common_p_halfwidth_kw=.24,independent_p_halfwidth_kw=.005,
    source_p_kw=2.,source_p_halfwidth_kw=.25,source_q_kvar=.7,source_q_halfwidth_kvar=.1,
    forecast='calibrated H meter mean and candidate H source mean at node1',
    tracking_error='exact execution for this declared simulator',
    error_model='known independent Gaussian, calibrated fixed offsets; paired future errors',
    public_voltage_repeat_counts=[1,9,1,9,1,9],other_future_repeat_count=5,
    device=dict(power_kw=6.,dt_hours=.25,eta_charge=.95,eta_discharge=.95,
        capacity_kwh=60.,initial_energy_kwh=24.,terminal_energy_kwh=26.85,
        travel_slots=1,setup_slots=1,drive_kwh=.2,max_moves=1,throughput_kwh=7.5,
        travel_cost=15.,setup_cost=15.,driver_budget=30.,cost_units='worker_minutes',min_arrival_throughput_kwh=.25),
    aggregation='uniform average of all12 directed tasks; individual and worst saved',
    exchange=dict(max_iterations=3,control_segments=4,residual_segments=4,max_segments=16,solver_seconds=3.,
        witness_policy='affine_preferred',backend='gurobi',
        total_seconds=15.,absolute_gap=.01,relative_gap=.01),
    fit=dict(multistart=2,max_nfev=250,tol=1e-10,linear_solver='exact'),
    ac_accept_gain=.01,proxy_channel_RMS_gate=.25,max_recenter=1,
    noiseless_refit_evaluation=True,pilot_seeds=list(range(988301,988305)),
    evaluation_seeds=list(range(988401,988413)),
    formal_calibration_count=0,statistical_rejection_rule=None,
    selector_uses_truth=False,selector_uses_future_noise=False)

def observation(physical,meter,white):
    values={}
    for k in CHANNELS:
        values[k]=np.asarray(physical[k])+np.asarray(getattr(meter,k+'_sigma'))*np.asarray(white[k])
        if k in ('p','q','v'): values[k]+=np.asarray(getattr(meter,'bias_'+k))
    return lk.Observations(**values,root_v=np.asarray(physical['root_v']))

def make_case(seed, family, *, short_edge=False):
    """Truth is used here only. Cable constants are public; lengths synthetic."""
    streams=np.random.SeedSequence(seed).spawn(5)
    seeds={k:int(s.generate_state(1)[0]) for k,s in zip(('asset','background','source','noise','policy'),streams)}
    candidate=RadialCandidate.from_supports('evaluation_truth',3,family)
    asset_rng=np.random.default_rng(seeds['asset']); rr=[];xx=[];assets=[]
    for support in candidate.supports:
        if len(support)==3: kind=2; length=.040
        elif len(support)==1:kind=0;length=(.080,.100,.120)[support[0]]
        else:kind=2;length=.002 if short_edge else .080
        length*=asset_rng.uniform(.95,1.05)
        rr.append(ASSET['R_ohm_km'][kind]*length/ZBASE)
        xx.append(ASSET['X_ohm_km'][kind]*length/ZBASE)
        assets.append(dict(support=support,cable_mm2=(50,120,150)[kind],length_km=length))
    t=CONFIG['history']+CONFIG['horizon'];rng=np.random.default_rng(seeds['background'])
    common=rng.uniform(-CONFIG['common_p_halfwidth_kw'],CONFIG['common_p_halfwidth_kw'],(t,1))
    p=(2.4+common+rng.uniform(-.005,.005,(t,3)))/BASE_KW
    q=(.8+.25*common+rng.uniform(-.0025,.0025,(t,3)))/BASE_KW
    sr=np.random.default_rng(seeds['source'])
    sp=(2.+sr.uniform(-.25,.25,t))/BASE_KW;sq=(.7+sr.uniform(-.1,.1,t))/BASE_KW
    root=np.ones(t);hm=model(3);nr=np.random.default_rng(seeds['noise'])
    white={k:nr.standard_normal((t,3) if k in ('p','q','v') else t) for k in CHANNELS}
    h=CONFIG['history'];physical=lk.ac_predict(candidate,rr,xx,p[:h],q[:h],sp[:h],sq[:h],root[:h])
    history=observation(physical,hm,{k:v[:h] for k,v in white.items()})
    counts=np.asarray(CONFIG['public_voltage_repeat_counts'])[:,None]
    fm=replace(hm,v_sigma=np.broadcast_to(hm.v_sigma/np.sqrt(counts),(CONFIG['horizon'],3)),
        **{k+'_sigma':getattr(hm,k+'_sigma')/np.sqrt(CONFIG['other_future_repeat_count']) for k in ('p','q','master_p','master_q')})
    fm,jm=_models_for_future(history,hm,CONFIG['horizon'],fm)
    return dict(seed=seed,seeds=seeds,candidate=candidate,family=family,r=np.asarray(rr),x=np.asarray(xx),
        p=p,q=q,source_p=sp,source_q=sq,root_v=root,white=white,history=history,hm=hm,fm=fm,jm=jm,
        assets=assets,short_edge=short_edge,truth_used_only_for_generation_and_evaluation=True)

def fit_options(domain):
    bounds=ASSET_BOUNDS if domain=='asset' else dict(r_bounds=(1e-6,.1),x_bounds=(1e-6,.1))
    return {**CONFIG['fit'],**bounds}

def prepare(case,domain):
    history=case['history'];bank,proposal=propose(history,run_wzzt=True,wzzt_time_limit=.3)
    fits,failed,cost=io.fit_all_candidates(bank,history,case['hm'],seed=case['seeds']['policy'],
        fit_options=fit_options(domain))
    return dict(candidates=bank,fits=fits,failures=failed,cost=cost,proposal=proposal)

def evaluate(case,prepared,choices,domain):
    """Paired real future errors, full H+N AC fits; no statistical threshold."""
    rows=[];cache={};mean_cache={};h=CONFIG['history'];bank=prepared['candidates'];hfits=prepared['fits']
    true_names=[c.name for c in bank if normalize_family(c.supports,3)==case['family']]
    for name,power in choices.items():
        row=dict(method=name,candidate_miss=len(true_names)!=1)
        if power is None: row['status']='operation_or_design_failure';rows.append(row);continue
        power=np.asarray(power,float).reshape(CONFIG['horizon'],3);row['power_kw']=power
        p=case['p'][h:]+power/BASE_KW
        try:
            physical=lk.ac_predict(case['candidate'],case['r'],case['x'],p,case['q'][h:],
                case['source_p'][h:],case['source_q'][h:],case['root_v'][h:])
            row['physical']={k:physical[k] for k in ('p','q','v','master_p','master_q','ac_equation_residual','power_balance_residual')}
            row['grid_net_kwh']=float(CONFIG['dt_hours']*power.sum())
            row['grid_throughput_kwh']=float(CONFIG['dt_hours']*np.abs(power).sum())
            row['status']='physical_voltage_failure'
            if np.min(physical['v'])<.9 or np.max(physical['v'])>1.1:rows.append(row);continue
            obs=observation(physical,case['fm'],{k:v[h:] for k,v in case['white'].items()})
            row['observations']=io.observations_record(obs)
            key=power.astype('<f8').tobytes();hit=key in cache
            if not hit:cache[key]=io.fit_all_candidates(bank,lk.concat_observations(case['history'],obs),case['jm'],
                initial=hfits,seed=case['seeds']['policy']+802,fit_options=fit_options(domain))
            fitted,failed,cost=cache[key]
            row.update(status='fit_failure' if failed or len(fitted)!=len(bank) else 'complete',cache_hit=hit,
                failures=failed,cost=cost,fits=[io.fit_record(c,fitted[c.name]) for c in bank if c.name in fitted])
            if row['status']=='complete':
                losses={n:float(f.loss) for n,f in fitted.items()};ranked=sorted(losses,key=lambda n:(losses[n],n))
                row.update(losses=losses,winner=ranked[0],candidate_ranking=ranked,
                    pair_signed_loss_differences={a+'__to__'+b:.5*(losses[b]-losses[a]) for a in losses for b in losses if a!=b})
                if len(true_names)==1:
                    true=true_names[0];margin=min(v for n,v in losses.items() if n!=true)-losses[true]
                    f=fitted[true];row.update(true_name=true,margin=margin,loglik_margin=.5*margin,
                        true_impedance_R_relative=float(np.linalg.norm(f.r-case['r'])/np.linalg.norm(case['r'])),
                        true_impedance_X_relative=float(np.linalg.norm(f.x-case['x'])/np.linalg.norm(case['x'])))
            if CONFIG['noiseless_refit_evaluation']:
                mean_obs=observation(physical,case['fm'],{k:np.zeros_like(v[h:]) for k,v in case['white'].items()})
                if key not in mean_cache:
                    mean_cache[key]=io.fit_all_candidates(bank,lk.concat_observations(case['history'],mean_obs),case['jm'],
                        initial=hfits,seed=case['seeds']['policy']+1202,fit_options=fit_options(domain))
                mean_fits,mean_failed,mean_cost=mean_cache[key]
                pair=[]
                for c in bank:
                    f=mean_fits.get(c.name)
                    if f is None:
                        pair.append(dict(candidate=c.name,success=False,reason='required_future_fit_missing',
                            attained_joint_SSE=None,added_half_SSE=None,future_channel_SSE=None,historical_SSE=None))
                        continue
                    future_cost={}
                    for k in CHANNELS:
                        residual=(np.asarray(f.predicted[k])[h:]-getattr(mean_obs,k))/np.asarray(getattr(case['fm'],k+'_sigma'))
                        future_cost[k]=float(np.sum(residual*residual))
                    hf=hfits.get(c.name);baseline_ok=hf is not None and hf.success and np.isfinite(hf.loss)
                    pair.append(dict(candidate=c.name,success=bool(f.success and baseline_ok),
                        reason=None if baseline_ok else 'required_historical_fit_failure',attained_joint_SSE=float(f.loss),
                        added_half_SSE=.5*(float(f.loss)-float(hf.loss)) if baseline_ok else None,future_channel_SSE=future_cost,
                        historical_SSE=float(f.loss)-sum(future_cost.values())))
                row['noiseless_AC_refit']=dict(fits=[io.fit_record(c,mean_fits[c.name]) for c in bank if c.name in mean_fits],
                    failures=mean_failed,cost=mean_cost,all_candidates=len(mean_fits)==len(bank) and not mean_failed,
                    all_historical_baselines=all(c.name in hfits and hfits[c.name].success for c in bank),
                    candidate_distances=pair,scope='offline H plus true noiseless future mean; selector never receives this',
                    future_mean_observations=io.observations_record(mean_obs))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            row.update(status='physical_or_fit_exception',error=type(error).__name__+':'+str(error))
        rows.append(row)
    return rows

def source_hashes():
    names=['ev_original_study.py','ev_original_domain_milp.py','ev_mobile_service_domain.py',
        'ev_original_ac_check.py','ev_original_protocol.md','ev_operations_milp.py',
        'ev_ac_moving_cross.py','ev_profile_ac_proxy.py','symmetric_ac_local.py',
        'information_context.py','gradient_price.py','gradient_price_study.py','likelihood.py','simulation.py','candidates.py',
        'ev_profile_chord_milp.py','ev_bilinear_score_exchange_v2.py','ev_bilinear_support_milp.py',
        'ev_bilinear_score_exchange.py','ev_bilinear_exchange.py','ev_original_mip_backend.py']
    names+=['../mobile_ev_probing/'+n for n in ('ac_general_v9.py','passive_bridge_v9.py','wzzt_adapter.py',
        'ac_shared_v8.py','local_shared_source_v6.py','design_v7.py')]
    names+=list(io.source_hashes())
    sources={n:(HERE/n) for n in names}
    for path in (HERE.parents[1]/'rnj_wzzt_core'/'rnj_wzzt').rglob('*.py'):
        sources[str(path.relative_to(HERE.parents[1])).replace('\\','/')]=path
    # Static local import closure; runtime/library versions are recorded separately.
    root=HERE.parents[1];search=(root,root/'rnj_wzzt_core',HERE,HERE.parent/'mobile_ev_probing')
    pending=list(sources.values());seen=set()
    def add(path):
        choices=(path.with_suffix('.py'),path/'__init__.py')
        for candidate in choices:
            if candidate.is_file() and candidate.resolve().is_relative_to(root):
                candidate=candidate.resolve()
                name=str(candidate.relative_to(root)).replace('\\','/') if candidate.is_relative_to(root/'rnj_wzzt_core') else os.path.relpath(candidate,HERE).replace('\\','/')
                if name not in sources:sources[name]=candidate;pending.append(candidate)
    while pending:
        path=pending.pop().resolve()
        if path in seen or path.suffix!='.py':continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for alias in node.names:
                    for directory in search:add(directory.joinpath(*alias.name.split('.')))
            elif isinstance(node,ast.ImportFrom):
                directories=(path.parent if node.level==1 else path.parents[node.level-1],) if node.level else search
                for directory in directories:
                    base=directory.joinpath(*(node.module or '').split('.'))
                    if node.module:add(base)
                    for alias in node.names:
                        if alias.name!='*':add(base/alias.name)
    return {n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()}

def service_domain(*,mobile,discharge):
    d=CONFIG['device'];access=np.ones((CONFIG['horizon'],3),bool)
    if not mobile:access[:,1:]=False
    cfg=MobileServiceConfig(n_slots=CONFIG['horizon'],n_sites=3,dt_hours=d['dt_hours'],
        p_charge_max_kw=d['power_kw'],p_discharge_max_kw=d['power_kw'],allow_discharge=discharge,
        battery_capacity_kwh=d['capacity_kwh'],initial_soc=d['initial_energy_kwh']/d['capacity_kwh'],
        terminal_energy_kwh=d['terminal_energy_kwh'],eta_charge=d['eta_charge'],eta_discharge=d['eta_discharge'],
        mobile=mobile,access=access,setup_slots=d['setup_slots'],travel_slots=d['travel_slots'],
        travel_energy_kwh=d['drive_kwh'],max_moves=d['max_moves'],max_throughput_kwh=d['throughput_kwh'],
        travel_cost=d['travel_cost'],setup_cost=d['setup_cost'],driver_budget=d['driver_budget'],cost_units=d['cost_units'],
        min_arrival_throughput_kwh=d['min_arrival_throughput_kwh'])
    return build_mobile_service_domain(cfg)

def realize_control(domain,power):
    """Find a full physical plan for a specified net curve, without route enumeration."""
    op=domain.as_dict();u=np.asarray(power,float).ravel()
    result=milp(np.zeros(len(op['lower'])),integrality=op['integrality'],bounds=Bounds(op['lower'],op['upper']),
        constraints=[*op['constraints'],LinearConstraint(op['u_map'],u,u)],options=dict(time_limit=3.,mip_rel_gap=1e-8))
    audit=domain.check(result)
    return dict(success=result.x is not None and audit['feasible'],x=result.x,audit=audit,
        status=int(result.status),message=str(result.message))

def operation_record(domain,point):
    return domain.check(point)

def _safe_raw_exchange(pairs,operation,offsets,initial_controls,initial_operation_points):
    # Explicit adapter kept here so its live signature is covered by integration tests.
    from .ev_original_domain_milp import exchange_original_domain
    return exchange_original_domain(pairs,operation['control_lower'],operation['control_upper'],operation=operation,offsets=offsets,
        initial_controls=initial_controls,initial_operation_points=initial_operation_points,
        weights=np.full(len(pairs),1/len(pairs)),**CONFIG['exchange'])

def select_controls(history,bank,hfits,hm,fm,jm,*,seed,domain):
    """Historical data/public hardware only: no path, true family or future noise."""
    from .ev_original_domain_milp import from_moving_model
    began=perf_counter();T=CONFIG['horizon'];n=history.n
    baseline=np.zeros((T,n));baseline[:,0]=(CONFIG['device']['terminal_energy_kwh']-
        CONFIG['device']['initial_energy_kwh'])/(CONFIG['device']['eta_charge']*T*CONFIG['dt_hours'])
    domains={name:service_domain(mobile=mobile,discharge=discharge) for name,mobile,discharge in (
        ('charge_fixed',False,False),('v2g_fixed',False,True),('charge_mobile',True,False),('v2g_mobile',True,True))}
    choices=dict(none=np.zeros_like(baseline),constant=baseline.copy())
    out=dict(uses_truth=False,uses_future_noise=False,methods={},failures={},context=None,
        predicted_AC={},all_candidates=len(bank),all_ordered_pairs=len(bank)*(len(bank)-1))
    baseline_point=realize_control(domains['charge_fixed'],baseline)
    out['constant_operation']=baseline_point['audit']
    if not baseline_point['success']:
        out['failures']['constant']='operation_infeasible';choices['constant']=None
        out['seconds']=perf_counter()-began;return choices,out
    fixed_quality=domains['charge_fixed'].solve_prices(np.tile(np.asarray(CONFIG['public_voltage_repeat_counts'])[:,None],(1,n)),time_limit=3.)
    mobile_quality=domains['v2g_mobile'].solve_prices(np.tile(np.asarray(CONFIG['public_voltage_repeat_counts'])[:,None],(1,n)),time_limit=3.)
    random=domains['v2g_mobile'].solve_prices(np.random.default_rng(seed+81).normal(size=(T,n)),time_limit=3.)
    for name,d,r in [('quality_fixed',domains['charge_fixed'],fixed_quality),('quality_mobile',domains['v2g_mobile'],mobile_quality),
                     ('random_mobile',domains['v2g_mobile'],random)]:
        audit=d.check(r);choices[name]=None if not audit['feasible'] else d.decode(r)['u_kw']
        out['methods'][name]=dict(operation=audit,solver_status=int(r.status),solver_message=str(r.message),
            linear_incumbent=r.objective_value,linear_bound=r.objective_bound)
    bounds={k:v for k,v in fit_options(domain).items() if k in ('r_bounds','x_bounds')}
    context=build_multisite_context(history,bank,hfits,hm,fm,baseline,-6.,6.,base_kw=BASE_KW,line_bounds=bounds)
    out['context']=context['diagnostics']
    if not context['success']:
        out['failures']['context']=context['diagnostics']['failure'];out['seconds']=perf_counter()-began
        for name in domains:choices[name]=None
        return choices,out
    pairs=[from_moving_model(m) for m in context['models']]
    offsets=[hfits[m['b']].loss for m in context['models']]
    ac_cache={}
    def predicted(power):
        key=np.asarray(power,float).astype('<f8').tobytes()
        if key not in ac_cache:
            ac_cache[key]=evaluate_multisite_predicted_action(history,bank,hfits,hm,fm,jm,power,
                base_kw=BASE_KW,fit_options=fit_options(domain),seed=seed+500)
        return ac_cache[key]
    baseline_ac=predicted(baseline)
    out['predicted_AC']['constant']=baseline_ac
    for name,power in choices.items():
        if name not in ('constant',) and power is not None:out['predicted_AC'][name]=predicted(power)
    for name,d in domains.items():
        op=d.as_dict();base=realize_control(d,baseline)
        prices=np.tile(np.asarray(CONFIG['public_voltage_repeat_counts'])[:,None],(1,n))
        quality=d.solve_prices(prices,time_limit=3.);qvalid=d.check(quality)['feasible']
        initial=[baseline.ravel()];points=[base['x']]
        if qvalid: initial.append(d.decode(quality)['u_kw'].ravel());points.append(quality.x)
        record=dict(operation_initial=base['audit'],raw=None,guard=None,operation=None,
            proxy_AC_checks=[],recentered=False,recenter_models=None)
        out['methods'][name]=record
        try:
            local_models=context['models'];local_pairs=pairs
            result=_safe_raw_exchange(local_pairs,op,offsets,initial,points);record['raw']=result
            raw_u=np.asarray(result['u'],float).reshape(T,n) if result.get('u') is not None else None
            if raw_u is None:
                record['guard']=dict(accepted=False,reason='no_raw_action');choices[name]=baseline.copy();continue
            def same_witness_checks(result,models):
                best=result['evaluations'][result['best_evaluation_index']]
                checks=[]
                for m,qp in zip(models,best['qp_results']):
                    acctx=make_ac_context(m,history,bank,hfits,hm,fm,voltage_limits=(.9,1.1))
                    check=evaluate_ac_witness(m,result['u'],qp['physical_parameter'],acctx)
                    checks.append(dict(a=m['a'],b=m['b'],check=check))
                return checks
            checks=same_witness_checks(result,local_models);record['proxy_AC_checks'].append(checks)
            gate=lambda rows:all(x['check']['complete_AC_voltage_feasible'] and
                x['check']['physical_box_feasible'] and x['check']['max_channel_RMS']<=CONFIG['proxy_channel_RMS_gate'] for x in rows)
            if not gate(checks) and CONFIG['max_recenter']:
                best=result['evaluations'][result['best_evaluation_index']];new_models=[]
                for m,qp,check in zip(local_models,best['qp_results'],checks):
                    if check['check']['max_channel_RMS']>CONFIG['proxy_channel_RMS_gate']:
                        acctx=make_ac_context(m,history,bank,hfits,hm,fm)
                        new_models.append(recenter_at_absolute(m,result['u'],qp['physical_parameter'],acctx))
                    else:new_models.append(m)
                record['before_recenter']=result
                record['recenter_models']=new_models;record['recentered']=True
                local_models=new_models;local_pairs=[from_moving_model(m) for m in local_models]
                result=_safe_raw_exchange(local_pairs,op,offsets,initial+[raw_u.ravel()],points)
                record['raw']=result
                raw_u=np.asarray(result['u'],float).reshape(T,n) if result.get('u') is not None else None
                if raw_u is None:raise ValueError('recentered proxy returned no executable action')
                checks=same_witness_checks(result,local_models);record['proxy_AC_checks'].append(checks)
            proxy_gate=gate(checks)
            realized=realize_control(d,raw_u);record['operation']=realized['audit']
            rawac=predicted(raw_u) if realized['success'] else None
            out['predicted_AC'][name+'__raw']=rawac
            complete=baseline_ac['success'] and rawac is not None and rawac['success']
            gain=rawac['mean_added_score']-baseline_ac['mean_added_score'] if complete else None
            accept=bool(complete and proxy_gate and gain>CONFIG['ac_accept_gain'])
            record['guard']=dict(accepted=accept,gain=gain,complete=complete,
                proxy_same_witness_gate=proxy_gate,
                reason='complete_AC_mean_gain' if accept else 'operation_or_AC_failure' if not complete else
                    'proxy_same_witness_gate_failed' if not proxy_gate else 'AC_gain_below_resolution',
                resolution=CONFIG['ac_accept_gain'])
            choices[name]=raw_u if accept else baseline.copy()
            out['predicted_AC'][name]=rawac if accept else baseline_ac
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,TypeError) as error:
            record['error']=type(error).__name__+':'+str(error);out['failures'][name]=record['error'];choices[name]=baseline.copy()
    out.update(seconds=perf_counter()-began,
        context_models=context['models'],unique_predicted_actions=len(ac_cache),
        predicted_fit_calls=sum(x['fit_calls'] for x in ac_cache.values()),
        predicted_ac_calls=sum(x['ac_calls'] for x in ac_cache.values()))
    return choices,out

def run_trial(seed,family,domain='asset',short_edge=False):
    began=perf_counter();case=make_case(seed,family,short_edge=short_edge);prepared=prepare(case,domain)
    choices,selection=select_controls(case['history'],prepared['candidates'],prepared['fits'],case['hm'],case['fm'],case['jm'],
        seed=case['seeds']['policy'],domain=domain)
    rows=evaluate(case,prepared,choices,domain)
    return dict(seed=seed,domain=domain,short_edge=short_edge,seeds=case['seeds'],
        history=io.observations_record(case['history']),historical_fits=[io.fit_record(c,prepared['fits'][c.name]) for c in prepared['candidates'] if c.name in prepared['fits']],
        historical_failures=prepared['failures'],historical_cost=prepared['cost'],proposal=prepared['proposal'],
        measurement_models={k:meter_to_record(case[k]) for k in ('hm','fm','jm')},selection=selection,rows=rows,
        truth={k:case[k] for k in ('family','r','x','p','q','source_p','source_q','root_v','white','assets')},
        source_interface=1,independent_unit='one history seed; all methods paired',seconds=perf_counter()-began,
        global_AC_MLE_certified=False,statistical_risk_calibrated=False)

def validate_evaluation_gate(gate,hashes,seeds):
    if not gate.get('ready_for_independent_evaluation'):
        raise ValueError('pilot gate has not passed')
    if gate.get('source_sha256')!=hashes:
        raise ValueError('pilot gate belongs to different scientific source bytes')
    if set(gate.get('pilot_seeds',[]))&set(seeds):
        raise ValueError('pilot and evaluation histories must be disjoint')
    if gate.get('evaluation_scope')!='independent descriptive AC evaluation; no calibrated rejection claim':
        raise ValueError('evaluation scope differs from declared evidence')
    return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','evaluation','stress'],default='pilot')
    parser.add_argument('--limit',type=int);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--domain',choices=['asset','numeric'],default='asset')
    parser.add_argument('--pilot-gate',type=Path);args=parser.parse_args()
    seeds=CONFIG['pilot_seeds'] if args.phase=='pilot' else CONFIG['evaluation_seeds'] if args.phase=='evaluation' else [988501,988502]
    args.output.mkdir(parents=True,exist_ok=True);hashes=source_hashes()
    if args.phase=='evaluation':
        if args.pilot_gate is None:raise ValueError('independent evaluation requires a saved pilot gate')
        validate_evaluation_gate(io.read_json(args.pilot_gate),hashes,seeds)
    freeze=dict(config=CONFIG,phase=args.phase,domain=args.domain,seeds=seeds,source_sha256=hashes,
        frozen_before_trials=True,independent_histories_not_future_repeats=True,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    fp=args.output/'freeze.json'
    if fp.exists():
        if io.read_json(fp)!=io.serial(freeze):raise ValueError('freeze differs; new evidence directory required')
    else:
        io.write_json(fp,freeze)
        snapshot={}
        for n,h in hashes.items():
            p=HERE/n if not n.startswith('rnj_wzzt_core/') else HERE.parents[1]/n
            raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source changed before snapshot: '+n)
            snapshot[n]=dict(sha256=h,bytes_base64=base64.b64encode(raw).decode('ascii'))
        io.write_json(args.output/'source_snapshot.json.gz',snapshot,compressed=True)
    for index,seed in enumerate(seeds[:args.limit]):
        path=args.output/f'trial_{seed}.json.gz'
        if path.exists():continue
        if source_hashes()!=hashes:raise ValueError('frozen source changed')
        family=FAMILIES[index%4] if args.phase!='stress' else FAMILIES[1+index%3]
        result=run_trial(seed,family,args.domain,short_edge=args.phase=='stress')
        after=source_hashes()
        if after!=hashes:
            result.update(config=CONFIG,source_before=hashes,source_after=after,source_changed_during_trial=True,phase=args.phase)
            io.write_json(args.output/('invalid_source_'+path.name),result,compressed=True)
            raise ValueError('source changed during trial; unfrozen result preserved separately')
        result.update(config=CONFIG,source_sha256=hashes,phase=args.phase)
        io.write_json(path,result,compressed=True)
        summary=dict(seed=seed,seconds=result['seconds'],selection_seconds=result['selection']['seconds'],
            failures=result['selection']['failures'],margins={r['method']:r.get('margin') for r in result['rows']},
            ac_gains={n:(r.get('guard') or {}).get('gain') for n,r in result['selection']['methods'].items() if 'guard' in r})
        io.write_json(args.output/'checkpoint.json',dict(completed=len(list(args.output.glob('trial_*.json.gz'))),planned=len(seeds),
            last_summary=summary,research_complete=False,statistical_risk_calibrated=False))
        print(json.dumps(io.serial(summary)),flush=True)

if __name__=='__main__':main()
