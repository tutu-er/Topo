"""Time-varying ordinary-load EV study; truth lives in generation/evaluation.

One trained line vector per topology. Deterministic baselines and all physical
domains are paired on the same held-out AC data and standardized meter errors.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from . import ev_original_study as base
from . import ev_fixed_train_study as old
from . import ev_fixed_train_design as design
from . import gradient_price_study as io
from . import likelihood as lk
from . import symmetric_ac_local as geometry
from . import ev_original_domain_milp as proxy
from . import ev_fixed_train_review as fit_replay
from . import ev_original_review as physical_replay
from .candidates import propose
from .ev_dynamic_loads import generate_background, fit_forecast
from .ev_fixed_train_fit import subset_observations, subset_model
from .information_context import meter_to_record

HERE = Path(__file__).resolve().parent
SELECT_MODES = ('none', 'constant', 'quality_fixed', 'quality_v2g_fixed',
    'quality_charge_mobile', 'quality_mobile', 'charge_fixed', 'v2g_fixed',
    'charge_mobile', 'v2g_mobile')
RENAME = {'quality_fixed':'quality_charge_fixed', 'quality_mobile':'quality_v2g_mobile',
    **{k:'design_'+k for k in design.DOMAIN_FLAGS}}
CONFIG = dict(version='ordinary_dynamic_fixed_lines_v1', training_points=96,
    validation_points=4, horizon=6, dt_hours=.25, meter_schedule='uniform',
    public_voltage_repeat_counts=[5]*6, forecast='ar1', forecast_gap=4,
    PQ_inputs_fixed=True, PQ_input_error_propagated=False, source_interface=1,
    training=dict(multistart=3, max_nfev=350, tol=1e-10, **base.ASSET_BOUNDS),
    source=dict(source_p_bounds=[-.5,.5], source_q_bounds=[-.5,.5]),
    profile=dict(multistart=2,max_nfev=180,tol=1e-10),
    exchange=dict(design.DEFAULT_EXCHANGE), acceptance='full_ac_score',
    quality_tie_break='early_stay',
    select_modes=list(SELECT_MODES), mean_policy_ablation=True,
    diagnostics=True, alpha=.1, development_seeds=list(range(991101,991105)),
    calibration_seeds=list(range(991301,991337)), evaluation_seeds=list(range(991201,991213)),
    scope='declared balanced LV cable-derived synthetic ordinary-load process; not field data')


def source_hashes():
    values=old.source_hashes()
    for name in ('ev_dynamic_loads.py','ev_dynamic_study.py','ev_dynamic_math.md',
                 'ev_dynamic_protocol.md','ev_dynamic_review.py','ev_dynamic_collect.py'):
        values[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return values


def freeze(directory, config):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes(); policy_id=io.identity(config,hashes)
    record=dict(config=config,source_sha256=hashes,policy_id=policy_id,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        base_commit='e6225d1')
    path=directory/'freeze.json'
    if path.exists():
        saved=io.read_json(path)
        if saved['policy_id']!=policy_id: raise ValueError('freeze differs; preserve it and use a new directory')
        return saved
    io.write_json(path,record)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/')
        else HERE/name).read_bytes()).decode() for name in hashes}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=hashes,source_base64=snapshot),compressed=True)
    return record


def make_case(seed, family_index, config):
    """Simulation truth never leaves this object for a selector argument."""
    assets=base.make_case(seed,base.FAMILIES[family_index])
    H,V,T=(config[k] for k in ('training_points','validation_points','horizon'))
    count=H+V+T
    data=generate_background(assets['seeds']['background'],count)
    p=np.asarray(data['p_kw'])/base.BASE_KW; q=np.asarray(data['q_kvar'])/base.BASE_KW
    sp=np.asarray(data['source_p_kw'])/base.BASE_KW; sq=np.asarray(data['source_q_kvar'])/base.BASE_KW
    root=np.asarray(data['root_v']); meter=base.model(3)
    rng=np.random.default_rng(assets['seeds']['noise'])
    white={k:rng.standard_normal((count,3) if k in ('p','q','v') else count) for k in base.CHANNELS}
    phy=lk.ac_predict(assets['candidate'],assets['r'],assets['x'],p,q,sp,sq,root)
    obs=base.observation(phy,meter,white)
    train=subset_observations(obs,slice(0,H)); validation=subset_observations(obs,slice(H,H+V))
    tm=subset_model(meter,obs,slice(0,H)); vm=subset_model(meter,obs,slice(H,H+V))
    counts=np.asarray(config['public_voltage_repeat_counts'],float)
    if counts.shape!=(T,) or np.any(counts<=0): raise ValueError('positive repeat counts per future slot required')
    fm=replace(meter,v_sigma=np.broadcast_to(meter.v_sigma/np.sqrt(counts[:,None]),(T,3)),
        **{k+'_sigma':getattr(meter,k+'_sigma')/np.sqrt(5.) for k in ('p','q','master_p','master_q')})
    return dict(seed=seed,family_index=family_index,assets=assets,background=data,
        p=p,q=q,source_p=sp,source_q=sq,root_v=root,white=white,
        train=train,validation=validation,tm=tm,vm=vm,fm=fm,future_start=H+V)


def highlow_curve(T,n):
    """Predeclared alternating-slot saturated charge; no model or data input."""
    u=np.zeros((T,n)); remain=(design.CONFIG['terminal_energy_kwh']-design.CONFIG['initial_energy_kwh'])/design.CONFIG['eta_charge']
    order=list(range(0,T,2))+list(range(1,T,2)); dt=design.CONFIG['dt_hours']
    for t in order:
        u[t,0]=min(design.CONFIG['command_power_kw'],remain/dt)
        remain-=u[t,0]*dt
    if remain>1e-9: raise ValueError('charging task outside highlow capacity')
    return u


def operation_record(power, mobile=False, discharge=False):
    T,n=np.asarray(power).shape
    domain=design._service(T,n,mobile=mobile,discharge=discharge)
    realized=design._realize(domain,power)
    if not realized['success']: return dict(success=False,audit=realized['audit'])
    decoded=domain.decode(realized['x']);dt=domain.config.dt_hours
    C=float(dt*np.sum(decoded['charge_kw']));D=float(dt*np.sum(decoded['discharge_kw']))
    arcs=decoded['selected_arcs'];drive=sum(a.driving_energy_kwh for a in arcs if a.kind=='travel')
    return dict(success=True,audit=realized['audit'],grid_energy_kwh=C-D,
        charging_kwh=C,discharging_kwh=D,throughput_kwh=C+D,driving_kwh=float(drive),
        worker_minutes=decoded['driver_cost'],moves=sum(a.kind=='travel' for a in arcs),
        energy_kwh=decoded['energy_kwh'],power_variance=float(np.var(np.sum(power,axis=1))),
        charging_efficiency=domain.config.eta_charge,discharging_efficiency=domain.config.eta_discharge)


def actual_curve(case, bank, fitted, validated, power, config, *, seed):
    a=case['assets'];start=case['future_start'];T=config['horizon']
    P=case['p'][start:]+power/base.BASE_KW;Q=case['q'][start:]
    phy=lk.ac_predict(a['candidate'],a['r'],a['x'],P,Q,case['source_p'][start:],case['source_q'][start:],case['root_v'][start:])
    white={k:v[start:] for k,v in case['white'].items()}
    actual=base.observation(phy,case['fm'],white)
    clean=base.observation(phy,case['fm'],{k:np.zeros_like(v) for k,v in white.items()})
    opts={**config['profile'],**config['source']}
    _,noisy=old.fit_bank(bank,actual,case['fm'],trained=fitted,options=opts,seed=seed)
    _,noiseless=old.fit_bank(bank,clean,case['fm'],trained=fitted,options=opts,seed=seed+100)
    val={r['candidate']:r['loss'] for r in validated['fits']}
    true_name=next((b.name for b in bank if b.supports==a['candidate'].supports),None)
    losses={r['candidate']:val[r['candidate']]+r['loss'] for r in noisy['fits']} if noisy['all_complete'] and validated['all_complete'] else {}
    cleanloss={r['candidate']:val[r['candidate']]+r['loss'] for r in noiseless['fits']} if noiseless['all_complete'] and validated['all_complete'] else {}
    def margin(values):
        return None if true_name not in values else min(v for k,v in values.items() if k!=true_name)-values[true_name]
    return dict(status='complete' if losses else 'fit_failure',M=margin(losses),clean_future_M=margin(cleanloss),
        clean_future_complete=bool(noiseless['all_complete'] and validated['all_complete']),
        clean_future_scope='future noise removed; the same noisy validation costs retained',
        loss_by_candidate=losses,noisy=noisy,noiseless=noiseless,
        observations=io.observations_record(actual),physical={k:phy[k] for k in ('p','q','v','root_v','master_p','master_q')})


def complete_ac_prices(context, power, *, seed=0, fit_options=None):
    """Envelope gradient at attained source fits; lines and covariance fixed."""
    evaluated=design.evaluate_fixed_action(context,power,seed=seed,fit_options=fit_options)
    if not evaluated['success']:return dict(success=False,prediction=evaluated)
    u=np.asarray(power,float);T,n=u.shape;width=2*n+2
    columns=np.array([t*width+j for t in range(T) for j in range(n)])
    total=dict(p=context['background']['p']+u/context['base_kw'],q=context['background']['q'])
    bank={c.name:c for c in context['bank']};references={};gradient=np.zeros(T*n);tasks=[]
    for model in context['models']:
        a,b=model['a'],model['b']
        if a not in references:
            references[a]=design._geometry(bank[a],context['lines'][a],total,context['root_v'],context['future_meter'],model['reference_source'])
        fit=evaluated['fit_results'][model['name']]
        source=np.column_stack((fit.source_p,fit.source_q)).ravel()
        competitor=design._geometry(bank[b],context['lines'][b],total,context['root_v'],context['future_meter'],source)
        reference=references[a];r=competitor['mean']-reference['mean']
        B=(competitor['state_jacobian'][:,columns]-reference['state_jacobian'][:,columns])/context['base_kw']
        price=B.T@r;gradient+=model['task_weight']*price
        tasks.append(dict(name=model['name'],a=a,b=b,weight=model['task_weight'],
            gradient_per_kw=price.reshape(T,n),half_SSE=float(r@r)/2,
            source_p=fit.source_p,source_q=fit.source_q,source_solver=fit.summary()))
    values=gradient.reshape(T,n)
    return dict(success=True,power_kw=u.copy(),gradient_per_kw=values,mean_half_SSE=evaluated['mean'],pairs=tasks,
        temporal_spread_by_site=np.ptp(values,axis=0),
        scope='envelope derivative of stable attained local source branches; no global AC certificate')


def _run_trial(seed, family_index, config=None):
    cfg=deepcopy(CONFIG if config is None else config);began=perf_counter();before=source_hashes()
    case=make_case(seed,family_index,cfg);tr=case['train'];tm=case['tm'];T=cfg['horizon'];n=tr.n
    bank,proposal=propose(tr,run_wzzt=True,wzzt_time_limit=.3)
    fits,training=old.fit_bank(bank,tr,tm,options={**cfg['training'],**cfg['source']},seed=case['assets']['seeds']['policy'])
    true_name=next((b.name for b in bank if b.supports==case['assets']['candidate'].supports),None)
    record=dict(schema='ev_ordinary_dynamic_v1',seed=seed,family_index=family_index,
        config=cfg,policy_id=io.identity(cfg,before),source_sha256=before,
        training=io.observations_record(tr),training_meter=meter_to_record(tm),training_result=training,
        validation=io.observations_record(case['validation']),validation_meter=meter_to_record(case['vm']),
        future_meter=meter_to_record(case['fm']),proposal=proposal,
        candidates=[dict(name=b.name,supports=b.supports) for b in bank],true_name=true_name,
        candidate_miss=true_name is None,background=case['background'],seeds=case['assets']['seeds'],
        truth=dict(r=case['assets']['r'],x=case['assets']['x'],supports=case['assets']['candidate'].supports),methods=[])
    if not training['all_complete']:
        record.update(status='training_failure',seconds=perf_counter()-began,source_unchanged=before==source_hashes());return io.serial(record)
    _,validated=old.fit_bank(bank,case['validation'],case['vm'],trained=fits,
        options={**cfg['profile'],**cfg['source']},seed=seed+100)
    record['validation_result']=validated
    fore=fit_forecast(tr,tm,fits,horizon=T,gap=cfg['forecast_gap'],base_kw=base.BASE_KW,kind=cfg['forecast'])
    record['forecast']=fore
    start=case['future_start']
    record['forecast_error']=dict(P_RMSE_kw=float(np.sqrt(np.mean((fore['p']-case['p'][start:])**2)))*base.BASE_KW,
        Q_RMSE_kvar=float(np.sqrt(np.mean((fore['q']-case['q'][start:])**2)))*base.BASE_KW,
        uses_future_only_for_offline_evaluation=True)
    highlow=highlow_curve(T,n)
    initials={'highlow':dict(power_kw=highlow,metadata=dict(deterministic=True,uses_validation=False,uses_truth=False,uses_future_noise=False))}
    opts=dict(seed=seed+200,fit_options={**cfg['profile'],**cfg['source']},exchange_options=cfg['exchange'],
        acceptance=cfg['acceptance'],baseline_initials=initials,quality_tie_break=cfg['quality_tie_break'])
    choices,selection=design.select_fixed_controls(tr,bank,fits,tm,case['fm'],
        forecast=fore,modes=cfg['select_modes'],**opts)
    plans={RENAME.get(k,k):v for k,v in choices.items()};plans['highlow']=highlow
    record['selection']=selection
    if cfg['mean_policy_ablation']:
        mean=fit_forecast(tr,tm,fits,horizon=T,gap=cfg['forecast_gap'],base_kw=base.BASE_KW,kind='mean')
        mc,ms=design.select_fixed_controls(tr,bank,fits,tm,case['fm'],forecast=mean,modes=('v2g_mobile',),**opts)
        plans['design_mean_v2g_mobile']=mc['v2g_mobile'];record['mean_selection']=ms;record['mean_forecast']=mean
    if cfg['diagnostics']:
        pure=plans.get('design_charge_fixed');bi=plans.get('design_v2g_fixed')
        if pure is not None:
            const=np.zeros_like(pure);const[:,0]=np.sum(pure)/T
            plans['amplitude_half_fixed']=const+.5*(pure-const)
        if bi is not None:plans['timing_reverse_fixed']=np.asarray(bi)[::-1].copy()
        reference=np.zeros((T,n));reference[:,0]=2.
        context=design.build_fixed_contexts(tr,bank,fits,tm,case['fm'],
            -design.CONFIG['command_power_kw'],design.CONFIG['command_power_kw'],reference,
            base_kw=base.BASE_KW,forecast=fore,**cfg['source'])
        record['price_diagnostics']={}
        if context['success']:
            for label,power in (('constant',reference),('design_v2g_fixed',bi)):
                if power is None:continue
                record['price_diagnostics'][label]=dict(
                    proxy=design.fixed_profile_gradient(context,power),
                    complete_AC=complete_ac_prices(context,power,seed=seed+200+500,
                        fit_options={**cfg['profile'],**cfg['source']}))
    cache={}
    for name,power in plans.items():
        row=dict(name=name,power_kw=power,status='selection_failure')
        if power is None:record['methods'].append(row);continue
        power=np.asarray(power,float)
        mobile='mobile' in name;discharge=('v2g' in name or name=='timing_reverse_fixed')
        op=(dict(success=True,battery_service=False,scope='no EV; matched observation length only')
            if name=='none' else operation_record(power,mobile=mobile,discharge=discharge))
        row['operation']=op
        if name!='none' and not op['success']:
            row['status']='operation_failure';record['methods'].append(row);continue
        key=design._control_key(power)
        if key not in cache:
            try:cache[key]=actual_curve(case,bank,fits,validated,power,cfg,seed=seed+500)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                cache[key]=dict(status='physical_or_fit_failure',M=None,
                    failure=type(exc).__name__+':'+str(exc))
        row.update(cache[key]);record['methods'].append(row)
    rows={r['name']:r for r in record['methods']}
    contrast=dict(timing=('design_v2g_fixed','timing_reverse_fixed'),
        amplitude=('design_charge_fixed','amplitude_half_fixed'),
        discharge=('design_v2g_fixed','design_charge_fixed'),
        movement=('design_v2g_mobile','design_v2g_fixed'),
        charge_movement=('design_charge_mobile','design_charge_fixed'),
        forecast_policy=('design_v2g_mobile','design_mean_v2g_mobile'))
    record['contrasts']={}
    for key,(left,right) in contrast.items():
        x,y=rows.get(left,{}),rows.get(right,{})
        record['contrasts'][key]=dict(left=left,right=right,
            delta_M=None if x.get('M') is None or y.get('M') is None else x['M']-y['M'],
            valid=x.get('M') is not None and y.get('M') is not None)
    record['training_parameter_error']={}
    if true_name:
        tf=fits[true_name];a=case['assets']
        record['training_parameter_error']=dict(R=float(np.linalg.norm(tf.r-a['r'])/np.linalg.norm(a['r'])),
            X=float(np.linalg.norm(tf.x-a['x'])/np.linalg.norm(a['x'])))
    record.update(status='complete' if validated['all_complete'] and all(r['status']=='complete' for r in record['methods']) else 'partial_failure',
        seconds=perf_counter()-began,source_unchanged=before==source_hashes(),
        unique_actual_curves=len(cache),validation_used_by_selector=False,
        future_noise_used_by_selector=False,one_trained_parameter_vector=True)
    return io.serial(record)


@contextmanager
def count_physics_calls():
    """Count actual batch AC calls, including imported geometry aliases."""
    original=lk.ac_predict; original_geo=geometry.ac_predict
    original_qp=proxy.evaluate_original_pair; original_design_qp=design.evaluate_original_pair
    original_replay=fit_replay.forward_physics;original_physics_replay=physical_replay.forward_physics
    counters=dict(AC_batches=0,AC_snapshots=0,AC_failures=0,QP_calls=0,
        independent_replay_AC_batches=0,independent_replay_AC_snapshots=0,independent_replay_AC_failures=0)
    def ac(*args,**kwargs):
        counters['AC_batches']+=1
        p=kwargs.get('p',args[3] if len(args)>3 else None)
        counters['AC_snapshots']+=0 if p is None else len(p)
        try:return original(*args,**kwargs)
        except Exception:
            counters['AC_failures']+=1;raise
    def qp(*args,**kwargs):
        counters['QP_calls']+=1;return original_qp(*args,**kwargs)
    def replay(*args,**kwargs):
        counters['independent_replay_AC_batches']+=1
        p=kwargs.get('p',args[3] if len(args)>3 else None)
        counters['independent_replay_AC_snapshots']+=0 if p is None else len(p)
        try:return original_replay(*args,**kwargs)
        except Exception:
            counters['independent_replay_AC_failures']+=1;raise
    lk.ac_predict=geometry.ac_predict=ac
    proxy.evaluate_original_pair=design.evaluate_original_pair=qp
    fit_replay.forward_physics=physical_replay.forward_physics=replay
    try:yield counters
    finally:
        lk.ac_predict=original;geometry.ac_predict=original_geo
        proxy.evaluate_original_pair=original_qp;design.evaluate_original_pair=original_design_qp
        fit_replay.forward_physics=original_replay;physical_replay.forward_physics=original_physics_replay
        counters['total_AC_batches']=counters['AC_batches']+counters['independent_replay_AC_batches']
        counters['total_AC_snapshots']=counters['AC_snapshots']+counters['independent_replay_AC_snapshots']


def run_trial(seed, family_index, config=None):
    cfg=deepcopy(CONFIG if config is None else config);before=source_hashes();began=perf_counter()
    with count_physics_calls() as counters:
        try:record=_run_trial(seed,family_index,cfg)
        except Exception as exc:
            # Preserve programming/physical failures as raw records, then stop
            # the CLI batch on driver failures for review before more runs.
            import traceback
            record=dict(schema='ev_ordinary_dynamic_v1',seed=seed,family_index=family_index,
                config=cfg,policy_id=io.identity(cfg,before),source_sha256=before,
                status='driver_failure',failure=type(exc).__name__+':'+str(exc),
                traceback=traceback.format_exc(),methods=[],seconds=perf_counter()-began)
    record['source_unchanged']=before==source_hashes()
    record['actual_solver_calls']=counters
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    parser.add_argument('--family-offset',type=int,default=0)
    parser.add_argument('--training-points',type=int,default=96)
    parser.add_argument('--meter-schedule',choices=('uniform','alternating'),default='uniform')
    parser.add_argument('--no-mean-ablation',action='store_true')
    parser.add_argument('--solver-seconds',type=float,default=2.)
    args=parser.parse_args();cfg=deepcopy(CONFIG);cfg.update(training_points=args.training_points,
        meter_schedule=args.meter_schedule,public_voltage_repeat_counts=[5]*6 if args.meter_schedule=='uniform' else [1,9]*3,
        mean_policy_ablation=not args.no_mean_ablation)
    cfg['exchange']['solver_seconds']=args.solver_seconds
    frozen=freeze(args.output,cfg)
    for index,seed in enumerate(args.seeds):
        path=args.output/f'trial_{seed}.json.gz'
        if path.exists():raise ValueError('existing trial retained; choose new output/seed')
        record=run_trial(seed,(index+args.family_offset)%4,cfg)
        if record['policy_id']!=frozen['policy_id'] or not record['source_unchanged']:raise RuntimeError('source/config changed during study')
        io.write_json(path,record,compressed=True)
        io.write_json(args.output/'checkpoint.json',dict(completed=index+1,planned=len(args.seeds),last_seed=seed,policy_id=frozen['policy_id']))
        print(json.dumps(dict(seed=seed,status=record['status'],seconds=record['seconds'],
            methods=[dict(name=r['name'],M=r.get('M'),status=r['status']) for r in record['methods']])),flush=True)
        if record['status']=='driver_failure':raise RuntimeError('driver failure saved; batch stopped for review')


if __name__=='__main__':main()
