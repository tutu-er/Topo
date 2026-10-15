"""Declared broad case registry; truth stays in generation/evaluation.

Ordinary train8/32/96 share the same 96-point training endpoint and future for
the same seed. Old scientific modules and generator globals are never changed.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import numpy as np
from . import likelihood as lk
from . import ev_original_study as base
from . import ev_fixed_train_design as fixed
from . import ev_rx_study as rx
from .ev_dynamic_loads import generate_background,fit_forecast
from .ev_fixed_train_fit import subset_observations,subset_model
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
from .simulation import model,generate
from .ev_physical_cases import case33_local_standard

CASES=('ordinary96','train32','train8','correlated96','rich32','alternating96',
       'no_source96','strong_source96','weak_numeric96','weak_asset96',
       'bias_misspec96','candidate_miss96','access96','long_horizon96',
       'case33local12','legacy_n5_12')
FORMAL_CASES=('ordinary96','train32','train8','correlated96','alternating96','strong_source96')
STRESS=('weak_asset96','bias_misspec96','candidate_miss96')


def candidate_bank(case_id):
    if case_id not in CASES:raise ValueError('unknown declared case')
    families=list(base.FAMILIES)
    if case_id=='legacy_n5_12':
        families=[(),((0,1),),((0,1),(0,1,2)),((1,2),(0,1,2))]
        n=5
    else:n=3
    if case_id=='candidate_miss96':families=families[:2]
    return [lk.RadialCandidate.from_supports('star' if not f else
        'clade_'+'_'.join(''.join(map(str,s)) for s in f),n,f) for f in families]


def case_config(case_id):
    if case_id not in CASES:raise ValueError('unknown declared case')
    cfg=deepcopy(rx.CONFIG)
    H=8 if case_id=='train8' else 32 if case_id in ('train32','rich32') else 12 if case_id.endswith('12') else 96
    T=8 if case_id=='long_horizon96' else 6
    cfg.update(case_id=case_id,training_points=H,validation_points=4,horizon=T,
        forecast_gap=4,training_domain='cone',forecast='ar1',base_kw=20.,
        public_voltage_repeat_counts=([1,9,1,9,1,9] if case_id=='alternating96' else
            [5]*6 if T==6 else [4,4,4,4,4,4,3,3]),
        stress=case_id in STRESS,whole_training_sequence=96,
        source_interface=1,root_fixed=True,PQ_inputs_fixed=True,
        future_sampling_budget_per_channel=30)
    cfg['training'].update(multistart=2,max_nfev=250,tol=1e-9)
    cfg['profile'].update(multistart=2,max_nfev=120,tol=1e-9)
    if case_id in ('weak_numeric96','case33local12','legacy_n5_12'):
        cfg['training'].update(r_bounds=[1e-6,.1],x_bounds=[1e-6,.1])
    if case_id in ('case33local12','legacy_n5_12'):
        cfg.update(ratio_bounds=[.05,1.5],forecast='mean')
    if case_id=='case33local12':cfg['base_kw']=1000.
    return cfg


def _rebuild_q(p,q,newp):
    return newp*np.divide(q,p,out=np.zeros_like(q),where=p!=0)


def make_case(case_id,seed,family_index):
    cfg=case_config(case_id);H,T=cfg['training_points'],cfg['horizon'];N=108
    fi=int(family_index)%4
    if case_id.startswith('weak_'):fi=1+fi%3
    if case_id=='candidate_miss96':fi=2+fi%2
    assets=base.make_case(seed,base.FAMILIES[fi],short_edge=case_id.startswith('weak_'))
    r,x,candidate=assets['r'],assets['x'],assets['candidate'];n=3
    background=generate_background(assets['seeds']['background'],N,n)
    p=background['p_kw']/20.;q=background['q_kvar']/20.
    sp=background['source_p_kw']/20.;sq=background['source_q_kvar']/20.
    source_provenance={'base':'ordinary_stationary_ar1_pf_v1','transforms':[]}
    if case_id=='correlated96':
        means=np.resize([2.4,2.8,2.1],n)/20.
        common=np.mean(p-means,axis=1,keepdims=True)
        newp=means+common+.02*(p-means-common)
        q=_rebuild_q(p,q,newp);p=newp
        source_provenance['transforms'].append('household deviations shrunk to 2%; common variation retained')
    if case_id=='rich32':
        rng=np.random.default_rng(assets['seeds']['background']+104729)
        p[:96]=(2.4+rng.uniform(-.6,.6,(96,n)))/20.
        q[:96]=(.8+rng.uniform(-.3,.3,(96,n)))/20.
        source_provenance['transforms'].append('independent rich training P/Q; ordinary future retained')
    if case_id=='no_source96':sp=np.zeros(N);sq=np.zeros(N)
    if case_id=='strong_source96':
        sp=.1+4*(sp-.1);sq=.035+4*(sq-.035)
        source_provenance['transforms'].append('fourfold source deviations; original fitting bounds retained')
    if case_id=='case33local12':
        physical=case33_local_standard();candidate=physical.candidate;r,x=physical.r,physical.x
        rng=np.random.default_rng(assets['seeds']['background']);common=rng.uniform(-.01,.01,(N,1))
        p=physical.p0*(1+common+rng.uniform(-.001,.001,(N,n)))
        q=physical.q0*(1+common+rng.uniform(-.001,.001,(N,n)))
        sr=np.random.default_rng(assets['seeds']['source'])
        sp=(10+sr.uniform(-2,2,N))/1000.;sq=(3+sr.uniform(-1,1,N))/1000.
        source_provenance=dict(standard=physical.provenance,time_series='synthetic correlated ordinary variation')
    if case_id=='legacy_n5_12':
        path=generate(seed,'strong',n=5,history_count=100,future_count=8)
        candidate,r,x=path.candidate,path.r,path.x;n=5
        p,q,sp,sq=path.p,path.q,path.source_p,path.source_q
        source_provenance=dict(generator='simulation.generate strong n5',predeclared_nested_family=True)
    meter=model(n);root=np.ones(N)
    sequences=np.random.SeedSequence(assets['seeds']['noise']).spawn(len(base.CHANNELS))
    white={k:np.random.default_rng(s).standard_normal((N,n) if k in ('p','q','v') else N)
        for k,s in zip(base.CHANNELS,sequences)}
    phy=lk.ac_predict(candidate,r,x,p,q,sp,sq,root)
    generating_meter=meter
    if case_id=='bias_misspec96':generating_meter=replace(meter,bias_v=np.linspace(-.00065,.00065,n))
    full=base.observation(phy,generating_meter,white)
    rows=slice(96-H,96);vrows=slice(96,100)
    training=subset_observations(full,rows);validation=subset_observations(full,vrows)
    tm=subset_model(meter,full,rows);vm=subset_model(meter,full,vrows)
    repeats=np.asarray(cfg['public_voltage_repeat_counts'])
    fm=replace(meter,v_sigma=np.broadcast_to(meter.v_sigma/np.sqrt(repeats[:,None]),(T,n)),
        **{k+'_sigma':getattr(meter,k+'_sigma')/np.sqrt(30/T) for k in ('p','q','master_p','master_q')})
    return dict(case_id=case_id,seed=int(seed),family_index=fi,config=cfg,train=training,
        validation=validation,tm=tm,vm=vm,fm=fm,p=p,q=q,source_p=sp,source_q=sq,
        root_v=root,white=white,future_start=100,base_kw=cfg['base_kw'],
        truth=dict(candidate=candidate,r=np.asarray(r),x=np.asarray(x)),
        provenance=source_provenance,bank=candidate_bank(case_id),
        truth_used_only_for_generation_and_evaluation=True)


def forecast_for(tr,tm,fits,cfg):
    if cfg['case_id']!='case33local12':
        return fit_forecast(tr,tm,fits,horizon=cfg['horizon'],gap=4,base_kw=cfg['base_kw'],kind=cfg['forecast'])
    # The old household 12-kW clipping domain is inappropriate at 120 kW.
    T=cfg['horizon'];p=tr.p-np.asarray(tm.bias_p);q=tr.q-np.asarray(tm.bias_q)
    return dict(p=np.tile(p.mean(axis=0),(T,1)),q=np.tile(q.mean(axis=0),(T,1)),root=np.ones(T),
        source_by_candidate={name:dict(source_p=np.full(T,np.mean(f.source_p)),
            source_q=np.full(T,np.mean(f.source_q))) for name,f in fits.items()},
        metadata=dict(training_only=True,source_interface=1,uses_validation=False,
            uses_truth=False,uses_future_noise=False,kind='declared un-clipped physical-case PQ mean',
            no_best_time_claim=True,base_kw=cfg['base_kw']))


def service_domain(case_id,T,n,*,mobile):
    if case_id!='access96':return fixed._service(T,n,mobile=mobile,discharge=True)
    access=np.ones((T,n),bool)
    access[:,1:]=False
    if mobile:access[3:,1:]=True
    c=fixed.CONFIG
    return build_mobile_service_domain(MobileServiceConfig(n_slots=T,n_sites=n,
        dt_hours=c['dt_hours'],p_charge_max_kw=c['command_power_kw'],p_discharge_max_kw=c['command_power_kw'],
        allow_discharge=True,battery_capacity_kwh=c['capacity_kwh'],
        initial_soc=c['initial_energy_kwh']/c['capacity_kwh'],terminal_energy_kwh=c['terminal_energy_kwh'],
        eta_charge=c['eta_charge'],eta_discharge=c['eta_discharge'],access=access,mobile=mobile,
        travel_slots=c['travel_slots'],setup_slots=c['setup_slots'],travel_energy_kwh=c['drive_kwh'],
        max_moves=c['max_moves'],max_throughput_kwh=c['throughput_kwh'],travel_cost=c['travel_worker_minutes'],
        setup_cost=c['setup_worker_minutes'],driver_budget=c['driver_budget_minutes'],
        cost_units='worker_minutes',min_arrival_throughput_kwh=c['min_arrival_throughput_kwh']))


def observe_future(case,power,*,clean=False):
    sl=slice(100,100+case['config']['horizon']);a=case['truth']
    pred=lk.ac_predict(a['candidate'],a['r'],a['x'],case['p'][sl]+np.asarray(power)/case['base_kw'],
        case['q'][sl],case['source_p'][sl],case['source_q'][sl],case['root_v'][sl])
    white={k:np.zeros_like(v[sl]) if clean else v[sl] for k,v in case['white'].items()}
    meter=case['fm']
    if case['case_id']=='bias_misspec96':meter=replace(meter,bias_v=np.linspace(-.00065,.00065,case['train'].n))
    return base.observation(pred,meter,white)
