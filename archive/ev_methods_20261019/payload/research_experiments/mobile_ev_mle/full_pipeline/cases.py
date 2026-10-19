"""Measured-import-driven synthetic complete radial cases, truth kept here."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from .. import likelihood as lk
from ..simulation import model
from ..ev_original_study import observation
from ..ev_fixed_train_fit import subset_observations,subset_model
from .data import synthetic_reactive_power

def _seed(seed,stage):
    return int(np.random.SeedSequence([int(seed),int(stage)]).generate_state(1)[0])

def clades(n,kind,seed):
    """Topology generation alone; never called by the planner."""
    rng=np.random.default_rng(seed);labels=list(map(int,rng.permutation(n)));out=[]
    def balanced(items):
        if len(items)<2:return
        if len(items)<n:out.append(tuple(sorted(items)))
        middle=len(items)//2;balanced(items[:middle]);balanced(items[middle:])
    if kind%4==0:balanced(labels)
    elif kind%4==2:
        cut=max(2,n//2-1);balanced(labels[:cut]);balanced(labels[cut:])
    elif kind%4==1:
        out=[tuple(sorted(labels[:k])) for k in range(2,n)]
    else:
        out=[tuple(sorted(labels[:2])),tuple(sorted(labels[2:4]))]
    return tuple(sorted(set(out),key=lambda x:(len(x),x)))

def asset_parameters(candidate,seed,weak):
    """Public cable ranges; lengths are explicitly synthetic, in kilometres."""
    rng=np.random.default_rng(seed)
    cable=np.array([[.206,.080],[.320,.085],[.641,.090]])
    idx=rng.integers(0,len(cable),candidate.m);length=rng.uniform(.03,.13,candidate.m)
    for j,s in enumerate(candidate.supports):
        if len(s)==candidate.n:length[j]=rng.uniform(.02,.05)
        elif len(s)>1 and weak:length[j]=rng.uniform(.005,.015)
    zbase=.4**2/.020
    r=cable[idx,0]*length/zbase;x=cable[idx,1]*length/zbase
    return r,x,dict(cable_R_X_ohm_per_km=cable.tolist(),length_km=length.tolist(),
        cable_index=idx.tolist(),lengths_synthetic=True,zbase_ohm=zbase,
        public_length_domain_km=[.005,.25],weak_shared_branches=bool(weak),
        source='declared public LV conductor range and synthetic positive lengths; not field line data')

def fit_options(config):
    return dict(r_bounds=(.206*.005/8.,.641*.25/8.),
        x_bounds=(.080*.005/8.,.090*.25/8.),ratio_bounds=(.080/.641,.090/.206),
        source_p_bounds=(-.5,.5),source_q_bounds=(-.5,.5),voltage_bounds=(.85,1.15),
        multistart=config['fit_multistart'],max_nfev=config['fit_max_nfev'],tol=config['fit_tol'])

def make_case(spec,window,config):
    """Simulator receives full measured-P window; selector never gets this dict."""
    H,T,n=config['H'],config['T'],int(spec['n']);seed=int(spec['seed'])
    physical_seed=int(spec.get('physical_seed',seed))
    p_raw=np.vstack((window['history']['p_kw'],window['future']['p_kw']))[:,:n]
    stamps=window['history']['timestamps_utc']+window['future']['timestamps_utc']
    if p_raw.shape!=(H+T,n):raise ValueError('window dimensions differ from frozen experiment')
    # Training mean only; the published raw values and all factors are retained.
    history_mean=np.mean(p_raw[:H],axis=0)
    factors=2./np.maximum(history_mean,.1)
    p_kw=p_raw*factors
    reactive=synthetic_reactive_power(p_kw,stamps,seed=_seed(physical_seed,10))
    q_kvar=reactive['q_kvar'];rng=np.random.default_rng(_seed(physical_seed,20))
    sp=np.zeros(H+T);sq=np.zeros(H+T)
    if spec['source_mode']=='DER':
        hours=np.asarray(pd.DatetimeIndex(pd.to_datetime(stamps,utc=True)).hour,float)
        hours+=np.asarray(pd.DatetimeIndex(pd.to_datetime(stamps,utc=True)).minute)/60.
        cloud=np.zeros(H+T);state=0.
        for t in range(H+T):
            state=.8*state+.08*rng.normal();cloud[t]=np.clip(.7+state,.1,1.)
        sp=-3.*np.maximum(np.sin(np.pi*(hours-6.)/12.),0)*cloud
        sq=.25*sp+.10*rng.normal(size=H+T)
    elif spec['source_mode']!='none':raise ValueError('registered source_mode required')
    family=clades(n,spec['cluster_index']%4,_seed(physical_seed,30))
    true=lk.RadialCandidate.from_supports('simulator_truth',n,family)
    r,x,assets=asset_parameters(true,_seed(physical_seed,40),weak=(spec['cluster_index']%4==3))
    base=config['base_kw'];p=p_kw/base;q=q_kvar/base;sp=sp/base;sq=sq/base;root=np.ones(H+T)
    physical=lk.ac_predict(true,r,x,p,q,sp,sq,root)
    mm=model(n)
    rng=np.random.default_rng(_seed(physical_seed,50))
    white={name:rng.standard_normal((H,n) if name in ('p','q','v') else H)
           for name in ('p','q','v','master_p','master_q')}
    hist_physical=lk.ac_predict(true,r,x,p[:H],q[:H],sp[:H],sq[:H],root[:H])
    hist=observation(hist_physical,mm,white)
    weight=config['total_precision_per_channel']/T
    fm=replace(mm,v_sigma=np.broadcast_to(np.asarray(mm.v_sigma)/np.sqrt(weight),(T,n)),
        **{name+'_sigma':getattr(mm,name+'_sigma')/np.sqrt(weight)
           for name in ('p','q','master_p','master_q')})
    return dict(spec=spec,history=hist,history_meter=mm,future_meter=fm,
        timestamps=window['history']['timestamps_utc'],future_timestamps=window['future']['timestamps_utc'],
        p=p,q=q,source_p=sp,source_q=sq,root_v=root,
        truth=dict(candidate=true,r=r,x=x),base_kw=base,H=H,T=T,
        provenance=dict(raw=window['provenance'],n=n,source_mode=spec['source_mode'],
            real_P_scope='measured building grid-import shape assigned as synthetic feeder load',
            Q=reactive['metadata'],source_Q_included=True,EV_Q=0.,assets=assets,
            scaling=dict(rule='2kW divided by max(history mean,0.1kW)',
                history_mean_kw=history_mean.tolist(),factors=factors.tolist(),future_used=False),
            actual_initial_voltage_range=[float(np.min(physical['v'])),float(np.max(physical['v']))],
            known_root=True,known_calibrated_bias=True,EV_tracking_exact=True,
            future_available_to_selector=False))

def planning_inputs(case):
    """Exact allowlist: no case metadata, truth, future values or source class."""
    return dict(history=case['history'],history_meter=case['history_meter'],future_meter=case['future_meter'],
        timestamps=case['timestamps'],horizon=case['T'],base_kw=case['base_kw'])

def actual_observation(case,power,white,*,clean=False):
    H,T=case['H'],case['T'];truth=case['truth'];power=np.asarray(power,float)
    physical=lk.ac_predict(truth['candidate'],truth['r'],truth['x'],
        case['p'][H:H+T]+power/case['base_kw'],case['q'][H:H+T],
        case['source_p'][H:H+T],case['source_q'][H:H+T],case['root_v'][H:H+T])
    noise={k:np.zeros_like(v) for k,v in white.items()} if clean else white
    obs=observation(physical,case['future_meter'],noise)
    return obs,physical,dict(p=case['p'][H:H+T],q=case['q'][H:H+T],
        source_p=case['source_p'][H:H+T],source_q=case['source_q'][H:H+T],root_v=case['root_v'][H:H+T])
