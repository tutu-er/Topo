"""Matched v3 windows/training lengths; original v2 generators are unchanged."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
from . import ev_mobile_cases_v2 as old
from . import ev_original_study as base
from . import likelihood as lk
from .ev_dynamic_loads import generate_background
from .ev_fixed_train_fit import subset_observations, subset_model
from .ev_mobile_service_domain import build_mobile_service_domain
from .simulation import model


def make_case(bank, seed, family_index, *, H=96, T=12):
    """Only generation/evaluation may receive this object, never a selector."""
    if bank not in ('n3', 'n4') or H not in (8, 96) or T not in (8, 12):
        raise ValueError('declared n3/n4, H8/96, T8/12 required')
    cid = 'remote8' if bank == 'n3' else 'twinblock12'
    cfg = old.case_config(cid, training_points=96)
    cfg.update(version='matched_mobile_v3', training_points=H, horizon=T,
        public_voltage_repeat_counts=[30./T]*T,
        future_metering_integration_seconds_per_slot=3000./T)
    candidate, r, x, seeds, assets = old._assets(seed, cid, family_index)
    n = candidate.n
    # Common 112-point trajectory also when T=8: changing an operation window
    # must not silently redraw the training record or the forecast origin.
    bg = generate_background(seeds['background'], 112, n)
    p, q = bg['p_kw']/base.BASE_KW, bg['q_kvar']/base.BASE_KW
    sp, sq = bg['source_p_kw']/base.BASE_KW, bg['source_q_kvar']/base.BASE_KW
    root = bg['root_v']; meter = model(n)
    sequences = np.random.SeedSequence(seeds['noise']).spawn(len(base.CHANNELS))
    white = {k: np.random.default_rng(s).standard_normal((112,n) if k in ('p','q','v') else 112)
             for k,s in zip(base.CHANNELS,sequences)}
    physical = lk.ac_predict(candidate,r,x,p,q,sp,sq,root)
    full = base.observation(physical,meter,white)
    tr, va = slice(96-H,96), slice(96,100)
    fm = replace(meter, v_sigma=np.broadcast_to(meter.v_sigma/np.sqrt(30./T),(T,n)),
        **{k+'_sigma':getattr(meter,k+'_sigma')/np.sqrt(30./T)
           for k in ('p','q','master_p','master_q')})
    return dict(case_id=cid,seed=int(seed),family_index=family_index,config=cfg,
        train=subset_observations(full,tr),validation=subset_observations(full,va),
        tm=subset_model(meter,full,tr),vm=subset_model(meter,full,va),fm=fm,
        p=p,q=q,source_p=sp,source_q=sq,root_v=root,white=white,future_start=100,
        base_kw=base.BASE_KW,truth=dict(candidate=candidate,r=r,x=x),bank=old.candidate_bank(cid),
        provenance=dict(data_kind='matched_v3_synthetic_ordinary_LV',assets=assets,seeds=seeds,
            source_interface=1,total_effective_precision=30.,H=H,T=T,
            true_labels_used_only_for_generation_and_evaluation=True))


def service_domain(T,n,*,mobile):
    template = old.service_domain('twinblock12',mobile=mobile)
    return build_mobile_service_domain(replace(template.config,n_slots=T,n_sites=n))


observe_future = old.observe_future
forecast_for = old.forecast_for
