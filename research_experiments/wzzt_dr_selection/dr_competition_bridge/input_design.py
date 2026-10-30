"""Ordinary input-design controls on the same price/response contract."""
from __future__ import annotations
import numpy as np
from .physical_bridge import forecast_mean, apply_response
from .pricing_tables import (enumerate_menus, single_node_menus, menu_tie_key,
                            payment, enumerate_training_outcomes, masks_from_key)


def design_data(history):
    forecast=forecast_mean(history)
    past=np.vstack([np.column_stack((d['P_terminal'].to_numpy(),d['Q_terminal'].to_numpy()))
                    for d in history['training']+history['validation']])
    scale=np.maximum(np.sqrt(np.mean(past*past,axis=0)),1e-12)
    natural=np.column_stack((forecast['P_pu'],forecast['Q_pu']))/scale
    past=past/scale
    base=past.T@past+2*natural.T@natural+1e-6*np.eye(past.shape[1])
    h=natural[80:84].sum(axis=0)-natural[76:80].sum(axis=0)
    return forecast,scale,base,h


def mean_scores(menus,history):
    """Exact Gram for the mean-response surrogate, not E logdet."""
    f,scale,base,h=design_data(history)
    probabilities=np.array([0.,.5,.75])
    out=[]
    for start in range(0,len(menus),512):
        block=np.asarray(menus[start:start+512])
        gram=np.broadcast_to(base,(len(block),*base.shape)).copy()
        for row in range(2):
            p=probabilities[block[:,row,:]]
            delta=np.concatenate((.2/f['base_kw']*p,.07/f['base_kw']*p),axis=1)/scale
            gram+=8*delta[:,:,None]*delta[:,None,:]+h[None,:,None]*delta[:,None,:]+delta[:,:,None]*h[None,None,:]
        signs,values=np.linalg.slogdet(gram)
        if not (signs>0).all():raise RuntimeError('nonpositive input Gram')
        out.extend(values.tolist())
    return np.asarray(out)


def select_max(menus,scores,seed):
    scores=np.asarray(scores)
    if not np.isfinite(scores).all():raise ValueError('unresolved input score')
    eligible=np.flatnonzero(scores>=scores.max()-1e-12)
    best=min(eligible,key=lambda i:(payment(menus[i])['expected'],menu_tie_key(menus[i],seed),int(i)))
    return dict(menu=np.asarray(menus[best]).tolist(),index=int(best),value=float(scores[best]),
                tied_indices=eligible.tolist(),candidate_count=len(menus))


def ordinary_plans(history,seed):
    n=len(history['labels'])
    menus=enumerate_menus(n)
    all_result=select_max(menus,mean_scores(menus,history),seed)
    single=single_node_menus(n)
    one=select_max(single,mean_scores(single,history),seed+1)
    sync=[]
    for i in range(n):
        for j in range(i+1,n):
            menu=np.zeros((2,n),dtype=np.uint8);menu[:,[i,j]]=1;sync.append(menu)
    sync=np.stack(sync)
    two=select_max(sync,mean_scores(sync,history),seed+2)
    rng=np.random.default_rng(seed+3)
    a,b=rng.choice(n,2,replace=False)
    rotate=np.zeros((2,n),dtype=int);rotate[0,a]=2;rotate[1,b]=2
    return dict(plans=dict(zero=np.zeros((2,n),dtype=int).tolist(),
                    single_input=one['menu'],sync_input=two['menu'],rotation=rotate.tolist(),
                    global_mean_input=all_result['menu']),
                receipts=dict(single_input=one,sync_input=two,global_mean_input=all_result,
                    rotation=dict(seed=seed+3,positions=[int(a),int(b)])),
                scope='ordinary mean-response input logdet; full45446 price menus,121 single,105 exactly-two-site synchronous')


def pair_stochastic_plan(history,active_labels,dependence,seed):
    labels=list(history['labels']);positions=[labels.index(v) for v in active_labels]
    small=enumerate_menus(len(positions));cache={}
    f,scale,_,_=design_data(history)
    past=np.vstack([np.column_stack((d['P_terminal'].to_numpy(),d['Q_terminal'].to_numpy()))
                    for d in history['training']+history['validation']])/scale
    gram_past=past.T@past+1e-6*np.eye(2*len(labels))
    values=[]
    for menu in small:
        value=0.
        for key,mass in enumerate_training_outcomes(menu,dependence):
            if key not in cache:
                masks=masks_from_key(key,len(positions));gram=gram_past.copy()
                for row in range(2):
                    z=np.zeros(len(labels),dtype=bool);z[positions]=masks[row]
                    price=np.zeros(len(labels),dtype=int);price[positions]=1
                    p,q,_=apply_response(f['P_pu'],f['Q_pu'],price,z,base_kw=f['base_kw'])
                    d=np.column_stack((p,q))/scale;gram+=d.T@d
                sign,score=np.linalg.slogdet(gram)
                if sign<=0:raise RuntimeError('nonpositive stochastic input Gram')
                cache[key]=float(score)
            value+=float(mass)*cache[key]
        values.append(value)
    selected=select_max(small,values,seed)
    full=np.zeros((2,len(labels)),dtype=int);full[:,positions]=selected['menu']
    return dict(menu=full.tolist(),active_labels=list(active_labels),record=selected,
                cached_training_patterns=len(cache),scope='exact E logdet over same two-site50 menus; no topology data')


def pure_design_check():
    """Check rank-update scores against direct full mean-power construction."""
    import pandas as pd
    rng=np.random.default_rng(8)
    history=dict(labels=[101,102,103],base_kw=200.,training=[],validation=[])
    for d in range(6):
        scene=dict(P_terminal=pd.DataFrame(.02+.005*rng.random((96,3))),
                   Q_terminal=pd.DataFrame(.005+.001*rng.random((96,3))))
        history['training' if d<3 else 'validation'].append(scene)
    menus=enumerate_menus(3)
    fast=mean_scores(menus,history)
    f,scale,_,_=design_data(history)
    past=np.vstack([np.column_stack((d['P_terminal'],d['Q_terminal'])) for d in history['training']+history['validation']])/scale
    for i in (0,1,19,len(menus)-1):
        gram=past.T@past+1e-6*np.eye(6)
        for row in range(2):
            prob=np.array([0.,.5,.75])[menus[i,row]]
            p=f['P_pu'].copy();q=f['Q_pu'].copy()
            for slots,sign in ((slice(76,80),-1),(slice(80,84),1)):
                p[slots]+=sign*.2/200*prob;q[slots]+=sign*.07/200*prob
            x=np.column_stack((p,q))/scale;gram+=x.T@x
        assert np.isclose(np.linalg.slogdet(gram)[1],fast[i],atol=1e-9,rtol=0)
    return dict(passed=True,mean_menus=168,direct_cases=4,solver_calls=0)
