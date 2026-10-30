"""Finite historical compensation faces and one-step WZZT price replay.

The planner accepts observed history, never an actual future or test topology.
Equal weights on finitely many endpoint predictions are a heuristic, not a
posterior or coverage assertion. Original nonnegative coefficient bounds stay.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
sys.path[:0]=[str(REPO),str(REPO/'rnj_wzzt_core'),str(REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate')]
import gate_driver as base
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays,matrices
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract
from controls import menus,enumerate_states
WAVE_KW=np.r_[np.full(4,-.2),np.full(4,.2)]

def family(point):return base.canonical(point['supports'])
def effective(point):
    return {tuple(s) for s,r,x in zip(point['supports'],point['r'],point['x']) if max(r,x)>1e-9}
def errors(point,target):return len(effective(point)^set(target))
def predict(p,q,point):
    r,x=matrices(p.shape[1],point['supports'],point['r'],point['x'])
    return p@r.T+q@x.T
def features(p,q,pool):
    z=np.zeros((len(pool),p.shape[1]))
    for row,s in zip(z,pool):row[list(s)]=1
    return np.column_stack([((p@v)[:,None]*v).ravel() for v in z]+[((q@v)[:,None]*v).ravel() for v in z])
def state_id(state):return ','.join(str(int(v)) for row in state for v in row)
def input_for_state(mean_p,kappa,state):
    p=np.tile(mean_p,(2,1));delta=np.zeros_like(p)
    for event,row in enumerate(state):delta[event*96+76:event*96+84]=WAVE_KW[:,None]*np.asarray(row)[None,:]/200.
    return p+delta,kappa*(p+delta)

def choose(points,cal):
    best=min(p['validation_mae'] for p in points)
    eligible=[(i,p) for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
    i,p=min(eligible,key=lambda v:(len(v[1]['supports']),v[1]['validation_mae'],v[0]))
    return contract(p,cal),p,i

def profile_endpoints(train,valid,pool,F,candidates,cap,v_cap,kappa,budget,out):
    """All parameters reoptimized in both train and validation constraints."""
    p,q,y=arrays(train);pv,qv,yv=arrays(valid);vf=features(pv,qv,pool)
    endpoints=[];receipts=[]
    for C,G in candidates:
        budget.check();lp=PoolLP(None,pool,data=(p,q,y))
        try:
            lp.model.Params.TimeLimit=min(15.,max(.001,budget.deadline-base.perf_counter()))
            before=len(lp.solves)
            try:lp.fit(G)
            finally:
                budget.calls+=len(lp.solves)-before
                receipts.append(dict(C=C,G=G,kind='initial_fixed_family_fit',LP_solves=deepcopy(lp.solves)))
                base.write(out,dict(receipts=receipts,endpoints=endpoints))
            # Do not discard a family using one arbitrary validation fit.
            k=len(pool);active=[lp.lookup[s] for s in G]
            upper=np.zeros(2*k);upper[active]=upper[np.asarray(active)+k]=2.
            lp.weights.UB=upper
            lp.model.addConstr(lp.t.sum()/y.size<=cap)
            tv=lp.model.addMVar(yv.size,lb=0.)
            lp.model.addConstr(vf@lp.weights-tv<=yv.ravel())
            lp.model.addConstr(vf@lp.weights+tv>=yv.ravel())
            lp.model.addConstr(tv.sum()/yv.size<=v_cap)
            j=lp.lookup[C]
            for direction in ('min','max'):
                budget.check();lp.model.Params.TimeLimit=min(15.,max(.001,budget.deadline-base.perf_counter()))
                sign=1 if direction=='min' else -1
                lp.model.setObjective(sign*(lp.weights[j]+kappa*lp.weights[j+k]))
                lp.model.optimize();budget.calls+=1
                row=dict(C=C,G=G,direction=direction,status=int(lp.model.Status),train_cap=cap,validation_cap=v_cap)
                if lp.model.Status==lp.gp.GRB.OPTIMAL:
                    w=np.asarray(lp.weights.X).copy();w[upper==0]=0.
                    tr=float(np.mean(abs(lp.features@w-y.ravel())))
                    va=float(np.mean(abs(vf@w-yv.ravel())))
                    row.update(theta=float(w[j]+kappa*w[j+k]),train_loss=tr,validation_loss=va,solver_objective=float(lp.model.ObjVal),
                               max_bound_violation=float(lp.model.BoundVio),max_constraint_violation=float(lp.model.ConstrVio))
                    if tr>cap+1e-8 or va>v_cap+1e-8 or w.min() < -1e-8 or w.max()>2+1e-8:
                        raise RuntimeError('profile endpoint failed independent primal checks')
                    w=np.clip(w,0.,2.)
                    point=dict(supports=[list(s) for s in G],r=w[active].tolist(),x=w[np.asarray(active)+k].tolist(),
                               origin='historical_compensation_'+direction,profile_C=list(C))
                    endpoints.append(point)
                elif lp.model.Status!=lp.gp.GRB.INFEASIBLE:
                    row['unresolved']=True
                receipts.append(row);base.write(out,dict(receipts=receipts,endpoints=endpoints))
        finally:lp.close()
    return endpoints,receipts

def run(history_packet,history_state,settings,budget,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    train,valid,days,hashes,_=base._observations(history_packet['training'],history_packet['validation'])
    hist=base._run_method('current_pool_wzzt',history_state,train,valid,history_state['pool'],settings,budget,
                          day_ids=days,current_state=history_state)
    base.write(out/'historical_wzzt.json',hist)
    if hist['status']!='complete' or not hist['all_scans_complete']:
        raise RuntimeError('historical WZZT not fully stopped')
    raw=hist['selected']['selected_uncontracted'];F=family(raw)
    pool=base.canonical(history_state['pool']);n=len(history_state['terminal_labels']);kappa=.35
    candidates=[]
    for C in pool:
        if len(C)<2 or C in F:continue
        G=base.canonical([s for s in F if base.compatible(s,C)]+[C])
        candidates.append((C,G))
    tau=1e-10+.001*float(raw['objective'])
    cap=float(raw['objective'])+tau
    v_cap=1.03*min(p['validation_mae'] for p in hist['points'])+1e-12
    endpoints,profile_receipts=profile_endpoints(train,valid,pool,F,candidates,cap,v_cap,kappa,budget,out/'profiles.json')
    models=[];model_keys={}
    for point in [raw,*endpoints]:
        r,x=matrices(n,point['supports'],point['r'],point['x']);K=r+kappa*x
        key=np.round(K,10).tobytes();target=effective(point)
        if key in model_keys:
            old=models[model_keys[key]]
            if set(map(tuple,old['target']))!=target:raise RuntimeError('same prediction matrix, inconsistent positive support')
            old['origins'].append(point.get('origin'));continue
        model_keys[key]=len(models)
        models.append(dict(id='m'+str(len(models)),point=point,K=K.tolist(),target=sorted(target),
                           origins=[point.get('origin')]))
    if any(r.get('unresolved') for r in profile_receipts):raise RuntimeError('profile face unresolved; no price fallback')
    p,q,yh=arrays(train);pv,qv,yv=arrays(valid)
    mean_p=p.reshape(3,96,n).mean(axis=0)
    # The declared synthetic load/DR model uses q=.35p. This is a planning
    # approximation, not a claim that noisy measured Q is exactly proportional.
    native=[base.canonical(t['family']) for t in history_state['native_bank']]
    families=tuple(dict.fromkeys([F,*native,*[G for C,G in candidates]]))
    menu_list=menus();states=sorted({s for menu in menu_list for s,prob in enumerate_states(menu)})
    inputs={s:input_for_state(mean_p,kappa,s) for s in states}
    profiles=dict(F=F,candidates=[dict(C=C,G=G,deleted=[s for s in F if s not in G]) for C,G in candidates],
        models=models,planning_families=families,train_cap=cap,validation_cap=v_cap,mean_forecast='three observed training days by quarter; Q=.35P declared synthetic forecast',
        source_information='historical observed only',history_hashes=hashes,
        finite_endpoint_weights='uniform heuristic after K deduplication; no posterior or full parameter coverage',
        calibration='historical raw; actual evaluation must rebuild current observed calibration')
    base.write(out/'design_state.json',profiles)
    scores={menu['id']:dict(expected_final_errors=0.,expected_raw_errors=0.,expected_full=0.,
                            expected_acceptance=0.,expected_payment=0.) for menu in menu_list}
    cache={}
    for m in models:
        K=np.asarray(m['K']);target=set(map(tuple,m['target']))
        for s in states:
            budget.check();pf,qf=inputs[s];yf=pf@K.T
            pp=np.concatenate([p,pf]);qq=np.concatenate([q,qf]);yy=np.concatenate([yh,yf])
            lp=PoolLP(None,pool,data=(pp,qq,yy))
            points={}
            try:
                view=base.SearchView(lp,pool,budget)
                for fam in families:points[fam]=view.fit(fam)
                current=points[F]
                best=None
                for C,G in candidates:
                    if best is None or points[G]['objective']<best['objective']-1e-12:best=points[G]
                accept=best is not None and current['objective']-best['objective']>1e-10+.001*current['objective']
                proposed=[points[f] for f in native]+[current]+([best] if accept else [])
                # cache per fixed executed training state, not per price menu
                info=dict(accepted=accept,points=proposed,LP_solves=lp.solves)
                base.write(out/'training_states'/f"{m['id']}_{state_id(s).replace(',','_')}.json",info)
                cache[m['id'],s]=info
            finally:lp.close()
        for menu in menu_list:
            law=enumerate_states(menu)
            acc=err=rawerr=full=0.
            for st,pt in law:
                info=cache[m['id'],st];acc+=pt*info['accepted']
                proposed=info['points']
                hist_val=[float(np.sum(abs(predict(pv,qv,z)-yv))) for z in proposed]
                # validation states are independent of training; no shared mask
                for sv,p_valid in law:
                    pf,qf=inputs[sv];yf=pf@K.T
                    measured=[]
                    for point,oldloss in zip(proposed,hist_val):
                        val=(oldloss+float(np.sum(abs(predict(pf,qf,point)-yf))))/(yv.size+yf.size)
                        measured.append(dict(point,validation_mae=val))
                    final,rawpoint,index=choose(measured,history_state['calibration'])
                    weight=pt*p_valid
                    err+=weight*errors(final,target);rawerr+=weight*errors(rawpoint,target)
                    full+=weight*(errors(final,target)==0)
            row=scores[menu['id']]
            row['expected_final_errors']+=err/len(models);row['expected_raw_errors']+=rawerr/len(models)
            row['expected_full']+=full/len(models);row['expected_acceptance']+=acc/len(models)
        base.write(out/'planning_progress.json',dict(completed_model=m['id'],models=len(models),LP_calls=budget.calls,scores=scores))
    probability=np.array([0.,.5,.75])
    for menu in menu_list:
        price=np.asarray(menu['price_rows'])
        scores[menu['id']]['expected_payment']=.4*float(np.sum(price*probability[price]))
    best=min(v['expected_final_errors'] for v in scores.values())
    eligible=[m for m in menu_list if scores[m['id']]['expected_final_errors']<=best+1e-12]
    pay=min(scores[m['id']]['expected_payment'] for m in eligible)
    tied=[m for m in eligible if scores[m['id']]['expected_payment']<=pay+1e-12]
    result=dict(status='complete',scores=scores,menus=menu_list,tied_menu_ids=[m['id'] for m in tied],
                model_count=len(models),family_count=len(families),unique_training_state_count=len(states),
                LP_calls=budget.calls,history_hashes=hashes,truth_or_actual_future_read=False,
                scope='finite face endpoints, one-step closure, historical calibration, mean-load/no-new-measurement-error prediction; actual full path still required')
    base.write(out/'price_design.json',result)
    return result
