"""Truth-free laminar-family -> radial AC bridge and bounded L1 correction.

This reuses the existing AC forward solver and fixed-family joint L1 LP.
No physical line table or topology is read. All topology comes from the supplied
family. Observed P/Q are held fixed: this is not the latent-P/Q AC likelihood.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
import numpy as np
import pandas as pd

from rnj_wzzt.models.network import TerminalizedNetwork
from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP,canonical,compatible
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays,matrices


def _validate_obs(obs):
    if not isinstance(obs,(tuple,list)) or len(obs)!=4:
        raise ValueError('obs must be normalized (P,Q,V,rootV)')
    p,q,v,root=obs
    if not all(isinstance(x,pd.DataFrame) for x in (p,q,v)) or not isinstance(root,pd.Series):
        raise ValueError('normalized labeled frames and root Series required')
    if not p.columns.equals(q.columns) or not p.columns.equals(v.columns):
        raise ValueError('terminal order must agree exactly')
    if not p.index.equals(q.index) or not p.index.equals(v.index) or not p.index.equals(root.index):
        raise ValueError('observed time order must agree exactly')
    if not all(np.isfinite(x.to_numpy()).all() for x in (p,q,v,root)):
        raise ValueError('observations must be finite')
    if (root.to_numpy()<=0).any() or (v.to_numpy()<=0).any():
        raise ValueError('positive voltage magnitudes required')
    return list(p.columns)


def _native(point,family):
    family=canonical(family)
    lookup={tuple(s):(float(r),float(x)) for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True)}
    if not set(lookup)<=set(family):raise ValueError('initial point has support outside fixed family')
    rr=np.asarray([lookup.get(s,(0.,0.))[0] for s in family]);xx=np.asarray([lookup.get(s,(0.,0.))[1] for s in family])
    if not np.isfinite(rr).all() or not np.isfinite(xx).all() or (rr<0).any() or (xx<0).any():
        raise ValueError('finite nonnegative response coefficients required')
    return dict(supports=[list(s) for s in family],r=rr.tolist(),x=xx.tolist())


def family_network(point,labels,*,base_kv,base_mva,root_label=1):
    """Each supplied support becomes its downstream edge; exact zero retained.

    WZZT coefficients describe squared-voltage response, so a branch's physical
    impedance is .5 * coefficient * (base_kv**2/base_mva), in ohm.
    """
    labels=list(labels);n=len(labels)
    if len(set(labels))!=n or root_label in labels or any(not isinstance(i,(int,np.integer)) for i in labels+[root_label]):
        raise ValueError('unique integer terminal/root labels required')
    if not np.isfinite(base_kv) or not np.isfinite(base_mva) or base_kv<=0 or base_mva<=0:
        raise ValueError('positive physical base_kv/base_mva required')
    family=canonical(point['supports'])
    if set((i,) for i in range(n))-set(family):raise ValueError('every terminal singleton must be supplied')
    if any(not s or min(s)<0 or max(s)>=n for s in family):raise ValueError('support index outside terminal order')
    if any(not compatible(family[:j],s) for j,s in enumerate(family)):raise ValueError('family must be laminar')
    native=_native(point,family)
    hidden=min([int(root_label),*map(int,labels),0])-1
    nodes={}
    for s in family:
        if len(s)==1:nodes[s]=int(labels[s[0]])
        else:nodes[s]=hidden;hidden-=1
    edges=[];zbase=float(base_kv**2/base_mva)
    for s,r,x in zip(family,native['r'],native['x'],strict=True):
        supers=[t for t in family if set(s)<set(t)]
        parent=nodes[min(supers,key=lambda t:(len(t),t))] if supers else int(root_label)
        edges.append(dict(from_bus=parent,to_bus=nodes[s],r_ohm=.5*r*zbase,x_ohm=.5*x*zbase,
                          is_true_closed=True,status=True,branch_type='candidate_support',
                          original_branch_id=len(edges),support_indices=tuple(s)))
    buses=[]
    for node in [int(root_label),*nodes.values()]:
        terminal=node in labels
        buses.append(dict(bus_id=node,bus_type='root' if node==root_label else 'observed_terminal' if terminal else 'hidden_internal',
            is_observed=node==root_label or terminal,has_load=terminal,is_terminal=terminal,pd_kw=0.,qd_kvar=0.))
    return TerminalizedNetwork(pd.DataFrame(buses),pd.DataFrame(edges),int(root_label),float(base_kv),float(base_mva),
        {int(i):int(i) for i in labels},{int(i):int(i) for i in labels},
        dict(case_name='observed_candidate',truth_used=False,coefficient_to_ohm_factor=.5*zbase,zero_impedances_preserved=True))


def predict_ac(point,obs,*,base_kv,base_mva,root_label=1,ac_max_iter=200,ac_tol=1e-12):
    labels=_validate_obs(obs)
    net=family_network(point,labels,base_kv=base_kv,base_mva=base_mva,root_label=root_label)
    p,q,v,root=obs
    ac=solve_ac_power_flow_timeseries(net,p,q,v_root=root,max_iter=ac_max_iter,tol=ac_tol)
    if not bool(ac['converged'].all()):raise RuntimeError('candidate AC did not converge on every observation')
    pred=ac['V_bus_mag'].loc[:,labels]
    drop=root.to_numpy()[:,None]**2-pred.to_numpy()**2
    observed=root.to_numpy()[:,None]**2-v.to_numpy()**2
    return dict(drop=drop,voltage=pred,mae=float(np.mean(np.abs(drop-observed))),
        voltage_rmse=float(np.sqrt(np.mean((pred.to_numpy()-v.to_numpy())**2))),
        convergence=dict(all_converged=True,rows=len(p),max_iterations=int(ac['iterations'].max()),
                         max_voltage_error=float(ac['max_voltage_error'].max())),net=net)


def _lp_native(lp,family):
    fitted=lp.fit(family)
    active=np.asarray([lp.lookup[s] for s in family]);k=len(lp.pool)
    values=np.maximum(lp.weights.X,0.)
    return dict(supports=[list(s) for s in family],r=values[active].tolist(),x=values[active+k].tolist()),fitted


def fit_family_ac(obs,family,*,base_kv,base_mva,root_label=1,initial_point=None,
                  coefficient_bound=float('inf'),outer_iterations=3,line_steps=(1.,.5,.25),
                  inner_abs_improvement=1e-12,zero_tolerance=1e-9,ac_max_iter=200,ac_tol=1e-12):
    """Jointly refit all coefficients on a fixed support, training data only.

    A failed initial AC fit returns success=False, objective=None and the
    original point. A caller must not substitute its linear loss as an AC score.
    Failed trials retain the last converged point and all failure receipts.
    """
    started=perf_counter();labels=_validate_obs(obs);family=canonical(family)
    if not isinstance(outer_iterations,int) or outer_iterations<0:raise ValueError('nonnegative iteration count required')
    if not line_steps or any(not 0<a<=1 for a in line_steps):raise ValueError('line steps must lie in (0,1]')
    p,q,y=arrays(obs,'raw');traces=[];lp_receipts=[];lp=None
    settings=dict(outer_iterations=outer_iterations,line_steps=list(line_steps),inner_abs_improvement=inner_abs_improvement,
        coefficient_bound='unbounded' if not np.isfinite(coefficient_bound) else float(coefficient_bound),
        zero_tolerance=zero_tolerance,ac_max_iter=ac_max_iter,ac_tol=ac_tol,
        objective='mean absolute observed squared-voltage-drop prediction error; raw observed P/Q fixed',
        optimization='bounded iterative AC correction and joint L1 refit; no global optimality claim',validation_used=False)
    try:
        if initial_point is None:
            lp=PoolLP(None,family,'raw',bound=coefficient_bound,data=(p,q,y))
            current,_=_lp_native(lp,family);lp_receipts.extend(lp.solves);lp.close();lp=None
        else:current=_native(initial_point,family)
        original=deepcopy(current)
        try:prediction=predict_ac(current,obs,base_kv=base_kv,base_mva=base_mva,root_label=root_label,ac_max_iter=ac_max_iter,ac_tol=ac_tol)
        except Exception as exc:
            return dict(success=False,status='initial_ac_failed',point=original,native_point=original,objective=None,
                error=repr(exc),settings=settings,LP_solves=lp_receipts,iterations=traces,seconds=perf_counter()-started)
        initial_ac=prediction['mae'];initial_convergence=prediction['convergence'];reason='iteration_limit'
        for iteration in range(outer_iterations):
            rmat,xmat=matrices(len(labels),current['supports'],current['r'],current['x'])
            correction=prediction['drop']-p@rmat.T-q@xmat.T
            row=dict(iteration=iteration,current_objective=prediction['mae'],trials=[],accepted=False)
            try:
                lp=PoolLP(None,family,'raw',bound=coefficient_bound,data=(p,q,y-correction))
                proposal,linearfit=_lp_native(lp,family);lp_receipts.extend(lp.solves);lp.close();lp=None
                row['linearized_subproblem_objective']=linearfit['objective']
            except Exception as exc:
                if lp is not None:lp_receipts.extend(lp.solves);lp.close();lp=None
                row['LP_error']=repr(exc);traces.append(row);reason='subproblem_failed_original_retained';break
            for alpha in line_steps:
                trial=dict(supports=current['supports'],
                    r=((1-alpha)*np.asarray(current['r'])+alpha*np.asarray(proposal['r'])).tolist(),
                    x=((1-alpha)*np.asarray(current['x'])+alpha*np.asarray(proposal['x'])).tolist())
                try:
                    pred=predict_ac(trial,obs,base_kv=base_kv,base_mva=base_mva,root_label=root_label,ac_max_iter=ac_max_iter,ac_tol=ac_tol)
                    row['trials'].append(dict(alpha=alpha,objective=pred['mae'],convergence=pred['convergence']))
                    if pred['mae']<prediction['mae']-inner_abs_improvement:
                        current,prediction=trial,pred;row['accepted']=True;row['accepted_alpha']=alpha;break
                except Exception as exc:row['trials'].append(dict(alpha=alpha,error=repr(exc),converged=False))
            traces.append(row)
            if not row['accepted']:reason='no_ac_improvement_original_retained';break
        kept=[i for i,s in enumerate(family) if len(s)==1 or max(current['r'][i],current['x'][i])>zero_tolerance]
        point=dict(supports=[list(family[i]) for i in kept],r=[current['r'][i] for i in kept],x=[current['x'][i] for i in kept],
                   objective=prediction['mae'],preprocess='raw_ac_corrected')
        return dict(success=True,status=reason,point=point,native_point=current,initial_native_point=original,
            initial_objective=initial_ac,objective=prediction['mae'],voltage_rmse=prediction['voltage_rmse'],
            initial_convergence=initial_convergence,final_convergence=prediction['convergence'],
            settings=settings,LP_solves=lp_receipts,iterations=traces,seconds=perf_counter()-started)
    finally:
        if lp is not None:lp.close()
