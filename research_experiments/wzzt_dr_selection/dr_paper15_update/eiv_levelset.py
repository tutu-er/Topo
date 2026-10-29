"""Finite AC impedance bounds preserving an incumbent measurement-loss level.

Only fixed raw-channel sigmas, observed data, candidate topology and an AC
feasible incumbent are used. This is an optimization level-set bound, not an
asset prior, a confidence interval or a guarantee that physical truth belongs
to the level set. Zero/negative-load cases can remain explicitly unresolved.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from rnj_wzzt.estimation.ac_measurement_fit import (
    ACMeasurementSigma, _aligned_frame, _sigma_array, _validate_network,
)
from rnj_wzzt.estimation.unmetered_inputs import _series_on_index, _terminal_values
from rnj_wzzt.models.ac_powerflow import _oriented_tree, solve_ac_power_flow_timeseries


def derive_eiv_levelset_bounds(net, observed, sigma, *, ac_max_iter=200, ac_tol=1e-12, bound_kind='triangle'):
    """Compute a safe common [0,B] ohm cap when passivity follows on the level set.

    Incumbent states equal measured P/Q/root, and impedance equals net's current
    values. One AC call checks this incumbent. Sigmas must be fixed absolute pu
    scales accepted by fit_ac_pqv. All unobserved bus loads are zero, as in that
    fitter with source=None; no shunt or DER model is added here.

    At usable time t every terminal P/Q lower envelope is strictly positive. For an
    edge with descendants C, receiving-end current has magnitude at least
    P_lower(C,t)/Uroot(t). Passive DistFlow gives
      |z I|^2 <= Uroot^2 - Lchild^2,
    where Lchild is max(0,max observed terminal lower voltages in C). Thus
      |z_ohm| <= Zbase * Uroot * sqrt(Uroot^2-Lchild^2) / P_lower(C,t).
    Minimize over usable times per edge, then maximize over edges for the
    scalar-bound-only existing API. Default uses the simpler and more
    conservative triangle 2*Uroot^2/P_lower cap; the stronger expression is
    optional and never required to obtain a finite cap.
    """
    if not isinstance(sigma,ACMeasurementSigma):raise ValueError('sigma must be ACMeasurementSigma')
    if bound_kind not in ('triangle','passive_distflow'):raise ValueError('unknown bound kind')
    reference=observed['P_terminal'];p=_terminal_values(reference,'P_terminal')
    q=_aligned_frame(observed['Q_terminal'],reference,'Q_terminal')
    v=_aligned_frame(observed['V_terminal'],reference,'V_terminal')
    root=_series_on_index(observed['root_voltage'],reference.index,'root_voltage')
    active=_validate_network(net,reference.columns)
    if sigma.root_v is None and observed.get('root_observation')=='noisy':raise ValueError('noisy root needs positive sigma.root_v')
    scales={k:_sigma_array(value,reference,'sigma.'+k,terminal=True) for k,value in
            [('p',sigma.p),('q',sigma.q),('v',sigma.v)]}
    scales['root']=np.zeros(len(root)) if sigma.root_v is None else _sigma_array(sigma.root_v,reference,'sigma.root_v',terminal=False)
    buses=net.buses['bus_id'].astype(int).tolist()
    pp=pd.DataFrame(p,index=reference.index,columns=reference.columns).reindex(columns=buses,fill_value=0.)
    qq=pd.DataFrame(q,index=reference.index,columns=reference.columns).reindex(columns=buses,fill_value=0.)
    ac=solve_ac_power_flow_timeseries(net,pp,qq,v_root=root,max_iter=ac_max_iter,tol=ac_tol)
    out=dict(status='prepared',finite_certified=False,api_usable=False,incumbent_AC_evaluations=1,
       implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       loss_definition='L=sum squared standardized raw measurement residuals; scipy_cost=L/2',
       bound_kind=bound_kind,assumptions=['balanced radial series-impedance network; R/X>=0; no shunts or negative fixed sources',
                    'all injections outside observed terminals are zero',
                    'fixed diagonal raw-channel sigmas; EIV states differ from measurements by their standardized residual',
                    'only times whose full terminal P/Q lower envelopes are strictly positive are used'],
       edges=[],scope='Optimization level set only; no statistical coverage, parameter-truth or global-convergence claim.')
    if not bool(ac['converged'].all()):
        out.update(status='not_certified_incumbent_AC_failure');return out
    pred=ac['V_bus_mag'].loc[:,reference.columns].to_numpy()
    Lbar=float(np.sum(((pred-v)/scales['v'])**2))
    heads=[x['label'] for x in ac['metadata']['branch_edges'] if x['from_bus']==int(net.root_bus)]
    for key,s in [('P0_measured',sigma.master_p),('Q0_measured',sigma.master_q)]:
        if s is not None:
            measurement=_series_on_index(observed[key],reference.index,key)
            channel='p' if key=='P0_measured' else 'q'
            prediction=ac[f'branch_{channel}_from_pu'][heads].sum(axis=1).to_numpy()
            std=_sigma_array(s,reference,'sigma.'+key,terminal=False)
            Lbar+=float(np.sum(((prediction-measurement)/std)**2))
    if not np.isfinite(Lbar):out.update(status='not_certified_nonfinite_incumbent_loss');return out
    b=float(np.sqrt(Lbar));pl=p-b*scales['p'];ql=q-b*scales['q']
    vu=root+b*scales['root'];vl=v-b*scales['v']
    valid_time=np.all(pl>0,axis=1)&np.all(ql>0,axis=1)&(vu>0)
    out.update(incumbent_loss_L=Lbar,incumbent_scipy_cost=.5*Lbar,standardized_residual_radius=b,
       usable_nonnegative_load_times=int(valid_time.sum()),time_count=len(p),
       minimum_P_lower_pu=float(np.min(pl)),minimum_Q_lower_pu=float(np.min(ql)),
       maximum_root_upper_pu=float(np.max(vu)))
    nodes,parent,children,zpu=_oriented_tree(net);pos={int(label):i for i,label in enumerate(reference.columns)}
    descendants={node:({pos[node]} if node in pos else set()) for node in nodes}
    for node in reversed(nodes):
        for child in children[node]:descendants[node].update(descendants[child])
    zbase=float(net.base_kv**2/net.base_mva)
    all_finite=True
    for child,par in parent.items():
        indices=sorted(descendants[child]);record=dict(parent=par,child=child,
            descendants=[int(reference.columns[i]) for i in indices],lower_ohm=0.)
        if not indices:
            record.update(status='not_certified_no_observed_downstream_load',upper_ohm=None);all_finite=False
        else:
            flow=pl[:,indices].sum(axis=1);child_lower=np.maximum(0.,np.max(vl[:,indices],axis=1))
            good=valid_time&(flow>0)
            if bound_kind=='passive_distflow':good=good&(vu>=child_lower)
            if not good.any():
                record.update(status='not_certified_zero_or_signed_levelset_flow',upper_ohm=None);all_finite=False
            else:
                ids=np.flatnonzero(good)
                weak=2*zbase*vu[ids]**2/flow[ids]
                strong=zbase*vu[ids]*np.sqrt(np.maximum(vu[ids]**2-child_lower[ids]**2,0.))/flow[ids]
                cap=weak if bound_kind=='triangle' else strong
                chosen=int(np.argmin(cap));row=int(ids[chosen]);upper=0. if cap[chosen]==0 else float(np.nextafter(cap[chosen],np.inf))
                initial_norm=float(abs(zpu[(par,child)])*zbase)
                inside=bool(initial_norm<=upper*(1+1e-9)+1e-12)
                record.update(status='certified' if inside else 'not_certified_incumbent_outside_numeric_bound',
                   upper_ohm=upper,triangle_upper_ohm=float(np.min(weak)),passive_distflow_upper_ohm=float(np.min(strong)),selected_time_index=row,
                   selected_timestamp=str(reference.index[row]),selected_downstream_P_lower_pu=float(flow[row]),
                   selected_root_upper_pu=float(vu[row]),selected_child_lower_pu=float(child_lower[row]),
                   initial_impedance_magnitude_ohm=initial_norm,incumbent_inside_bound=inside)
                all_finite=all_finite and inside and np.isfinite(upper)
        out['edges'].append(record)
    if all_finite and out['edges']:
        upper=max(e['upper_ohm'] for e in out['edges']);usable=bool(upper>0)
        initial_components=net.branches.loc[active,['r_ohm','x_ohm']].to_numpy(dtype=float)
        if (initial_components>upper).any():
            out.update(status='not_certified_numeric_cap_excludes_initial_component',global_upper_ohm=None)
            return out
        out.update(status='certified' if usable else 'certified_zero_levelset_cap',finite_certified=True,
                   api_usable=usable,r_bounds_ohm=[0.,upper],x_bounds_ohm=[0.,upper],global_upper_ohm=upper,
                   active_edge_count=len(active),base_impedance_ohm=zbase,
                   interpretation='Scalar cap preserves all candidate points with L<=incumbent L under the stated passivity conditions; it is not an externally calibrated impedance range.')
    else:out.update(status='not_certified',global_upper_ohm=None)
    return out
