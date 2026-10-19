"""Independent physical-scale and EV/ESS operation mechanism audit.

All sites are compared as a declared electrical experiment, not selected by
truth. Continuous quality/random-price LPs optimize public affine criteria
and charging service only. No topology fit or rejection claim is made.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import linprog

from .ev_physical_cases import (case33_local_standard, case33_local_record,
    impedance_ohm_to_pu, power_kw_to_pu)
from .likelihood import ac_predict
from .ev_service_milp import (EVServiceConstraints, canonical_service_plan,
    check_service_plan, operational_data)
from .ev_bilinear_score_exchange import charging_service_domain

HERE=Path(__file__).resolve().parent
T=4
CURVES={'constant':np.array([3.,3.,3.,3.]),'alternating':np.array([6.,0.,6.,0.])}
SIGMA_V=.00012


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def physical_response(case,site,power_kw,*,base_mva=None):
    """Same ohms/kW/kvar at a consistently changed per-unit power base."""
    u=np.asarray(power_kw,float)
    if u.ndim!=1 or not len(u) or not np.isfinite(u).all() or not 0<=site<case.candidate.n:
        raise ValueError('finite time power and valid site required')
    mva=case.base_mva if base_mva is None else float(base_mva)
    if not np.isfinite(mva) or mva<=0:raise ValueError('positive finite base_mva required')
    power_base=1000*mva; zbase=case.base_kv**2/mva
    r_ohm=case.r*case.zbase_ohm; x_ohm=case.x*case.zbase_ohm
    p_kw=np.tile(case.p0*case.base_kw,(len(u),1)); q_kvar=np.tile(case.q0*case.base_kw,(len(u),1))
    p_kw[:,site]+=u
    answer=ac_predict(case.candidate,r_ohm/zbase,x_ohm/zbase,p_kw/power_base,
        q_kvar/power_base,source_p=np.zeros(len(u)),source_q=np.zeros(len(u)),root_v=np.ones(len(u)))
    return answer,dict(base_kv=case.base_kv,base_mva=mva,base_kw=power_base,zbase_ohm=zbase)


def electrical_audit():
    case=case33_local_standard(); baseline,_=physical_response(case,0,np.zeros(T)); rows=[]
    calls=1
    for site in range(case.candidate.n):
        for name,u in CURVES.items():
            ev,units=physical_response(case,site,u); ess,_=physical_response(case,site,u); calls+=2
            dv=ev['v']-baseline['v']; same={k:float(np.max(abs(ev[k]-ess[k]),initial=0.))
                for k in ('p','q','v','voltage','master_p','master_q')}
            rebases=[]
            for mva in (.02,10.):
                rebased,runit=physical_response(case,site,u,base_mva=mva);calls+=1
                rebases.append(dict(units=runit,
                    max_voltage_abs_difference=float(np.max(abs(rebased['voltage']-ev['voltage']))),
                    max_master_p_kw_difference=float(np.max(abs(rebased['master_p']*runit['base_kw']-ev['master_p']*case.base_kw))),
                    max_master_q_kvar_difference=float(np.max(abs(rebased['master_q']*runit['base_kw']-ev['master_q']*case.base_kw))),
                    ac_equation_residual=rebased['ac_equation_residual'],power_balance_residual=rebased['power_balance_residual']))
            rows.append(dict(site=site,source_bus=case.provenance['terminal_source_buses'][site],curve=name,u_kw=u,
                units=units,grid_energy_kwh=.25*float(u.sum()),
                ratio_to_site_static_p_kw=u/(case.p0[site]*case.base_kw),
                terminal_v_pu=ev['v'],delta_terminal_v_pu=dv,
                delta_terminal_v_volts=dv*case.base_kv*1000,
                delta_v_noise_units=dv/SIGMA_V,
                max_abs_delta_v_pu=float(np.max(abs(dv))),
                max_abs_delta_v_noise_units=float(np.max(abs(dv))/SIGMA_V),
                all_voltage_white_L2=float(np.linalg.norm(dv/SIGMA_V)),
                root_power_increment_kw=(ev['master_p']-baseline['master_p'])*case.base_kw,
                root_reactive_increment_kvar=(ev['master_q']-baseline['master_q'])*case.base_kw,
                same_PQ_EV_ESS_max_differences=same,rebase_checks=rebases,
                ac_equation_residual=ev['ac_equation_residual'],power_balance_residual=ev['power_balance_residual']))
    return dict(case=case33_local_record(),static_background_p_kw=case.p0*case.base_kw,
        static_background_q_kvar=case.q0*case.base_kw,root_v_pu=1.,generated_source_p_kw=0.,generated_source_q_kvar=0.,
        source_physical_interface='node1, unchanged for all sites',voltage_sigma_pu=SIGMA_V,
        sigma_is_declared_example_meter_precision=True,observation_noise_draws=0,
        baseline_terminal_v_pu=baseline['v'],rows=rows,ac_calls=calls,
        same_location_same_PQ_equivalence_max=max(max(r['same_PQ_EV_ESS_max_differences'].values()) for r in rows),
        consistent_rebase_voltage_max=max(c['max_voltage_abs_difference'] for r in rows for c in r['rebase_checks']),
        site_comparison_purpose='mechanism response comparison; not site optimizer or topology rejection experiment')


def service_case(kind,*,base_kw=1000.):
    """Physical kW/kWh operations converted to existing pu/pu-hour API.

    Preposition explicitly spends one 15-minute trip before the four-slot
    observation window. Its 0.2kWh is still deducted by the service model.
    """
    known={
        'fixed_ess_initial_connected':(0,False,False,False,-1),
        'fixed_ess_remote_connected':(2,False,False,False,-1),
        'ev_initial_early_worker':(0,True,False,False,0),
        'ev_initial_late_worker':(0,True,False,False,1),
        'ev_initial_very_late_worker':(0,True,False,False,2),
        'ev_remote_in_window_early_worker':(2,True,True,False,1),
        'ev_remote_in_window_late_worker':(2,True,True,False,2),
        'ev_remote_preposition_early_worker':(2,True,True,True,0),
        'ev_remote_preposition_late_worker':(2,True,True,True,1),
        'ev_remote_preposition_no_worker':(2,True,True,True,4)}
    if kind not in known:raise ValueError('unknown operation case')
    if not np.isfinite(base_kw) or base_kw<=0:raise ValueError('positive finite base_kw required')
    site,is_ev,travelled,preposition,worker_slot=known[kind]
    off=np.ones((3,3))-np.eye(3); travel=off.astype(int)
    expense=.2*off/base_kw if travelled else np.zeros((3,3))
    # travel_slots describes remaining travel inside the observation window.
    # Preposition has a recorded one-slot trip outside that window, not free travel.
    if not is_ev or preposition:travel=np.zeros((3,3),int)
    worker=np.zeros((T,3),bool)
    if not is_ev:worker[:]=True
    elif worker_slot<T:worker[worker_slot,site]=True
    service=EVServiceConstraints(terminal_energy=26.85/base_kw,dt_hours=.25,
        p_charge_max=6./base_kw,p_discharge_max=0.,battery_capacity=60./base_kw,
        initial_soc=.4,soc_min=.1,soc_max=.9,eta_charge=.95,eta_discharge=.95,
        availability=np.ones((T,3),bool),travel_slots=travel,travel_energy=expense,
        initial_site=0,worker_setup_available=worker,setup_slots=1 if is_ev else 0,
        departure_slot=4,return_energy=0.,departure_reserve=0.)
    return service,site,dict(kind=kind,device='EV' if is_ev else 'fixed ESS',
        physical_trip_slots=1 if travelled else 0,trip_energy_kwh=.2 if travelled else 0.,
        travel_before_observation_slots=1 if preposition else 0,
        within_observation_travel_slots=0 if preposition else (1 if travelled else 0),
        observation_hours=1.,total_declared_task_hours=1.25 if preposition else 1.,
        setup_slots=1 if is_ev else 0,worker_available_slot=None if worker_slot<0 or worker_slot>=T else worker_slot,
        return_trip_included=False,fixed_ESS_assumed_already_installed_and_connected=not is_ev,
        base_kw=base_kw,matched_terminal_battery_kwh=26.85)


def solve_affine_curve(prices,domain):
    """Continuous LP with independently recomputed primal/dual evidence."""
    low,high=np.asarray(domain['lower']),np.asarray(domain['upper'])
    constraint=domain['constraints']; A=np.asarray(constraint.A);lo=np.asarray(constraint.lb);hi=np.asarray(constraint.ub)
    equality=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
    up=np.isfinite(hi)&~equality; down=np.isfinite(lo)&~equality
    aub=np.vstack((A[up],-A[down])); bub=np.r_[hi[up],-lo[down]]
    ae=A[equality];be=hi[equality]; objective=-np.asarray(prices,float)
    solved=linprog(objective,A_ub=aub,b_ub=bub,A_eq=ae,b_eq=be,bounds=list(zip(low,high)),method='highs')
    answer=dict(success=bool(solved.success),status=int(solved.status),message=str(solved.message),
        u_kw=None if solved.x is None else solved.x,affine_objective=None if solved.fun is None else -float(solved.fun),
        solver='scipy.linprog/HiGHS',complete_curve_enumeration=False,
        objective_scope='declared public quality or random affine prices; not topology score')
    if solved.success:
        u=np.asarray(solved.x); yu=np.asarray(solved.ineqlin.marginals); ye=np.asarray(solved.eqlin.marginals)
        zl=np.asarray(solved.lower.marginals);zu=np.asarray(solved.upper.marginals)
        dual=float(bub@yu+be@ye+low@zl+high@zu)
        stationarity=objective-aub.T@yu-ae.T@ye-zl-zu
        primal=float(objective@u)
        answer.update(minimization_primal=primal,minimization_dual_numeric=dual,
            primal_dual_gap_numeric=primal-dual,
            dual_stationarity_inf=float(np.max(abs(stationarity),initial=0.)),
            operation_violation=float(max(0.,np.max(low-u),np.max(u-high),
                np.max(lo-A@u),np.max(A@u-hi))))
    return answer


def operation_audit(*,base_kw=1000.,random_seed=988201):
    kinds=('fixed_ess_initial_connected','fixed_ess_remote_connected',
        'ev_initial_early_worker','ev_initial_late_worker','ev_initial_very_late_worker',
        'ev_remote_in_window_early_worker','ev_remote_in_window_late_worker',
        'ev_remote_preposition_early_worker','ev_remote_preposition_late_worker','ev_remote_preposition_no_worker')
    quality=np.array([1.,9.,1.,9.]); random=np.random.default_rng(random_seed).standard_normal(T)
    rows=[]
    for kind in kinds:
        service,site,context=service_case(kind,base_kw=base_kw)
        data=operational_data(site,T,3,service)
        domain=charging_service_domain(site,T,3,service,power_scale=base_kw)
        required_grid=data['required_charging_gain']*base_kw/service.eta_charge
        capacity=.25*float(np.sum(data['charge']))*base_kw
        row=dict(context=context,site=site,operations=data,
            allowed_slots=int(np.count_nonzero(data['charge'])),required_grid_energy_kwh=required_grid,
            maximum_grid_energy_kwh=capacity,analytic_energy_capacity_feasible=capacity+1e-10>=required_grid,
            input_service=asdict(service),domain_precheck_success=domain['success'],plans={})
        if not domain['success']:
            row.update(reason=domain['reason'],service_feasible=False)
        else:
            for method,prices in (('quality_lp',quality),('random_price_lp',random)):
                solved=solve_affine_curve(prices,domain)
                if solved['success']:
                    plan=canonical_service_plan(site,solved['u_kw']/base_kw,3,service,solver=dict(method=method))
                    checked=check_service_plan(plan,service)
                    solved.update(validation=checked,plan=plan.summary(),
                        grid_energy_kwh=.25*float(np.sum(solved['u_kw'])),
                        terminal_energy_kwh=checked['terminal_battery_energy']*base_kw,
                        efficiency_loss_kwh=checked['efficiency_loss_pu_hours']*base_kw)
                row['plans'][method]=solved
            if domain.get('constant_u') is not None:
                plan=domain['constant_plan']; checked=check_service_plan(plan,service)
                row['plans']['constant_accessible']=dict(success=True,u_kw=domain['constant_u'],
                    grid_energy_kwh=.25*float(np.sum(domain['constant_u'])),
                    terminal_energy_kwh=checked['terminal_battery_energy']*base_kw,
                    validation=checked,plan=plan.summary())
            else:
                row['plans']['constant_accessible']=dict(success=False,status=domain['constant_status'])
            row['service_feasible']=bool(row['plans']['quality_lp']['success'] and row['plans']['quality_lp']['validation']['feasible'])
        rows.append(row)
    return dict(horizon_slots=T,dt_hours=.25,battery_kwh=60.,initial_battery_kwh=24.,
        terminal_battery_kwh=26.85,eta_charge=.95,p_charge_max_kw=6.,quality_prices=quality,
        random_prices=random,random_seed=random_seed,rows=rows,
        unit_note='existing API power is pu and battery/travel energy is pu-hours; controls passed to LP are kW',
        matching='terminal battery service matched; mobile grid energy includes compensation for driving',
        topology_score_or_rejection_evaluated=False)


def audit():
    began=perf_counter(); electrical=electrical_audit(); operations=operation_audit()
    return dict(schema='case33_derived_ev_ess_spatial_mechanism_v1',electrical=electrical,operations=operations,
        source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in (
            'ev_ev_ess_spatial_audit.py','ev_physical_cases.py','likelihood.py','ev_service_milp.py','ev_bilinear_score_exchange.py')},
        seconds=perf_counter()-began,new_topology_fits=0,new_topology_designs=0,
        existing_frozen_trials_modified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(serial(result),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(ac_calls=result['electrical']['ac_calls'],
        EV_ESS_equivalence_max=result['electrical']['same_location_same_PQ_equivalence_max'],
        rebase_voltage_error=result['electrical']['consistent_rebase_voltage_max'],
        max_6kw_voltage_sigma_by_site={r['site']:r['max_abs_delta_v_noise_units']
            for r in result['electrical']['rows'] if r['curve']=='alternating'},
        operation_feasible={r['context']['kind']:r['service_feasible'] for r in result['operations']['rows']},
        seconds=result['seconds']),ensure_ascii=False))


if __name__=='__main__':main()
