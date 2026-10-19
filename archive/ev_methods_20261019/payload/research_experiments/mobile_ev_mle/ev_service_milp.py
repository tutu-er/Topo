"""Battery-service-fair single-site EV/ESS operations.

Power is pu and energy is pu-hours. ``terminal_energy`` is the absolute
battery energy before the declared return trip, fixed independently of the
chosen curve/site. It is not a grid-side net-energy constraint. Setup may be
performed only after vehicle arrival during a declared worker window; once
setup is complete, the worker need not remain. Earliest feasible setup
weakly enlarges all later continuous-power opportunities in this fixed-site
model. No complete EV route or power-curve library is enumerated.

The affine step has an LP bound if only charging is enabled, and a MILP
bound if charging/discharging exclusion needs binaries. These bounds apply
only to that frozen affine subproblem.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from scipy.sparse import coo_matrix
from .single_site_milp import SingleSiteConstraints, SingleSitePlan


@dataclass(frozen=True)
class EVServiceConstraints:
    terminal_energy: float
    dt_hours: float = .25
    p_charge_max: Any = .30
    p_discharge_max: Any = 0.
    battery_capacity: float = 3.
    initial_soc: float = .4
    soc_min: float = .1
    soc_max: float = .9
    eta_charge: float = .95
    eta_discharge: float = .95
    q_over_p: float = 0.
    availability: Any = None
    travel_slots: Any = None
    travel_energy: Any = None
    initial_site: int = 0
    worker_setup_available: Any = None
    setup_slots: Any = 0
    departure_slot: Any = None
    return_energy: Any = 0.
    departure_reserve: float = 0.

    def arrays(self, n_slots, n_sites):
        # Reuse existing public cap/travel validation without changing its
        # frozen grid-energy semantics or sources.
        scalar = [self.terminal_energy, self.departure_reserve]
        if not np.isfinite(scalar).all() or self.departure_reserve < 0:
            raise ValueError('terminal_energy and departure_reserve must be finite; reserve nonnegative')
        common = {k: getattr(self,k) for k in (
            'dt_hours','p_charge_max','p_discharge_max','battery_capacity',
            'initial_soc','soc_min','soc_max','eta_charge','eta_discharge',
            'q_over_p','availability','travel_slots','travel_energy','initial_site')}
        arrays = SingleSiteConstraints(energy=0., **common).arrays(n_slots,n_sites)
        if not self.soc_min*self.battery_capacity <= self.terminal_energy <= self.soc_max*self.battery_capacity:
            raise ValueError('terminal_energy must satisfy battery SOC limits')
        _extended_arrays(self,n_slots,n_sites)
        return arrays


def _extended_arrays(c, n_slots, n_sites):
    worker = np.ones((n_slots,n_sites),bool) if c.worker_setup_available is None else np.asarray(c.worker_setup_available)
    if worker.shape != (n_slots,n_sites) or not np.isin(worker,[False,True]).all():
        raise ValueError('worker_setup_available must be bool (time,site)')
    values=[]
    for name, default in (('setup_slots',0),('departure_slot',n_slots),('return_energy',0.)):
        value=getattr(c,name)
        value=default if value is None else value
        try: a=np.broadcast_to(np.asarray(value,float),(n_sites,)).copy()
        except ValueError as error: raise ValueError(name+' must be scalar or (site,)') from error
        if not np.isfinite(a).all() or (a<0).any(): raise ValueError(name+' must be finite and nonnegative')
        if name != 'return_energy' and not np.equal(a,np.floor(a)).all():
            raise ValueError(name+' must contain integers')
        if name == 'departure_slot' and (a > n_slots).any():
            raise ValueError('departure_slot outside horizon')
        values.append(a.astype(int) if name != 'return_energy' else a)
    return worker.astype(bool), *values


def operational_data(site, n_slots, n_sites, constraints):
    charge,discharge,access,travel,mobility=constraints.arrays(n_slots,n_sites)
    if not isinstance(site,(int,np.integer)) or not 0 <= site < n_sites:
        raise ValueError('site outside site set')
    worker,setup,departure,returned=_extended_arrays(constraints,n_slots,n_sites)
    arrival=int(travel[constraints.initial_site,site]); finish=int(departure[site])
    expense=float(mobility[constraints.initial_site,site]); duration=int(setup[site])
    setup_start = None
    if duration == 0:
        setup_start=arrival if arrival <= finish else None
    else:
        for start in range(arrival,finish-duration+1):
            # A fixed-site setup completed earlier dominates a later setup:
            # it provides a superset of possible power slots at equal cost.
            if np.all(worker[start:start+duration,site]) and np.all(access[start:start+duration,site]):
                setup_start=start; break
    ready=n_slots+1 if setup_start is None else setup_start+duration
    allowed=access[:,site].copy()
    allowed[:min(ready,n_slots)]=False; allowed[finish:]=False
    return dict(charge=charge[:,site]*allowed,discharge=discharge[:,site]*allowed,
        allowed=allowed,arrival_slot=arrival,setup_start=setup_start,
        setup_slots=duration,ready_slot=ready,departure_slot=finish,
        travel_energy=expense,return_energy=float(returned[site]),
        post_drive_energy=constraints.initial_soc*constraints.battery_capacity-expense,
        service_battery_gain=constraints.terminal_energy-constraints.initial_soc*constraints.battery_capacity,
        required_charging_gain=constraints.terminal_energy-(constraints.initial_soc*constraints.battery_capacity-expense),
        post_return_energy=constraints.terminal_energy-float(returned[site]))


def _site_data(site,n_slots,n_sites,constraints):
    d=operational_data(site,n_slots,n_sites,constraints)
    return d['charge'],d['discharge'],d['ready_slot'],d['travel_energy']


def _plan(site,u,n_sites,constraints,tau,expense,solver=None):
    u=np.asarray(u,float)
    p=np.zeros((len(u),n_sites));p[:,site]=u
    d=operational_data(site,len(u),n_sites,constraints)
    sites=np.where(d['allowed'],site,-1).astype(int)
    initial=constraints.initial_soc-expense/constraints.battery_capacity
    soc=np.r_[initial,initial+np.cumsum(constraints.dt_hours*(
        constraints.eta_charge*np.maximum(u,0)-np.maximum(-u,0)/constraints.eta_discharge)/constraints.battery_capacity)]
    return SingleSitePlan(site,p,constraints.q_over_p*p,sites,soc,solver or {})


def canonical_service_plan(site,u,n_sites,constraints,solver=None):
    """Keep the injection curve and rebuild metadata for a declared domain.

    This performs no optimization and claims no feasibility. Call
    ``check_service_plan`` afterwards. In particular, a changed driving
    energy changes initial battery state and can invalidate the fixed task.
    The original plan/context should remain saved by the caller.
    """
    power=np.asarray(u,float)
    if power.ndim != 1 or not len(power) or not np.isfinite(power).all():
        raise ValueError('finite nonempty net power vector required')
    d=operational_data(site,len(power),n_sites,constraints)
    metadata={} if solver is None else dict(solver)
    metadata['canonical_service_context']=dict(ready_slot=d['ready_slot'],
        departure_slot=d['departure_slot'],travel_energy_pu_hours=d['travel_energy'],
        unchanged_injection=True)
    return _plan(site,power,n_sites,constraints,d['ready_slot'],d['travel_energy'],metadata)


def check_service_plan(plan,constraints,*,tolerance=2e-7):
    """Independent net-power reconstruction, including terminal and return service."""
    if not np.isfinite(tolerance) or tolerance < 0: raise ValueError('finite nonnegative tolerance required')
    issues=[]
    try:
        p,q,sites,soc=(np.asarray(getattr(plan,k)) for k in ('p','q','sites','soc'))
        if p.ndim != 2 or not all(p.shape) or q.shape != p.shape or sites.shape != (len(p),) or soc.shape != (len(p)+1,):
            return dict(feasible=False,issues=['shape'])
        if not all(np.isfinite(a).all() for a in (p,q,sites,soc)):
            return dict(feasible=False,issues=['nonfinite'])
        d=operational_data(plan.site,len(p),p.shape[1],constraints)
        u=p[:,plan.site]
        if np.any(abs(np.delete(p,plan.site,axis=1)) > tolerance): issues.append('multiple_sites')
        if np.any(abs(q-constraints.q_over_p*p) > tolerance): issues.append('reactive_power')
        if not np.array_equal(sites,np.where(d['allowed'],plan.site,-1)): issues.append('connection_path')
        if d['ready_slot'] >= d['departure_slot']: issues.append('setup_or_arrival_outside_window')
        if np.any(u > d['charge']+tolerance) or np.any(u < -d['discharge']-tolerance): issues.append('power_or_access')
        cp,dp=np.maximum(u,0),np.maximum(-u,0)
        charge=float(constraints.dt_hours*cp.sum()); discharge=float(constraints.dt_hours*dp.sum())
        loss=float((1-constraints.eta_charge)*charge+(1/constraints.eta_discharge-1)*discharge)
        expected=_plan(plan.site,u,p.shape[1],constraints,d['ready_slot'],d['travel_energy']).soc
        if np.max(abs(soc-expected)) > tolerance: issues.append('soc_dynamics')
        if np.any(expected < constraints.soc_min-tolerance) or np.any(expected > constraints.soc_max+tolerance): issues.append('soc')
        terminal=float(expected[-1]*constraints.battery_capacity)
        if abs(terminal-constraints.terminal_energy) > tolerance: issues.append('terminal_battery_service')
        post_return=terminal-d['return_energy']
        if post_return < max(constraints.departure_reserve,constraints.soc_min*constraints.battery_capacity)-tolerance:
            issues.append('return_or_departure_reserve')
        # Independently check solver split flows if supplied; a fictitious
        # simultaneous charge/discharge pair cannot hide behind net power.
        if 'split_charge_pu' in plan.solver or 'split_discharge_pu' in plan.solver:
            sp=np.asarray(plan.solver.get('split_charge_pu',[]),float)
            sd=np.asarray(plan.solver.get('split_discharge_pu',[]),float)
            if sp.shape != u.shape or sd.shape != u.shape or not np.isfinite(sp).all() or not np.isfinite(sd).all():
                issues.append('split_flow_shape')
            elif np.max(abs(sp-cp)) > tolerance or np.max(abs(sd-dp)) > tolerance:
                issues.append('simultaneous_or_wrong_split_flow')
        return dict(feasible=not issues,issues=issues,grid_charge_pu_hours=charge,
            grid_discharge_pu_hours=discharge,grid_net_pu_hours=charge-discharge,
            efficiency_loss_pu_hours=loss,throughput_pu_hours=charge+discharge,
            travel_energy_pu_hours=d['travel_energy'],return_energy_pu_hours=d['return_energy'],
            initial_battery_energy=constraints.initial_soc*constraints.battery_capacity,
            post_drive_energy=d['post_drive_energy'],terminal_battery_energy=terminal,
            post_return_battery_energy=post_return,service_battery_gain=d['service_battery_gain'],
            battery_net_charging_gain=charge-discharge-loss,
            accounting_error=terminal-(constraints.initial_soc*constraints.battery_capacity-d['travel_energy']+charge-discharge-loss),
            ready_slot=d['ready_slot'],arrival_slot=d['arrival_slot'],setup_start=d['setup_start'],
            setup_slots=d['setup_slots'],departure_slot=d['departure_slot'],recomputed_soc=expected.tolist())
    except (ValueError,TypeError,AttributeError,IndexError) as error:
        return dict(feasible=False,issues=['invalid_input'],error=str(error))


def _failure(status,message,**extra):
    return dict(success=False,status=status,plan=None,incumbent=None,best_bound=None,
                absolute_gap=None,relative_gap=None,termination=message,**extra)


def _service_precheck(d,c):
    if d['ready_slot'] >= d['departure_slot']:
        return 'arrival or setup outside connection horizon'
    if d['post_drive_energy'] < c.soc_min*c.battery_capacity-2e-7:
        return 'driving violates SOC minimum before charging'
    if d['post_return_energy'] < max(c.departure_reserve,c.soc_min*c.battery_capacity)-2e-7:
        return 'fixed terminal service insufficient for return/departure reserve'
    return None


def constant_site_plan(site,n_slots,n_sites,constraints):
    """Uniform accessible power delivering the same absolute battery terminal energy."""
    d=operational_data(site,n_slots,n_sites,constraints)
    reason=_service_precheck(d,constraints)
    if reason:return _failure('operational_infeasible',reason,operations=d)
    delta=d['required_charging_gain']; positive=delta>=0
    possible=d['charge']>0 if positive else d['discharge']>0
    count=int(possible.sum())
    if not count and abs(delta)>1e-12:
        return _failure('operational_infeasible','no accessible charging/discharging slots',operations=d)
    u=np.zeros(n_slots)
    if count:
        grid=delta/constraints.eta_charge if positive else delta*constraints.eta_discharge
        u[possible]=grid/(constraints.dt_hours*count)
    plan=_plan(site,u,n_sites,constraints,d['ready_slot'],d['travel_energy'],dict(method='uniform_battery_service'))
    checked=check_service_plan(plan,constraints)
    if not checked['feasible']:
        return _failure('constant_infeasible','uniform power violates battery service constraints',validation=checked)
    return dict(success=True,status='constant_feasible',plan=plan,incumbent=None,best_bound=None,
        absolute_gap=None,relative_gap=None,termination='analytic uniform allocation',validation=checked)


def _frozen_solver(objective,integer,lower,upper,matrix,los,his,options):
    """Use LP for charging only; preserve primal/dual evidence in either case."""
    if np.any(integer):
        return milp(objective,integrality=integer,bounds=Bounds(lower,upper),
            constraints=LinearConstraint(matrix,los,his),options=options)
    los,his=np.asarray(los,float),np.asarray(his,float)
    equality=np.isfinite(los)&np.isfinite(his)&(los==his)
    upper_rows=np.isfinite(his)&~equality; lower_rows=np.isfinite(los)&~equality
    from scipy.sparse import vstack
    aub=vstack([matrix[upper_rows],-matrix[lower_rows]],format='csr')
    bub=np.r_[his[upper_rows],-los[lower_rows]]
    result=linprog(objective,A_ub=aub,b_ub=bub,A_eq=matrix[equality],b_eq=his[equality],
        bounds=np.column_stack([lower,upper]),method='highs',
        options=dict(time_limit=options['time_limit'],presolve=options['presolve']))
    result.mip_dual_bound=None; result.mip_gap=None; result.mip_node_count=None
    result.lp_dual_objective=None; result.lp_dual_stationarity_inf=None
    result.lp_dual_sign_error=None
    if result.x is not None and result.status == 0:
        yu=np.asarray(result.ineqlin.marginals); ye=np.asarray(result.eqlin.marginals)
        zl=np.asarray(result.lower.marginals); zu=np.asarray(result.upper.marginals)
        lf=np.isfinite(lower); uf=np.isfinite(upper)
        dual=float(bub@yu+his[equality]@ye+lower[lf]@zl[lf]+upper[uf]@zu[uf])
        stationarity=objective-aub.T@yu-matrix[equality].T@ye-zl-zu
        signs=max(float(np.max(yu,initial=0)),float(np.max(-zl,initial=0)),float(np.max(zu,initial=0)))
        result.lp_dual_objective=dual
        result.lp_dual_stationarity_inf=float(np.max(abs(stationarity),initial=0))
        result.lp_dual_sign_error=signs
        if result.lp_dual_stationarity_inf<=1e-7 and signs<=1e-7:
            result.mip_dual_bound=dual
    return result

def solve_site_step(site, n_slots, n_sites, constraints, base_power=None,
                    trust_radius=None, intercepts=None, gradients=None, *,
                    time_limit=10., mip_rel_gap=1e-8):
    """Maximize min_k intercept[k]+gradient[k]@(u-base_power).

    The returned incumbent is recomputed on decoded net injection. A
    best_bound is an upper bound for this frozen affine LP/MILP objective. A
    feasible time-limit incumbent is executable but retains its termination.
    """
    d=operational_data(site,n_slots,n_sites,constraints)
    reason=_service_precheck(d,constraints)
    if reason:return _failure('operational_infeasible',reason,operations=d)
    charge,discharge,tau,expense=d['charge'],d['discharge'],d['ready_slot'],d['travel_energy']
    initial=d['post_drive_energy']/constraints.battery_capacity
    if not np.isfinite(time_limit) or time_limit <= 0 or not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError('positive finite time limit and nonnegative finite gap required')
    base = np.zeros(n_slots) if base_power is None else np.asarray(base_power,float)
    if base.shape != (n_slots,) or not np.isfinite(base).all():
        raise ValueError('base_power must be finite (time,)')
    if gradients is None:
        ints = np.atleast_1d(np.asarray(0. if intercepts is None else intercepts,float))
        gs = np.zeros((len(ints),n_slots))
    else:
        gs = np.asarray(gradients,float)
        if gs.ndim == 1:gs = gs[None,:]
        ints = np.zeros(len(gs)) if intercepts is None else np.atleast_1d(np.asarray(intercepts,float))
    if gs.ndim != 2 or gs.shape[1] != n_slots or not len(gs) or ints.shape != (len(gs),) or not np.isfinite(gs).all() or not np.isfinite(ints).all():
        raise ValueError('finite intercepts (K,) and gradients (K,time) required')
    if trust_radius is not None and (not np.isfinite(trust_radius) or trust_radius < 0):
        raise ValueError('trust_radius must be finite and nonnegative')
    cp=np.arange(n_slots);dp=np.arange(n_slots,2*n_slots)
    offset=2*n_slots
    binary=bool(np.any(discharge > 0))
    mode=np.arange(offset,offset+n_slots) if binary else None
    if binary:offset += n_slots
    soc=np.arange(offset,offset+n_slots+1);eta=offset+n_slots+1;nv=eta+1
    lower=np.full(nv,-np.inf);upper=np.full(nv,np.inf);integer=np.zeros(nv,int)
    lower[cp]=0;upper[cp]=charge;lower[dp]=0;upper[dp]=discharge
    lower[soc]=constraints.soc_min;upper[soc]=constraints.soc_max
    if binary:lower[mode]=0;upper[mode]=1;integer[mode]=1
    rows=[];los=[];his=[]
    def add(row,lo=-np.inf,hi=np.inf):rows.append(row);los.append(lo);his.append(hi)
    add({int(soc[0]):1},initial,initial)
    for t in range(n_slots):
        add({int(soc[t+1]):1,int(soc[t]):-1,
             int(cp[t]):-constraints.dt_hours*constraints.eta_charge/constraints.battery_capacity,
             int(dp[t]):constraints.dt_hours/(constraints.eta_discharge*constraints.battery_capacity)},0,0)
        if binary:
            add({int(cp[t]):1,int(mode[t]):-charge[t]},hi=0)
            add({int(dp[t]):1,int(mode[t]):discharge[t]},hi=discharge[t])
        if trust_radius is not None:
            add({int(cp[t]):1,int(dp[t]):-1},base[t]-trust_radius,base[t]+trust_radius)
    terminal_soc=constraints.terminal_energy/constraints.battery_capacity
    add({int(soc[-1]):1},terminal_soc,terminal_soc)
    for intercept,g in zip(ints,gs):
        add({eta:1,**{int(i):-float(v) for i,v in zip(cp,g)},**{int(i):float(v) for i,v in zip(dp,g)}},hi=float(intercept-g@base))
    rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(rows),nv)).tocsr()
    objective=np.zeros(nv);objective[eta]=-1
    solve_started=perf_counter()
    try:
        result=_frozen_solver(objective,integer,lower,upper,matrix,los,his,
            dict(time_limit=float(time_limit),mip_rel_gap=float(mip_rel_gap),presolve=True))
    except (RuntimeError,ValueError,FloatingPointError) as error:
        return _failure('solver_failure',str(error),solver_status=None,solve_seconds=perf_counter()-solve_started)
    diagnostics=dict(solver_status=int(result.status),solver_success=bool(result.success),solve_seconds=perf_counter()-solve_started,
        solver_message=str(result.message),mip_node_count=getattr(result,'mip_node_count',None),
        mip_gap=getattr(result,'mip_gap',None),binary_modes=binary,
        variable_count=nv,constraint_count=len(rows),objective_scope='frozen affine MILP' if binary else 'frozen affine LP',
        solver_kind='MILP' if binary else 'LP',
        lp_dual_objective=getattr(result,'lp_dual_objective',None),
        lp_dual_stationarity_inf=getattr(result,'lp_dual_stationarity_inf',None),
        lp_dual_sign_error=getattr(result,'lp_dual_sign_error',None),
        terminal_energy_pu_hours=constraints.terminal_energy,
        setup_start=d['setup_start'],ready_slot=d['ready_slot'],arrival_slot=d['arrival_slot'])
    if result.x is None or not np.isfinite(result.x).all():
        return _failure('infeasible' if result.status == 2 else 'solver_failure',str(result.message),**diagnostics)
    u=np.asarray(result.x)[cp]-np.asarray(result.x)[dp]
    plan=_plan(site,u,n_sites,constraints,tau,expense,diagnostics)
    # Use the solver SOC in the independent validator, so split-flow mistakes
    # or inaccurate constraints cannot be hidden by reconstructing net SOC.
    plan.soc=np.asarray(result.x)[soc].copy()
    diagnostics.update(split_charge_pu=np.asarray(result.x)[cp].tolist(),
        split_discharge_pu=np.asarray(result.x)[dp].tolist())
    validation=check_service_plan(plan,constraints)
    trust_ok=trust_radius is None or bool(np.max(abs(u-base)) <= trust_radius+2e-7)
    value=float(np.min(ints+gs@(u-base)))
    raw=-float(result.fun) if result.fun is not None else None
    raw_bound=getattr(result,'mip_dual_bound',None)
    bound=-float(raw_bound) if raw_bound is not None and np.isfinite(raw_bound) else raw if result.status == 0 and binary else None
    gap=None if bound is None else bound-value
    eta_ok=float(result.x[eta]) <= value+2e-6
    bounds_ok=gap is None or gap >= -2e-6
    dual_ok=binary or result.status != 0 or bound is not None
    feasible=bool(validation['feasible'] and trust_ok and eta_ok and bounds_ok and dual_ok and np.isfinite(value))
    if feasible:status='optimal' if result.status == 0 else 'feasible_incumbent'
    else:status='invalid_incumbent'
    diagnostics.update(raw_incumbent=raw,raw_best_bound=None if raw_bound is None else -float(raw_bound),
        affine_value=value,objective_eta=float(result.x[eta]),trust_ok=trust_ok,
        objective_eta_ok=eta_ok,bound_consistent=bounds_ok,dual_evidence_ok=dual_ok)
    return dict(success=feasible,status=status,plan=plan,incumbent=value,
        best_bound=bound,absolute_gap=gap,
        relative_gap=None if gap is None else gap/max(1.,abs(value)),
        termination=str(result.message),validation=validation,**diagnostics)


@dataclass(frozen=True)
class SearchSettings:
    max_iterations: int = 3
    max_starts: int = 3
    initial_trust_pu: float = .15
    minimum_trust_pu: float = .003
    gain_tolerance: float = 1e-6
    milp_time_limit: float = 5.
    gradient_mode: str = "full"

    def validate(self):
        if self.max_iterations < 0 or not 1 <= self.max_starts <= 3:
            raise ValueError("nonnegative iterations and one to three starts required")
        if not (0 < self.minimum_trust_pu <= self.initial_trust_pu
                and self.gain_tolerance >= 0 and self.milp_time_limit > 0):
            raise ValueError("invalid trust region, gain tolerance or MILP budget")
        if self.gradient_mode not in ("full", "pq_only"):
            raise ValueError("gradient_mode must be full or pq_only")


def evaluation_record(result):
    """Keep numerical evidence, omitting only in-memory FitResult objects."""
    return {key: value for key, value in result.items() if key != "fit_results"}


def _operation_record(result):
    return {key: value for key, value in result.items() if key != "plan"}


def optimize_service_curves(evaluator, constraints, *, sites=None, settings=None):
    """Compare a few sites, optimizing continuous power at each one.

    The evaluator sees only historical observations and declared forecasts.
    This function never calls an action-library enumeration routine. The two
    optional nonconstant starts are generated by operational optimization.
    """
    began = perf_counter()
    settings = settings or SearchSettings()
    settings.validate()
    horizon, n = evaluator.horizon, evaluator.history.n
    site_list = tuple(range(n) if sites is None else sites)
    if not site_list or len(set(site_list)) != len(site_list):
        raise ValueError("nonempty distinct sites required")
    if any(not isinstance(j, (int, np.integer)) or not 0 <= j < n for j in site_list):
        raise ValueError("site outside terminal set")
    if not np.isclose(constraints.q_over_p, evaluator.q_over_p):
        raise ValueError("operation and physical Q/P must agree")
    cache, evaluations, site_records, failures = {}, [], [], []
    site_plans, site_evaluations = {}, {}
    fit_calls = ac_calls = 0

    def evaluate(plan):
        nonlocal fit_calls, ac_calls
        key = (plan.site, np.asarray(plan.p[:, plan.site], dtype="<f8").tobytes())
        if key not in cache:
            check = check_service_plan(plan, constraints)
            if not check["feasible"]:
                raise ValueError("operation solver returned infeasible plan: "+str(check))
            result = evaluator.evaluate(plan.site, evaluator.base_kva*plan.p[:, plan.site])
            cache[key] = result
            fit_calls += int(result.get("fit_calls", 0))
            ac_calls += int(result.get("total_ac_calls", 0))
            evaluations.append(evaluation_record(result))
            if not result["success"]:
                failures.append(dict(site=int(plan.site), kind="AC_comparison",
                                     details=result.get("failures", [])))
        return cache[key]

    best_plan = best_evaluation = None
    best_constant_plan = best_constant_evaluation = None
    for site in site_list:
        constant = constant_site_plan(site, horizon, n, constraints)
        record = dict(site=int(site), constant_operation=_operation_record(constant),
                      starts=[], steps=[], failures=[], stop_reasons=[])
        site_records.append(record)
        baseline_plan = constant.get("plan")
        baseline = None
        if not constant["success"]:
            failure = dict(site=int(site), kind="constant_operation", details=_operation_record(constant))
            failures.append(failure); record["failures"].append(failure)
            # A constant amplitude may violate time-varying caps while a
            # shaped curve remains feasible. Try the operational feasible set.
            feasible = solve_site_step(site, horizon, n, constraints,
                intercepts=np.zeros(1), gradients=np.zeros((1, horizon)),
                time_limit=settings.milp_time_limit)
            record["feasible_initialization"] = _operation_record(feasible)
            if not feasible["success"]:
                record["stop_reasons"].append("no_feasible_operational_initialization")
                failures.append(dict(site=int(site), kind="initialization_MILP", details=_operation_record(feasible)))
                continue
            initial_plan = feasible["plan"]
        else:
            initial_plan = baseline_plan
            baseline = evaluate(baseline_plan)
            if baseline["success"]:
                if best_constant_evaluation is None or baseline["score"] > best_constant_evaluation["score"]:
                    best_constant_plan, best_constant_evaluation = baseline_plan, baseline
            else:
                record["failures"].append(dict(kind="constant_AC_comparison_failed"))
        record["constant_score"] = None if baseline is None else baseline.get("score")
        local_best_plan = baseline_plan if baseline is not None and baseline["success"] else None
        local_best_evaluation = baseline if baseline is not None and baseline["success"] else None
        best_seed_score = None
        starts = [initial_plan]
        # These seeds address zero first-order prices; they are generated by
        # two small optimization problems, not a stored library of profiles.
        pattern = np.where(np.arange(horizon) % 2 == 0, 1., -1.)
        seed_keys = {initial_plan.p.tobytes()}
        for sign in (1., -1.):
            if len(starts) >= settings.max_starts:
                break
            generated = solve_site_step(site, horizon, n, constraints,
                base_power=initial_plan.p[:, site], trust_radius=None,
                intercepts=np.zeros(1), gradients=(sign*pattern)[None, :],
                time_limit=settings.milp_time_limit)
            record["starts"].append(dict(kind="optimized_alternating_seed",
                                         sign=sign, operation=_operation_record(generated)))
            if not generated["success"]:
                failure = dict(site=int(site), kind="seed_MILP", details=_operation_record(generated))
                failures.append(failure); record["failures"].append(failure)
                continue
            plan = generated["plan"]
            if plan.p.tobytes() not in seed_keys:
                starts.append(plan); seed_keys.add(plan.p.tobytes())
        for start_index, start_plan in enumerate(starts):
            current_plan, current = start_plan, evaluate(start_plan)
            record["starts"].append(dict(kind="accepted_seed_input", index=start_index,
                                         score=current.get("score"), success=current["success"]))
            if not current["success"]:
                record["stop_reasons"].append("start_AC_comparison_failed")
                continue
            best_seed_score = float(current["score"]) if best_seed_score is None else max(best_seed_score, float(current["score"]))
            if local_best_evaluation is None or current["score"] > local_best_evaluation["score"]:
                local_best_plan, local_best_evaluation = current_plan, current
            trust = settings.initial_trust_pu
            for iteration in range(settings.max_iterations):
                pairs = current["pairs"]
                field = "gradient_kw" if settings.gradient_mode == "full" else "pq_only_gradient_kw"
                gradients = evaluator.base_kva*np.asarray([p[field] for p in pairs], float)
                intercepts = np.asarray([p["score"] for p in pairs], float)
                if gradients.shape != (len(pairs), horizon) or not np.isfinite(gradients).all():
                    raise ValueError("necessary pair gradients missing or nonfinite")
                proposed = solve_site_step(site, horizon, n, constraints,
                    base_power=current_plan.p[:, site], trust_radius=trust,
                    intercepts=intercepts, gradients=gradients,
                    time_limit=settings.milp_time_limit)
                step = dict(start=start_index, iteration=iteration, trust_radius_pu=trust,
                            before_score=float(current["score"]), gradient_mode=settings.gradient_mode,
                            milp=_operation_record(proposed), accepted=False)
                record["steps"].append(step)
                if not proposed["success"]:
                    failure = dict(site=int(site), kind="step_MILP", details=_operation_record(proposed))
                    failures.append(failure); record["failures"].append(failure)
                    record["stop_reasons"].append("step_MILP_failed"); break
                proposal_plan = proposed["plan"]
                distance = float(np.max(np.abs(proposal_plan.p-current_plan.p)))
                step["power_change_inf_pu"] = distance
                if distance <= 1e-9:
                    record["stop_reasons"].append("affine_step_stationary"); break
                trial = evaluate(proposal_plan)
                step["trial_score"] = trial.get("score")
                step["trial_success"] = trial["success"]
                gain = (float(trial["score"]-current["score"]) if trial["success"] else None)
                step["actual_local_score_gain"] = gain
                if trial["success"] and gain > settings.gain_tolerance:
                    step["accepted"] = True
                    current_plan, current = proposal_plan, trial
                    caps = constraints.arrays(horizon, n)
                    cap_span = float(np.max(caps[0][:, site]+caps[1][:, site]))
                    trust = min(cap_span, 1.5*trust)
                    if local_best_evaluation is None or current["score"] > local_best_evaluation["score"]:
                        local_best_plan, local_best_evaluation = current_plan, current
                else:
                    trust *= .5
                    if trust < settings.minimum_trust_pu:
                        record["stop_reasons"].append("trust_region_below_minimum"); break
            else:
                record["stop_reasons"].append("iteration_budget")
        if local_best_evaluation is None:
            record["stop_reasons"].append("all_AC_initializations_failed")
            continue
        record["selected_score"] = float(local_best_evaluation["score"])
        record["best_seed_score"] = best_seed_score
        record["gradient_gain_over_best_seed"] = float(local_best_evaluation["score"]-best_seed_score)
        record["gain_over_site_constant"] = (float(local_best_evaluation["score"]-baseline["score"])
            if baseline is not None and baseline["success"] else None)
        record["selected_plan"] = local_best_plan.summary()
        site_plans[int(site)], site_evaluations[int(site)] = local_best_plan, local_best_evaluation
        if best_evaluation is None or local_best_evaluation["score"] > best_evaluation["score"]:
            best_plan, best_evaluation = local_best_plan, local_best_evaluation
    success = best_plan is not None
    return dict(success=success, plan=best_plan, evaluation=best_evaluation,
                best_constant_plan=best_constant_plan, best_constant_evaluation=best_constant_evaluation,
                selected_site=None if best_plan is None else int(best_plan.site),
                score=None if best_evaluation is None else float(best_evaluation["score"]),
                gain_over_best_constant=None if not success or best_constant_evaluation is None else
                    float(best_evaluation["score"]-best_constant_evaluation["score"]),
                site_records=site_records, evaluations=evaluations, failures=failures,
                site_plans=site_plans, site_evaluations=site_evaluations,
                best_seed_score=max((r["best_seed_score"] for r in site_records if "best_seed_score" in r),default=None),
                gradient_gain_over_best_seed=None if not success else float(best_evaluation["score"]-
                    max(r["best_seed_score"] for r in site_records if "best_seed_score" in r)),
                fit_calls=fit_calls, total_ac_calls=ac_calls, seconds=perf_counter()-began,
                settings=asdict(settings), search="continuous_battery_service_affine_LP_or_MILP_with_AC_acceptance",
                constant_guard="retained same-battery-terminal-service feasible incumbent",
                bound_scope="affine operational subproblem only",
                global_design_optimality_certified=False)


def _jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable(v) for v in value]
    return value


def search_record(result):
    return _jsonable({key: (value.summary() if key in ("plan", "best_constant_plan") and value is not None
                  else evaluation_record(value) if key in ("evaluation", "best_constant_evaluation") and value is not None
                  else {str(j): p.summary() for j, p in value.items()} if key == "site_plans"
                  else {str(j): evaluation_record(e) for j, e in value.items()} if key == "site_evaluations"
                  else value) for key, value in result.items()})