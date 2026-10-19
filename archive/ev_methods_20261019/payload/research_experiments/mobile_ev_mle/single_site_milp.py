"""One connection site per EV round; continuous power selected by a MILP.

Power is pu and energy is pu-hours. The affine max-min objective is only the
current optimization surrogate. Its MILP bound applies to this frozen model.
Travel is fixed by the chosen site: its energy is deducted before slot zero,
and no injection is possible during the first travel_slots slots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class SingleSiteConstraints:
    dt_hours: float = .25
    p_charge_max: Any = .30
    p_discharge_max: Any = 0.
    energy: float = .15
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

    def arrays(self, n_slots: int, n_sites: int):
        if (not isinstance(n_slots, (int, np.integer)) or n_slots < 1
                or not isinstance(n_sites, (int, np.integer)) or n_sites < 1):
            raise ValueError('positive integer n_slots and n_sites required')
        numbers = [self.dt_hours, self.battery_capacity, self.eta_charge,
                   self.eta_discharge, self.energy, self.initial_soc,
                   self.soc_min, self.soc_max, self.q_over_p]
        if not np.isfinite(numbers).all():
            raise ValueError('constraint scalars must be finite')
        if (self.dt_hours <= 0 or self.battery_capacity <= 0
                or not 0 < self.eta_charge <= 1
                or not 0 < self.eta_discharge <= 1):
            raise ValueError('invalid time, battery capacity, or efficiency')
        if not 0 <= self.soc_min <= self.initial_soc <= self.soc_max <= 1:
            raise ValueError('SOC limits must contain initial SOC')
        if (not isinstance(self.initial_site, (int, np.integer))
                or not 0 <= self.initial_site < n_sites):
            raise ValueError('initial site outside site set')
        shape = (n_slots, n_sites)
        caps = []
        for name in ('p_charge_max', 'p_discharge_max'):
            try:
                cap = np.broadcast_to(np.asarray(getattr(self, name), float), shape).copy()
            except ValueError as error:
                raise ValueError(name+' must broadcast to (time,site)') from error
            if not np.isfinite(cap).all() or (cap < 0).any():
                raise ValueError(name+' must be finite and nonnegative')
            caps.append(cap)
        if self.energy < 0 and not np.any(caps[1] > 0):
            raise ValueError('negative energy requires discharge capability')
        access = np.ones(shape, bool) if self.availability is None else np.asarray(self.availability)
        if access.shape != shape or not np.isin(access, [False, True]).all():
            raise ValueError('availability must be bool (time,site)')
        travel = np.ones((n_sites,n_sites), int)-np.eye(n_sites,dtype=int) if self.travel_slots is None else np.asarray(self.travel_slots)
        mobility = .01*(np.ones((n_sites,n_sites))-np.eye(n_sites)) if self.travel_energy is None else np.asarray(self.travel_energy,float)
        if (travel.shape != (n_sites,n_sites) or not np.isfinite(travel).all()
                or (travel < 0).any() or not np.equal(travel,np.floor(travel)).all()
                or np.any(np.diag(travel) != 0)):
            raise ValueError('travel_slots requires nonnegative integer matrix with zero diagonal')
        if (mobility.shape != (n_sites,n_sites) or not np.isfinite(mobility).all()
                or (mobility < 0).any() or np.any(np.diag(mobility) != 0)):
            raise ValueError('travel_energy requires nonnegative matrix with zero diagonal')
        return caps[0], caps[1], access.astype(bool), travel.astype(int), mobility


@dataclass
class SingleSitePlan:
    site: int
    p: np.ndarray
    q: np.ndarray
    sites: np.ndarray
    soc: np.ndarray
    solver: dict = field(default_factory=dict)

    def summary(self):
        return dict(site=int(self.site),p=np.asarray(self.p).tolist(),
                    q=np.asarray(self.q).tolist(),sites=np.asarray(self.sites).tolist(),
                    soc=np.asarray(self.soc).tolist(),solver=self.solver)


def _site_data(site, n_slots, n_sites, constraints):
    charge, discharge, access, travel, mobility = constraints.arrays(n_slots,n_sites)
    if not isinstance(site,(int,np.integer)) or not 0 <= site < n_sites:
        raise ValueError('site outside site set')
    tau = int(travel[constraints.initial_site,site])
    expense = float(mobility[constraints.initial_site,site])
    allowed = access[:,site].copy()
    allowed[:min(tau,n_slots)] = False
    return charge[:,site]*allowed, discharge[:,site]*allowed, tau, expense


def _plan(site, u, n_sites, constraints, tau, expense, solver=None):
    u = np.asarray(u,float)
    p = np.zeros((len(u),n_sites));p[:,site] = u
    q = constraints.q_over_p*p
    sites = np.full(len(u),site,int);sites[:min(tau,len(u))] = -1
    cp,dp = np.maximum(u,0),np.maximum(-u,0)
    initial = constraints.initial_soc-expense/constraints.battery_capacity
    soc = np.r_[initial,initial+np.cumsum(constraints.dt_hours*(constraints.eta_charge*cp-dp/constraints.eta_discharge)/constraints.battery_capacity)]
    return SingleSitePlan(site,p,q,sites,soc,solver or {})


def check_single_site_plan(plan, constraints, *, tolerance=2e-7):
    """Recompute operations from net injections, independently of MILP rows.

    soc[0] is the post-driving SOC; the independently supplied initial_soc is
    the pre-driving state. This checks both states and every slot boundary.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    issues=[]
    try:
        p,q,sites,soc = (np.asarray(getattr(plan,key)) for key in ('p','q','sites','soc'))
        if p.ndim != 2 or not all(p.shape) or q.shape != p.shape or sites.shape != (len(p),) or soc.shape != (len(p)+1,):
            return dict(feasible=False,issues=['shape'])
        if not all(np.isfinite(a).all() for a in (p,q,sites,soc)):
            return dict(feasible=False,issues=['nonfinite'])
        charge,discharge,tau,expense = _site_data(plan.site,len(p),p.shape[1],constraints)
        u = p[:,plan.site]
        if np.any(abs(np.delete(p,plan.site,axis=1)) > tolerance):issues.append('multiple_sites')
        if np.any(abs(q-constraints.q_over_p*p) > tolerance):issues.append('reactive_power')
        expected_sites = np.full(len(p),plan.site,int);expected_sites[:min(tau,len(p))] = -1
        if not np.array_equal(sites,expected_sites):issues.append('connection_path')
        if tau >= len(p):issues.append('arrival_outside_horizon')
        if np.any(u > charge+tolerance) or np.any(u < -discharge-tolerance):issues.append('power_or_access')
        energy = float(constraints.dt_hours*np.sum(u))
        if abs(energy-constraints.energy) > tolerance:issues.append('energy')
        # No use of the solver's split charging/discharging variables.
        expected = _plan(plan.site,u,p.shape[1],constraints,tau,expense).soc
        if np.max(abs(soc-expected)) > tolerance:issues.append('soc_dynamics')
        if np.any(expected < constraints.soc_min-tolerance) or np.any(expected > constraints.soc_max+tolerance):issues.append('soc')
        return dict(feasible=not issues,issues=issues,energy_pu_hours=energy,
                    travel_slots=tau,travel_energy_pu_hours=expense,
                    initial_soc=constraints.initial_soc,soc_before_charging=float(expected[0]),
                    recomputed_soc=expected.tolist())
    except (ValueError,TypeError,AttributeError,IndexError) as error:
        return dict(feasible=False,issues=['invalid_input'],error=str(error))


def _failure(status, message, **extra):
    return dict(success=False,status=status,plan=None,incumbent=None,best_bound=None,
                absolute_gap=None,relative_gap=None,termination=message,**extra)


def constant_site_plan(site, n_slots, n_sites, constraints):
    """Same grid-energy task at a uniform power in all available arrival slots."""
    charge,discharge,tau,expense = _site_data(site,n_slots,n_sites,constraints)
    if tau >= n_slots:
        return _failure('operational_infeasible','arrival outside horizon')
    post = constraints.initial_soc-expense/constraints.battery_capacity
    if post < constraints.soc_min-2e-7:
        return _failure('operational_infeasible','driving violates SOC minimum')
    possible = charge > 0 if constraints.energy >= 0 else discharge > 0
    count = int(possible.sum())
    if count == 0 and constraints.energy != 0:
        return _failure('operational_infeasible','no accessible charging/discharging slots')
    u = np.zeros(n_slots)
    if count:u[possible] = constraints.energy/(constraints.dt_hours*count)
    plan = _plan(site,u,n_sites,constraints,tau,expense,dict(method='uniform_available_slots'))
    check = check_single_site_plan(plan,constraints)
    if not check['feasible']:
        return _failure('constant_infeasible','uniform power violates operational constraints',validation=check)
    return dict(success=True,status='constant_feasible',plan=plan,incumbent=None,
                best_bound=None,absolute_gap=None,relative_gap=None,
                termination='analytic uniform allocation',validation=check)


def solve_site_step(site, n_slots, n_sites, constraints, base_power=None,
                    trust_radius=None, intercepts=None, gradients=None, *,
                    time_limit=10., mip_rel_gap=1e-8):
    """Maximize min_k intercept[k]+gradient[k]@(u-base_power).

    The returned incumbent is recomputed on the decoded net injection. A
    best_bound is an upper bound for this frozen affine MILP objective. A
    feasible time-limit incumbent is executable but retains its termination.
    """
    charge,discharge,tau,expense = _site_data(site,n_slots,n_sites,constraints)
    if tau >= n_slots:
        return _failure('operational_infeasible','arrival outside horizon')
    initial = constraints.initial_soc-expense/constraints.battery_capacity
    if initial < constraints.soc_min-2e-7:
        return _failure('operational_infeasible','driving violates SOC minimum')
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
    add({**{int(i):constraints.dt_hours for i in cp},**{int(i):-constraints.dt_hours for i in dp}},constraints.energy,constraints.energy)
    for intercept,g in zip(ints,gs):
        add({eta:1,**{int(i):-float(v) for i,v in zip(cp,g)},**{int(i):float(v) for i,v in zip(dp,g)}},hi=float(intercept-g@base))
    rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(rows),nv)).tocsr()
    objective=np.zeros(nv);objective[eta]=-1
    solve_started=perf_counter()
    try:
        result=milp(objective,integrality=integer,bounds=Bounds(lower,upper),
                    constraints=LinearConstraint(matrix,los,his),
                    options=dict(time_limit=float(time_limit),mip_rel_gap=float(mip_rel_gap),presolve=True))
    except (RuntimeError,ValueError,FloatingPointError) as error:
        return _failure('solver_failure',str(error),solver_status=None,solve_seconds=perf_counter()-solve_started)
    diagnostics=dict(solver_status=int(result.status),solver_success=bool(result.success),solve_seconds=perf_counter()-solve_started,
        solver_message=str(result.message),mip_node_count=getattr(result,'mip_node_count',None),
        mip_gap=getattr(result,'mip_gap',None),binary_modes=binary,
        variable_count=nv,constraint_count=len(rows),objective_scope='frozen affine MILP')
    if result.x is None or not np.isfinite(result.x).all():
        return _failure('infeasible' if result.status == 2 else 'solver_failure',str(result.message),**diagnostics)
    u=np.asarray(result.x)[cp]-np.asarray(result.x)[dp]
    plan=_plan(site,u,n_sites,constraints,tau,expense,diagnostics)
    # Use the solver SOC in the independent validator, so split-flow mistakes
    # or inaccurate constraints cannot be hidden by reconstructing net SOC.
    plan.soc=np.asarray(result.x)[soc].copy()
    validation=check_single_site_plan(plan,constraints)
    trust_ok=trust_radius is None or bool(np.max(abs(u-base)) <= trust_radius+2e-7)
    value=float(np.min(ints+gs@(u-base)))
    raw=-float(result.fun) if result.fun is not None else None
    raw_bound=getattr(result,'mip_dual_bound',None)
    bound=-float(raw_bound) if raw_bound is not None and np.isfinite(raw_bound) else raw if result.status == 0 else None
    gap=None if bound is None else bound-value
    eta_ok=float(result.x[eta]) <= value+2e-6
    bounds_ok=gap is None or gap >= -2e-6
    feasible=bool(validation['feasible'] and trust_ok and eta_ok and bounds_ok and np.isfinite(value))
    if feasible:status='optimal' if result.status == 0 else 'feasible_incumbent'
    else:status='invalid_incumbent'
    diagnostics.update(raw_incumbent=raw,raw_best_bound=None if raw_bound is None else -float(raw_bound),
        affine_value=value,objective_eta=float(result.x[eta]),trust_ok=trust_ok,
        objective_eta_ok=eta_ok,bound_consistent=bounds_ok)
    return dict(success=feasible,status=status,plan=plan,incumbent=value,
        best_bound=bound,absolute_gap=gap,
        relative_gap=None if gap is None else gap/max(1.,abs(value)),
        termination=str(result.message),validation=validation,**diagnostics)
