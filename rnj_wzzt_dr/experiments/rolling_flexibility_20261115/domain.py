"""Quarter-hour equipment physics with firm service and uncertain extra calls.

All actions are changes from the original measured-net device decomposition.
The firm service controller is an explicit new assumption. Only additional
requests are multiplied by the device response factor; past delivery is fixed.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
import numpy as np

from ..equipment_flex_20261115.domain import EquipmentCase


DT = .25
ETA_ENDPOINTS = (.9, 1.)
BAND_BUFFER = 1e-7


def window(clock, start, end):
    return ((clock >= start) & (clock < end) if start < end
            else (clock >= start) | (clock < end))


class Fleet:
    def __init__(self, spec, labels, base_kw, band_arrays=None):
        # Reuse the validated catalog/units/prices, not its hourly physics.
        catalog = EquipmentCase(spec, labels, np.zeros((96, 2*len(labels))), base_kw)
        for key in ('spec', 'labels', 'N', 'R', 'P_R', 'Q_R', 'resources', 'B',
                    'p_index', 'q_index', 'device_ids', 'devices', 'base_kw', 'config'):
            setattr(self, key, deepcopy(getattr(catalog, key)))
        self.dt, self.H, self.repeat = DT, 96, 1
        self.shape = (96, self.R)
        self.clock = (8+(np.arange(96)+.5)*DT) % 24
        self.original_baseline = np.zeros(self.shape)
        self.original_baseline[:, :self.P_R] = catalog.device_p0
        for did, q in self.q_index.items():
            self.original_baseline[:, q] = catalog.device_q0[:, self.p_index[did]]
        self.fee_plus = np.repeat(catalog.fee_plus, 4, axis=0)
        self.fee_minus = np.repeat(catalog.fee_minus, 4, axis=0)
        original_spec_hash = self.config.pop('spec_content_sha256', None)
        self.config.update(schema='rolling_equipment_firm_service_v1', original_spec_content_sha256=original_spec_hash,
            time=dict(start_hour=8, hours=24, decision_hours=DT, measurement_minutes=15),
            response='Firm service is directly controlled; one latent daily eta per device multiplies additional P/Q requests only',
            cost_scope='Requested activation optimizes service+extra relative to original baseline; realized activation bills service+eta*extra',
            baseline_rule='Original measured net plus the total delta; no device baseline is added twice')
        self.lower = np.zeros(self.shape); self.upper = np.zeros(self.shape)
        self.states = {}; self.circles = []
        for device in self.devices:
            did, kind = device['id'], device['kind']; j = self.p_index[did]
            power = float(device['power_kw']); p0 = self.original_baseline[:, j]
            if kind == 'pv':
                available = np.asarray(device['available_generation_kw'], float)
                self.upper[:, j] = available
                q_online = available > 0
            elif kind == 'ev':
                online = window(self.clock, *device['connected_clock_hours'])
                self.lower[:, j] = -p0
                self.upper[:, j] = power*online-p0
                target = float(device['required_battery_energy_kwh'])
                self.states[did] = dict(column=j, kind=kind, initial=0., target=target,
                    minimum=0., maximum=target, a=np.ones(96), g=np.zeros(96),
                    c=np.full(96, DT*device['charging_efficiency']),
                    departure_step=int(np.flatnonzero(online)[-1]+1), units='kWh')
            elif kind == 'battery':
                charge = window(self.clock, *device['charging_clock_hours'])
                discharge = window(self.clock, *device['discharging_clock_hours'])
                self.upper[charge, j] = power; self.lower[discharge, j] = -power
                energy = float(device['energy_kwh'])
                c = DT*np.where(charge, device['charging_efficiency'],
                                np.where(discharge, 1/device['discharging_efficiency'], 0.))
                self.states[did] = dict(column=j, kind=kind, initial=energy*device['initial_soc'],
                    target=energy*device['terminal_soc'], minimum=energy*device['minimum_soc'],
                    maximum=energy*device['maximum_soc'], a=np.ones(96), g=np.zeros(96),
                    c=c, energy_kwh=energy, units='kWh')
                q_online = np.zeros(96, bool)
                for start, stop in device['reactive_clock_windows']:
                    q_online |= window(self.clock, start, stop)
            elif kind == 'thermal':
                self.lower[:, j] = -p0; self.upper[:, j] = power-p0
                rho_h = float(device['hourly_temperature_retention']); rho = rho_h**DT
                C = float(device['heat_capacity_kwh_per_C'])
                gain = DT/C if rho_h == 1 else (1-rho)/((1-rho_h)*C)
                demand = np.asarray(device['heat_demand_kw'], float)
                ambient = float(device['ambient_temperature_C']); cop = float(device['cop'])
                g = (1-rho)*ambient-gain*demand
                self.states[did] = dict(column=j, kind=kind,
                    initial=float(device['initial_temperature_C']),
                    target=float(device['terminal_temperature_C']),
                    minimum=float(device['minimum_temperature_C']),
                    maximum=float(device['maximum_temperature_C']),
                    a=np.full(96, rho), g=g, c=np.full(96, gain*cop), units='degC',
                    rho_hour=rho_h, rho_15min=rho, input_gain_15min=gain,
                    effective_C_15min=DT/gain,
                    discretization='Exact four-way subdivision of the original hourly linear map; original heat demand retained')
                expected = rho*device['initial_temperature_C']+g+gain*cop*p0
                if not np.allclose(expected, device['initial_temperature_C'], atol=1e-11, rtol=0):
                    raise ValueError('Quarter-hour thermal baseline is not stationary')
            else:
                raise ValueError('Unknown device kind')
            if kind in ('pv', 'battery'):
                q = self.q_index[did]; cap = float(device['reactive_cap_kvar'])
                self.lower[q_online, q] = -cap; self.upper[q_online, q] = cap
                for t in range(96):
                    self.circles.append(dict(t=t, p=j, q=q, S=float(device['inverter_kva']), device=did))
        self.reachable = {}
        for did, state in self.states.items():
            lo = np.empty(97); hi = np.empty(97); lo[96] = hi[96] = state['target']
            j = state['column']
            pmin = self.original_baseline[:, j]+self.lower[:, j]
            pmax = self.original_baseline[:, j]+self.upper[:, j]
            for t in range(95, -1, -1):
                a, g, c = state['a'][t], state['g'][t], state['c'][t]
                lo[t] = max(state['minimum'], (lo[t+1]-g-c*pmax[t])/a)
                hi[t] = min(state['maximum'], (hi[t+1]-g-c*pmin[t])/a)
            if np.any(lo > hi+1e-10):
                raise ValueError('Declared original service is unreachable: '+did)
            self.reachable[did] = dict(lower=lo, upper=hi,
                assumption='Future firm service can use the declared contracted active-power limits; future backup Q=0')
        self.kappa = self.band = None
        if band_arrays is not None:
            if tuple(np.asarray(band_arrays['labels']).tolist()) != self.labels:
                raise ValueError('Band label order differs')
            if not np.array_equal(np.asarray(band_arrays['decision_hours']), (8+np.arange(24)) % 24):
                raise ValueError('Band hourly order differs from 08:00 origin')
            self.kappa = np.asarray(band_arrays['kappa'], float).copy()
            hourly = np.asarray(band_arrays['band_kvar'], float)
            if self.kappa.shape != (self.N,) or hourly.shape != (24, self.N) or not np.isfinite(hourly).all() or np.any(hourly < 0) or not np.isfinite(self.kappa).all():
                raise ValueError('Invalid historical band')
            self.band = np.repeat(hourly, 4, axis=0)
        if not self.audit_prefix(np.zeros(self.shape))['passed']:
            raise ValueError('Original quarter-hour baseline fails service tasks')
        self.config['quarter_hour_thermal']={did:{key:s[key] for key in
            ('rho_hour','rho_15min','input_gain_15min','effective_C_15min','discretization')}
            for did,s in self.states.items() if s['kind']=='thermal'}

    def _array(self, value, rows=None):
        value = np.asarray(value, float)
        if value.ndim != 2 or value.shape[1] != self.R or value.shape[0] > 96 or (rows is not None and len(value) != rows) or not np.isfinite(value).all():
            raise ValueError('Expected finite step-by-resource delta')
        return value

    def activation_fee(self, delta, start=0):
        delta = self._array(delta)
        if not isinstance(start, (int, np.integer)) or start < 0 or start+len(delta) > 96:
            raise ValueError('Invalid fee time slice')
        s = slice(start, start+len(delta))
        return float(DT*np.sum(self.fee_plus[s]*np.maximum(delta, 0)+self.fee_minus[s]*np.maximum(-delta, 0)))

    def band_audit(self, delta, start=0):
        delta = self._array(delta)
        if self.band is None:return dict(passed=True, available=False)
        pq = delta@self.B.T; departure = pq[:, self.N:]-pq[:, :self.N]*self.kappa
        width = self.band[start:start+len(delta)]
        error = np.abs(departure)-width
        tolerance = 64*np.finfo(float).eps*(np.abs(pq[:, self.N:])+np.abs(pq[:, :self.N]*self.kappa)+width)
        return dict(passed=bool(np.all(error <= tolerance)), available=True,
                    maximum_violation_kvar=float(np.maximum(error, 0).max(initial=0)),
                    departure_kvar=departure.tolist(), bandwidth_kvar=width.tolist())

    def audit_prefix(self, prefix_delta, tolerance=1e-7):
        delta = self._array(prefix_delta); k = len(delta)
        gross = self.original_baseline[:k]+delta
        box = float(max(0., np.max(self.lower[:k]-delta, initial=0), np.max(delta-self.upper[:k], initial=0)))
        circle = max((max(0., (gross[r['t'], r['p']]**2+gross[r['t'], r['q']]**2)/r['S']**2-1)
                      for r in self.circles if r['t'] < k), default=0.)
        states = {}; state_error = reach_error = terminal_error = 0.
        for did, s in self.states.items():
            values = [s['initial']]
            for t in range(k):values.append(s['a'][t]*values[-1]+s['g'][t]+s['c'][t]*gross[t, s['column']])
            values = np.asarray(values)
            violation = max(0., s['minimum']-float(values.min()), float(values.max())-s['maximum'])
            reach = self.reachable[did]
            gap = max(0., float(reach['lower'][k]-values[-1]), float(values[-1]-reach['upper'][k]))
            terminal = abs(float(values[-1]-s['target'])) if k == 96 else None
            state_error = max(state_error, violation); reach_error = max(reach_error, gap)
            terminal_error = max(terminal_error, terminal or 0.)
            states[did] = dict(kind=s['kind'], units=s['units'], trajectory=values.tolist(),
                current=float(values[-1]), target=s['target'], terminal_error=terminal,
                reachable_lower=float(reach['lower'][k]), reachable_upper=float(reach['upper'][k]),
                remaining_reachable=gap <= tolerance)
            if s['kind']=='battery':states[did]['soc']=(values/s['energy_kwh']).tolist()
            if s['kind']=='ev':states[did]['remaining_energy_kwh']=float(s['target']-values[-1])
        physical = max(box, circle, state_error, terminal_error) <= tolerance
        return dict(passed=bool(physical and reach_error <= tolerance), physical_passed=bool(physical),
            remaining_reachable=bool(reach_error <= tolerance), completed_steps=k, terminal_checked=k==96,
            max_box_violation=box, max_circle_relative_violation=float(circle),
            max_state_violation=state_error, max_terminal_error=terminal_error,
            max_reachability_violation=reach_error, states=states, band=self.band_audit(delta),
            activation_yuan=self.activation_fee(delta),
            firm_service_assumption=True, baseline_added=False)

    def _tail(self, prefix_delta):
        prefix = self._array(prefix_delta); k = len(prefix); n = 96-k; dim = n*self.R
        status = self.audit_prefix(prefix)
        eq, beq, inequalities, bounds = [], [], [], []
        for did, s in self.states.items():
            constant = status['states'][did]['current']; row = np.zeros(dim); j = s['column']
            for h, t in enumerate(range(k, 96)):
                row = s['a'][t]*row; row[h*self.R+j] += s['c'][t]
                constant = s['a'][t]*constant+s['g'][t]+s['c'][t]*self.original_baseline[t,j]
                inequalities.extend((row.copy(), -row.copy()))
                bounds.extend((s['maximum']-constant, constant-s['minimum']))
            if n: eq.append(row); beq.append(s['target']-constant)
        return dict(prefix=prefix.copy(), k=k, H=n, dimension=dim, status=status,
            A_eq=np.asarray(eq).reshape(len(eq),dim), b_eq=np.asarray(beq),
            A_ub=np.asarray(inequalities).reshape(len(inequalities),dim), b_ub=np.asarray(bounds),
            lower=self.lower[k:].copy(), upper=self.upper[k:].copy())

    def _attach_tail(self, model, tail, name='service'):
        u = model.addMVar((tail['H'],self.R), lb=tail['lower'], ub=tail['upper'], name=name)
        flat = u.reshape(tail['dimension'])
        if len(tail['b_eq']):model.addConstr(tail['A_eq']@flat == tail['b_eq'])
        if len(tail['b_ub']):model.addConstr(tail['A_ub']@flat <= tail['b_ub'])
        for r in self.circles:
            if r['t'] < tail['k']:continue
            h=r['t']-tail['k']; b=self.original_baseline[r['t']]
            model.addConstr(((b[r['p']]+u[h,r['p']])/r['S'])**2+
                            ((b[r['q']]+u[h,r['q']])/r['S'])**2 <= 1.)
        fee = self._fee_expression(model, u, tail['k'], name)
        return u, fee

    def _fee_expression(self, model, delta, start, name):
        absolute = model.addMVar(delta.shape, lb=0, name=name+'_abs')
        model.addConstr(absolute >= delta); model.addConstr(absolute >= -delta)
        plus=self.fee_plus[start:start+delta.shape[0]]; minus=self.fee_minus[start:start+delta.shape[0]]
        return DT*((plus-minus)*delta+(plus+minus)*absolute).sum()/2

    def deterministic_service_tail(self, prefix_delta, *, prefix_audit=None):
        """Construct a physical backup, independently of an optimizer or H.

        The backward intervals encode the original terminal tasks. At every
        step, project the original baseline P onto powers that land in the next
        interval. Future Q is zero. This is a feasible service construction,
        not an activation-cost optimum and not a node-P/Q-band guarantee.
        """
        began=perf_counter(); prefix=self._array(prefix_delta); k=len(prefix)
        checked=self.audit_prefix(prefix) if prefix_audit is None else prefix_audit
        receipt=dict(method='baseline_projection_into_next_reachable_interval',
            completed_steps=k, optimizer_invoked=False, future_gross_Q_zero=True,
            original_terminal_targets_retained=True)
        if not checked['passed']:
            return None,dict(receipt,feasible=False,status='prefix_infeasible_or_unreachable',
                             seconds=perf_counter()-began)
        point=np.zeros((96-k,self.R))
        for q in self.q_index.values():point[:,q]=-self.original_baseline[k:,q]
        inverter_limits={row['p']:row['S'] for row in self.circles}
        failure=None
        for did,s in self.states.items():
            j=s['column']; value=checked['states'][did]['current']; reach=self.reachable[did]
            for h,t in enumerate(range(k,96)):
                base=s['a'][t]*value+s['g'][t]; c=s['c'][t]
                lower=self.original_baseline[t,j]+self.lower[t,j]
                upper=self.original_baseline[t,j]+self.upper[t,j]
                if j in inverter_limits:
                    lower=max(lower,-inverter_limits[j]); upper=min(upper,inverter_limits[j])
                if c:
                    first=(reach['lower'][t+1]-base)/c
                    second=(reach['upper'][t+1]-base)/c
                    lower=max(lower,min(first,second));upper=min(upper,max(first,second))
                    roundoff=128*np.finfo(float).eps*max(1.,abs(lower),abs(upper),abs(base/c))
                    if lower>upper+roundoff:
                        failure=dict(device=did,step=t,reason='empty_next_state_power_interval',
                                     lower_power_kw=float(lower),upper_power_kw=float(upper));break
                    # A collapsed equality can invert by floating-point roundoff.
                    power=float(np.clip(self.original_baseline[t,j],min(lower,upper),max(lower,upper)))
                else:
                    roundoff=128*np.finfo(float).eps*max(1.,abs(base))
                    if base<reach['lower'][t+1]-roundoff or base>reach['upper'][t+1]+roundoff:
                        failure=dict(device=did,step=t,reason='zero_gain_next_state_unreachable');break
                    power=float(np.clip(self.original_baseline[t,j],lower,upper))
                point[h,j]=power-self.original_baseline[t,j]
                value=base+c*power
            if failure is not None:break
        audit=None if failure is not None else self.audit_prefix(np.vstack((prefix,point)))
        feasible=bool(audit is not None and audit['passed'])
        return (point if feasible else None),dict(receipt,feasible=feasible,
            status='verified_physical_tail' if feasible else 'construction_failed',failure=failure,
            physical_audit=audit,band=self.band_audit(point,k) if feasible else None,
            cost_yuan=self.activation_fee(point,k) if feasible else None,seconds=perf_counter()-began)

    def service_plan(self, prefix_delta, *, env, seconds=30., warm_tail=None):
        began=perf_counter(); tail=self._tail(prefix_delta); k=tail['k']; prefix=tail['prefix']
        receipt=dict(firm_service=True, original_terminal_targets_retained=True, completed_steps=k,
                     state_before=tail['status'], budget_seconds=float(seconds), objective='remaining activation relative to original baseline',
                     reactive_service='Q is allowed by device contracts and explicitly audited; positive activation prices favor Q=0, not an assumed identity')
        if not np.isfinite(seconds) or seconds <= 0:raise ValueError('Positive service budget required')
        if not tail['status']['passed']:
            return None, dict(receipt,status='prefix_infeasible_or_unreachable',feasible=False,seconds=perf_counter()-began)
        zero=np.zeros((tail['H'],self.R)); zcheck=self.audit_prefix(np.vstack((prefix,zero)))
        if zcheck['passed']:
            return zero,dict(receipt,status='zero_original_delta_service',feasible=True,optimizer_invoked=False,
                cost_yuan=0.,physical_audit=zcheck,band=self.band_audit(zero,k),seconds=perf_counter()-began)
        initial,constructed=self.deterministic_service_tail(prefix,prefix_audit=tail['status'])
        receipt['deterministic_backup']=constructed
        source='deterministic_interval_projection' if initial is not None else None
        if warm_tail is not None:
            candidate=self._array(warm_tail,tail['H'])
            if self.audit_prefix(np.vstack((prefix,candidate)))['passed'] and (initial is None or self.activation_fee(candidate,k)<self.activation_fee(initial,k)):
                initial=candidate.copy();source='verified_previous_tail'
        point=initial;status=None;invoked=False;error=None
        budget_exhausted=perf_counter()-began>=seconds
        if not budget_exhausted:
            try:
                import gurobipy as gp
                with gp.Model('firm_remaining_service',env=env) as model:
                    for key,value in dict(OutputFlag=0,Threads=1,FeasibilityTol=1e-9,NumericFocus=1).items():model.setParam(key,value)
                    u,fee=self._attach_tail(model,tail)
                    model.setObjective(fee)
                    if initial is not None:u.Start=initial
                    remaining=seconds-(perf_counter()-began)
                    if remaining>0:
                        invoked=True;model.Params.TimeLimit=max(.001,remaining);model.optimize()
                        status=int(model.Status)
                        if model.SolCount:
                            candidate=np.asarray(u.X)
                            if self.audit_prefix(np.vstack((prefix,candidate)))['passed'] and (point is None or self.activation_fee(candidate,k)<self.activation_fee(point,k)):
                                point=candidate;source='convex_service_qcp'
                    else:budget_exhausted=True
            except Exception as exc:
                import traceback
                error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        return point,dict(receipt,status='feasible' if point is not None else 'unresolved_service',
            feasible=point is not None,native_status=status,optimizer_invoked=invoked,source=source,
            optimizer_exception=error,budget_exhausted_before_optimize=budget_exhausted,
            feasible_initial_retained=bool(point is not None and source!='convex_service_qcp'),
            cost_yuan=None if point is None else self.activation_fee(point,k),
            physical_audit=None if point is None else self.audit_prefix(np.vstack((prefix,point))),
            band=None if point is None else self.band_audit(point,k),seconds=perf_counter()-began)

    def remaining(self, prefix_delta, service_delta, gains, *, caution_fraction=1., paid_capacity=None):
        return RemainingCase(self,prefix_delta,service_delta,gains,caution_fraction,paid_capacity)


class RemainingCase:
    def __init__(self,fleet,prefix,service,gains,caution_fraction=1.,paid_capacity=None):
        self.fleet=fleet; self.tail=fleet._tail(prefix); self.k=self.tail['k']; self.H=self.tail['H']
        if self.H <= 0:raise ValueError('No remaining planning intervals')
        for name in ('N','R','P_R','Q_R','B','labels','resources','device_ids','p_index','q_index','base_kw','config'):
            setattr(self,name,getattr(fleet,name))
        self.dt=DT;self.repeat=1;self.shape=(self.H,self.R);self.dimension=self.H*self.R
        self.service_delta=fleet._array(service,self.H).copy()
        if not fleet.audit_prefix(np.vstack((self.tail['prefix'],self.service_delta)))['passed']:
            raise ValueError('Firm service tail must satisfy original physical tasks')
        gains=np.asarray(gains,float)
        if gains.shape==(self.P_R,):gains=np.array([gains[self.p_index[r['physical_device_id']]] for r in self.resources])
        if gains.shape!=(self.R,) or not np.isfinite(gains).all() or np.any((gains<.9)|(gains>1)):
            raise ValueError('Gains must lie in [.9,1]')
        for did,q in self.q_index.items():
            if gains[q]!=gains[self.p_index[did]]:raise ValueError('Device P/Q gains must agree')
        if not np.isfinite(caution_fraction) or not 0<=caution_fraction<=1:raise ValueError('Caution fraction must be in [0,1]')
        self.gains=gains.copy();self.caution_fraction=float(caution_fraction);self.paid_capacity=deepcopy(paid_capacity)
        self.clock=fleet.clock[self.k:];self.fee_plus=fleet.fee_plus[self.k:];self.fee_minus=fleet.fee_minus[self.k:]
        self.lower=(self.tail['lower']-self.service_delta)/gains
        self.upper=(self.tail['upper']-self.service_delta)/gains
        for eta in ETA_ENDPOINTS:
            self.lower[0]=np.maximum(self.lower[0],(self.tail['lower'][0]-self.service_delta[0])/eta)
            self.upper[0]=np.minimum(self.upper[0],(self.tail['upper'][0]-self.service_delta[0])/eta)
        self.lower[0]*=caution_fraction;self.upper[0]*=caution_fraction
        scale=np.tile(gains,self.H);origin=self.service_delta.ravel()
        self.A_eq=self.tail['A_eq']*scale;self.b_eq=self.tail['b_eq']-self.tail['A_eq']@origin
        self.A_ub=self.tail['A_ub']*scale;self.b_ub=self.tail['b_ub']-self.tail['A_ub']@origin
        # First-step reachability is physical, independent of the fitted H.
        extra_A=[];extra_b=[]
        for did,s in fleet.states.items():
            j=s['column']; current=self.tail['status']['states'][did]['current']
            constant=s['a'][self.k]*current+s['g'][self.k]+s['c'][self.k]*(fleet.original_baseline[self.k,j]+self.service_delta[0,j])
            reach=fleet.reachable[did]
            for eta in ETA_ENDPOINTS:
                row=np.zeros(self.dimension);row[j]=s['c'][self.k]*eta
                extra_A.extend((row,-row));extra_b.extend((reach['upper'][self.k+1]-constant,constant-reach['lower'][self.k+1]))
        self.band_C=None
        if fleet.band is not None:
            self.band_C=fleet.B[self.N:]-fleet.kappa[:,None]*fleet.B[:self.N]
            block=np.kron(np.eye(self.H),self.band_C);width=fleet.band[self.k:].ravel()*(1-BAND_BUFFER)
            extra_A.extend((block*scale,-block*scale));extra_b.extend((width-block@origin,width+block@origin))
        if extra_A:
            aa=np.vstack([np.atleast_2d(a) for a in extra_A]);bb=np.concatenate([np.atleast_1d(b) for b in extra_b])
            self.A_ub=np.vstack((self.A_ub,aa));self.b_ub=np.r_[self.b_ub,bb]

    def array(self,u):return self.fleet._array(u,self.H)
    def request_delta(self,u):return self.service_delta+self.array(u)
    def total_delta(self,u):return self.service_delta+self.array(u)*self.gains
    def fee(self,u):return self.fleet.activation_fee(self.request_delta(u),self.k)
    def nominal_activation_fee(self,u):return self.fleet.activation_fee(self.total_delta(u),self.k)
    def response_effect(self,H,base_kw):return np.asarray(H)@self.B@np.diag(self.gains)/base_kw
    def shift_pq_pu(self,u,base_kw):return (self.array(u)*self.gains)@self.B.T/base_kw
    def aggregate_total(self,u):
        value=self.total_delta(u)@self.B.T
        return value[:,:self.N],value[:,self.N:]

    def audit(self,u,tolerance=1e-7):
        u=self.array(u);nominal=self.total_delta(u)
        domain=self.fleet.audit_prefix(np.vstack((self.tail['prefix'],nominal)),tolerance)
        endpoints=[]
        for eta in ETA_ENDPOINTS:
            step=self.service_delta[0]+eta*u[0]
            check=self.fleet.audit_prefix(np.vstack((self.tail['prefix'],step)),tolerance)
            endpoints.append(dict(eta=eta,**check))
        band=self.fleet.band_audit(nominal,self.k)
        box=max(0.,float(np.max(self.lower-u)),float(np.max(u-self.upper)))
        return dict(passed=bool(domain['passed'] and all(r['passed'] for r in endpoints) and band['passed'] and box<=tolerance),
            physical_passed=domain['passed'],nominal_physical=domain,first_step_endpoints=endpoints,
            first_step_robust_reachable=all(r['passed'] for r in endpoints),band=band,
            max_extra_box_violation=box,request_activation_yuan=self.fee(u),
            nominal_activation_yuan=self.nominal_activation_fee(u),
            service_band=self.fleet.band_audit(self.service_delta,self.k),
            scope='Nominal future domain plus per-device first-step eta endpoints and firm future service reachability; no network safety certificate')

    def attach(self,model,name='rolling_extra'):
        u=model.addMVar(self.shape,lb=self.lower,ub=self.upper,name=name);flat=u.reshape(self.dimension)
        if len(self.b_eq):model.addConstr(self.A_eq@flat==self.b_eq)
        if len(self.b_ub):model.addConstr(self.A_ub@flat<=self.b_ub)
        for r in self.fleet.circles:
            if r['t']<self.k:continue
            h=r['t']-self.k;p,q,S=r['p'],r['q'],r['S']
            b=self.fleet.original_baseline[r['t']]+self.service_delta[h]
            model.addConstr(((b[p]+self.gains[p]*u[h,p])/S)**2+((b[q]+self.gains[q]*u[h,q])/S)**2<=1.)
            if h==0:
                for eta in ETA_ENDPOINTS:
                    model.addConstr(((b[p]+eta*u[0,p])/S)**2+((b[q]+eta*u[0,q])/S)**2<=1.)
        return u,self.fleet._fee_expression(model,self.service_delta+u,self.k,name)
