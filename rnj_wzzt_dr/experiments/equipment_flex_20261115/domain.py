"""Convex four-equipment domain in increments of an existing net P/Q day.

EVs never discharge. Battery charge/discharge hours are fixed before planning.
The physical domain is linear constraints plus convex inverter circles. The
topology-evidence constraints added by the existing optimizer remain nonconvex.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..customer_flexibility_20261113.model import CustomerCase, window
from .resources import content_hash


class EquipmentCase(CustomerCase):
    def __init__(self, spec, labels, forecast, base_kw):
        self.spec = deepcopy(spec)
        claimed = self.spec.pop('spec_content_sha256', None)
        if claimed is None or content_hash(self.spec) != claimed:
            raise ValueError('Equipment specification content hash mismatch')
        self.spec['spec_content_sha256'] = claimed
        self.labels = tuple(map(int, labels)); self.N = len(self.labels)
        if list(self.labels) != spec['expected_terminal_labels'] or len(set(self.labels)) != self.N:
            raise ValueError('Equipment and observation labels/order differ')
        self.forecast = np.asarray(forecast, float).copy()
        if self.forecast.shape != (96, 2*self.N) or not np.isfinite(self.forecast).all() or not np.isfinite(base_kw) or base_kw <= 0:
            raise ValueError('Expected finite 96-by-2N net forecast and positive common power base')
        self.base_kw = float(base_kw)
        self.H, self.repeat, self.dt = 24, 4, 1.
        time = spec['time']
        if (time.get('start_hour') != 8 or time.get('hours') != 24 or
                time.get('decision_hours') != 1 or time.get('measurement_minutes') != 15):
            raise ValueError('This adapter requires 08:00 start, 24 hourly controls and 96 quarter-hour samples')
        self.clock = (8.+np.arange(24)+.5) % 24.
        self.fraction = float(spec['participation_fraction'])
        self.devices = deepcopy(spec['devices'])
        self.device_ids = [row['id'] for row in self.devices]
        if not self.devices or len(set(self.device_ids)) != len(self.devices):
            raise ValueError('Nonempty unique physical devices required')
        self.P_R = len(self.devices)
        q_devices = [j for j, row in enumerate(self.devices) if row['kind'] in ('pv', 'battery')]
        self.Q_R = len(q_devices); self.R = self.P_R+self.Q_R
        self.shape, self.dimension = (24, self.R), 24*self.R
        self.resources = []
        self.p_index, self.q_index = {}, {}
        for axis, indices in [('P', range(self.P_R)), ('Q', q_devices)]:
            for device_i in indices:
                device = self.devices[device_i]
                index = len(self.resources)
                (self.p_index if axis == 'P' else self.q_index)[device['id']] = index
                self.resources.append(dict(id=device['id'] if axis == 'P' else device['id']+'_q',
                    physical_device_id=device['id'], device_id=device['id'],
                    old_resource_id=device.get('old_resource_id') if axis == 'P' else None,
                    kind=device['kind'] if axis == 'P' else 'reactive', device_kind=device['kind'],
                    node_label=int(device['node_label']), axis=axis,
                    action_unit='kW' if axis == 'P' else 'kvar',
                    contract_kw=device['power_kw'] if axis == 'P' else None,
                    capacity_kvar=device.get('reactive_cap_kvar') if axis == 'Q' else None))
                if axis == 'P':
                    self.resources[-1].update(
                        fee_positive_yuan_per_kwh=float(device['fee_positive_yuan_per_kwh']),
                        fee_negative_yuan_per_kwh=float(device['fee_negative_yuan_per_kwh']))
                else:
                    self.resources[-1].update(
                        fee_positive_yuan_per_kvarh=float(spec['reactive_fee_yuan_per_kvarh']),
                        fee_negative_yuan_per_kvarh=float(spec['reactive_fee_yuan_per_kvarh']))
        self.action_units = tuple(row['action_unit'] for row in self.resources)
        self.node = np.asarray([self.labels.index(row['node_label']) for row in self.resources], int)
        self.B = np.zeros((2*self.N, self.R))
        for j, node in enumerate(self.node):
            self.B[node if j < self.P_R else self.N+node, j] = 1.
        self.baseline = np.zeros(self.shape)  # Increment origin, never added to net load.
        self.lower = np.zeros(self.shape); self.upper = np.zeros(self.shape)
        self.fee_plus = np.zeros(self.shape); self.fee_minus = np.zeros(self.shape)
        self.device_p0 = np.column_stack([row['baseline_p_kw'] for row in self.devices])
        self.device_q0 = np.column_stack([row['baseline_q_kvar'] for row in self.devices])
        if self.device_p0.shape != (96, self.P_R) or self.device_q0.shape != self.device_p0.shape or not np.isfinite(self.device_p0).all() or not np.isfinite(self.device_q0).all():
            raise ValueError('Finite 96-point device baselines required')
        # This simple model declares all reference Q=0 and zero idle battery P.
        if np.any(self.device_q0 != 0.):
            raise ValueError('This implementation requires zero device baseline Q')
        eq, eq_rhs, inequalities, bounds = [], [], [], []
        self.state_rows, self.circles = {}, []
        mult = np.ones(24)
        price = spec['pricing']
        for a, b in price['peak_clock_windows']:
            mult[window(self.clock, a, b)] = price['clock_hour_multiplier']['morning_evening_peak']
        for a, b in price['night_clock_windows']:
            mult[window(self.clock, a, b)] = price['clock_hour_multiplier']['night']
        for j, device in enumerate(self.devices):
            power = float(device['power_kw']); kind = device['kind']; did = device['id']
            if not np.isfinite(power) or power <= 0:
                raise ValueError('Positive finite device rated active power required')
            p0 = self.device_p0[:, j].reshape(24, 4)
            self.fee_plus[:, j] = mult*float(device['fee_positive_yuan_per_kwh'])
            self.fee_minus[:, j] = mult*float(device['fee_negative_yuan_per_kwh'])
            if kind == 'pv':
                available = np.asarray(device['available_generation_kw'], float).reshape(24, 4)
                if not np.array_equal(p0, -available) or np.any(available < 0) or np.any(available > power):
                    raise ValueError('PV baseline must generate exactly the declared available gross power')
                self.upper[:, j] = available.min(axis=1)
                q_online = np.all(available > 0., axis=1)
            elif kind == 'ev':
                online = window(self.clock, *device['connected_clock_hours'])
                if np.any(p0 < 0) or np.any(p0 > power) or np.any(p0[~online] != 0):
                    raise ValueError('EV baseline must charge within power and connection limits')
                self.lower[:, j] = -p0.min(axis=1)
                self.upper[online, j] = power-p0[online].max(axis=1)
                row = np.zeros(self.shape); row[:, j] = self.dt*float(device['charging_efficiency'])
                eq.append(row.ravel()); eq_rhs.append(0.)
                expected = float(device['charging_efficiency'])*self.device_p0[:, j].sum()/4.
                if not np.isclose(expected, device['required_battery_energy_kwh'], atol=1e-10, rtol=1e-12):
                    raise ValueError('EV baseline does not satisfy its declared energy need')
            elif kind == 'thermal':
                if np.any(p0 < 0) or np.any(p0 > power):
                    raise ValueError('Thermal baseline exceeds heater power')
                self.lower[:, j] = -p0.min(axis=1)
                self.upper[:, j] = power-p0.max(axis=1)
                rho = float(device['hourly_temperature_retention'])
                capacity = float(device['heat_capacity_kwh_per_C'])
                cop = float(device['cop']); initial = float(device['initial_temperature_C'])
                if not 0 < rho <= 1 or capacity <= 0 or cop != 1.:
                    raise ValueError('Expected resistive heat, positive capacitance and retention in (0,1]')
                demand = np.asarray(device['heat_demand_kw'], float)
                expected = cop*self.device_p0[:, j]-(1.-rho)*capacity*(initial-device['ambient_temperature_C'])
                if demand.shape != (96,) or np.any(demand < 0) or not np.allclose(demand, expected, rtol=1e-12, atol=1e-12):
                    raise ValueError('Fixed thermal demand does not maintain the declared baseline temperature')
                rows = np.zeros((24, self.dimension))
                for h in range(24):
                    rows[h, np.arange(h+1)*self.R+j] = cop*self.dt*rho**np.arange(h, -1, -1)
                low = capacity*(float(device['minimum_temperature_C'])-initial)
                high = capacity*(float(device['maximum_temperature_C'])-initial)
                inequalities.extend(rows); bounds.extend(np.full(24, high))
                inequalities.extend(-rows); bounds.extend(np.full(24, -low))
                eq.append(rows[-1]); eq_rhs.append(0.)
                self.state_rows[did] = dict(rows=rows, kind=kind, initial=initial,
                    scale=capacity, lower=float(device['minimum_temperature_C']), upper=float(device['maximum_temperature_C']))
            elif kind == 'battery':
                if np.any(p0 != 0):
                    raise ValueError('This battery scenario declares an idle active-power baseline')
                charge = window(self.clock, *device['charging_clock_hours'])
                discharge = window(self.clock, *device['discharging_clock_hours'])
                if np.any(charge & discharge):
                    raise ValueError('Prelocked charge and discharge windows must be disjoint')
                self.upper[charge, j] = power; self.lower[discharge, j] = -power
                eta_c, eta_d = float(device['charging_efficiency']), float(device['discharging_efficiency'])
                energy = float(device['energy_kwh'])
                if not 0 < eta_c <= 1 or not 0 < eta_d <= 1 or energy <= 0:
                    raise ValueError('Positive energy and physical battery efficiencies required')
                coefficient = np.where(charge, eta_c, np.where(discharge, 1./eta_d, 0.))
                rows = np.zeros((24, self.dimension))
                for h in range(24):
                    rows[h, np.arange(h+1)*self.R+j] = self.dt*coefficient[:h+1]
                initial = float(device['initial_soc']); terminal = float(device['terminal_soc'])
                if initial != terminal or float(device.get('baseline_self_discharge_per_hour', 0.)) != 0.:
                    raise ValueError('Idle battery zero action requires equal initial/terminal SOC and declared zero self-discharge')
                low = energy*(float(device['minimum_soc'])-initial)
                high = energy*(float(device['maximum_soc'])-initial)
                inequalities.extend(rows); bounds.extend(np.full(24, high))
                inequalities.extend(-rows); bounds.extend(np.full(24, -low))
                eq.append(rows[-1]); eq_rhs.append(0.)
                self.state_rows[did] = dict(rows=rows, kind=kind, initial=initial,
                    scale=energy, lower=float(device['minimum_soc']), upper=float(device['maximum_soc']))
                q_online = np.zeros(24, bool)
                for a, b in device['reactive_clock_windows']:
                    q_online |= window(self.clock, a, b)
            else:
                raise ValueError('Unsupported equipment kind: '+kind)
            if kind in ('pv', 'battery'):
                q = self.q_index[did]; cap = float(device['reactive_cap_kvar']); S = float(device['inverter_kva'])
                if not np.isfinite(S+cap) or S <= 0 or cap < 0 or cap > S:
                    raise ValueError('Invalid inverter rating or reactive contract')
                self.lower[q_online, q] = -cap; self.upper[q_online, q] = cap
                self.fee_plus[:, q] = float(spec['reactive_fee_yuan_per_kvarh'])
                self.fee_minus[:, q] = self.fee_plus[:, q]
                for h in range(24):
                    for k in range(4):
                        self.circles.append(dict(device_id=did, hour=h, quarter=k,
                            p_index=j, q_index=q, p0=float(p0[h, k]), q0=0., rating_kva=S))
        self.A_eq = np.asarray(eq, float).reshape(-1, self.dimension); self.b_eq = np.asarray(eq_rhs, float)
        self.A_ub = np.asarray(inequalities, float).reshape(-1, self.dimension); self.b_ub = np.asarray(bounds, float)
        if np.any(self.lower > 0) or np.any(self.upper < 0) or np.any(self.lower > self.upper):
            raise ValueError('Declared device baselines must make zero action feasible')
        self.config = deepcopy(spec)
        self.config.update(resources=deepcopy(self.resources), B=self.B.tolist(),
            action_units=list(self.action_units), equipment_devices=deepcopy(self.devices),
            physical_domain='Convex linear power/time/energy/temperature constraints and four-point inverter circles; battery direction prelocked',
            electrical_limits='Network AC voltage/current checked separately; no true topology enters equipment configuration')
        if not self.audit(np.zeros(self.shape))['passed']:
            raise ValueError('Zero increments fail declared equipment physics')

    @property
    def qratio(self):
        raise AttributeError('P and Q are explicit device columns; use B/response_effect')

    def response_effect(self, difference, base_kw):
        difference = np.asarray(difference, float)
        if difference.shape[-1] != 2*self.N or not np.isfinite(difference).all() or base_kw <= 0:
            raise ValueError('Invalid response matrix or common base')
        return difference@self.B/base_kw

    def aggregate(self, delivery):
        delivery = np.asarray(delivery, float)
        if delivery.shape != (self.R, 96) or not np.isfinite(delivery).all():
            raise ValueError('Expected finite resource-by-96 delivery')
        pq = delivery.T@self.B.T
        return pq[:, :self.N], pq[:, self.N:]

    def device_states(self, u):
        u = self.array(u); x = u.ravel(); repeated = np.repeat(u, 4, axis=0)
        result = {}
        for j, device in enumerate(self.devices):
            did = device['id']; q = self.q_index.get(did)
            p = self.device_p0[:, j]+repeated[:, j]
            reactive = self.device_q0[:, j]+(repeated[:, q] if q is not None else 0.)
            row = dict(kind=device['kind'], node_label=device['node_label'],
                p_kw=p.tolist(), q_kvar=np.asarray(reactive).tolist(),
                baseline_p_kw=self.device_p0[:, j].tolist(), baseline_q_kvar=self.device_q0[:, j].tolist())
            if q is not None:
                row['apparent_power_kva'] = np.sqrt(p*p+reactive*reactive).tolist()
                row['inverter_rating_kva'] = float(device['inverter_kva'])
            if did in self.state_rows:
                info = self.state_rows[did]
                state = np.r_[info['initial'], info['initial']+info['rows']@x/info['scale']]
                row['soc' if device['kind'] == 'battery' else 'temperature_C'] = state.tolist()
                row['terminal_error'] = float(state[-1]-state[0])
                row['minimum_allowed'], row['maximum_allowed'] = info['lower'], info['upper']
            if device['kind'] == 'ev':
                row.update(delivered_battery_energy_kwh=float(p.sum()/4.*device['charging_efficiency']),
                    required_battery_energy_kwh=float(device['required_battery_energy_kwh']), v2g=False)
            result[did] = row
        return result

    def baseline_decomposition(self, forecast=None):
        value = self.forecast if forecast is None else np.asarray(forecast, float)
        if value.shape != (96, 2*self.N) or not np.isfinite(value).all():
            raise ValueError('Expected finite net per-unit forecast')
        p0, q0 = np.zeros((96, self.N)), np.zeros((96, self.N))
        for j, device in enumerate(self.devices):
            node = self.labels.index(device['node_label'])
            p0[:, node] += self.device_p0[:, j]; q0[:, node] += self.device_q0[:, j]
        residual_p = value[:, :self.N]*self.base_kw-p0
        residual_q = value[:, self.N:]*self.base_kw-q0
        old_negative = value[:, :self.N]*self.base_kw < 0
        return dict(scope='Planning forecast only; synthetic disaggregation, not measured component loads',
            baseline_added_to_net=False, device_p_kw=p0.tolist(), device_q_kvar=q0.tolist(),
            residual_background_p_kw=residual_p.tolist(), residual_background_q_kvar=residual_q.tolist(),
            residual_negative_p_cells=int(np.count_nonzero(residual_p < 0)),
            new_negative_background_cells=int(np.count_nonzero((residual_p < 0)&~old_negative)),
            total_cells=int(residual_p.size), minimum_background_p_kw=float(residual_p.min()),
            negative_background_energy_kwh=float(-np.minimum(residual_p, 0.).sum()/4.),
            interpretation='Residual includes uncontracted generation and load; negative values are reported, never clipped or relabelled pure consumption')

    def amounts(self, u):
        u = self.array(u)
        return dict(active_abs_kwh=float(np.abs(u[:, :self.P_R]).sum()),
                    reactive_abs_kvarh=float(np.abs(u[:, self.P_R:]).sum()))

    def fee_breakdown(self, u):
        u = self.array(u)
        cells = self.dt*(self.fee_plus*np.maximum(u, 0.)+self.fee_minus*np.maximum(-u, 0.))
        return dict(active_yuan=float(cells[:, :self.P_R].sum()), reactive_yuan=float(cells[:, self.P_R:].sum()),
                    reservation_yuan=0., total_yuan=float(cells.sum()))

    def quantity_summary(self, u):
        return dict(**self.amounts(u), **self.fee_breakdown(u))

    def audit(self, u, tolerance=1e-7):
        u = self.array(u); x = u.ravel()
        eq = float(np.max(np.abs(self.A_eq@x-self.b_eq), initial=0.))
        linear = max(0., float(np.max(self.lower-u)), float(np.max(u-self.upper)),
            float(np.max(self.A_ub@x-self.b_ub, initial=0.)))
        circle_values = [(row['p0']+u[row['hour'], row['p_index']])**2+
            (row['q0']+u[row['hour'], row['q_index']])**2-row['rating_kva']**2 for row in self.circles]
        circle = max(0., float(np.max(circle_values, initial=0.)))
        # Normalize kVA^2 violation by rating^2 so small aggregate devices are
        # not admitted with a large relative capacity error.
        relative = max((max(0., v)/row['rating_kva']**2 for v, row in zip(circle_values, self.circles)), default=0.)
        return dict(passed=bool(linear <= tolerance and eq <= tolerance and relative <= tolerance),
            max_linear_inequality_violation=linear, max_linear_equality_error=eq,
            max_inverter_squared_violation_kva2=circle, max_inverter_relative_violation=float(relative),
            service_fee_yuan=self.fee(u), amounts=self.amounts(u), fee_breakdown=self.fee_breakdown(u),
            baseline_added=False, device_states=self.device_states(u),
            feasibility_scope='Declared physical equipment powers, online windows, SOC/thermal/EV energy and inverter circles; no network AC certificate')

    def attach(self, model, name='equipment'):
        u = model.addMVar(self.dimension, lb=self.lower.ravel(), ub=self.upper.ravel(), name=name+'_delta')
        absolute = model.addMVar(self.dimension, lb=0., name=name+'_absolute')
        if len(self.b_eq): model.addConstr(self.A_eq@u == self.b_eq)
        if len(self.b_ub): model.addConstr(self.A_ub@u <= self.b_ub)
        model.addConstr(absolute >= u); model.addConstr(absolute >= -u)
        hourly = u.reshape(self.shape)
        for row in self.circles:
            h, p, q, rating = row['hour'], row['p_index'], row['q_index'], row['rating_kva']
            # Dimensionless encoding; baseline feasible, no hidden headroom box.
            model.addConstr(((hourly[h, p]+row['p0'])/rating)**2+
                ((hourly[h, q]+row['q0'])/rating)**2 <= 1., name=f"{name}_{row['device_id']}_{h}_{row['quarter']}_circle")
        fee = self.dt*((self.fee_plus-self.fee_minus).ravel()@u+
            (self.fee_plus+self.fee_minus).ravel()@absolute)/2.
        return hourly, fee

    def execute(self, u, actual_net_p_kw, multipliers):
        u = self.array(u); actual = np.asarray(actual_net_p_kw, float)
        if actual.shape != (96, self.N) or not np.isfinite(actual).all():
            raise ValueError('Finite actual net active power is required for execution bookkeeping')
        eta = np.asarray(multipliers, float)
        if eta.shape == (self.P_R,):
            eta = np.asarray([eta[self.device_ids.index(row['physical_device_id'])] for row in self.resources])
        if eta.shape != (self.R,) or not np.isfinite(eta).all() or np.any((eta < .9)|(eta > 1.)):
            raise ValueError('One eta in [.9,1] per device, or per-column identical device eta, required')
        for did, q in self.q_index.items():
            if eta[q] != eta[self.p_index[did]]:
                raise ValueError('The same physical inverter P/Q must share one whole-day response multiplier')
        if not self.audit(u)['passed']:
            raise ValueError('Requested action fails the physical equipment domain')
        delivered = u*eta[None, :]
        audit = self.audit(delivered)
        if not audit['passed']:
            raise ArithmeticError('Shared whole-device response violated a convex equipment constraint')
        return delivered, dict(resource_audit=audit, multipliers=eta.tolist(),
            multipliers_by_device={did:float(eta[self.p_index[did]]) for did in self.device_ids},
            requested_device_states=self.device_states(u), delivered_device_states=self.device_states(delivered),
            delivered_abs_kwh=self.amounts(delivered)['active_abs_kwh'],
            delivered_abs_kvarh=self.amounts(delivered)['reactive_abs_kvarh'],
            response_abs_error_kwh=float(np.abs(u[:, :self.P_R]-delivered[:, :self.P_R]).sum()),
            response_abs_error_kvarh=float(np.abs(u[:, self.P_R:]-delivered[:, self.P_R:]).sum()),
            pv_clipped_kwh=0., device_availability_clipping=False, baseline_added=False,
            availability_scope='Same declared deterministic gross device baseline/availability; no extra future weather model or net-export proxy',
            net_bookkeeping='actual_net_PQ + aggregate(delivered_increment), with no device baseline addition')

    def device_baseline_pq_kw(self):
        raise RuntimeError('Net observations already include devices; inspect baseline_decomposition, never add the baseline again')

    def linear_optimum(self, direction, *, least_fee=False):
        raise NotImplementedError('Device domain contains inverter circles; a linear relaxation is not a full-domain optimum. Use audited seed candidates or a convex QCP.')
