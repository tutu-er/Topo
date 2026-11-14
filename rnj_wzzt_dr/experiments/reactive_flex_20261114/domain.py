"""Independent bounded reactive services appended to the unchanged P contracts.

Each P column is measured in kW; each Q column is measured in kvar. Positive
values increase net consumption/absorption. Reactive injection is negative.
The resource map B converts these native units to [P, Q] at measured terminals.
No q/p ratio represents a pure-Q resource. No additional device baseline is
added to the already measured net profiles.
"""
from copy import deepcopy

import numpy as np

from ..customer_flexibility_20261113.model import CustomerCase, window
from ..expected_logodds_admm_20261113.evidence import Evidence
from ..measured_pq_flex_20261114.domain import IncrementCase


class PQIncrementCase(CustomerCase):
    """Old P domain times independent bidirectional Q boxes.

    q_resources is an explicit list with id, node_label, capacity_kvar,
    fee_positive_yuan_per_kvarh and fee_negative_yuan_per_kvarh. Optional
    available_clock_hours=[start,end] restricts a half-open daily window;
    omitted or the full list 0,...,23 means all day. Optional hourly_fee_multiplier has 24 values in
    the decision order 08:00--08:00. Reservation fees are explicitly zero.

    Q capacity is a declared controllable compensator or separately reserved
    inverter capability; the contract does not infer spare kVA from registered
    PV kW. Independent Q boxes require that reserved capability to be available
    over the declared window. Converter losses and hidden device states are not
    added. Actual network effects are evaluated by the unchanged AC simulator.
    """

    def __init__(self, context, labels, forecast, base_kw, *, q_resources, fraction=.1):
        self.p_case = IncrementCase(context, labels, forecast, base_kw, fraction)
        p = self.p_case
        self.labels, self.N = p.labels, p.N
        if len(set(self.labels)) != self.N:
            raise ValueError('Measured terminal labels must be unique')
        self.H, self.repeat, self.dt = p.H, p.repeat, p.dt
        self.clock, self.fraction = p.clock.copy(), p.fraction
        self.P_R = p.R
        self.q_resources = self._reactive_catalog(q_resources)
        self.Q_R = len(self.q_resources)
        self.resources = deepcopy(p.resources)+self.q_resources
        self.R = len(self.resources)
        if len({r['id'] for r in self.resources}) != self.R:
            raise ValueError('P/Q resource IDs must be unique')
        self.shape, self.dimension = (self.H, self.R), self.H*self.R
        self.node = np.array([self.labels.index(int(r['node_label'])) for r in self.resources])
        self.action_units = tuple(['kW']*self.P_R+['kvar']*self.Q_R)
        self.B = np.zeros((2*self.N, self.R))
        for j, node in enumerate(self.node):
            self.B[node if j < self.P_R else self.N+node, j] = 1.
        self.baseline = np.zeros(self.shape)
        self.lower, self.upper = np.zeros(self.shape), np.zeros(self.shape)
        self.lower[:, :self.P_R], self.upper[:, :self.P_R] = p.lower, p.upper
        self.fee_plus, self.fee_minus = np.zeros(self.shape), np.zeros(self.shape)
        self.fee_plus[:, :self.P_R], self.fee_minus[:, :self.P_R] = p.fee_plus, p.fee_minus
        for q, resource in enumerate(self.q_resources, start=self.P_R):
            hours = resource.get('available_clock_hours')
            available = np.ones(self.H, bool) if hours is None else window(self.clock, *hours)
            self.lower[available, q] = -resource['capacity_kvar']
            self.upper[available, q] = resource['capacity_kvar']
            mult = np.asarray(resource.get('hourly_fee_multiplier', np.ones(self.H)), float)
            self.fee_plus[:, q] = mult*resource['fee_positive_yuan_per_kvarh']
            self.fee_minus[:, q] = mult*resource['fee_negative_yuan_per_kvarh']
        self.A_eq, self.A_ub = self._expand(p.A_eq), self._expand(p.A_ub)
        self.b_eq, self.b_ub = p.b_eq.copy(), p.b_ub.copy()
        self.energy_rows = {j: dict(info, rows=self._expand(info['rows']))
            for j, info in p.energy_rows.items()}
        self.config = deepcopy(p.config)
        self.config.update(id='registered_P_plus_reserved_Q_increment_contract',
            resources=deepcopy(self.resources), action_units=list(self.action_units),
            reactive='Independent pure-Q bidirectional boxes at measured terminals; positive absorption, negative injection',
            reactive_capacity_source='Explicit declared Q reserve; no inference from registered PV kW or true network',
            reactive_reservation_fee_yuan_per_kvar_day=0.,
            mixed_unit_reporting='P activation in kWh and Q activation in kvarh are reported separately',
            response='P columns preserve original order and execution; append Q response multipliers without changing the P stream')

    @property
    def qratio(self):
        raise AttributeError('Pure-Q columns have no q/p ratio; use response_effect(D, base_kw) or B')

    def _reactive_catalog(self, rows):
        result = []
        for original in rows:
            row = deepcopy(original)
            if not isinstance(row.get('id'), str) or not row['id']:
                raise ValueError('Reactive resource needs a nonempty string ID')
            if row.get('node_label') not in self.labels:
                raise ValueError('Reactive resource must be at an explicitly measured terminal')
            for key in ('capacity_kvar', 'fee_positive_yuan_per_kvarh', 'fee_negative_yuan_per_kvarh'):
                value = float(row[key])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('Positive finite reactive capacity and activation prices required')
                row[key] = value
            if row.get('reservation_fee_yuan_per_kvar_day', 0.) != 0.:
                raise ValueError('This activation-only domain declares zero Q reservation fee')
            hours = row.get('available_clock_hours')
            if hours is not None and list(hours) == list(range(24)):
                hours = row['available_clock_hours'] = [0, 24]
            if hours is not None and (len(hours) != 2 or not np.isfinite(hours).all()
                    or not all(0 <= hour <= 24 for hour in hours) or hours[0] == hours[1]):
                raise ValueError('Reactive window requires two distinct hours in [0,24]')
            mult = np.asarray(row.get('hourly_fee_multiplier', np.ones(self.H)), float)
            if mult.shape != (self.H,) or not np.isfinite(mult).all() or np.any(mult <= 0):
                raise ValueError('Reactive hourly fee multipliers must be 24 positive finite values')
            row.update(kind='reactive', action_unit='kvar',
                reservation_fee_yuan_per_kvar_day=0.,
                capacity_interpretation='Declared independently available Q contract, not measured PV inverter headroom')
            result.append(row)
        return result

    def _expand(self, rows):
        rows = np.asarray(rows, float)
        extended = np.zeros((len(rows), self.H, self.R))
        extended[:, :, :self.P_R] = rows.reshape(len(rows), self.H, self.P_R)
        return extended.reshape(len(rows), self.dimension)

    def response_effect(self, difference, base_kw):
        """Map frozen voltage-response differences to native P/Q action units."""
        difference = np.asarray(difference, float)
        if (difference.ndim < 2 or difference.shape[-1] != 2*self.N
                or not np.isfinite(difference).all() or not np.isfinite(base_kw) or base_kw <= 0):
            raise ValueError('Expected finite response matrix and positive common power base')
        return difference@self.B/base_kw

    def aggregate(self, delivery):
        delivery = np.asarray(delivery, float)
        if delivery.shape != (self.R, self.H*self.repeat) or not np.isfinite(delivery).all():
            raise ValueError('Expected finite resource-by-time delivery in each column native unit')
        pq = delivery.T@self.B.T
        return pq[:, :self.N], pq[:, self.N:]

    def evidence(self, H, target, competitors, forecast, sigma2, base_kw, gamma):
        H, forecast = np.asarray(H, float), np.asarray(forecast, float)
        if (H.ndim != 3 or H.shape[1:] != (self.N, 2*self.N) or not np.isfinite(H).all()
                or forecast.shape != (self.H*self.repeat, 2*self.N) or not np.isfinite(forecast).all()
                or not np.isfinite(sigma2) or sigma2 <= 0):
            raise ValueError('Frozen prediction dimensions or residual scale invalid')
        difference = H[target][None]-H[np.asarray(competitors, int)]
        normal = np.sqrt(2*sigma2)
        base = np.einsum('tc,knc->ktn', forecast, difference)/normal
        effect = self.response_effect(difference, base_kw)/normal
        scales = np.maximum(1., np.linalg.norm(base, axis=(1, 2)))
        return Evidence(base/scales[:, None, None], effect/scales[:, None, None], scales, gamma)

    def attach(self, model, name='pq_service'):
        """Linear resource constraints, exact piecewise-linear mixed-unit fee."""
        u = model.addMVar(self.dimension, lb=self.lower.ravel(), ub=self.upper.ravel(), name=name+'_delta_native_units')
        absolute = model.addMVar(self.dimension, lb=0., name=name+'_absolute_native_units')
        model.addConstr(self.A_eq@u == self.b_eq)
        model.addConstr(self.A_ub@u <= self.b_ub)
        model.addConstr(absolute >= u)
        model.addConstr(absolute >= -u)
        fee = self.dt*((self.fee_plus-self.fee_minus).ravel()@u+
            (self.fee_plus+self.fee_minus).ravel()@absolute)/2
        return u.reshape(self.shape), fee

    def amounts(self, u):
        u = self.array(u)
        return dict(active_abs_kwh=float(self.dt*np.abs(u[:, :self.P_R]).sum()),
            reactive_abs_kvarh=float(self.dt*np.abs(u[:, self.P_R:]).sum()))

    def fee_breakdown(self, u):
        u = self.array(u)
        cost = self.dt*(self.fee_plus*np.maximum(u, 0)+self.fee_minus*np.maximum(-u, 0))
        return dict(active_yuan=float(cost[:, :self.P_R].sum()),
            reactive_yuan=float(cost[:, self.P_R:].sum()), reservation_yuan=0.,
            total_yuan=float(cost.sum()))

    def quantity_summary(self, u):
        """Separate P/Q quantities and service fees; never add kWh to kvarh."""
        return dict(**self.amounts(u), **self.fee_breakdown(u))

    def audit(self, u, tolerance=1e-7):
        u = self.array(u)
        result = self.p_case.audit(u[:, :self.P_R], tolerance)
        x = u.ravel()
        eq = float(np.max(np.abs(self.A_eq@x-self.b_eq), initial=0.))
        violation = max(0., float(np.max(self.lower-u)), float(np.max(u-self.upper)),
            float(np.max(self.A_ub@x-self.b_ub, initial=0.)))
        result.update(passed=bool(violation <= tolerance and eq <= tolerance),
            max_linear_inequality_violation=violation, max_linear_equality_error=eq,
            service_fee_yuan=self.fee(u), amounts=self.amounts(u), fee_breakdown=self.fee_breakdown(u),
            reactive_bounds_kvar={r['id']:dict(capacity=r['capacity_kvar'],
                maximum_absolute_action=float(np.max(np.abs(u[:, self.P_R+j]))))
                for j, r in enumerate(self.q_resources)},
            feasibility_scope='Original P contracts and declared independent Q reserve; AC voltage/thermal feasibility evaluated separately')
        return result

    def execute(self, u, actual_net_p_kw, multipliers):
        """Retain original P response/clipping; independently scale bounded Q."""
        u, eta = self.array(u), np.asarray(multipliers, float)
        if eta.shape != (self.R,) or not np.isfinite(eta).all() or np.any((eta < 0) | (eta > 1)):
            raise ValueError('One finite multiplier in [0,1] per original-P then appended-Q resource required')
        if not self.audit(u)['passed']:
            raise ValueError('Requested action violates declared P/Q resource contracts')
        p_delivered, receipt = self.p_case.execute(u[:, :self.P_R], actual_net_p_kw, eta[:self.P_R])
        q_delivered = u[:, self.P_R:]*eta[None, self.P_R:]
        delivered = np.column_stack((p_delivered, q_delivered))
        audit = self.audit(delivered)
        if not audit['passed']:
            raise ArithmeticError('Delivery violates declared P/Q contracts')
        receipt.update(resource_audit=audit, multipliers=eta,
            delivered_abs_kwh=self.amounts(delivered)['active_abs_kwh'],
            delivered_abs_kvarh=self.amounts(delivered)['reactive_abs_kvarh'],
            response_abs_error_kvarh=float(self.dt*np.abs(u[:, self.P_R:]-q_delivered).sum()),
            mixed_quantity_sum_prohibited=True)
        return delivered, receipt

    def device_baseline_pq_kw(self):
        raise RuntimeError('Measured net profiles already include devices; no P/Q baseline may be added')
