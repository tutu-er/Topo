"""Linear customer-device domain and adapter to the existing evidence metric.

u[h,r] is the change in device net consumption, in kW, relative to its
announced baseline. PV consumption is negative. The horizon is 08:00-08:00.
No true network, impedance, voltage, or candidate quality enters this module.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from ... import io


def window(clock, start, end):
    """Half-open clock window, including a crossing of midnight."""
    if start == end:
        raise ValueError("A clock window must have distinct endpoints")
    return ((clock >= start) & (clock < end) if start < end
            else (clock >= start) | (clock < end))


class CustomerCase:
    def __init__(self, config=None, labels=None):
        self.config = deepcopy(config if config is not None else
                               io.read(Path(__file__).with_name("case.json")))
        c = self.config
        self.labels = tuple(c["expected_terminal_labels"] if labels is None else labels)
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("Duplicate terminal labels")
        if set(self.labels) != set(c["expected_terminal_labels"]):
            raise ValueError("Terminal IDs must match the declared resource case")
        self.resources = c["resources"]
        if not self.resources or len({r['id'] for r in self.resources}) != len(self.resources):
            raise ValueError("Resource IDs must be nonempty and unique")
        self.R = len(self.resources)
        self.N = len(self.labels)
        self.H = c['time']['hours']
        self.dt = c['time']['decision_hours']
        self.repeat = int(60*self.dt/c['time']['measurement_minutes'])
        if self.H != 24 or self.dt != 1 or self.repeat != 4:
            raise ValueError("This case uses 24 hourly decisions and 96 measurements")
        self.clock = (c['time']['start_hour']+np.arange(self.H)+.5) % 24
        self.shape = (self.H, self.R)
        self.dimension = self.H*self.R
        self.node = np.array([self.labels.index(r['node_label']) for r in self.resources])
        self.qratio = np.array([r.get('q_over_p', np.tan(np.arccos(r.get('power_factor', 1.))))
                                for r in self.resources])
        self.baseline = np.zeros(self.shape)
        minimum = np.zeros(self.shape)
        maximum = np.zeros(self.shape)
        self.solar_available = np.zeros(self.shape)
        self.thermal = {}
        eq, rhs, ub, bound = [], [], [], []
        for j, r in enumerate(self.resources):
            kind = r['kind']
            if kind == 'pv':
                solar = c['solar']
                phase = (self.clock-solar['sunrise_hour'])/(solar['sunset_hour']-solar['sunrise_hour'])
                shape = np.where((phase > 0) & (phase < 1), np.maximum(0, np.sin(np.pi*phase)), 0.)
                available = np.minimum(r['dc_peak_kwp']*shape**solar['shape_exponent'],
                                       r['inverter_kva']/np.sqrt(1+self.qratio[j]**2))
                fraction = float(r['baseline_available_fraction'])
                if not 0 <= fraction <= 1:
                    raise ValueError("PV baseline must lie within available production")
                if fraction < 1 and not r.get('baseline_curtailment_provenance'):
                    raise ValueError("Pre-existing PV curtailment needs an explicit provenance")
                self.solar_available[:, j] = available
                self.baseline[:, j] = -fraction*available
                minimum[:, j] = -available
            elif kind == 'ev':
                connected = window(self.clock, *r['connected_clock_hours'])
                charging = window(self.clock, *r['baseline_charging_clock_hours'])
                if np.any(charging & ~connected) or not charging.any():
                    raise ValueError("Baseline EV charging must be inside its connection window")
                if not 0 < r['efficiency'] <= 1 or r['battery_energy_required_kwh'] <= 0:
                    raise ValueError("Positive EV energy and efficiency in (0,1] are required")
                p = r['battery_energy_required_kwh']/(r['efficiency']*self.dt*charging.sum())
                self.baseline[charging, j] = p
                maximum[connected, j] = r['power_kw']
                row = np.zeros(self.shape)
                row[:, j] = r['efficiency']*self.dt
                eq.append(row.ravel())
                rhs.append(0.)
            elif kind == 'thermal':
                t = c['thermal']
                p0 = np.full(self.H, t['baseline_power_kw']['default'])
                p0[window(self.clock, *t['morning_hours'])] = t['baseline_power_kw']['morning']
                p0[window(self.clock, *t['evening_hours'])] = t['baseline_power_kw']['evening']
                self.baseline[:, j] = p0
                maximum[:, j] = r['power_kw']
                rho = t['hourly_temperature_retention']
                capacity = t['water_litres']*t['water_heat_capacity_kJ_per_kg_K']/3600.
                if not 0 < rho <= 1 or capacity <= 0 or t['cop'] <= 0:
                    raise ValueError("Invalid thermal retention, capacitance or COP")
                # z is stored-heat deviation from the reference temperature.
                # z[h+1] = rho*z[h] + COP*u[h]*dt, z[0]=z[24]=0.
                rows = np.zeros((self.H, self.dimension))
                for h in range(self.H):
                    rows[h, np.arange(h+1)*self.R+j] = t['cop']*self.dt*rho**np.arange(h, -1, -1)
                low = capacity*(t['minimum_temperature_C']-t['initial_temperature_C'])
                high = capacity*(t['maximum_temperature_C']-t['initial_temperature_C'])
                ub.extend(rows)
                bound.extend(np.full(self.H, high))
                ub.extend(-rows)
                bound.extend(np.full(self.H, -low))
                eq.append(rows[-1])
                rhs.append(0.)
                heat = t['cop']*p0-capacity*(1-rho)*(t['initial_temperature_C']-t['ambient_temperature_C'])/self.dt
                if np.any(heat < 0):
                    raise ValueError("Reference thermal demand would be negative")
                self.thermal[j] = dict(capacity_kwh_per_C=capacity, rows=rows,
                                       heat_demand_kw=heat, lower_heat_kwh=low, upper_heat_kwh=high)
            else:
                raise ValueError("Unknown resource kind: "+kind)
        self.lower, self.upper = minimum-self.baseline, maximum-self.baseline
        if not all(np.isfinite(a).all() for a in (self.baseline, self.lower, self.upper, self.qratio)):
            raise ValueError("Finite device parameters required")
        if np.any(self.lower > self.upper) or np.any(self.lower > 0) or np.any(self.upper < 0):
            raise ValueError("The announced baseline must be device feasible")
        self.A_eq = np.array(eq).reshape(-1, self.dimension)
        self.b_eq = np.array(rhs)
        self.A_ub = np.array(ub).reshape(-1, self.dimension)
        self.b_ub = np.array(bound)
        price = c['pricing']
        mult = np.full(self.H, price['clock_hour_multiplier']['default'])
        for w in price['peak_clock_windows']:
            mult[window(self.clock, *w)] = price['clock_hour_multiplier']['morning_evening_peak']
        for w in price['night_clock_windows']:
            mult[window(self.clock, *w)] = price['clock_hour_multiplier']['night']
        self.fee_plus = mult[:, None]*np.array([r['fee_positive_yuan_per_kwh'] for r in self.resources])
        self.fee_minus = mult[:, None]*np.array([r['fee_negative_yuan_per_kwh'] for r in self.resources])
        if np.any(self.fee_plus <= 0) or np.any(self.fee_minus <= 0):
            raise ValueError("Strictly positive service quotes are required")

    def array(self, u):
        u = np.asarray(u, float)
        if u.shape != self.shape or not np.isfinite(u).all():
            raise ValueError("Expected finite hourly-resource changes with shape "+str(self.shape))
        return u

    def delivery(self, u):
        """Resource-by-96-slot changes for the existing evidence interface."""
        return np.repeat(self.array(u), self.repeat, axis=0).T

    def aggregate(self, delivery):
        delivery = np.asarray(delivery, float)
        if delivery.shape != (self.R, self.H*self.repeat) or not np.isfinite(delivery).all():
            raise ValueError("Expected resource-by-96-slot finite kW delivery")
        p = np.zeros((self.H*self.repeat, self.N))
        q = np.zeros_like(p)
        for j, node in enumerate(self.node):
            p[:, node] += delivery[j]
            q[:, node] += self.qratio[j]*delivery[j]
        return p, q

    def shift_pq_pu(self, u, base_kw):
        if not np.isfinite(base_kw) or base_kw <= 0:
            raise ValueError("A positive nominal system base in kW is required")
        p, q = self.aggregate(self.delivery(u))
        return np.column_stack((p, q))/base_kw

    def device_baseline_pq_kw(self):
        return self.aggregate(np.repeat(self.baseline, self.repeat, axis=0).T)

    def temperatures(self, u):
        u = self.array(u)
        t = self.config['thermal']
        answer = {}
        for j, info in self.thermal.items():
            z = 0.
            values = [t['initial_temperature_C']]
            for h in range(self.H):
                z = t['hourly_temperature_retention']*z+t['cop']*u[h, j]*self.dt
                values.append(t['initial_temperature_C']+z/info['capacity_kwh_per_C'])
            answer[self.resources[j]['id']] = np.array(values)
        return answer

    def fee(self, u):
        u = self.array(u)
        return float(self.dt*np.sum(self.fee_plus*np.maximum(u, 0)+self.fee_minus*np.maximum(-u, 0)))

    def audit(self, u, tolerance=1e-7):
        u = self.array(u)
        x = u.ravel()
        p = self.baseline+u
        eq = float(np.max(np.abs(self.A_eq@x-self.b_eq), initial=0.))
        violation = max(0., float(np.max(self.lower-u)), float(np.max(u-self.upper)),
                        float(np.max(self.A_ub@x-self.b_ub, initial=0.)))
        ev = {r['id']: float(r['efficiency']*self.dt*p[:, j].sum())
              for j, r in enumerate(self.resources) if r['kind'] == 'ev'}
        temp = self.temperatures(u)
        return dict(passed=violation <= tolerance and eq <= tolerance,
                    max_linear_inequality_violation=violation, max_linear_equality_error=eq,
                    ev_delivered_battery_kwh=ev,
                    thermal_temperature_ranges_C={k: [float(v.min()), float(v.max())] for k, v in temp.items()},
                    thermal_terminal_error_C=max((abs(v[-1]-v[0]) for v in temp.values()), default=0.),
                    service_fee_yuan=self.fee(u))

    def linear_optimum(self, direction, *, least_fee=False):
        """Minimize a linear functional over the full coupled device domain.

        The optional second LP preserves its optimum to 1e-8 and minimizes
        service compensation. This is not a topology-evidence optimizer.
        """
        direction = self.array(direction).ravel()
        bounds = list(zip(self.lower.ravel(), self.upper.ravel()))
        result = linprog(direction, A_ub=self.A_ub, b_ub=self.b_ub,
                         A_eq=self.A_eq, b_eq=self.b_eq, bounds=bounds, method='highs')
        if not result.success:
            raise RuntimeError("Resource LP failed: "+result.message)
        if least_fee:
            n = self.dimension
            eye = np.eye(n)
            A = np.block([[self.A_ub, np.zeros_like(self.A_ub)],
                          [eye, -eye], [-eye, -eye],
                          [direction[None], np.zeros((1, n))]])
            b = np.r_[self.b_ub, np.zeros(2*n), result.fun+1e-8]
            costs = self.dt*np.r_[(self.fee_plus-self.fee_minus).ravel()/2,
                                  (self.fee_plus+self.fee_minus).ravel()/2]
            second = linprog(costs, A_ub=A, b_ub=b, A_eq=np.c_[self.A_eq, np.zeros_like(self.A_eq)],
                             b_eq=self.b_eq, bounds=bounds+[(0., None)]*n, method='highs')
            if not second.success:
                raise RuntimeError("Least-fee resource LP failed: "+second.message)
            u = second.x[:n].reshape(self.shape)
        else:
            u = result.x.reshape(self.shape)
        receipt = self.audit(u)
        if not receipt['passed']:
            raise ArithmeticError("Resource LP fails the original device audit")
        return u, dict(linear_objective=float(direction@u.ravel()), resource_audit=receipt,
                       first_stage_optimal_value=float(result.fun), original_domain_global_LP_optimum=True)

    def attach(self, model, name='customer'):
        """Add only linear constraints and the exact piecewise-linear fee."""
        u = model.addMVar(self.dimension, lb=self.lower.ravel(), ub=self.upper.ravel(), name=name+'_delta_kw')
        absolute = model.addMVar(self.dimension, lb=0., name=name+'_absolute_kw')
        model.addConstr(self.A_eq@u == self.b_eq)
        model.addConstr(self.A_ub@u <= self.b_ub)
        model.addConstr(absolute >= u)
        model.addConstr(absolute >= -u)
        fee = self.dt*((self.fee_plus-self.fee_minus).ravel()@u+
                       (self.fee_plus+self.fee_minus).ravel()@absolute)/2
        return u.reshape(self.shape), fee

    def evidence(self, H, target, competitors, forecast, sigma2, base_kw, gamma):
        """Reuse the existing frozen-H expected-log-odds calculation."""
        from ..expected_logodds_admm_20261113.evidence import build
        if not np.isfinite(base_kw) or base_kw <= 0 or not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError("Positive finite power base and residual variance required")
        H = np.asarray(H, float)
        forecast = np.asarray(forecast, float)
        if H.ndim != 3 or H.shape[1:] != (self.N, 2*self.N) or not np.isfinite(H).all():
            raise ValueError("Frozen H must use the same terminal order as this case")
        if forecast.shape != (self.H*self.repeat, 2*self.N) or not np.isfinite(forecast).all():
            raise ValueError("Expected the full no-DFS net P/Q forecast in per unit")
        return build(H, target, competitors, forecast, sigma2, self, base_kw, gamma)

    def attach_evidence(self, model, u, evidence):
        """Exact hourly QCQP form of the existing 96-observation gain."""
        if evidence.base.shape[1:] != (self.H*self.repeat, self.N) or evidence.effect.shape[2] != self.R:
            raise ValueError("Evidence dimensions differ from this resource case")
        for j in range(len(evidence.base)):
            effect = evidence.effect[j]
            Q = self.repeat*(effect.T@effect)*evidence.scales[j]**2
            a = evidence.base[j].reshape(self.H, self.repeat, self.N).sum(1)@effect*evidence.scales[j]**2
            model.addConstr(sum(u[h]@Q@u[h] for h in range(self.H))+2*(a*u).sum()
                            >= evidence.gain_required+evidence.buffer)

    def grid_energy_value(self, net_p_kw):
        """Separate loss-free terminal net-energy value, not a service fee."""
        net_p_kw = np.asarray(net_p_kw, float)
        if net_p_kw.shape != (self.H*self.repeat, self.N) or not np.isfinite(net_p_kw).all():
            raise ValueError("Expected 96-by-terminal finite net power in kW")
        price = self.config['pricing']
        buy = np.full(self.H, price['grid_buy_yuan_per_kwh']['day'])
        buy[window(self.clock, 0, 7)] = price['grid_buy_yuan_per_kwh']['night']
        buy[window(self.clock, 17, 22)] = price['grid_buy_yuan_per_kwh']['evening']
        buy = np.repeat(buy, self.repeat)
        total = net_p_kw.sum(1)
        return float(self.dt/self.repeat*np.sum(buy*np.maximum(total, 0)-
                     price['grid_sell_to_buy_ratio']*buy*np.maximum(-total, 0)))
