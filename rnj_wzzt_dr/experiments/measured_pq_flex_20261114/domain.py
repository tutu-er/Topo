"""Sparse incremental service contracts on an already measured net baseline.

Registered capacities are measured-source metadata; participation, availability,
energy buffers and yuan quotes below are declared service assumptions. No device
baseline is added to net P/Q, and no R/X or true topology is read here.
"""
from copy import deepcopy
import numpy as np

from ..customer_flexibility_20261113.model import CustomerCase, window


PRICING = dict(clock_hour_multiplier=dict(default=1., morning_evening_peak=1.25, night=.8),
    peak_clock_windows=[[7,10],[17,22]], night_clock_windows=[[0,7]],
    reservation_fee_yuan_per_kw_day=0.,
    grid_buy_yuan_per_kwh=dict(night=.3, day=.5, evening=.8), grid_sell_to_buy_ratio=.9)


def catalog(context, fraction=.1):
    """Select capacity leaders, with deterministic label ties, before outcomes."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Participation fraction must be in (0,1]')
    registrations = {int(row['feeder']):row for row in context['resource_metadata_at_history_start']}
    rows = []
    for kind, field, count, price in [('pv','PV_systems_kW',3,.30),
                                     ('ev','EV_chargers_kW',2,.10),
                                     ('thermal','hot_water_tanks_kW',2,.15)]:
        available = []
        for node in context['node_mapping']:
            registered = float(registrations[int(node['feeder_id'])].get(field,0.) or 0.)
            scaled = registered*float(node['scale'])
            if not np.isfinite(scaled) or scaled < 0:
                raise ValueError('Invalid scaled registered capacity')
            if scaled > 0:
                available.append(dict(kind=kind, node_label=int(node['label']),
                    feeder_id=int(node['feeder_id']), metadata_field=field,
                    registration_date=registrations[int(node['feeder_id'])]['date'],
                    registered_kw=registered, source_to_network_scale=float(node['scale']),
                    scaled_registered_kw=scaled, contract_kw=fraction*scaled,
                    id=kind+'_'+str(node['label']), q_over_p=0.,
                    fee_positive_yuan_per_kwh=price, fee_negative_yuan_per_kwh=price))
        rows.extend(sorted(available, key=lambda r:(-r['scaled_registered_kw'],r['node_label']))[:count])
    if not rows:
        raise ValueError('No registered resources in this input')
    return rows


class IncrementCase(CustomerCase):
    """CustomerCase-compatible linear delta domain; baseline is never injected.

    EV: 18:00–07:00, net daily delta energy zero, 2-hour cumulative buffer.
    Resistance hot water: 24h, e[h+1]=.98e[h]+u[h], one-hour buffer, final e=0.
    PV: curtailment only 10:00–14:00; cap also uses forecast net export.
    """
    def __init__(self, context, labels, forecast, base_kw, fraction=.1):
        self.labels = tuple(map(int,labels))
        self.N = len(self.labels)
        if tuple(int(row['label']) for row in context['node_mapping']) != self.labels:
            raise ValueError('Resource metadata and observation labels/order differ')
        forecast = np.asarray(forecast,float)
        if forecast.shape != (96,2*self.N) or not np.isfinite(forecast).all() or base_kw <= 0:
            raise ValueError('Expected finite 96-row per-unit net P/Q forecast')
        self.resources = catalog(context,fraction)
        self.R, self.H, self.repeat, self.dt = len(self.resources),24,4,1.
        self.shape, self.dimension = (self.H,self.R),self.H*self.R
        self.clock = (8+np.arange(self.H)+.5)%24
        self.node = np.array([self.labels.index(row['node_label']) for row in self.resources])
        self.qratio = np.zeros(self.R)
        self.fraction = float(fraction)
        self.baseline = np.zeros(self.shape)  # algebraic delta origin, not a device load
        self.lower = np.zeros(self.shape);self.upper = np.zeros(self.shape)
        self.energy_rows = {};eq=[];rhs=[];ub=[];bounds=[]
        forecast_export = np.maximum(0.,-forecast[:,:self.N]*base_kw).reshape(24,4,self.N).min(1)
        for j,row in enumerate(self.resources):
            cap=row['contract_kw'];kind=row['kind']
            if kind=='pv':
                self.upper[:,j]=np.where(window(self.clock,10,14),np.minimum(cap,forecast_export[:,self.node[j]]),0.)
                continue
            available=window(self.clock,18,7) if kind=='ev' else np.ones(24,dtype=bool)
            self.lower[available,j]=-cap;self.upper[available,j]=cap
            rho=1. if kind=='ev' else .98
            buffer=(2. if kind=='ev' else 1.)*cap
            rows=np.zeros((24,self.dimension))
            for h in range(24):
                rows[h,np.arange(h+1)*self.R+j]=rho**np.arange(h,-1,-1)*self.dt
            self.energy_rows[j]=dict(rows=rows,rho=rho,buffer_kwh=buffer)
            ub.extend(rows);bounds.extend(np.full(24,buffer))
            ub.extend(-rows);bounds.extend(np.full(24,buffer))
            eq.append(rows[-1]);rhs.append(0.)
        self.A_eq=np.asarray(eq,float).reshape(-1,self.dimension);self.b_eq=np.asarray(rhs,float)
        self.A_ub=np.asarray(ub,float).reshape(-1,self.dimension);self.b_ub=np.asarray(bounds,float)
        price=deepcopy(PRICING)
        mult=np.ones(24)
        for a,b in price['peak_clock_windows']:mult[window(self.clock,a,b)]=1.25
        for a,b in price['night_clock_windows']:mult[window(self.clock,a,b)]=.8
        self.fee_plus=mult[:,None]*np.array([r['fee_positive_yuan_per_kwh'] for r in self.resources])
        self.fee_minus=mult[:,None]*np.array([r['fee_negative_yuan_per_kwh'] for r in self.resources])
        self.config=dict(id='registered_increment_contract',expected_terminal_labels=list(self.labels),
            resources=self.resources,pricing=price,participation_fraction=fraction,
            placement='Top3 PV, top2 EV, top2 hot-water scaled registered capacities; label tie-break; no candidate/true-network inputs',
            baseline='Already present measured net P/Q. No extra device baseline is added.',
            availability='Synthetic contracted delta envelopes; registration alone does not measure available flexibility',
            PV='Synthetic PV-availability contract, additionally capped by net export as a proxy: 10–14, min quarter-hour forecast export for quoting and actual net export for clipping. Net export may also contain batteries/CHP; it is not measured PV availability.',
            EV='18–07, signed shift, zero daily energy change, +/-2 contract-kW-hours cumulative buffer',
            thermal='Resistance hot-water aggregation, pure-P, rho=.98, +/-1 contract-kW-hour buffer, terminal restoration',
            response='Independent whole-day resource multiplier uniform[.9,1]; synthetic PV delivery additionally clipped by actual net-export proxy',
            costs='Illustrative activation compensation; absolute EV/thermal moves including rebound; capacity reservation not priced')

    def audit(self,u,tolerance=1e-7):
        u=self.array(u);x=u.ravel()
        eq=float(np.max(np.abs(self.A_eq@x-self.b_eq),initial=0.))
        violation=max(0.,float(np.max(self.lower-u)),float(np.max(u-self.upper)),
                      float(np.max(self.A_ub@x-self.b_ub,initial=0.)))
        energy={self.resources[j]['id']:dict(maximum_abs_kwh=float(np.max(np.abs(info['rows']@x))),
            final_kwh=float((info['rows']@x)[-1]),buffer_kwh=info['buffer_kwh']) for j,info in self.energy_rows.items()}
        return dict(passed=violation<=tolerance and eq<=tolerance,
            max_linear_inequality_violation=violation,max_linear_equality_error=eq,
            energy_buffers=energy,service_fee_yuan=self.fee(u),baseline_added=False,
            feasibility_scope='Declared aggregate service contract, not unobserved individual device states')

    def execute(self,u,actual_net_p_kw,multipliers):
        """Execution-only PV clipping; no actual P is used to build the quote."""
        u=self.array(u);p=np.asarray(actual_net_p_kw,float);eta=np.asarray(multipliers,float)
        if p.shape!=(96,self.N) or eta.shape!=(self.R,) or not np.isfinite(p).all() or not np.isfinite(eta).all() or np.any((eta<0)|(eta>1)):
            raise ValueError('Invalid actual net P or resource multiplier')
        before=u*eta[None];delivered=before.copy()
        export=np.maximum(0.,-p).reshape(24,4,self.N).min(1)
        excess=[]
        for j,r in enumerate(self.resources):
            if r['kind']=='pv':
                cap=np.minimum(r['contract_kw'],export[:,self.node[j]])
                excess.append(float(np.maximum(before[:,j]-cap,0).sum()*self.dt))
                delivered[:,j]=np.minimum(delivered[:,j],cap)
        audit=self.audit(delivered)
        if not audit['passed']:raise ArithmeticError('Execution violates declared incremental contract')
        return delivered,dict(resource_audit=audit,PV_clipped_kwh=float(sum(excess)),
            PV_clipped_cells=int((before-delivered>1e-9).sum()),multipliers=eta,
            response_abs_error_kwh=float(np.abs(u-before).sum()*self.dt),
            delivered_abs_kwh=float(np.abs(delivered).sum()*self.dt))

    def device_baseline_pq_kw(self):
        raise RuntimeError('Measured net profiles already include devices; adding a baseline is prohibited')
