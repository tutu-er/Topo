"""Registered active-service expansion with unchanged reactive capability.

All fees in this module are activation fees. A separate objective wrapper may
price purchased availability; the capacities here are maximum capabilities.
"""
from copy import deepcopy

import numpy as np

from ..customer_flexibility_20261113.model import window
from ..measured_pq_flex_20261114.domain import IncrementCase
from ..reactive_flex_20261114.domain import PQIncrementCase


class CatalogPCase(IncrementCase):
    """Same linear P physics with a prelocked explicit registration catalogue."""
    def __init__(self,context,labels,forecast,base_kw,resources,*,fraction=.1):
        super().__init__(context,labels,forecast,base_kw,fraction)
        original=deepcopy(self.resources)
        rows=deepcopy(resources)
        self.original_P_count=len(original)
        if rows[:len(original)]!=original:
            raise ValueError('The original seven P resources must be an unchanged prefix')
        if len({row['id'] for row in rows})!=len(rows):
            raise ValueError('Duplicate P resource ID')
        for row in rows:
            if row['node_label'] not in self.labels or row['kind'] not in ('pv','ev','thermal'):
                raise ValueError('Expected a registered P resource at a measured node')
            if not np.isfinite(row['contract_kw']) or row['contract_kw']<=0 or row.get('q_over_p')!=0.:
                raise ValueError('Pure-P positive finite capability required')
            if abs(row['contract_kw']-fraction*row['scaled_registered_kw'])>1e-12:
                raise ValueError('P capability differs from the locked registration fraction')
            for key in ('fee_positive_yuan_per_kwh','fee_negative_yuan_per_kwh'):
                if not np.isfinite(row[key]) or row[key]<=0:
                    raise ValueError('Positive finite activation fee required')
        self.resources=rows
        self.R=len(rows);self.shape=(self.H,self.R);self.dimension=self.H*self.R
        self.node=np.array([self.labels.index(row['node_label']) for row in rows])
        self.qratio=np.zeros(self.R);self.baseline=np.zeros(self.shape)
        self.lower=np.zeros(self.shape);self.upper=np.zeros(self.shape)
        forecast=np.asarray(forecast,float)
        export=np.maximum(0.,-forecast[:,:self.N]*base_kw).reshape(24,4,self.N).min(1)
        self.energy_rows={};eq=[];rhs=[];ub=[];bounds=[]
        for j,row in enumerate(rows):
            cap=row['contract_kw'];kind=row['kind']
            if kind=='pv':
                self.upper[:,j]=np.where(window(self.clock,10,14),np.minimum(cap,export[:,self.node[j]]),0.)
                continue
            available=window(self.clock,18,7) if kind=='ev' else np.ones(24,bool)
            self.lower[available,j]=-cap;self.upper[available,j]=cap
            rho=1. if kind=='ev' else .98
            buffer=(2. if kind=='ev' else 1.)*cap
            energy=np.zeros((24,self.dimension))
            for h in range(24):
                energy[h,np.arange(h+1)*self.R+j]=rho**np.arange(h,-1,-1)*self.dt
            self.energy_rows[j]=dict(rows=energy,rho=rho,buffer_kwh=buffer)
            ub.extend(energy);bounds.extend(np.full(24,buffer))
            ub.extend(-energy);bounds.extend(np.full(24,buffer))
            eq.append(energy[-1]);rhs.append(0.)
        self.A_eq=np.asarray(eq,float).reshape(-1,self.dimension);self.b_eq=np.asarray(rhs,float)
        self.A_ub=np.asarray(ub,float).reshape(-1,self.dimension);self.b_ub=np.asarray(bounds,float)
        price=self.config['pricing'];mult=np.ones(24)
        for a,b in price['peak_clock_windows']:mult[window(self.clock,a,b)]=1.25
        for a,b in price['night_clock_windows']:mult[window(self.clock,a,b)]=.8
        self.fee_plus=mult[:,None]*np.array([r['fee_positive_yuan_per_kwh'] for r in rows])
        self.fee_minus=mult[:,None]*np.array([r['fee_negative_yuan_per_kwh'] for r in rows])
        self.config.update(resources=deepcopy(rows),original_P_count=self.original_P_count,
            appended_P_count=self.R-self.original_P_count,
            placement='Original P prefix plus prelocked registered additional sites; no new-day voltage or truth used',
            capability='contract_kw is the maximum offered service envelope; purchased daily availability is a separate planning variable',
            physical_energy_buffers='Original per-device physical energy buffers retained; no extra baseline or energy added')


class RegisteredPQCase(PQIncrementCase):
    """PQIncrementCase API with explicit P catalogue and the original Q layer."""
    def __init__(self,context,labels,forecast,base_kw,*,p_resources,q_resources,fraction=.1):
        super().__init__(context,labels,forecast,base_kw,q_resources=q_resources,fraction=fraction)
        old_p_count=self.P_R
        q_lower=self.lower[:,old_p_count:].copy();q_upper=self.upper[:,old_p_count:].copy()
        q_plus=self.fee_plus[:,old_p_count:].copy();q_minus=self.fee_minus[:,old_p_count:].copy()
        original_config=deepcopy(self.config)
        self.p_case=CatalogPCase(context,labels,forecast,base_kw,p_resources,fraction=fraction)
        p=self.p_case;self.original_P_count=p.original_P_count;self.P_R=p.R
        self.resources=deepcopy(p.resources)+deepcopy(self.q_resources)
        self.R=len(self.resources);self.shape=(self.H,self.R);self.dimension=self.H*self.R
        self.node=np.array([self.labels.index(int(r['node_label'])) for r in self.resources])
        self.action_units=tuple(['kW']*self.P_R+['kvar']*self.Q_R)
        self.B=np.zeros((2*self.N,self.R))
        for j,node in enumerate(self.node):self.B[node if j<self.P_R else self.N+node,j]=1.
        self.baseline=np.zeros(self.shape)
        self.lower=np.column_stack((p.lower,q_lower));self.upper=np.column_stack((p.upper,q_upper))
        self.fee_plus=np.column_stack((p.fee_plus,q_plus));self.fee_minus=np.column_stack((p.fee_minus,q_minus))
        self.A_eq=self._expand(p.A_eq);self.A_ub=self._expand(p.A_ub)
        self.b_eq=p.b_eq.copy();self.b_ub=p.b_ub.copy()
        self.energy_rows={j:dict(info,rows=self._expand(info['rows'])) for j,info in p.energy_rows.items()}
        self.config=original_config
        self.config.update(resources=deepcopy(self.resources),action_units=list(self.action_units),
            original_P_count=self.original_P_count,appended_P_count=self.P_R-self.original_P_count,
            placement=p.config['placement'],
            reactive='Independent pure-Q maximum capabilities; positive absorption and negative injection. Purchased daily availability is determined separately.',
            reactive_capacity_source='Original 36 kvar maximum offered capability unchanged; this does not assert that all 36 kvar is purchased',
            P_capability=p.config['capability'],physical_energy_buffers=p.config['physical_energy_buffers'],
            prices_in_this_domain='Activation only; daily availability price is added by the separate objective wrapper',
            response='Original P stream773 prefix preserved, added P draws appended; all original Q stream774 draws and Q order preserved',
            rotation='Original P logical indices unchanged; Q uses original P offset7, independent of added P count')

    def fixed_rotation_direction(self):
        """Keep original resource directions when new P columns precede Q."""
        phase=np.r_[np.arange(self.P_R),self.original_P_count+np.arange(self.Q_R)]
        return np.where((np.arange(self.H)[:,None]//2+phase[None,:])%2==0,1.,-1.)

    def audit(self,u,tolerance=1e-7):
        result=super().audit(u,tolerance)
        result['feasibility_scope']='Original/expanded P physical energy constraints and maximum Q capability; purchased reserve and price checked separately'
        return result


def create_case(entry,ctx,forecast):
    return RegisteredPQCase(ctx['resource'],ctx['labels'],forecast,ctx['base'],
        p_resources=entry['p_resources'],q_resources=entry['q_resources'],
        fraction=float(entry.get('physical_contract_fraction',.1)))
