"""Historical P/Q-consistent planning domain, with unchanged physical devices.

Only training P/Q determine the additional convex quadratic constraint. The
physical_case must still be used for actual delivery/AC feasibility checks:
this restriction concerns extrapolation of a frozen identification model.
"""
from time import perf_counter
import hashlib
import json
import numpy as np
from ... import io
from ..customer_flexibility_20261113.model import CustomerCase


def training_fingerprint(labels,p,q):
    """Stable content hashes; never feed an object array to contract_hash."""
    power=np.stack((np.asarray(p,dtype=np.float64),np.asarray(q,dtype=np.float64)))
    names=json.dumps(list(labels),ensure_ascii=False,separators=(',',':'),default=io.plain)
    return dict(training_pq_sha256=io.contract_hash(power),
        training_labels_sha256=hashlib.sha256(names.encode('utf-8')).hexdigest(),
        training_hash_schema='numeric_arrays_v2')


class PQTrustedCase(CustomerCase):
    def __init__(self,physical_case,training,*,base_kw=200.,scale=1.):
        began=perf_counter()
        if not np.isfinite(base_kw) or base_kw<=0 or not np.isfinite(scale) or scale<=0:
            raise ValueError('Positive finite power base and departure amplitude multiplier required')
        if tuple(training.labels)!=physical_case.labels:
            raise ValueError('Training and resource terminal order differ')
        super().__init__(physical_case.config,physical_case.labels)
        self.physical_case=physical_case
        self.base_kw=float(base_kw);self.trust_scale=float(scale)
        p,q=np.asarray(training.p,float),np.asarray(training.q,float)
        denominator=np.sum(p*p,axis=0)
        self.kappa=np.divide(np.sum(p*q,axis=0),denominator,
            out=np.zeros(self.N),where=denominator>0)
        residual=q-p*self.kappa[None]
        self.training_departure_variance=np.mean(residual*residual,axis=0)
        # C maps hourly device kW changes to node kvar departures from the
        # historical per-node relation Δq_n = kappa_n Δp_n.
        self.departure=np.zeros((self.N,self.R))
        self.departure[self.node,np.arange(self.R)]=self.qratio-self.kappa[self.node]
        self.trust_Q=self.departure.T@self.departure
        self.trust_limit=float(self.trust_scale**2*self.H*self.base_kw**2*self.training_departure_variance.sum())
        self.trust_receipt=dict(source=io.ref(__file__),rule='sum_hn(delta_q_hn-kappa_n*delta_p_hn)^2 <= scale^2*H*base_kw^2*sum_n mean_train(q_n-kappa_n*p_n)^2',
            **training_fingerprint(training.labels,p,q),
            training_rows=len(p),decision_hours=self.H,measurements_per_hour=self.repeat,
            base_kw=self.base_kw,scale=self.trust_scale,limit=self.trust_limit,
            kappa=self.kappa.tolist(),training_departure_rms_pu=np.sqrt(self.training_departure_variance).tolist(),
            zero_training_p_columns=np.flatnonzero(denominator==0).tolist(),
            constraint='One convex PSD quadratic constraint in hourly resource controls',
            units='Hourly sum of squared kvar departures; not kWh procurement energy',
            interpretation='Historical magnitude chosen as a design restriction, not a statistical coverage guarantee',
            execution_scope='Planning restriction only; use physical_case for device execution and report delivered trust separately',
            construction_seconds=perf_counter()-began)
        self.config['planning_pq_trust']=dict(scale=self.trust_scale,base_kw=self.base_kw,
            training_pq_sha256=self.trust_receipt['training_pq_sha256'])

    def trust_audit(self,u,tolerance=1e-7):
        u=self.array(u)
        # Independent direct node aggregation, rather than evaluating trust_Q.
        p,q=self.physical_case.aggregate(self.physical_case.delivery(u))
        departure=q-p*self.kappa[None]
        energy=float(np.sum(departure*departure)/self.repeat)
        quadratic=float(np.einsum('hi,ij,hj',u,self.trust_Q,u))
        if abs(energy-quadratic)>1e-9*(1+energy):
            raise ArithmeticError('Direct P/Q departure and quadratic expansion differ')
        return dict(passed=energy<=self.trust_limit+tolerance,energy=energy,limit=self.trust_limit,
            margin=self.trust_limit-energy,scale=self.trust_scale,
            relative_usage=None if self.trust_limit==0 else energy/self.trust_limit,
            maximum_node_departure_kvar=float(np.max(abs(departure),initial=0.)),
            expansion_error=abs(energy-quadratic),planning_only=True)

    def audit(self,u,tolerance=1e-7):
        physical=self.physical_case.audit(u,tolerance)
        trust=self.trust_audit(u,tolerance)
        return dict(physical,physical_passed=physical['passed'],trusted_direction=trust,
            passed=bool(physical['passed'] and trust['passed']))

    def physical_audit(self,u,tolerance=1e-7):
        return self.physical_case.audit(u,tolerance)

    def attach(self,model,name='pq_trusted_customer'):
        u,fee=super().attach(model,name)
        model.addConstr(sum(u[h]@self.trust_Q@u[h] for h in range(self.H))<=self.trust_limit,
            name=name+'_historical_PQ_departure')
        return u,fee

    def linear_optimum(self,*args,**kwargs):
        raise RuntimeError('The trusted domain is quadratic: generate LP seeds/rotation with physical_case, then audit or radially scale them')
