"""Observed-history AC responses; fixed candidate lines and source forecasts."""
from itertools import combinations
from time import perf_counter
import numpy as np
from .. import likelihood as lk
from .. import ev_mobile_v4_cases as forecasts


def output(physical):
    return np.column_stack((physical['v'], physical['master_p'], physical['master_q']))


def sigmas(meter, T, n):
    return np.column_stack((lk._array(meter.v_sigma, (T,n), 'v_sigma'),
        lk._array(meter.master_p_sigma, (T,), 'master_p_sigma'),
        lk._array(meter.master_q_sigma, (T,), 'master_q_sigma')))


def power_jacobian(candidate, physical, base_kw):
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('positive kW base required')
    derivative = lk._response_derivatives(candidate, physical, True)
    columns = slice(2*candidate.m, 2*candidate.m+candidate.n)
    return np.concatenate((derivative['v'][:,:,columns],
        derivative['master_p'][:,None,columns], derivative['master_q'][:,None,columns]), axis=1)/base_kw


class FixedResponses:
    """No simulator or true parameters accepted; same forecast for all actions."""
    def __init__(self, candidates, history, meter, future_meter, fitted, *, horizon, base_kw):
        began=perf_counter(); self.bank=tuple(candidates); self.base_kw=float(base_kw)
        if not fitted.complete:
            raise ValueError('all candidate historical fits required')
        self.states={c.name:dict(r=fitted.fits[c.name].r.copy(), x=fitted.fits[c.name].x.copy(),
            source_p=fitted.fits[c.name].source_p.copy(), source_q=fitted.fits[c.name].source_q.copy(),
            success=True) for c in self.bank}
        self.forecast=forecasts.forecast_for(history, meter, self.states, T=horizon, base_kw=base_kw)
        self.shape=(horizon, history.n); self.sigma=sigmas(future_meter, *self.shape)
        self.voltage_bounds=np.asarray(fitted.options['voltage_bounds'],float)
        self.cache={}; self.ac_calls=0; self.ac_states=0; self.derivative_calls=0
        self.zero=self.responses(np.zeros(self.shape), derivatives=True)
        self.seconds=perf_counter()-began

    def responses(self, power, *, derivatives=False):
        power=np.asarray(power,float)
        if power.shape != self.shape or not np.isfinite(power).all():
            raise ValueError('finite time/site powers required')
        key=np.ascontiguousarray(power).tobytes()
        if key not in self.cache:
            rows={}; p=self.forecast['p']+power/self.base_kw
            for c in self.bank:
                s=self.states[c.name]; source=self.forecast['source_by_candidate'][c.name]
                physical=lk.ac_predict(c,s['r'],s['x'],p,self.forecast['q'],
                    source['source_p'],source['source_q'],self.forecast['root_v'])
                magnitudes=np.abs(physical['voltage'])
                if not (self.voltage_bounds[0] <= np.min(magnitudes)
                        and np.max(magnitudes) <= self.voltage_bounds[1]):
                    raise RuntimeError('fixed forecast AC violates declared voltage domain')
                rows[c.name]=dict(mean=output(physical),physical=physical)
                self.ac_calls+=1; self.ac_states+=self.shape[0]
            self.cache[key]=rows
        rows=self.cache[key]
        if derivatives:
            for c in self.bank:
                if 'jacobian' not in rows[c.name]:
                    rows[c.name]['jacobian']=power_jacobian(c,rows[c.name]['physical'],self.base_kw)
                    self.derivative_calls+=1
        return rows

    def scores(self, power):
        rows=self.responses(power); total={}; incremental={}
        for a,b in combinations(sorted(rows),2):
            key=a+'__vs__'+b
            gap=rows[a]['mean']-rows[b]['mean']
            change=gap-(self.zero[a]['mean']-self.zero[b]['mean'])
            total[key]=.5*float(np.sum((gap/self.sigma)**2))
            incremental[key]=.5*float(np.sum((change/self.sigma)**2))
        return dict(total=total,incremental=incremental,
            total_score=min(total.values()),incremental_score=min(incremental.values()),
            physical_feasible=True,source_fixed_to_forecast=True,lines_fixed=True)

    def record(self):
        return dict(forecast=self.forecast,states=self.states,sigmas=self.sigma,
            zero_responses={name:{k:v for k,v in row.items() if k!='physical'}
                            for name,row in self.zero.items()},
            ac_calls=self.ac_calls,ac_states=self.ac_states,derivative_calls=self.derivative_calls,
            observed_only=True,parameters='one main historical fit per candidate',
            covariance='independent time/channel public meter sigma; no line uncertainty inflation',
            source_profiled_in_planning=False)
