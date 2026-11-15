"""Evaluator-only AC, metering, observable gain updates and whole-day bills."""
from __future__ import annotations

import numpy as np

from ... import io
from ...observations import Observation
from ..history288_pq_noise_20261115 import run as history
from ..measured_pq_flex_20261114 import full_ac


class DeliveryEstimator:
    """Only actual device telemetry is admitted, never the simulator's eta."""
    def __init__(self, fleet, initial=.95, lower=.9):
        self.fleet, self.lower = fleet, float(lower)
        self.gains = np.full(fleet.R, float(initial))
        self.numerator, self.denominator = {}, {}

    def update(self, extra_request, extra_delivered):
        request, delivered = np.asarray(extra_request, float), np.asarray(extra_delivered, float)
        if request.shape != (self.fleet.R,) or delivered.shape != request.shape:
            raise ValueError('Expected one measured request/delivery row')
        for did in self.fleet.device_ids:
            columns = [i for i,r in enumerate(self.fleet.resources) if r['physical_device_id']==did]
            # One common device gain, with normalized P/Q so units do not weight the fit.
            scales = np.maximum(1e-12, np.asarray([
                self.fleet.resources[i].get('contract_kw') or self.fleet.resources[i].get('capacity_kvar') or 1.
                for i in columns], float))
            x, y = request[columns]/scales, delivered[columns]/scales
            self.numerator[did] = self.numerator.get(did,0.) + float(x@y)
            self.denominator[did] = self.denominator.get(did,0.) + float(x@x)
            if self.denominator[did] > 1e-16:
                value = self.numerator[did]/self.denominator[did]
                self.gains[columns] = np.clip(value,self.lower,1.)
        return self.gains.copy()


class CostLedger:
    def __init__(self, fleet, kappa_p=.5, kappa_q=.5):
        self.fleet = fleet
        self.prices = np.r_[np.full(fleet.P_R,kappa_p),np.full(fleet.R-fleet.P_R,kappa_q)]
        self.paid_capacity = np.zeros(fleet.R)
        self.activation, self.capacity, self.grid = 0.,0.,0.

    def settle(self, step, request_tail, actual_delta, root_kw, service_tail=None):
        request, actual = np.asarray(request_tail,float), np.asarray(actual_delta,float)
        if request.ndim!=2 or request.shape[1]!=self.fleet.R or actual.shape!=(1,self.fleet.R):
            raise ValueError('Invalid settlement action dimensions')
        before=self.paid_capacity.copy()
        service=np.zeros_like(request) if service_tail is None else np.asarray(service_tail,float)
        if service.shape!=request.shape:raise ValueError('Service/request reservation shapes differ')
        new_peak=np.maximum(self.paid_capacity,np.maximum(np.max(np.abs(request),axis=0),
                            np.max(np.abs(service),axis=0)))
        capacity=float(self.prices@(new_peak-self.paid_capacity))
        self.paid_capacity=new_peak
        activation=float(self.fleet.activation_fee(actual,start=step))
        clock=(8.+(step+.5)*.25)%24
        rates=self.fleet.spec['pricing']['grid_buy_yuan_per_kwh']
        buy=float(rates['night' if clock<7 else 'evening' if 17<=clock<22 else 'day'])
        grid=float(.25*(buy*max(root_kw,0.)-.9*buy*max(-root_kw,0.)))
        self.activation+=activation; self.capacity+=capacity; self.grid+=grid
        return dict(activation_yuan=activation,availability_increment_yuan=capacity,
            root_energy_yuan=grid,cumulative_activation_yuan=self.activation,
            cumulative_availability_yuan=self.capacity,cumulative_service_yuan=self.activation+self.capacity,
            cumulative_root_energy_yuan=self.grid,paid_capacity_before=before,
            paid_capacity=self.paid_capacity.copy())


class Evaluator:
    """Construct only after the first request seal; expose just the current row."""
    def __init__(self, entry, fleet, repeat):
        self.entry,self.fleet,self.repeat=entry,fleet,int(repeat)
        history.old.WORK.setdefault('contexts',{})
        self._ctx=history.old.context(entry)
        arrays=history.paired.arrays(history.old.checked(entry['evaluation_noise']))
        self._noise=history.evaluation_noises(arrays,repeat,fleet.N)
        self._eta,self._lineage=history.device_multipliers(fleet,entry,repeat)
        self.next_step=0

    def save_randomness(self,path):
        np.savez_compressed(path,eta=self._eta,**{f'noise{i}':v for i,v in enumerate(self._noise)})
        return dict(randomness=io.ref(path),physical_PQ_share_device_eta=True,
            same_latent_gain_all_96_rounds=True,device_lineage=self._lineage)

    def step(self, step, service_first, extra_first):
        if step!=self.next_step:
            raise ValueError('Evaluator cannot expose a future or repeated physical row')
        service,extra=np.asarray(service_first,float),np.asarray(extra_first,float)
        if service.shape!=(self.fleet.R,) or extra.shape!=service.shape:
            raise ValueError('One quarter-hour device action required')
        delta=service+self._eta*extra
        node=self.fleet.B@delta
        c=self._ctx
        p=c['p'][step:step+1]+node[:self.fleet.N][None,:]/c['base']
        q=c['q'][step:step+1]+node[self.fleet.N:][None,:]/c['base']
        root=c['root'][step:step+1]
        if np.all(delta==0.):
            saved=c['baseline_state']
            state={key:(value[step:step+1].copy() if np.asarray(value).ndim and len(value)==96 else value)
                   for key,value in saved.items()}
            checks=dict(status='reused_sealed_zero_row',thermal_rating_check=False)
        else:
            state,checks=full_ac.solve(c['network'],c['labels'],p,q,root)
        voltage=state['terminal_voltage_pu']
        noise=tuple(value[step:step+1] for value in self._noise)
        pm,qm,y,vm,rm=history.old.meter(p,q,voltage,root,noise,
            pq_sigma=history.pq_sd(self.entry),voltage_sigma=.0005)
        self.next_step+=1
        return dict(observation=Observation(pm,qm,y,c['labels'],vm,rm),
            clean_P=p,clean_Q=q,clean_V=voltage,clean_root=root,
            base_P=c['p'][step:step+1],base_Q=c['q'][step:step+1],
            total_actual_delta=delta,extra_actual=self._eta*extra,ac=checks,
            ac_state=state,root_kw=float(state['root_supply_kw'][0]),
            minimum_voltage=float(np.min(state['voltage_magnitude_pu'])),
            maximum_voltage=float(np.max(state['voltage_magnitude_pu'])))

    def final_evaluation(self, observation):
        if self.next_step!=96:
            raise ValueError('No final truth evaluation before all 96 observations')
        result,_,_=history.old.score_observation(self._ctx,observation)
        return result
