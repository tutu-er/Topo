"""Independent equipment-domain and paired AC audit from sealed arrays.

No production device domain, optimizer, identifier, selector or AC solver is
called. Device operating states are reconstructed from declared parameters;
real topology is opened only for post-execution score/electrical evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from ..single_target_20261115 import audit as single
from ..node_time_pq_20261115 import audit as band

base, panel, prob, previous, ROOT = single.base, single.panel, single.prob, single.previous, single.ROOT


def injection_map(config, labels, base_kw):
    """Rebuild the physical P/Q column map from device and node identities."""
    resources, units = config['resources'], config['action_units']
    if len(resources) != len(units) or len({r['id'] for r in resources}) != len(resources):
        raise ValueError('Resource column identity is not unique')
    n = len(labels); mapping = np.zeros((2*n, len(resources)))
    for j, (r, unit) in enumerate(zip(resources, units, strict=True)):
        if unit not in ('kW', 'kvar') or r['node_label'] not in labels:
            raise ValueError('Invalid device action unit/node')
        mapping[labels.index(r['node_label'])+(n if unit == 'kvar' else 0), j] = 1./float(base_kw)
    return mapping


def states(power, *, initial, dt, retention=1., demand=None, eta_charge=1., eta_discharge=1.):
    """Direct energy recursion, including all samples and the terminal state."""
    power = np.asarray(power, float)
    if power.ndim != 1 or not np.isfinite(power).all() or dt <= 0:
        raise ValueError('Finite one-dimensional physical power is required')
    need = np.zeros_like(power) if demand is None else np.asarray(demand, float)
    if need.shape != power.shape or not 0 < retention <= 1 or not 0 < min(eta_charge, eta_discharge) <= 1:
        raise ValueError('Invalid demand/efficiency/retention')
    answer = [float(initial)]
    for value, draw in zip(power, need, strict=True):
        rate = eta_charge*max(value, 0.)+min(value, 0.)/eta_discharge
        answer.append(retention*answer[-1]+float(dt)*(rate-draw))
    return np.asarray(answer)


def circle_values(p, q, rating):
    p,q = np.asarray(p,float), np.asarray(q,float)
    if p.shape != q.shape or rating <= 0 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Invalid inverter power/rating')
    excess=p*p+q*q-float(rating)**2
    return dict(maximum_apparent_power_kva=float(np.sqrt(p*p+q*q).max(initial=0.)),
        maximum_squared_capacity_violation=max(0.,float(excess.max(initial=0.))),
        minimum_squared_headroom=float((-excess).min(initial=np.inf)))


def power_shift(config, action, labels, base_kw, points=96):
    action=np.asarray(action,float)
    if action.shape != (24,len(config['resources'])) or not np.isfinite(action).all() or points % 24:
        raise ValueError('Invalid hourly device action')
    return np.repeat(action@injection_map(config,labels,base_kw).T,points//24,axis=0)


def quantity(action, config):
    action=np.asarray(action,float); units=np.asarray(config['action_units'])
    return dict(active_abs_kwh=float(abs(action[:,units=='kW']).sum()),
        reactive_abs_kvarh=float(abs(action[:,units=='kvar']).sum()))


def energy_bill(root_power_kw):
    """Original illustrative grid energy tariff, independent of service fees."""
    root=np.asarray(root_power_kw,float)
    if root.shape!=(96,) or not np.isfinite(root).all():
        raise ValueError('Expected 96 quarter-hour root powers')
    clock=(8+np.arange(96)*.25)%24
    buy=np.where(clock<7,.3,np.where((clock>=17)&(clock<22),.8,.5))
    return float(.25*np.sum(buy*np.maximum(root,0.)-.9*buy*np.maximum(-root,0.)))


def window(clock, hours):
    a,b=hours
    return (clock>=a)&(clock<b) if a<b else (clock>=a)|(clock<b)


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fees(config, request, delivered, kp=.5, kq=.5):
    """Read device prices, not planner's linear objective coefficients."""
    pricing=config['pricing'];clock=(8+np.arange(24)+.5)%24
    multiplier=np.ones(24)
    for hours in pricing['peak_clock_windows']:
        multiplier[window(clock,hours)]=pricing['clock_hour_multiplier']['morning_evening_peak']
    for hours in pricing['night_clock_windows']:
        multiplier[window(clock,hours)]=pricing['clock_hour_multiplier']['night']
    devices={d['id']:d for d in config['devices']};units=config['action_units']
    peak=np.max(abs(request),axis=0);answer={}
    for stage,u in [('planned',request),('settled',delivered)]:
        costs={'active_activation_yuan':0.,'reactive_activation_yuan':0.,
               'active_availability_yuan':0.,'reactive_availability_yuan':0.}
        for col,(r,unit) in enumerate(zip(config['resources'],units,strict=True)):
            if unit=='kW':
                d=devices[r['physical_device_id']]
                cost=np.sum(multiplier*(np.maximum(u[:,col],0)*d['fee_positive_yuan_per_kwh']+
                                        np.maximum(-u[:,col],0)*d['fee_negative_yuan_per_kwh']))
                prefix='active';capacity=kp
            else:
                cost=np.sum(abs(u[:,col]))*config['reactive_fee_yuan_per_kvarh']
                prefix='reactive';capacity=kq
            costs[prefix+'_activation_yuan']+=float(cost)
            costs[prefix+'_availability_yuan']+=float(peak[col]*capacity)
        costs['activation_yuan']=costs['active_activation_yuan']+costs['reactive_activation_yuan']
        costs['availability_yuan']=costs['active_availability_yuan']+costs['reactive_availability_yuan']
        costs['total_yuan']=costs['activation_yuan']+costs['availability_yuan']
        costs['reserved_P_kw']=float(sum(v for v,k in zip(peak,units) if k=='kW'))
        costs['reserved_Q_kvar']=float(sum(v for v,k in zip(peak,units) if k=='kvar'))
        answer[stage]=costs
    return answer


def device_values(config, action):
    """Direct absolute device power and energy, independent of linear rows."""
    u=np.asarray(action,float);resources=config['resources'];devices=config['devices']
    if u.shape!=(24,len(resources)) or not np.isfinite(u).all():raise ValueError('Invalid equipment action')
    bydevice={d['id']:{} for d in devices}
    for j,r in enumerate(resources):
        bydevice[r['physical_device_id']][config['action_units'][j]]=j
    clock=(8+np.arange(24)+.5)%24;result={};violations={}
    for d in devices:
        did=d['id'];columns=bydevice[did];pcol=columns['kW'];qcol=columns.get('kvar')
        p0=np.asarray(d['baseline_p_kw'],float);q0=np.asarray(d['baseline_q_kvar'],float)
        p=p0+np.repeat(u[:,pcol],4);q=q0+(0 if qcol is None else np.repeat(u[:,qcol],4))
        power=float(d['power_kw']);kind=d['kind'];checks={}
        row=dict(kind=kind,node_label=d['node_label'],p_kw=p,q_kvar=q,baseline_p_kw=p0,baseline_q_kvar=q0)
        hourly=p.reshape(24,4).mean(axis=1)
        if kind=='pv':
            available=np.asarray(d['available_generation_kw'],float)
            checks['PV_generation_range']=max(0.,float(np.max(-available-p)),float(np.max(p)))
            checks['PV_curtailment_only']=max(0.,float(np.max(p0-p)))
            online=np.all(available.reshape(24,4)>0,axis=1)
        elif kind=='ev':
            online=window(clock,d['connected_clock_hours'])
            checks['EV_power']=max(0.,float((-p).max()),float((p-power).max()))
            checks['EV_offline_power']=float(np.max(abs(p.reshape(24,4)[~online]),initial=0.))
            energy=float(.25*p.sum()*d['charging_efficiency'])
            checks['EV_departure_energy']=abs(energy-d['required_battery_energy_kwh'])
            row.update(delivered_battery_energy_kwh=energy,required_battery_energy_kwh=d['required_battery_energy_kwh'],v2g=False)
        elif kind=='thermal':
            checks['thermal_power']=max(0.,float((-p).max()),float((p-power).max()))
            rho=d['hourly_temperature_retention'];capacity=d['heat_capacity_kwh_per_C'];ambient=d['ambient_temperature_C']
            heat=np.asarray(d['heat_demand_kw']).reshape(24,4).mean(1)
            temperature=[float(d['initial_temperature_C'])]
            for value,need in zip(hourly,heat,strict=True):
                temperature.append(ambient+rho*(temperature[-1]-ambient)+(d['cop']*value-need)/capacity)
            t=np.asarray(temperature)
            checks['temperature_energy_range']=max(0.,float((d['minimum_temperature_C']-t).max()),float((t-d['maximum_temperature_C']).max()))*capacity
            checks['thermal_terminal_energy']=abs(t[-1]-d['terminal_temperature_C'])*capacity
            row.update(temperature_C=t,terminal_error=float(t[-1]-t[0]),minimum_allowed=d['minimum_temperature_C'],maximum_allowed=d['maximum_temperature_C'])
        elif kind=='battery':
            charge=window(clock,d['charging_clock_hours']);discharge=window(clock,d['discharging_clock_hours'])
            lo=np.where(discharge,-power,0.);hi=np.where(charge,power,0.)
            checks['prelocked_battery_power']=max(0.,float((lo[:,None]-p.reshape(24,4)).max()),float((p.reshape(24,4)-hi[:,None]).max()))
            e=states(hourly,initial=d['initial_soc']*d['energy_kwh'],dt=1.,
                eta_charge=d['charging_efficiency'],eta_discharge=d['discharging_efficiency'])
            soc=e/d['energy_kwh']
            checks['battery_energy_range']=max(0.,float((d['minimum_soc']-soc).max()),float((soc-d['maximum_soc']).max()))*d['energy_kwh']
            checks['battery_terminal_energy']=abs(e[-1]-d['terminal_soc']*d['energy_kwh'])
            row.update(soc=soc,terminal_error=float(soc[-1]-soc[0]),minimum_allowed=d['minimum_soc'],maximum_allowed=d['maximum_soc'])
            online=np.zeros(24,bool)
            for hours in d['reactive_clock_windows']:online|=window(clock,hours)
        else:raise ValueError('Unsupported device kind')
        if qcol is None:
            checks['no_uncontrolled_Q']=float(np.max(abs(q),initial=0.))
        else:
            cap=float(d['reactive_cap_kvar']);checks['reactive_service_cap']=max(0.,float((abs(q)-cap).max()))
            checks['reactive_offline']=float(np.max(abs(q.reshape(24,4)[~online]),initial=0.))
            circle=circle_values(p,q,d['inverter_kva'])
            checks['relative_inverter_circle']=circle['maximum_squared_capacity_violation']/d['inverter_kva']**2
            row.update(apparent_power_kva=np.sqrt(p*p+q*q),inverter_rating_kva=d['inverter_kva'])
        result[did]=row;violations[did]=checks
    maximum=max((x for r in violations.values() for x in r.values()),default=0.)
    return dict(passed=maximum<=1e-7,maximum_violation=float(maximum),violations=violations,
        states=result,quantities=quantity(u,config),
        scope='Original declared equipment constraints. Power/energy residual tolerance1e-7; inverter violation normalized by S squared.')


def amplitude_box(config):
    """Outer domain reconstructed from equipment powers and connection hours."""
    lower=np.zeros((24,len(config['resources'])));upper=lower.copy();clock=(8.5+np.arange(24))%24
    for d in config['devices']:
        p=next(j for j,r in enumerate(config['resources']) if r['physical_device_id']==d['id'] and r['action_unit']=='kW')
        p0=np.asarray(d['baseline_p_kw']).reshape(24,4);rating=d['power_kw'];kind=d['kind']
        if kind=='pv':
            g=np.asarray(d['available_generation_kw']).reshape(24,4);upper[:,p]=g.min(1);online=(g>0).all(1)
        elif kind in ('ev','thermal'):
            lower[:,p]=-p0.min(1)
            online=window(clock,d['connected_clock_hours']) if kind=='ev' else np.ones(24,bool)
            upper[online,p]=rating-p0[online].max(1)
        else:
            upper[window(clock,d['charging_clock_hours']),p]=rating
            lower[window(clock,d['discharging_clock_hours']),p]=-rating
            online=np.zeros(24,bool)
            for hours in d['reactive_clock_windows']:online|=window(clock,hours)
        if kind in ('pv','battery'):
            q=next(j for j,r in enumerate(config['resources']) if r['physical_device_id']==d['id'] and r['action_unit']=='kvar')
            lower[online,q]=-d['reactive_cap_kvar'];upper[online,q]=d['reactive_cap_kvar']
    return lower,upper


def interval_upper(Q,linear,constant,lower,upper):
    """Independent interval sum; cross terms evaluated at four box corners."""
    value=np.longdouble(constant);absolute=abs(value)
    for h in range(24):
        for i in range(lower.shape[1]):
            lo,hi=np.longdouble(lower[h,i]),np.longdouble(upper[h,i])
            term=max(linear[h,i]*lo,linear[h,i]*hi);value+=term;absolute+=abs(term)
            sqlo=0 if lo<=0<=hi else min(lo*lo,hi*hi);sqhi=max(lo*lo,hi*hi)
            term=max(Q[h,i,i]*sqlo,Q[h,i,i]*sqhi);value+=term;absolute+=abs(term)
            for j in range(i+1,lower.shape[1]):
                products=[a*b for a in (lo,hi) for b in (np.longdouble(lower[h,j]),np.longdouble(upper[h,j]))]
                term=max((Q[h,i,j]+Q[h,j,i])*p for p in products);value+=term;absolute+=abs(term)
    padding=1e-10*max(1.,float(absolute))
    return dict(upper=float(value)+padding,roundoff_padding=padding)


def independent_box_checks(model,forecast,config,base_kw,mode):
    mapping=injection_map(config,model['labels'],base_kw);lo,hi=amplitude_box(config)
    sd=model['forecast_relative_sd']*abs(forecast) if mode=='mgf_forecast' else np.zeros_like(forecast)
    score=model['score'];sigma=model['sigma2'];T,n=forecast.shape[0],model['n']
    tie=max(1e-7,(score['cells']+T*n)*score['reference_mse']*score['validation_tie']/(2*sigma));rows=[]
    for target in model['targets']:
        i=model['representatives'][target];js=[j for j,f in enumerate(model['families']) if f!=model['families'][i]]
        for j in js:
            alpha=.1/len(js);h=(model['history'][j]-model['history'][i])/(2*sigma)
            integral=prob.PairIntegral(model['H'][i],model['H'][j],model['means'][target],sigma,model['covariance'],sd,h)
            lam=integral.selected_lambda;eigen=1+2*lam*integral.eigenvalues
            G=integral.projected_G
            F=lam*integral.Q[None]-2*lam**2*np.einsum('tki,tk,tkj->tij',G,1/eigen,G,optimize=True)
            constant=np.log(alpha)+lam*(h-tie)+.5*np.log(eigen).sum()+np.einsum('ti,tij,tj->',forecast,F,forecast)
            Q=np.einsum('ir,tij,js->trs',mapping,F,mapping,optimize=True).reshape(24,4,len(lo[0]),len(lo[0])).sum(1)
            linear=2*np.einsum('ir,tij,tj->tr',mapping,F,forecast,optimize=True).reshape(24,4,len(lo[0])).sum(1)
            check=interval_upper(Q,linear,constant,lo,hi)
            check.update(row=len(rows),target_native=i,competitor_native=j,impossible=check['upper'] < -1e-7,
                zero_margin=float(constant),lambda_selected=lam,
                scope='Outer equipment amplitude box, relaxing energy/SOC, inverter circle, node/hour band and all other gates; fixed lambda and equal pair risk only.')
            rows.append(check)
    return rows


def self_test():
    """Small independent arithmetic counterexamples; no project experiment."""
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    e=states([2.,-1.805],initial=3.,dt=1.,eta_charge=.95,eta_discharge=.95)
    check('battery_efficiency_and_terminal',np.allclose(e,[3.,4.9,3.]))
    check('all_samples_retained',len(states([1.,1.,1.],initial=0,dt=.25))==4)
    e=states([1.,0.],initial=2.,dt=1.,retention=.98,demand=[.96,0.])
    check('thermal_retention_and_demand',np.allclose(e,[2.,2.,1.96]))
    check('inverter_boundary',circle_values([3.],[4.],5.)['maximum_squared_capacity_violation']==0.)
    check('inverter_violation',circle_values([3.],[4.1],5.)['maximum_squared_capacity_violation']>0.)
    config=dict(resources=[dict(id='P',node_label=7),dict(id='Q',node_label=7)],action_units=['kW','kvar'])
    u=np.zeros((24,2));u[0]=[2.,-3.]
    shift=power_shift(config,u,[7],200.)
    check('native_units_and_four_quarters',np.array_equal(shift[:4],np.tile([.01,-.015],(4,1))) and not shift[4:].any())
    check('P_Q_not_summed',quantity(u,config)==dict(active_abs_kwh=2.,reactive_abs_kvarh=3.))
    config['resources'][0].update(fee_positive_yuan_per_kwh=.1,fee_negative_yuan_per_kwh=.1)
    config['resources'][1].update(fee_positive_yuan_per_kvarh=.05,fee_negative_yuan_per_kvarh=.05)
    bill=base.procurement(config,u,u*.9,.5,.5)
    check('capacity_from_request',bill['settled']['availability_yuan']==2.5)
    check('activation_from_delivery',np.isclose(bill['settled']['activation_yuan'],.9*(.25+.15)))
    check('grid_export_is_credit',energy_bill(np.ones(96))>0 and energy_bill(-np.ones(96))<0)
    Q=np.zeros((24,2,2));Q[0]=[[-2.,.7],[.7,1.]];lin=np.zeros((24,2));lin[0]=[.3,-.2]
    lo=np.zeros((24,2));hi=lo.copy();lo[0]=[-1.,-.5];hi[0]=[.8,1.2]
    upper=interval_upper(Q,lin,-.1,lo,hi)['upper']
    mesh=np.array(np.meshgrid(np.linspace(-1,.8,23),np.linspace(-.5,1.2,29))).reshape(2,-1).T
    check('quadratic_interval_outer_bound',upper>=max(np.einsum('ni,ij,nj->n',mesh,Q[0],mesh)+mesh@lin[0]-.1))
    C=150*2*4.186/3600;UA=2/1200
    check('manufacturer_scaled_thermal_retention',np.isclose(1-UA/C,.9952221691352126,atol=1e-12))
    return dict(status='passed',checks=checks)


class Audit(single.Audit):
    def __init__(self):
        super().__init__()
        self.band_baseline_records=[];self.zero_records=[];self.device_checks=[]

    def tree(self, saved, actual, prefix):
        self.require(set(saved)==set(actual),'State keys differ: '+prefix)
        for key,value in actual.items():
            name=prefix+'.'+key
            if isinstance(value,dict):self.tree(saved[key],value,name)
            elif isinstance(value,(str,bool)) or value is None:self.require(saved[key]==value,'State differs: '+name)
            else:self.same(name,saved[key],value,atol=1e-9,rtol=1e-10)

    def equipment(self,entry,config,spec,forecast):
        self.require(spec['schema']=='equipment_flex_v1','Unknown equipment schema')
        claimed=spec['spec_content_sha256'];body={k:v for k,v in spec.items() if k!='spec_content_sha256'}
        self.require(content_hash(body)==claimed,'Equipment content hash differs')
        self.require(all(config[k]==v for k,v in spec.items()),'Config changes declared equipment specification')
        context=base.read(self.checked(entry['resource_context']))
        self.require(content_hash(context)==spec['context_content_sha256'],'Registration context changed')
        registration={int(r['feeder']):r for r in context['resource_metadata_at_history_start']}
        fields=[('pv','PV_systems_kW',3),('ev','EV_chargers_kW',2),('thermal','hot_water_tanks_kW',2),('battery','batteries_kW',2)]
        selected=[]
        for kind,field,count in fields:
            candidates=[]
            for node in context['node_mapping']:
                r=registration[node['feeder_id']]
                self.require(r['date'][:10]<=entry['start_date'][:10],'Future registration used')
                capacity=float(r.get(field,0.) or 0.)*node['scale']
                if capacity>0:candidates.append((capacity,node['label'],node,r))
            selected.extend((kind,*row) for row in sorted(candidates,key=lambda x:(-x[0],x[1]))[:count])
        self.require(len(selected)==len(spec['devices']),'Device-selection denominator differs')
        clock=(8+(np.arange(96)+.5)/4)%24;doy=datetime.fromisoformat(entry['start_date']).timetuple().tm_yday
        daylight=12+4*np.sin(2*np.pi*(doy-80)/365.25);rise=12-daylight/2
        phase=(clock-rise)/daylight;solar=np.where((phase>0)&(phase<1),np.maximum(0,np.sin(np.pi*phase))**1.5,0.)
        for d,(kind,capacity,label,node,r) in zip(spec['devices'],selected,strict=True):
            self.require(d['kind']==kind and d['node_label']==label and d['id']==kind+'_'+str(label),'Device placement identity differs')
            self.same('registered_scaled_P',d['scaled_registered_kw'],capacity,atol=0,rtol=0)
            self.same('participating_device_P',d['power_kw'],.1*capacity,atol=0,rtol=0)
            P=d['power_kw'];expected=np.zeros(96)
            if kind=='pv':
                expected=-P*solar
                self.same('synthetic_gross_PV',d['available_generation_kw'],-expected,atol=1e-13)
            elif kind=='ev':
                expected[window(clock,[22,4])]=.5*P
                self.require(d['v2g'] is False and d['connected_clock_hours']==[18,7],'Ordinary EV availability differs')
                self.same('EV_energy_need',d['required_battery_energy_kwh'],.95*3*P,atol=1e-12)
            elif kind=='thermal':
                expected[:]=.2*P;expected[window(clock,[6,9])]=.4*P;expected[window(clock,[17,22])]=.5*P
                C=150*P*4.186/3600
                self.same('tank_heat_capacity',d['heat_capacity_kwh_per_C'],C,atol=1e-12)
                UA=P/1200.
                self.same('standby_loss_coefficient',d['equivalent_thermal_loss_coefficient_kw_per_C'],UA,atol=1e-12)
                self.same('thermal_retention_from_UA',d['hourly_temperature_retention'],1-UA/C,atol=1e-12)
                self.same('fixed_heat_demand',d['heat_demand_kw'],expected-UA*(47.5-20),atol=1e-12)
            else:
                self.require(d['charging_clock_hours']==[10,16] and d['discharging_clock_hours']==[18,23], 'Battery direction windows changed')
                self.require(d['reactive_clock_windows']==[[10,16],[18,23]],'Battery Q outside the declared online windows')
                self.same('battery_energy_two_hours',d['energy_kwh'],2*P,atol=1e-12)
            self.same('device_baseline_'+d['id'],d['baseline_p_kw'],expected,atol=1e-12)
            self.same('device_Q_baseline_'+d['id'],d['baseline_q_kvar'],np.zeros(96),atol=0,rtol=0)
            if kind in ('pv','battery'):
                self.same('inverter_S',d['inverter_kva'],P,atol=0,rtol=0)
                self.same('inverter_Q_service_cap',d['reactive_cap_kvar'],.4*P,atol=0,rtol=0)
        labels=spec['expected_terminal_labels'];mapping=injection_map(config,labels,entry['base_kw'])
        self.same('device_map_B',config['B'],mapping*entry['base_kw'],atol=0,rtol=0)
        self.require(len(config['resources'])==len(spec['devices'])+sum(d['kind'] in ('pv','battery') for d in spec['devices']),'Extra free-Q or missing device column')
        zero=device_values(config,np.zeros((24,len(config['resources']))))
        self.require(zero['passed'],'Zero fails the declared device baselines')
        self.tree(base.read(self.checked(entry['baseline_device_states'])),zero['states'],'prepared_zero_states')
        decomposition=base.read(self.checked(entry['baseline_decomposition']))
        p0=np.zeros((96,len(labels)));q0=p0.copy()
        for d in spec['devices']:
            i=labels.index(d['node_label']);p0[:,i]+=np.asarray(d['baseline_p_kw']);q0[:,i]+=np.asarray(d['baseline_q_kvar'])
        residualP=forecast[:,:len(labels)]*entry['base_kw']-p0
        residualQ=forecast[:,len(labels):]*entry['base_kw']-q0
        for key,value in [('device_p_kw',p0),('device_q_kvar',q0),('residual_background_p_kw',residualP),('residual_background_q_kvar',residualQ)]:
            self.same('unchanged_net_decomposition_'+key,decomposition[key],value,atol=1e-12)
        self.require(decomposition['baseline_added_to_net'] is False,'Device baseline was added twice')
        self.device_checks.append(dict(id=entry['id'],devices=len(spec['devices']),columns=len(config['resources']),
            device_kinds=dict(Counter(d['kind'] for d in spec['devices'])),zero_feasible=True,
            residual_negative_P_cells=int((residualP<0).sum()),new_negative_P_cells=int(((residualP<0)&(forecast[:,:len(labels)]>=0)).sum()),
            baseline_scope='Synthetic participating-device decomposition of the unchanged net profile; not measured gross component trajectories.'))

    def band_arrays(self,entry,seal):
        train=base.native.arrays(self.checked(seal['train']));sampling=base.read(self.checked(seal['sampling_source']))
        self.require(seal['sampling_source']==entry['files']['observations_sealed.json'],'Training sampling source changed')
        b=band.reconstruct_band(train['P'],train['Q'],sampling['sampling']['train']['timestamps'],
            sampling['sampling']['new_day']['timestamps'][::4])
        arrays=base.native.arrays(self.checked(seal['bandwidth_npz']))
        self.same('frozen_band_slope',arrays['kappa'],b['slope'],atol=1e-10)
        self.same('frozen_band_width',arrays['band_kvar'],b['width_pu']*entry['base_kw'],atol=1e-10)
        previous_band=base.read(self.checked(entry['node_time_bandwidth']))
        original=base.native.arrays(self.checked(previous_band['arrays']))
        for key in arrays:self.same('old_band_arrays_'+key,arrays[key],original[key],atol=0,rtol=0)
        return arrays

    def action(self,entry,config,band_arrays,u):
        dv=device_values(config,u);shift=power_shift(config,u,config['expected_terminal_labels'],entry['base_kw'])
        n=len(config['expected_terminal_labels']);hourly=shift.reshape(24,4,2*n)[:,0,:]
        bd=band.direct_band(hourly[:,:n],hourly[:,n:],band_arrays['kappa'],band_arrays['band_kvar']/entry['base_kw'])
        return dv,bd,shift

    def randomness(self,folder,prior,entry,protocol,config,oldconfig,repeat):
        random=base.native.arrays(self.file(folder/f'execution{repeat:02d}'/'paired_randomness.npz'))
        old=base.native.arrays(self.file(prior/f'execution{repeat:02d}'/'paired_randomness.npz'))
        self.same('old_eta_reference',random['original_eta'],old['eta'],atol=0,rtol=0)
        for k in range(4):self.same('paired_noise_'+str(k),random['noise'+str(k)],old['noise'+str(k)],atol=0,rtol=0)
        oldP={r['id']:(j,r) for j,(r,unit) in enumerate(zip(oldconfig['resources'],oldconfig['action_units'])) if unit=='kW'}
        eta=np.zeros(len(config['resources']))
        for d in config['devices']:
            if d['id'] in oldP:
                j,r=oldP[d['id']];self.require(r['node_label']==d['node_label'] and r['kind']==d['kind'],'Response mapping changes device identity')
                value=old['eta'][j]
            else:
                self.require(d['kind']=='battery','Unexpected newly randomized device')
                key=[protocol['response_namespace'],int(protocol['seed']),int(entry['index']),0,int(repeat),d['id']]
                digest=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest()
                value=np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4').tolist())).uniform(.9,1.)
            for j,r in enumerate(config['resources']):
                if r['physical_device_id']==d['id']:eta[j]=value
        self.same('whole_device_PQ_shared_response',random['eta'],eta,atol=0,rtol=0)
        self.references(base.read(self.file(folder/f'execution{repeat:02d}'/'randomness_lineage.json')))
        return random

    def case(self,folder,entry,p,metadata,branches,buses):
        self.where=str(folder);self.references(entry)
        seal=base.read(self.file(folder/'seal.json'));self.references(seal)
        for key,expected in [('score',entry['score']),('train',entry['files']['train.npz']),('validation',entry['files']['validation.npz'])]:
            self.require(seal[key]==expected,'Frozen historical input changed: '+key)
        ctx=self.context(entry);prior=Path(entry['prior_directory'])
        oldseal=base.read(self.checked(entry['prior_seal']));oldconfig=base.read(self.checked(oldseal['resource_case']))
        planner=base.native.arrays(self.checked(seal['planner']));forecast=planner['forecast']
        oldplanner=base.native.arrays(self.checked(oldseal['planner']))
        bandseal=base.read(self.checked(entry['band_baseline_seal']))
        self.same('original_forecast',forecast,oldplanner['forecast'],atol=0,rtol=0)
        self.same('latest_band_forecast',forecast,base.native.arrays(self.checked(bandseal['planner']))['forecast'],atol=0,rtol=0)
        policies=single.policy_names(p);self.require(set(planner)==set(policies+['forecast']),'New action list differs')
        config=base.read(self.checked(seal['resource_case']));spec=base.read(self.checked(seal['equipment_spec']))
        self.require(spec==base.read(self.checked(entry['equipment_spec'])),'Equipment changed after preparation')
        self.equipment(entry,config,spec,forecast);bands=self.band_arrays(entry,seal)
        noise=panel.noise_parameters(entry,p)
        model=single.historical_target(prob.build_model(ctx['score'],base.native.arrays(self.checked(seal['validation'])),ctx['native'],
            meter_relative_sd=noise['meter_relative_sd'],forecast_relative_sd=noise['forecast_relative_sd']))
        self.input_records.append(self.coverage(entry,ctx,model))
        seeds=base.native.arrays(self.checked(seal['shared_initial_points']));original=base.native.arrays(self.checked(seal['original_initial_points']))
        previous_seeds=base.native.arrays(self.checked(entry['initial_points']))
        self.require(set(original)==set(previous_seeds),'Original common-pool identity differs')
        for key in original:self.same('original_seed_'+key,original[key],previous_seeds[key],atol=0,rtol=0)
        oldP={r['id']:j for j,(r,unit) in enumerate(zip(oldconfig['resources'],oldconfig['action_units'])) if unit=='kW'}
        expected_seeds={'zero':np.zeros_like(planner[policies[0]])}
        for name,value in original.items():
            mapped=np.zeros_like(expected_seeds['zero'])
            for j,(r,unit) in enumerate(zip(config['resources'],config['action_units'])):
                if unit=='kW' and r['physical_device_id'] in oldP:mapped[:,j]=value[:,oldP[r['physical_device_id']]]
            dv,bd,_=self.action(entry,config,bands,mapped)
            if dv['passed'] and bd['passed'] and not any(np.array_equal(mapped,x) for x in expected_seeds.values()):expected_seeds['mapped_'+name]=mapped
        self.require(set(seeds)==set(expected_seeds),'Unannounced/final-policy seed entered common portfolio')
        for key in seeds:self.same('mapped_seed_'+key,seeds[key],expected_seeds[key],atol=0,rtol=0)
        plans=base.read(self.checked(seal['plans']));self.require(set(plans)==set(policies),'Missing quotes')
        saved_audits=base.read(self.checked(seal['prediction_audits']));kp=p['reservation_prices']['P'];kq=p['reservation_prices']['Q']
        forecast_values={};action_values={};seed_cache={}
        def seed_check(name,mode):
            if (name,mode) not in seed_cache:
                dv,bd,shift=self.action(entry,config,bands,seeds[name]);gate=self.gates(model,forecast,forecast+shift,mode)
                seed_cache[name,mode]=(dict(gate,feasible=gate['feasible'] and dv['passed'] and bd['passed']),fees(config,seeds[name],seeds[name],kp,kq)['planned']['activation_yuan'])
            return seed_cache[name,mode]
        for name in policies:
            u=planner[name];q=plans[name];mode=name.removeprefix('single_')
            self.same('frozen_mode_action',base.native.arrays(self.file(folder/(name+'_action.npz')))['action'],u,atol=0,rtol=0)
            self.require(base.read(self.file(folder/(name+'_plan.json')))==q,'Mode receipt changed after sealing')
            dv,bd,shift=self.action(entry,config,bands,u);self.require(dv['passed'] and bd['passed'],'Executed request violates original physical/band domain')
            bill=fees(config,u,u,kp,kq)['planned'];direct=self.gates(model,forecast,forecast+shift,mode,saved_audits[mode][name])
            self.require(q['initial_points']==seal['shared_initial_points'],'Quote used a different seed portfolio')
            super().single_quote(entry,name,q,u,direct,bill,seeds,seed_check)
            quote_row=self.quote_records[-1];certificate=quote_row['certificate']
            if q.get('resource_box_checks'):
                box=independent_box_checks(model,forecast,config,entry['base_kw'],mode)
                self.require(len(box)==len(q['resource_box_checks']),'Missing box certificate rows')
                for actual,saved in zip(box,q['resource_box_checks'],strict=True):
                    self.same('independent_probability_box_upper',saved['upper'],actual['upper'],atol=1e-7)
                    self.require(actual['impossible']==saved['impossible'],'Box infeasibility flag differs')
                if q['status']=='infeasible_by_probability_box_bound':self.require(any(r['impossible'] for r in box),'No independent negative box certificate')
                quote_row['independent_resource_box_checks']=box
            encoded=bd['encoded_passed'] and all(r['margin']>=1e-8-1e-9 for r in direct['rows']) and all(g['gain']>=-1e-9 for g in direct['guards'])
            certificate.update(encoded_domain_feasible=encoded,original_domain_feasible=dv['passed'] and bd['passed'] and direct['feasible'],
                bound_scope='New device domain plus encoded original node/hour band and single-target gates; activation fee only.',
                lower_bound_valid_for_old_free_Q_domain=False)
            if not encoded and certificate.get('lower_bound_yuan') not in (None,0):
                certificate.update(raw_encoded_lower_bound_yuan=certificate['lower_bound_yuan'],lower_bound_yuan=0.,
                    relative_gap=1. if bill['activation_yuan']>0 else 0.,lower_bound_withdrawn=True)
            quote_row.update(planning_band=bd,requested_devices=dv,shared_preparation_seconds=q.get('shared_preparation_seconds'),
                total_method_seconds_with_shared_preparation=q.get('total_method_seconds_with_shared_preparation'),
                zero_direct_gate=self.gates(model,forecast,forecast,mode),
                zero_normal_direct_gate=self.gates(model,forecast,forecast,'normal_fixed'),
                historical_target_internal_correct=self.input_records[-1]['history_correct'])
            forecast_values[name]=self.evaluate(ctx,forecast+shift);action_values[name]=(dv,bd,bill)
        done=base.read(self.file(folder/'complete.json'));rows=base.read(self.checked(done['results']))
        expected={(r,n) for r in range(p['repeats']) for n in policies}
        self.require(len(rows)==len(expected) and {(r['repeat'],r['strategy']) for r in rows}==expected,'Execution denominator differs')
        zero_forecast=self.evaluate(ctx,forecast);zeros={}
        for row in rows:
            repeat,name=row['repeat'],row['strategy'];path=folder/f'execution{repeat:02d}'/name
            record=dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=name,source=str(path),original_status=row['status'])
            try:
                self.require(base.read(self.file(path/'receipt.json'))==row,'Execution ledger differs')
                if row['status']!='complete':
                    record.update(audit_status='recorded_failure',error=row.get('error'),evaluation={'native_correct':False});self.records.append(record);continue
                self.references(row);q=plans[name]
                self.require(row['quote_zero_fallback']==q['zero_fallback'] and row['quote_new_constraints_feasible']==q['new_gate_feasible'],'Executed quote flags differ')
                random=self.randomness(folder,prior,entry,p,config,oldconfig,repeat)
                obs=base.native.arrays(self.file(path/'observations.npz'));u=planner[name];delivered=u*random['eta'][None]
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0)
                self.same('shared_device_delivery',obs['delivered_action'],delivered,atol=0,rtol=0)
                dv,bd,shift=self.action(entry,config,bands,delivered);self.require(dv['passed'],'Delivered device limits fail')
                requested=action_values[name][0]
                state_json=base.read(self.checked(row['device_states']['json']))
                self.tree(state_json['requested'],requested['states'],'requested_states')
                self.tree(state_json['delivered'],dv['states'],'delivered_states')
                state_arrays=base.native.arrays(self.checked(row['device_states']['arrays']))
                for key,value in state_arrays.items():
                    reference=state_json
                    for part in key.split('__'):reference=reference[part]
                    self.same('state_archive_'+key,value,reference,atol=0,rtol=0)
                self.require(row['delivery']['device_availability_clipping'] is False and row['delivery']['pv_clipped_kwh']==0,'Unannounced availability clipping')
                self.same('true_net_P',obs['true_P'],ctx['baseline_P']+shift[:,:ctx['n']],atol=1e-13)
                self.same('true_net_Q',obs['true_Q'],ctx['baseline_Q']+shift[:,ctx['n']:],atol=1e-13)
                for label,k,sd in [('P',0,noise['pq_relative_sd']),('Q',1,noise['pq_relative_sd']),('V',2,noise['meter_relative_sd']),('root',3,noise['meter_relative_sd'])]:
                    self.same('meter_'+label,obs[label],obs['true_'+label]*(1+sd*random['noise'+str(k)]),atol=1e-13)
                self.same('squared_voltage_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                bill=fees(config,u,delivered,kp,kq)
                for when in ('planned','settled'):
                    self.costs(row[when+'_cost'],bill[when],when)
                    self.same(when+'_fee',row[when+'_fee_yuan'],bill[when]['total_yuan'],atol=1e-8)
                for field,value in [('requested_quantities',u),('delivered_quantities',delivered)]:
                    for key,expected_quantity in quantity(value,config).items():self.same(field+'_'+key,row[field][key],expected_quantity,atol=1e-9)
                evaluation,values=self.score(ctx,obs,row);evaluation['full_native_diagnostic']=previous.full_native_diagnostic(ctx,values)
                self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                if repeat not in zeros:
                    zpath=prior/f'execution{repeat:02d}'/'no_dfs';zo=base.native.arrays(self.file(zpath/'observations.npz'))
                    zr=base.read(self.file(zpath/'receipt.json'));ze,zv=self.score(ctx,zo,zr)
                    zeros[repeat]=(zo,ze,zv,zr)
                zero_obs,zero_evaluation,zero_values,zero_row=zeros[repeat]
                if not np.count_nonzero(delivered):
                    for key in ('P','Q','Y','V','root','true_P','true_Q','true_V','true_root'):
                        self.same('zero_preserves_original_'+key,obs[key],zero_obs[key],atol=0,rtol=0)
                self.reported_evidence(ctx,row['evidence'],values,zero_values,forecast_values[name],zero_forecast)
                changes=base.native.arrays(self.file(path/'pair_changes.npz'))
                for key in ('J','J_without_common','J_common','J_cross','bias','meter','observed_logodds','observed_logodds_without_common'):
                    expected_changes=[self.directed(ctx,values,i,j)[key]-self.directed(ctx,zero_values,i,j)[key] for i,j in combinations(range(len(ctx['H'])),2)]
                    self.same('pair_changes_'+key,changes[key],expected_changes,atol=1e-8)
                delta=np.c_[obs['true_P']-zero_obs['true_P'],obs['true_Q']-zero_obs['true_Q']]
                dy=obs['true_root'][:,None]**2-obs['true_V']**2-zero_obs['true_root'][:,None]**2+zero_obs['true_V']**2
                errorK=np.square(dy[None]-np.einsum('tc,knc->ktn',delta,model['means'])).mean((1,2))/model['sigma2']
                errorH=np.square(dy[None]-np.einsum('tc,knc->ktn',delta,model['H'][model['representatives']])).mean((1,2))/model['sigma2']
                for key,value in [('actual_increment_response_error_K_over_sigma2_by_representative',errorK),('actual_increment_response_error_H_over_sigma2_by_representative',errorH)]:self.same(key,row[key],value,atol=1e-9)
                ac=base.native.arrays(self.checked(row['physical_state']))
                if 'time_slice' in row['physical_state']:
                    start,end=row['physical_state']['time_slice'];length=len(ac['voltage_complex_pu'])
                    ac={k:v[start:end] if v.ndim and len(v)==length else v for k,v in ac.items()}
                electrical,branch_values,bus_values=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                cost_energy=energy_bill(ac['root_supply_kw']);self.same('root_energy_cost',row['root_energy_value_yuan'],cost_energy,atol=1e-8)
                zero_ac=base.native.arrays(self.checked(zero_row['physical_state']))
                if 'time_slice' in zero_row['physical_state']:
                    start,end=zero_row['physical_state']['time_slice'];length=len(zero_ac['voltage_complex_pu'])
                    zero_ac={k:v[start:end] if v.ndim and len(v)==length else v for k,v in zero_ac.items()}
                zero_energy=energy_bill(zero_ac['root_supply_kw'])
                self.same('baseline_root_energy_cost',row['baseline_root_energy_value_yuan'],zero_energy,atol=1e-8)
                self.same('root_energy_cost_change',row['root_energy_value_change_yuan'],cost_energy-zero_energy,atol=1e-8)
                voltage_ok=electrical['cells_below_095']+electrical['cells_above_105']==0
                devP=np.zeros_like(ctx['baseline_P']);devQ=np.zeros_like(ctx['baseline_Q'])
                for device in config['devices']:
                    node=ctx['labels'].index(device['node_label'])
                    devP[:,node]+=device['baseline_p_kw'];devQ[:,node]+=device['baseline_q_kvar']
                residualP=ctx['baseline_P']*entry['base_kw']-devP
                record.update(audit_status='passed',requested_devices=requested,delivered_devices=dv,
                    planning_band=action_values[name][1],delivered_band=bd,planned_cost=bill['planned'],settled_cost=bill['settled'],
                    requested_violations={},delivered_violations={},evaluation=evaluation,electrical=electrical,
                    branches=branch_values,buses=bus_values,voltage_screen_passed=voltage_ok,
                    baseline_root_energy_value_yuan=zero_energy,root_energy_value_yuan=cost_energy,root_energy_value_change_yuan=cost_energy-zero_energy,
                    service_plus_root_energy_change_yuan=bill['settled']['total_yuan']+cost_energy-zero_energy,
                    actual_net_baseline_decomposition=dict(negative_background_P_cells=int((residualP<0).sum()),
                        newly_negative_P_cells=int(((residualP<0)&(ctx['baseline_P']>=0)).sum()),
                        minimum_background_P_kw=float(residualP.min()),scope='Actual net profile minus declared device baseline, post-execution only; no baseline is added to the net.'),
                    recovered_and_voltage_screen_passed=bool(evaluation['native_correct'] and voltage_ok),
                    quote_new_gate_feasible=q['new_gate_feasible'],quote_zero_fallback=q['zero_fallback'],base_mode=q['mode'],is_rotation=False,
                    actual_H_increment_error_over_sigma2=errorH.tolist(),actual_K_increment_error_over_sigma2=errorK.tolist(),
                    thermal_rating_status='unavailable_not_verified',reused_action_source=row.get('reused_action_source'),baseline_source=row['baseline_source'])
            except Exception as exc:record.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(record)


def planning_boundary(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    runner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call)]
    solves=[n for n in calls if ast.unparse(n.func)=='engine.plan']
    seals=[n for n in calls if ast.unparse(n.func)=='io.write' and n.args and isinstance(n.args[0],ast.BinOp)
        and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='seal.json']
    executes=[n for n in calls if ast.unparse(n.func)=='execute']
    forbidden=[n for n in calls if ast.unparse(n.func) in ('old.context','paired.old.context','physical_response')]
    executor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    ec=[n for n in ast.walk(executor) if isinstance(n,ast.Call)]
    checks=[n for n in ec if ast.unparse(n.func)=='paired.check_refs']
    physical=[n for n in ec if ast.unparse(n.func)=='paired.old.context']
    expected=['case',"planning['score']","planning['val']",'forecast',"planning['base']"]
    passed=(len(solves)==len(seals)==len(executes)==1 and not forbidden
        and solves[0].lineno<seals[0].lineno<executes[0].lineno
        and [ast.unparse(a) for a in solves[0].args]==expected
        and len(physical)==1 and any(c.lineno<physical[0].lineno for c in checks))
    return dict(passed=passed,source=base.ref(path),scope='History/forecast-only quote; checked action seal precedes opening physical network and future observations.')


def run(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not (source/'complete.json').exists():raise ValueError('Completed run required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py')
    audit=Audit();p=base.read(audit.file(source/'protocol.json'))
    entries=base.read(audit.checked(p['inputs']));complete=base.read(audit.file(source/'complete.json'))
    audit.require(complete['source_unchanged'] and p['oracle'] is False,'Changed source or unexpected oracle')
    policies=single.policy_names(p);audit.require(p['policies']==policies,'Policy list differs')
    for item in p['sources']:
        relative=Path(item['path']).resolve().relative_to(ROOT);snapshot=source/'source_snapshot'/relative
        audit.file(snapshot,item['sha256']);audit.source_snapshots[(str(Path(item['path']).resolve()),item['sha256'])]=snapshot
        current=base.ref(item['path'])
        if current['sha256']!=item['sha256']:
            audit.post_execution_source_changes.append(dict(executed_source=item,snapshot=base.ref(snapshot),current_source=current))
    boundary=planning_boundary(source/'source_snapshot'/p['driver_relative_path'])
    audit.require(boundary['passed'],'Planning information boundary differs')
    original=base.read(audit.checked(p['source_protocol']))
    audit.require(p['seed']==original['seed'] and p['forecasts']==1,'Original paired random stream changed')
    oldentries={e['id']:e for e in base.read(audit.checked(original['inputs']))}
    for entry in entries:
        for key in ('score','native_copy','files','physical_cache','p_resources','q_resources','index','base_kw','meter'):
            audit.require(entry[key]==oldentries[entry['id']][key],'Frozen input differs: '+key)
        audit.require(not any(k.startswith('oracle') for k in entry),'Oracle information in entry')
    selected={e['id'] for e in entries};band_audit=base.read(audit.checked(p['baseline_audit']))
    grid_audit=base.read(audit.checked(p['grid_baseline_audit']))
    audit.require(band_audit['status']=='passed' and band_audit['audited_rows']==144,'Band baseline audit incomplete')
    audit.require(grid_audit['status']=='passed' and grid_audit['audited_rows']==4608,'Grid baseline audit incomplete')
    for src,target,allowed in [(band_audit,audit.band_baseline_records,set(policies)),(grid_audit,audit.zero_records,{'no_dfs'})]:
        with audit.checked(src['records']).open(encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['id'] in selected and row['repeat']<p['repeats'] and row['strategy'] in allowed:target.append(row)
        audit.require(all(r['audit_status']=='passed' for r in target),'Baseline contains unverified rows')
    expected_rows=len(entries)*len(policies)*p['repeats'];expected_zero=len(entries)*p['repeats']
    audit.require(expected_rows==p['expected_rows']==p['expected_band_baseline_rows']==len(audit.band_baseline_records),'New/band denominator differs')
    audit.require(expected_zero==p['expected_zero_baseline_rows']==len(audit.zero_records),'Zero denominator differs')
    audit.require(p['expected_quotes']==len(entries)*len(policies),'Quote denominator differs')
    success={r['id'] for r in complete.get('receipts',[])};failures={r['id'] for r in complete.get('failed_tasks',[])}
    audit.require(not(success&failures) and success|failures==selected,'Missing or duplicate task identity')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),source_protocol=base.ref(source/'protocol.json'),
        planning_boundary=boundary,expected_rows=expected_rows,scope='Independent saved-array device dynamics, unchanged net baseline, gates, full-native scores, paired noise, fees and full AC conservation; no new solve or AC.',
        environment=dict(python=sys.version,numpy=np.__version__),helpers=[base.ref(m.__file__) for m in (single,band,panel,previous,base,base.native,base.operation,prob)])
    base.write(out/'protocol.json',own)
    for entry in entries:
        if entry['id'] in failures:
            for repeat in range(p['repeats']):
                for name in policies:audit.records.append(dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=name,audit_status='recorded_failure',original_status='task_failed',evaluation={'native_correct':False}))
            continue
        try:
            casepath=source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
            metadata=base.read(audit.file(casepath/'metadata.json'))
            with audit.file(casepath/'branches.csv').open(encoding='utf-8-sig',newline='') as stream:branches=list(csv.DictReader(stream))
            with audit.file(casepath/'buses.csv').open(encoding='utf-8-sig',newline='') as stream:buses=list(csv.DictReader(stream))
            audit.case(source/'cases'/entry['id'],entry,p,metadata,branches,buses)
        except Exception as exc:audit.task_errors.append(dict(id=entry['id'],error=repr(exc),traceback=traceback.format_exc()))
        base.write(out/'progress.json',dict(rows=len(audit.records),expected=expected_rows,task_errors=len(audit.task_errors),row_errors=sum(r['audit_status']=='failed' for r in audit.records)))
    for path,sha in audit.hashes.items():
        if base.ref(path)['sha256']!=sha:audit.task_errors.append(dict(path=path,error='Input changed during audit'))
    passed=len(audit.records)==expected_rows and not audit.task_errors and all(r['audit_status'] in ('passed','recorded_failure') for r in audit.records)
    collections={'records':audit.records,'quotes':audit.quote_records,'band_baseline_records':audit.band_baseline_records,'zero_records':audit.zero_records}
    for name,rows in collections.items():
        with (out/(name+'.jsonl')).open('w',encoding='utf8') as stream:
            for row in rows:panel.json_line(stream,row)
    result=dict(status='passed' if passed else 'failed',protocol=base.ref(out/'protocol.json'),source_protocol=base.ref(source/'protocol.json'),source_inputs=p['inputs'],
        expected_rows=expected_rows,audited_rows=len(audit.records),band_baseline_rows=len(audit.band_baseline_records),zero_rows=len(audit.zero_records),
        audit_status_counts=dict(Counter(r['audit_status'] for r in audit.records)),input_coverage=audit.input_records,device_checks=audit.device_checks,task_errors=audit.task_errors,
        original_failed_tasks=complete.get('failed_tasks',[]),original_quote_exceptions=complete.get('quote_exceptions',0),original_execution_failed_rows=complete.get('failed_rows',0),
        post_execution_source_changes=audit.post_execution_source_changes,
        maximum_numeric_errors=dict(audit.maxima),maximum_electrical_errors=dict(audit.operation.maxima),counts=dict(audit.counts),
        evidence_scope='Conditional model sufficient gates and actual AC outcomes kept separate. Fixed-lambda allocated-risk box failure certifies only that sufficient QCQP.',
        thermal_scope='Declared device thermal/SOC dynamics checked; line/transformer thermal ratings unavailable.',
        input_refs=[dict(path=path,sha256=sha) for path,sha in audit.hashes.items()],
        **{name:base.ref(out/(name+'.jsonl')) for name in collections})
    base.write(out/'report.json',result);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run');parser.add_argument('--out');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(self_test())
    else:
        if not args.run or not args.out:parser.error('--run and --out are required')
        result=run(args.run,args.out)
        print({k:result[k] for k in ('status','expected_rows','audited_rows','task_errors')})
        if result['status']!='passed':raise SystemExit(1)
