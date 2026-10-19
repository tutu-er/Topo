"""Complete AC both-model profiled margins and envelope prices.

The constructor accepts observed-only objects.  No truth, future arrays, or
simulation case is accepted.  R/X are retained as physical response scenarios;
both candidate costs refit shared lines and same-interface source P/Q.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np

from .. import likelihood as lk
from .. import ev_ac_repair_refit as refit
from .. import ev_mobile_v4_cases as forecasts
from ..ev_fixed_train_fit import _FixedInputProblem
from ..ev_rx_fit import _RxMap, fit_rx_inputs
from .config import CONFIG
from .block_adapter import build_block_adapter


def _state(fit):
    return {name: np.asarray(getattr(fit, name), float).copy()
            for name in ('r', 'x', 'source_p', 'source_q')}


def _problem_options(options):
    return {k: options[k] for k in ('r_bounds','x_bounds','source_p_bounds',
                                   'source_q_bounds','voltage_bounds')}


def _output(physical):
    return np.column_stack((physical['v'], physical['master_p'], physical['master_q']))


def _sigmas(meter, t, n):
    return np.column_stack((lk._array(meter.v_sigma, (t,n), 'v_sigma'),
        lk._array(meter.master_p_sigma, (t,), 'master_p_sigma'),
        lk._array(meter.master_q_sigma, (t,), 'master_q_sigma')))


def _power_derivative(candidate, physical, base_kw):
    d = lk._response_derivatives(candidate, physical, True)
    columns = slice(2*candidate.m, 2*candidate.m+candidate.n)
    return np.concatenate((d['v'][:,:,columns], d['master_p'][:,None,columns],
                           d['master_q'][:,None,columns]), axis=1)/base_kw


def _future_observation(physical, meter):
    t, n = physical['p'].shape
    return lk.Observations(p=physical['p']+lk._array(meter.bias_p,(t,n),'bias_p'),
        q=physical['q']+lk._array(meter.bias_q,(t,n),'bias_q'),
        v=physical['v']+lk._array(meter.bias_v,(t,n),'bias_v'),
        root_v=physical['root_v'], master_p=physical['master_p'], master_q=physical['master_q'])


def _weak_scenarios(candidate, D, dm, main, options, config, seed):
    """Profile source at legal weak physical line perturbations, retaining misses."""
    began=perf_counter(); counters={}; attempts=[]
    states=[dict(label='main', state=_state(main), history_loss=float(main.loss))]
    base=_FixedInputProblem(candidate,D,dm,initial=main,call_counter=counters,**_problem_options(options))
    vector=np.r_[main.r,main.x,np.column_stack((main.source_p,main.source_q)).ravel()]
    J=base.jacobian(vector); lines=J[:,:2*candidate.m]; source=J[:,2*candidate.m:]
    left, singular, _=np.linalg.svd(source,full_matrices=False)
    rank=int(np.count_nonzero(singular>1e-10*singular[0])) if len(singular) else 0
    span=np.r_[np.broadcast_to(np.asarray(options['r_bounds'][1])-np.asarray(options['r_bounds'][0]),(candidate.m,)),
               np.broadcast_to(np.asarray(options['x_bounds'][1])-np.asarray(options['x_bounds'][0]),(candidate.m,))]
    projected=(lines-left[:,:rank]@(left[:,:rank].T@lines))*span[None,:]
    _, values, right=np.linalg.svd(projected,full_matrices=False)
    mapping=_RxMap(candidate.m,options['r_bounds'],options['x_bounds'],options['ratio_bounds'])
    center=np.r_[main.r, main.x]
    target=int(config['max_alternatives_per_candidate'])
    for di, direction in enumerate(right[::-1][:int(config['weak_directions'])]):
        for sign in (-1.,1.):
            if len(states)-1>=target: break
            for backtrack in range(int(config['support_backtracks'])):
                fraction=config['weak_direction_fraction']/(2**backtrack)
                raw=center+sign*fraction*span*direction/max(abs(direction).max(),1e-15)
                rr, alpha=mapping.inverse(raw[:candidate.m],raw[candidate.m:])
                rr,xx,_,_=mapping.forward(rr,alpha)
                change=float(np.max(abs(np.r_[rr,xx]-center)/span))
                event=dict(direction=di,sign=sign,fraction=fraction,relative_change=change)
                if change<1e-6:
                    event['failure']='projection_collapsed_to_center';attempts.append(event);break
                try:
                    fit=fit_rx_inputs(candidate,D,dm,initial=main,fixed_r=rr,fixed_x=xx,
                        seed=seed+len(attempts),call_counter=counters,**dict(options,multistart=1))
                    event.update(loss=float(fit.loss),success=bool(fit.success),
                                 support_increase=float(fit.loss-main.loss))
                    accepted=bool(fit.success and mapping.contains(fit.r,fit.x)
                        and fit.loss<=main.loss+config['history_support_sse'])
                    event['accepted']=accepted
                    if accepted:
                        states.append(dict(label='weak_'+str(di)+'_'+str(int(sign)),
                            state=_state(fit), history_loss=float(fit.loss)))
                        attempts.append(event);break
                except Exception as exc:
                    event['failure']=type(exc).__name__+': '+str(exc)
                attempts.append(event)
        if len(states)-1>=target:break
    return states,dict(attempts=attempts,retained=len(states),
        alternative_support_available=len(states)>1, counters=counters,
        scaled_profile_line_singular_values=values,seconds=perf_counter()-began,
        coverage_claim=False, source_refitted_at_each_line=True)


class MarginObjective:
    def __init__(self, bank, D, dm, fm, historyfits, config=None, seed=0):
        self.bank=tuple(bank);self.D=D;self.dm=dm;self.fm=fm
        if len(self.bank)<2:raise ValueError('at least two candidate models required')
        self.history=historyfits;self.config=dict(CONFIG,**(config or {}));self.seed=int(seed)
        if not historyfits.complete:raise ValueError('all history candidates must be retained and fitted')
        if tuple(c.name for c in self.bank)!=historyfits.candidate_names:
            raise ValueError('candidate bank and history cache differ')
        self.options=dict(historyfits.options);self.cache={};self.calls=0;self.seconds=0.;self.counters={}
        self.blocks=build_block_adapter(self.bank)
        self.pairs={(p['reference'],p['competitor']) for p in self.blocks['directed_pairs']}
        self.scenarios=[];self.support={};self.setup_seconds=0.
        began=perf_counter()
        for i,c in enumerate(self.bank):
            states,details=_weak_scenarios(c,D,dm,historyfits.fits[c.name],self.options,
                                          self.config,self.seed+1000*i)
            self.support[c.name]=details
            for s in states:
                # Forecast is tied to this historical source/line explanation.
                f=forecasts.forecast_for(D,dm,{c.name:dict(s['state'],success=True)},
                    T=self.config['horizon'],base_kw=self.config['base_kw'])
                self.scenarios.append(dict(candidate=c,label=s['label'],state=s['state'],
                    history_loss=s['history_loss'],forecast=f))
        self.setup_seconds=perf_counter()-began

    def scenario_record(self):
        return dict(blocks=self.blocks,support=self.support,scenarios=[dict(reference=s['candidate'].name,
            label=s['label'],state=s['state'],history_loss=s['history_loss'],
            forecast=s['forecast']) for s in self.scenarios],
            observed_only=True, uncertainty='legal historical weak line directions with refitted source',
            no_calibrated_coverage=True, setup_seconds=self.setup_seconds,
            future_load_scenarios=False, future_source_forecast='history-only harmonic regression')

    def _evaluate_scenario(self, power, si, scenario, sigma):
        a=scenario['candidate'];f=scenario['forecast'];state=scenario['state']
        src=f['source_by_candidate'][a.name]
        self.counters['direct_ac_calls']=self.counters.get('direct_ac_calls',0)+1
        physical=lk.ac_predict(a,state['r'],state['x'],
            f['p']+power/self.config['base_kw'],f['q'],src['source_p'],src['source_q'],f['root_v'])
        voltage=abs(physical['voltage'])
        if voltage.min()<self.options['voltage_bounds'][0] or voltage.max()>self.options['voltage_bounds'][1]:
            raise RuntimeError('reference forecast AC violates declared voltage bounds')
        future=_future_observation(physical,self.fm)
        result=refit.evaluate_refits(self.bank,self.D,self.dm,future,self.fm,self.history,
            self.options,self.seed+100000+si*1000)
        for k,v in result['counters'].items():self.counters[k]=self.counters.get(k,0)+v
        entry=dict(reference=a.name,label=scenario['label'],refits=result)
        if not result['complete']:
            return [],entry,result['failures']
        refmean=_output(physical);dy=_power_derivative(a,physical,self.config['base_kw'])
        losses={};parts={}
        for candidate in self.bank:
            retained=result['comparisons'][candidate.name]['joint'];st=retained['state']
            self.counters['direct_ac_calls']=self.counters.get('direct_ac_calls',0)+1
            fitted=lk.ac_predict(candidate,np.asarray(st['r']),np.asarray(st['x']),
                f['p']+power/self.config['base_kw'],f['q'],
                np.asarray(st['source_p'])[-len(power):],np.asarray(st['source_q'])[-len(power):],f['root_v'])
            self.counters['direct_ac_calls']+=1
            past=lk.ac_predict(candidate,np.asarray(st['r']),np.asarray(st['x']),
                self.D.p-lk._array(self.dm.bias_p,self.D.p.shape,'bias_p'),
                self.D.q-lk._array(self.dm.bias_q,self.D.q.shape,'bias_q'),
                np.asarray(st['source_p'])[:self.D.t],np.asarray(st['source_q'])[:self.D.t],self.D.root_v)
            vmin=min(abs(fitted['voltage']).min(),abs(past['voltage']).min())
            vmax=max(abs(fitted['voltage']).max(),abs(past['voltage']).max())
            if (vmin<=self.options['voltage_bounds'][0]+1e-8
                or vmax>=self.options['voltage_bounds'][1]-1e-8):
                raise RuntimeError('active voltage boundary requires a constraint multiplier price')
            e=refmean-_output(fitted)
            dm=_power_derivative(candidate,fitted,self.config['base_kw'])
            parts[candidate.name]=np.einsum('to,toj->tj',e/sigma**2,dy-dm)
            losses[candidate.name]=float(retained['loss'])
        planes=[dict(pair=(a.name,b.name),scenario=si,
            value=.5*(losses[b.name]-losses[a.name]),gradient=parts[b.name]-parts[a.name])
            for b in self.bank if (a.name,b.name) in self.pairs]
        entry['joint_losses']=losses
        return planes,entry,None

    def get(self,u):
        power=np.asarray(u,float)
        if power.shape!=(self.config['horizon'],self.D.n) or not np.isfinite(power).all():
            raise ValueError('finite [horizon,terminals] EV power required')
        key=sha256(power.astype('<f8').tobytes()).hexdigest()
        if key in self.cache:return self.cache[key]
        began=perf_counter();planes=[];records=[];failures={};sigma=_sigmas(self.fm,len(power),self.D.n)
        for si,scenario in enumerate(self.scenarios):
            try:
                scenario_planes,entry,failure=self._evaluate_scenario(power,si,scenario,sigma)
                planes.extend(scenario_planes);records.append(entry)
                if failure is not None:failures[str(si)]=failure
            except Exception as exc:
                failure=type(exc).__name__+': '+str(exc)
                failures[str(si)]=failure
                records.append(dict(reference=scenario['candidate'].name,
                    label=scenario['label'],failure=failure))
        values={}
        for p in planes:
            pair=tuple(p['pair']);values[pair]=min(values.get(pair,np.inf),p['value'])
        expected=len(self.pairs)
        complete=not failures and len(values)==expected
        score=float(np.mean([min(v,self.config['margin_cap']) for v in values.values()])) if complete else None
        record=dict(complete=complete,score=score,planes=planes,pair_margins=[dict(pair=k,value=v) for k,v in values.items()],
            failures=failures,scenario_fits=records,power_kw=power.copy(),seconds=perf_counter()-began,
            global_optimality_certified=False,gradient_scope='envelope at retained local inner fit',
            score_is_probability=False)
        self.calls+=1;self.seconds+=record['seconds'];self.cache[key]=record
        return record


__all__=['MarginObjective']
