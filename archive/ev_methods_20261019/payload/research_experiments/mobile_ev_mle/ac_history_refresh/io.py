"""Explicit observed-only inputs for a historical full-AC posterior refresh.

This module never reads an old posterior, a future fitted line, a true line,
or a future measurement while preparing a fit/forecast. Saved fixed plans
are read as decisions, not re-optimized. Evaluation data has a separate API.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .. import ev_bayes_response_ac as old
from .. import ev_bayes_response_truncation as trunc
from .. import ev_systematic_refit as refit
from ..information_context import meter_from_record
from ..likelihood import Observations, RadialCandidate, concat_observations


OBSERVATION_KEYS = ('p','q','v','root_v','master_p','master_q','source_p','source_q')
ELECTRICAL_KEYS = ('r_bounds','x_bounds','ratio_bounds','source_p_bounds',
                   'source_q_bounds','voltage_bounds')
PLAN_LABELS = ('constant','SQMD','BCMD')


class InputFailure(ValueError):
    """A saved history lacks the observed inputs required for this model."""


def _serial(value):
    if isinstance(value,np.ndarray):return _serial(value.tolist())
    if isinstance(value,np.generic):return _serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):_serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_serial(v) for v in value]
    return value


def fingerprint(value):
    return sha256(json.dumps(_serial(value),sort_keys=True,separators=(',',':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def load_record(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz'
          else path.open('r',encoding='utf-8')) as stream:
        return json.load(stream)


def save_record(path,record):
    """Create a new B record without replacing any existing evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    serialized=_serial(record)
    def nonfinite(value,prefix=''):
        if isinstance(value,np.ndarray):value=value.tolist()
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,float) and not np.isfinite(value):return [prefix]
        if isinstance(value,dict):
            return [path for key,row in value.items() for path in nonfinite(row,prefix+'/'+str(key))]
        if isinstance(value,(tuple,list)):
            return [path for key,row in enumerate(value) for path in nonfinite(row,prefix+'/'+str(key))]
        return []
    invalid=nonfinite(record)
    if invalid:
        serialized['serialization_nonfinite']=dict(count=len(invalid),paths=invalid[:100],
            replacement=None,scope='nonfinite diagnostics retained as explicit nulls; status/failure unchanged')
    encoded=json.dumps(serialized,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
    if path.suffix=='.gz':
        with path.open('xb') as target:
            with gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as stream:
                stream.write(encoded.encode('utf-8'))
    else:
        with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(encoded)


def _observation(record,label):
    if not isinstance(record,dict) or set(record)!=set(OBSERVATION_KEYS):
        raise InputFailure(label+' requires the exact saved observation schema')
    if record['source_p'] is not None or record['source_q'] is not None:
        raise InputFailure('this refresh keeps the original unmetered P/Q source model')
    result=Observations(**{key:None if record[key] is None else np.asarray(record[key],float)
                           for key in OBSERVATION_KEYS})
    if result.master_p is None or result.master_q is None:
        raise InputFailure(label+' requires both master P and master Q')
    return result


def _public_options(raw):
    config=raw.get('config',{})
    supplied=config.get('posterior_options',{})
    # Only options belonging to the original probabilistic model are copied.
    # Optimizer tolerances in refit_options do not change that model.
    options={key:deepcopy(supplied[key]) for key in old.DEFAULT_OPTIONS if key in supplied}
    public=raw.get('refit_options',{})
    if not all(key in public for key in ('r_bounds','x_bounds','ratio_bounds')):
        public=raw.get('refit_history',{}).get('fit_options',{})
    if not all(key in public for key in ('r_bounds','x_bounds','ratio_bounds')):
        raise InputFailure('public original R/X and ratio bounds are required')
    options.update({key:deepcopy(public[key]) for key in ELECTRICAL_KEYS if key in public})
    seed=raw.get('spec',{}).get('seed',0)
    options['seed']=int(np.random.SeedSequence([int(seed),5000,20261017]).generate_state(1)[0])
    return old._options(options)


def _latent_from_readings(observations,meter):
    p=observations.p-np.broadcast_to(np.asarray(meter.bias_p,float),observations.p.shape)
    q=observations.q-np.broadcast_to(np.asarray(meter.bias_q,float),observations.q.shape)
    # Master minus measured terminal totals contains losses. This is ONLY a
    # starting value, never a source observation or a process-prior mean.
    return np.c_[p,q,observations.master_p-p.sum(1),observations.master_q-q.sum(1)]


def _initial(candidate,history,meter,training,train_meter,line_prior,raw):
    T,n=history.t,history.n;diagnostics=[]
    def accept(state,label,training_only=False):
        if not isinstance(state,dict):raise ValueError('missing state')
        r,x=(np.asarray(state[key],float) for key in ('r','x'))
        if r.shape!=(candidate.m,) or x.shape!=r.shape or np.any(r<=0) or np.any(x<=0):
            raise ValueError('invalid R/X shape or sign')
        eta=np.empty(2*candidate.m);eta[0::2]=np.log(r);eta[1::2]=np.log(x/r)
        if not np.isfinite(eta).all() or not bool(trunc.contains_eta(eta,line_prior['bounds'],tolerance=1e-7)):
            raise ValueError('initial R/X lies outside the original public domain')
        latent=_latent_from_readings(history,meter)
        count=training.t if training_only else T
        for key,column in (('source_p',-2),('source_q',-1)):
            value=np.asarray(state[key],float)
            if value.shape!=(count,) or not np.isfinite(value).all():
                raise ValueError('historical source state length mismatch')
            latent[:count,column]=value
        if training_only:
            for key,columns in (('p',slice(0,n)),('q',slice(n,2*n))):
                if key in state:
                    value=np.asarray(state[key],float)
                    if value.shape!=(count,n) or not np.isfinite(value).all():
                        raise ValueError('training P/Q initial shape mismatch')
                    latent[:count,columns]=value
        return dict(status='ok',eta=eta,latent=latent,source=label,
            source_proxy_initialization_only=True,source_proxy_in_likelihood=False,
            uses_future_data=False,rejected_initializations=diagnostics)
    fitted=raw.get('refit_history',{})
    if fitted.get('observed_history_only') is True:
        row=fitted.get('candidates',{}).get(candidate.name,{})
        if row.get('usable',row.get('success',False)):
            try:return accept(row.get('state'),'observed_H_plus_V_conditional_fit')
            except (KeyError,ValueError,TypeError) as exc:diagnostics.append(str(exc))
    for row in raw.get('training_result',{}).get('fits',[]):
        if row.get('candidate',row.get('name'))!=candidate.name or not row.get('success',False):continue
        try:return accept(row,'training_only_fit_with_observed_evidence_proxy',True)
        except (KeyError,ValueError,TypeError) as exc:diagnostics.append(str(exc))
    return dict(status='initialization_failure',uses_future_data=False,
        rejected_initializations=diagnostics,reason='no valid history-only or training-only initial state')


def prepare_history(raw,*,base_kw=20.,future_root_v=1.,plan_labels=PLAN_LABELS):
    """Return inputs for MAP/forecast, with no labels or future data container.

    `base_kw` and `future_root_v` are explicit public task rules. The original
    process hyperparameters are reconstructed from training readings alone;
    no successful A context/posterior is needed.
    """
    required=('observed_training','observed_evidence','train_meter','evidence_meter',
              'future_meter','candidate_names','candidate_structures')
    missing=[key for key in required if key not in raw]
    if missing:raise InputFailure('missing observed inputs: '+','.join(missing))
    training=_observation(raw['observed_training'],'training')
    evidence=_observation(raw['observed_evidence'],'evidence')
    if training.n!=evidence.n:raise InputFailure('history terminal dimensions disagree')
    tm,em,fm=(meter_from_record(raw[key]) for key in ('train_meter','evidence_meter','future_meter'))
    history=concat_observations(training,evidence)
    dm=refit.concat_models((training,evidence),(tm,em))
    names=list(raw['candidate_names'])
    if not names or len(set(names))!=len(names):raise InputFailure('unique complete candidate names required')
    structures=raw['candidate_structures']
    if len(structures)!=len(names) or {row['name'] for row in structures}!=set(names):
        raise InputFailure('candidate structures must exactly match the complete bank')
    lookup={row['name']:row for row in structures};candidates=[]
    for name in names:
        row=lookup[name]
        if int(row['n'])!=history.n:raise InputFailure('candidate terminal dimension mismatch')
        candidates.append(RadialCandidate.from_supports(name,history.n,row['supports']))
    horizon=int(raw.get('config',{}).get('horizon',0))
    if horizon<1:raise InputFailure('public positive horizon required')
    if not np.isfinite(base_kw) or base_kw<=0:raise InputFailure('positive public power base required')
    root=np.broadcast_to(np.asarray(future_root_v,float),(horizon,)).copy()
    if not np.isfinite(root).all() or np.any(root<=0):raise InputFailure('positive public future root required')
    options=_public_options(raw)
    hyper=old.estimate_process_hyperparameters(training,tm,start_time=0,options=options)
    process_mean,K=old.process_kernel(hyper,history.t+horizon,start_time=0)
    if np.any(process_mean!=0):raise InputFailure('the original process prior mean must remain zero')
    width=2*history.n+2;split=history.t*width
    line_priors={c.name:trunc.line_prior(c.m,options['r_bounds'],options['x_bounds'],
                                       options['ratio_bounds']) for c in candidates}
    initial={c.name:_initial(c,history,dm,training,tm,line_priors[c.name],raw) for c in candidates}
    plans={};plan_failures={}
    for label in plan_labels:
        row=raw.get('plans',{}).get(label)
        if row is None:plan_failures[label]='saved_plan_missing';continue
        try:
            power=np.asarray(row['power'],float)
            if power.shape!=(horizon,history.n) or not np.isfinite(power).all():
                raise ValueError('finite T by n fixed power required')
            key=str(row['action_key'])
            plans[label]=dict(action_key=key,domain=str(row['domain']),power=power.copy())
        except (KeyError,ValueError,TypeError) as exc:plan_failures[label]=str(exc)
    identity=dict(spec=deepcopy(raw.get('spec',{})),candidate_names=names,
        candidate_structures=structures,initializations=initial,
        observed_training={k:getattr(training,k) for k in OBSERVATION_KEYS},
        observed_evidence={k:getattr(evidence,k) for k in OBSERVATION_KEYS},
        train_meter=raw['train_meter'],evidence_meter=raw['evidence_meter'],
        future_meter=raw['future_meter'],options=options,base_kw=float(base_kw),future_root_v=root,
        horizon=horizon,plans=plans)
    return dict(training=training,evidence=evidence,history=history,train_meter=tm,
        evidence_meter=em,history_meter=dm,future_meter=fm,candidates=tuple(candidates),
        candidate_names=names,options=options,hyperparameters=hyper,
        K_D=K[:split,:split],K_FD=K[split:,:split],K_FF=K[split:,split:],
        line_priors=line_priors,initializations=initial,plans=plans,
        requested_plan_labels=tuple(plan_labels),plan_failures=plan_failures,
        horizon=horizon,n=history.n,base_kw=float(base_kw),future_root_v=root,root_v=root.copy(),
        spec=deepcopy(raw.get('spec',{})),observed_input_sha256=fingerprint(identity),
        scope='observed H+V, original public prior and training-only process, saved fixed plans',
        old_posterior_required=False,old_topology_weights_used=False,
        source_interface=1,unit_PF_EV=True)


def evaluation_payload(raw,prepared):
    """Read saved future measurements only after predictions have been fixed.

    True labels are used exclusively by the downstream scorer. Future joint
    refits, true R/X, future latent inputs, and standardized noise are ignored.
    """
    names=prepared['candidate_names'];true_name=raw.get('true_name')
    true_index=names.index(true_name) if true_name in names else None
    supplied=raw.get('true_index')
    if true_index is not None and supplied is not None and int(supplied)!=true_index:
        raise InputFailure('saved true index/name mismatch')
    expected=prepared['horizon']*(3*prepared['n']+2);actions={}
    for key in dict.fromkeys(plan['action_key'] for plan in prepared['plans'].values()):
        row=raw.get('actions',{}).get(key);regimes={}
        if row is not None:
            for regime in ('regular','shift'):
                actual=row.get('regimes',{}).get(regime)
                if actual is None:continue
                values=np.asarray(actual['values'],float);clean=np.asarray(actual['clean'],float)
                if values.ndim!=2 or values.shape[0]<1 or values.shape[1]!=expected or clean.shape!=(expected,) or not np.isfinite(values).all() or not np.isfinite(clean).all():
                    raise InputFailure('invalid saved future measurement shape')
                regimes[regime]=dict(values=values.copy(),clean=clean.copy())
        actions[key]=regimes
    return dict(true_name=true_name,true_index=true_index,candidate_miss=true_index is None,
        actions=actions,scope='saved measurements and labels, evaluation only; no new samples')
