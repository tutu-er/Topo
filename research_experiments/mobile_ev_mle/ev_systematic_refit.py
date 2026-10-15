"""Paired fixed-training-lines versus shared-line refitting, conditional on P/Q.

H=training, V=validation already arrived, E=new EV measurements.  Both routes
use the same known biases, diagonal V/root-PQ output model, root voltage, source
interface/box and R/X box intersected with the same X/R cone.  All P/Q readings
are direct inputs and EV must already be included.  No simulator/truth objects
are accepted.  Fits are local attained results, with feasible witnesses kept.

For S_D(theta,s)=sum of whitened squared output residuals on block D:
  fixed_total = S_H(theta_H,s_H)+min_sV S_V(theta_H,sV)+min_sE S_E(theta_H,sE)
  joint_pre   = min_theta,sH,sV (S_H+S_V)
  joint_total = min_theta,sH,sV,sE (S_H+S_V+S_E).
The adapter reports attained values and their three block contributions.  It
does not treat different nuisance dimensions as calibrated evidence by itself.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter

import numpy as np

from . import likelihood as lk
from .ev_fixed_train_fit import OUTPUT_CHANNELS, _FixedInputProblem
from .ev_rx_fit import DEFAULT_RATIO_BOUNDS, _RxMap, fit_rx_inputs


_DEFAULT_OPTIONS = dict(r_bounds=(1e-6,.1),x_bounds=(1e-6,.1),
    ratio_bounds=DEFAULT_RATIO_BOUNDS,source_p_bounds=(-.5,.5),source_q_bounds=(-.5,.5),
    voltage_bounds=(.85,1.15),multistart=2,max_nfev=250,tol=1e-10)
_PROBLEM_OPTIONS = ('r_bounds','x_bounds','source_p_bounds','source_q_bounds','voltage_bounds')
_DIAGNOSTIC_KEYS = ('objective','measurement_channels','line_parameters_fixed','P_Q_inputs_fixed',
    'P_Q_measurement_error_propagated','input_EV_already_in_total_PQ','source_node',
    'source_spatial_location_shared','line_domain','ratio_constraint_enforced','ratio_bounds',
    'ratio_constraints','shared_parameter_count','source_parameter_count','joint_parameter_count',
    'residual_count','output_normalizer','initial_loss','restarts','solver','solver_calls',
    'total_nfev','ac_evaluations','local_jacobian_rank','local_shared_rank','local_source_rank',
    'active_bound_count','ac_equation_residual','power_balance_residual','failure_kind','seconds',
    'global_optimality_certified','global_loss_lower_bound')


def _get(item,name,default=None):
    return item.get(name,default) if isinstance(item,Mapping) else getattr(item,name,default)


def _options(values):
    result=dict(_DEFAULT_OPTIONS)
    if values is not None:
        unknown=set(values)-set(result)
        if unknown:raise ValueError('unsupported fit options: '+str(sorted(unknown)))
        result.update(values)
    # A single common physical source box is intentional in this adapter.
    for name in ('source_p_bounds','source_q_bounds'):
        if np.asarray(result[name],float).shape!=(2,):
            raise ValueError(name+' must be a common scalar lower/upper box')
    return result


def concat_models(observations,models):
    """Broadcast each meter on its own block before concatenating its times."""
    observations,models=tuple(observations),tuple(models)
    if not observations or len(observations)!=len(models) or len({o.n for o in observations})!=1:
        raise ValueError('one compatible model per nonempty observation block required')
    values={}
    for obs,model in zip(observations,models):
        if not isinstance(obs,lk.Observations) or not isinstance(model,lk.MeasurementModel):
            raise TypeError('Observations and MeasurementModel required')
        if model.fit_bias or model.calibration is not None:
            raise ValueError('known calibrated biases required; no joint calibration likelihood')
    for name in ('p','q','v','master_p','master_q','source_p','source_q'):
        blocks=[getattr(model,name+'_sigma') for model in models]
        if all(value is None for value in blocks):values[name+'_sigma']=None
        elif any(value is None for value in blocks):
            raise ValueError(name+' sigma must be present consistently across blocks')
        else:values[name+'_sigma']=np.concatenate([lk._array(value,
            (obs.t,obs.n) if name in ('p','q','v') else (obs.t,),name+'_sigma')
            for obs,value in zip(observations,blocks)])
    for name in ('p','q','v'):
        values['bias_'+name]=np.concatenate([lk._array(getattr(model,'bias_'+name),
            (obs.t,obs.n),'bias_'+name) for obs,model in zip(observations,models)])
    return lk.MeasurementModel(**values)


def _fingerprint(candidate,observations,models,options):
    digest=sha256()
    digest.update(repr((candidate.name,candidate.supports,candidate.n)).encode())
    for obs,model in zip(observations,models):
        for name in ('p','q','v','root_v','master_p','master_q','source_p','source_q'):
            value=getattr(obs,name)
            digest.update(name.encode())
            if value is None:digest.update(b'None')
            else:
                array=np.asarray(value,dtype='<f8')
                digest.update(repr(array.shape).encode());digest.update(array.tobytes())
        expanded=concat_models((obs,),(model,))
        for name in expanded.__dataclass_fields__:
            value=getattr(expanded,name)
            if value is None:digest.update(b'None')
            elif isinstance(value,bool):digest.update(str(value).encode())
            else:
                array=np.asarray(value,dtype='<f8')
                digest.update(repr(array.shape).encode());digest.update(array.tobytes())
    digest.update(json.dumps(options,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode())
    return digest.hexdigest()


def _state(item):
    return {name:np.asarray(_get(item,name),float).copy()
            for name in ('r','x','source_p','source_q')}


def _join_state(r,x,items):
    return dict(r=np.asarray(r,float).copy(),x=np.asarray(x,float).copy(),
        source_p=np.concatenate([np.asarray(_get(item,'source_p'),float) for item in items]),
        source_q=np.concatenate([np.asarray(_get(item,'source_q'),float) for item in items]))


def _summary(fit):
    if fit is None:return None
    result={name:_get(fit,name) for name in ('candidate','loss','loglik','success','status',
        'message','nfev','optimality')}
    result.update({name:value.tolist() for name,value in _state(fit).items()})
    diagnostics=_get(fit,'diagnostics',{})
    result['diagnostics']={k:diagnostics[k] for k in _DIAGNOSTIC_KEYS if k in diagnostics}
    return result


def _score(candidate,observations,model,state,options,lengths,names,counter=None):
    """Rebuild output residuals at a supplied state, with no new optimization."""
    problem=_FixedInputProblem(candidate,observations,model,fixed_r=state['r'],fixed_x=state['x'],
        call_counter=counter,**{k:options[k] for k in _PROBLEM_OPTIONS})
    mapping=_RxMap(candidate.m,options['r_bounds'],options['x_bounds'],options['ratio_bounds'])
    sp=lk._array(state['source_p'],(observations.t,),'source_p')
    sq=lk._array(state['source_q'],(observations.t,),'source_q')
    vector=np.c_[sp,sq].ravel()
    source_feasible=bool(np.all(vector>=problem.lower-1e-12) and np.all(vector<=problem.upper+1e-12))
    residual=problem.evaluate(vector).reshape(observations.t,observations.n+2)
    phy=problem.cache['physical'];voltage=abs(phy['voltage'])
    physical=bool(np.min(voltage)>=options['voltage_bounds'][0] and np.max(voltage)<=options['voltage_bounds'][1])
    domain=mapping.contains(state['r'],state['x'])
    block={};start=0
    for name,length in zip(names,lengths):
        row=residual[start:start+length];start+=length
        block[name]=dict(loss=float(np.sum(row*row)),residual=row.tolist(),time_count=int(length),
            channel_loss=dict(v=float(np.sum(row[:,:observations.n]**2)),
                master_p=float(np.sum(row[:,observations.n]**2)),master_q=float(np.sum(row[:,observations.n+1]**2))))
    if start!=observations.t:raise ValueError('block lengths must partition all observations')
    loss=float(np.sum(residual*residual))
    return dict(loss=loss,loglik=-.5*(loss+problem.normalizer),normalizer=problem.normalizer,
        blocks=block,residual_count=int(residual.size),physical_voltage_feasible=physical,
        source_box_feasible=source_feasible,line_domain_feasible=domain,
        usable=bool(physical and source_feasible and domain and np.isfinite(loss)),
        ac_equation_residual=float(phy['ac_equation_residual']),
        power_balance_residual=float(phy['power_balance_residual']))


def _stage(candidate,observations,model,options,witnesses,lengths,names,seed,counter):
    began=perf_counter();feasible=[]
    for label,state in witnesses:
        try:score=_score(candidate,observations,model,state,options,lengths,names,counter)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError):continue
        if score['usable']:feasible.append((score['loss'],label,state,score))
    if not feasible:raise RuntimeError('joint stage has no legal feasible comparison witness')
    _,label,warm,warm_score=min(feasible,key=lambda item:item[0])
    fitted=None;error=None
    try:fitted=fit_rx_inputs(candidate,observations,model,initial=warm,
        seed=seed,call_counter=counter,**options)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        error=type(exc).__name__+':'+str(exc)
    selected_state=warm;selected_score=warm_score;origin='feasible_witness:'+label
    fitted_score=None;fit_success=False
    if fitted is not None:
        try:fitted_score=_score(candidate,observations,model,_state(fitted),options,lengths,names,counter)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError):pass
        if fitted_score is not None and fitted_score['usable']:
            tolerance=1e-8*(1+abs(warm_score['loss']))
            fit_success=bool(fitted.success and fitted_score['loss']<=warm_score['loss']+tolerance)
            if fitted_score['loss']<=warm_score['loss']:
                selected_state=_state(fitted);selected_score=fitted_score
                origin='local_optimizer' if fitted.success else 'failed_optimizer_attainment'
    result=dict(**selected_score,success=fit_success,optimization_failure=not fit_success,
        selected_origin=origin,state={k:v.tolist() for k,v in selected_state.items()},
        feasible_witness=warm_score,witness_origin=label,optimizer=_summary(fitted),optimizer_error=error,
        optimizer_recomputed_score=fitted_score,seconds=perf_counter()-began,
        global_optimality_certified=False,global_loss_lower_bound=None,
        loss_scope='best retained feasible attainment; upper bound on candidate profile minimum')
    return result,selected_state


@dataclass
class RefitHistory:
    """Ephemeral, action-independent prefit cache; never a simulator container."""
    fingerprint:str
    training_fit:object
    fixed_validation_fit:object
    fixed_pre:dict
    fixed_state:dict
    joint_pre:dict
    joint_state:dict
    seconds:float


def prepare_refit_history(candidate,training,validation,training_model,validation_model,*,
        training_fit=None,fit_options=None,seed=0,call_counter=None):
    """Fit/cache H+V once; supplied training fit must document cone/direct P/Q."""
    began=perf_counter();options=_options(fit_options)
    if training_fit is None:
        training_fit=fit_rx_inputs(candidate,training,training_model,seed=seed,
            call_counter=call_counter,**options)
    diagnostics=_get(training_fit,'diagnostics',{})
    if not (_get(training_fit,'success',False) and diagnostics.get('P_Q_inputs_fixed')
            and diagnostics.get('ratio_constraint_enforced')):
        raise ValueError('successful training fit with documented direct P/Q and X/R cone required')
    if _get(training_fit,'candidate',candidate.name)!=candidate.name:
        raise ValueError('training fit belongs to a different candidate')
    trained=_state(training_fit)
    train_score=_score(candidate,training,training_model,trained,options,(training.t,),('training',),call_counter)
    if not train_score['usable']:raise ValueError('training state not feasible under declared common model/domain')
    supplied_loss=float(_get(training_fit,'loss',np.nan))
    if not np.isfinite(supplied_loss) or abs(supplied_loss-train_score['loss'])>1e-6*(1+abs(supplied_loss)):
        raise ValueError('supplied training loss does not match conditional output model')
    validation_fit=fit_rx_inputs(candidate,validation,validation_model,
        initial=dict(r=trained['r'],x=trained['x']),fixed_r=trained['r'],fixed_x=trained['x'],
        seed=seed+1,call_counter=call_counter,**options)
    if not np.isfinite(validation_fit.loss):raise RuntimeError('validation source fit has no finite attainment')
    obs=lk.concat_observations(training,validation)
    model=concat_models((training,validation),(training_model,validation_model))
    fixed_state=_join_state(trained['r'],trained['x'],(training_fit,validation_fit))
    fixed_pre=_score(candidate,obs,model,fixed_state,options,(training.t,validation.t),('training','validation'),call_counter)
    fixed_pre.update(success=bool(validation_fit.success and fixed_pre['usable']),
        training_fit=_summary(training_fit),validation_fit=_summary(validation_fit))
    joint_pre,joint_state=_stage(candidate,obs,model,options,[('fixed_training_theta',fixed_state)],
        (training.t,validation.t),('training','validation'),seed+2,call_counter)
    return RefitHistory(_fingerprint(candidate,(training,validation),(training_model,validation_model),options),
        training_fit,validation_fit,fixed_pre,fixed_state,joint_pre,joint_state,perf_counter()-began)


def compare_fixed_and_joint(candidate,training,validation,future,training_model,validation_model,future_model,*,
        training_fit=None,history_cache=None,fit_options=None,seed=0,call_counter=None):
    """Compare the same realized action/data; both total SSEs count H+V+E.

    A history_cache prepared once per candidate is reusable across actions.
    Future fits are inference, after the action is fixed; no returned future
    information may be fed back into the preceding planning decision.
    """
    began=perf_counter();options=_options(fit_options);cache_reused=history_cache is not None
    history=history_cache if cache_reused else prepare_refit_history(candidate,training,validation,
        training_model,validation_model,training_fit=training_fit,fit_options=options,
        seed=seed,call_counter=call_counter)
    fingerprint=_fingerprint(candidate,(training,validation),(training_model,validation_model),options)
    if not isinstance(history,RefitHistory) or history.fingerprint!=fingerprint:
        raise ValueError('history cache differs in candidate/data/meter/domain/fit options')
    if cache_reused and training_fit is not None and any(not np.array_equal(_state(training_fit)[k],
            _state(history.training_fit)[k]) for k in ('r','x','source_p','source_q')):
        raise ValueError('supplied training fit differs from cached training state')
    trained=_state(history.training_fit);fixed_began=perf_counter()
    future_fit=fit_rx_inputs(candidate,future,future_model,
        initial=dict(r=trained['r'],x=trained['x']),fixed_r=trained['r'],fixed_x=trained['x'],
        seed=seed+3,call_counter=call_counter,**options)
    if not np.isfinite(future_fit.loss):raise RuntimeError('future source fit has no finite attainment')
    obs=lk.concat_observations(training,validation,future)
    model=concat_models((training,validation,future),(training_model,validation_model,future_model))
    lengths=(training.t,validation.t,future.t);names=('training','validation','future')
    fixed_state=_join_state(trained['r'],trained['x'],(history.training_fit,history.fixed_validation_fit,future_fit))
    fixed_post=_score(candidate,obs,model,fixed_state,options,lengths,names,call_counter)
    fixed_post.update(success=bool(history.fixed_pre['success'] and future_fit.success and fixed_post['usable']),
        future_fit=_summary(future_fit),state={k:v.tolist() for k,v in fixed_state.items()})
    fixed_seconds=perf_counter()-fixed_began
    warm_joint=_join_state(history.joint_state['r'],history.joint_state['x'],(history.joint_state,future_fit))
    joint_post,_=_stage(candidate,obs,model,options,
        [('fixed_training_theta',fixed_state),('previous_joint_theta',warm_joint)],lengths,names,
        seed+4,call_counter)
    fixed_holdout=sum(fixed_post['blocks'][k]['loss'] for k in ('validation','future'))
    normalizer=bool(abs(fixed_post['normalizer']-joint_post['normalizer'])<1e-10)
    nested=bool(joint_post['loss']<=fixed_post['loss']+1e-8*(1+abs(fixed_post['loss'])))
    return dict(candidate=candidate.name,fixed=dict(pre=history.fixed_pre,post=fixed_post,
        fixed_total=fixed_post['loss'],fixed_holdout=fixed_holdout,
        future_increment=fixed_post['blocks']['future']['loss']),
        joint=dict(pre=history.joint_pre,post=joint_post,joint_total=joint_post['loss'],
            profile_increment=joint_post['loss']-history.joint_pre['loss']),
        success=bool(fixed_post['success'] and history.joint_pre['success'] and joint_post['success']),
        checks=dict(same_observations_and_normalizer=normalizer,joint_feasible_attainment_no_worse_than_fixed=nested,
            shared_theta_across_all_joint_blocks=True,source_interface=1,PQ_direct_inputs=True,
            PQ_input_error_propagated=False,EV_already_in_total_PQ=True,
            common_ratio_bounds=list(options['ratio_bounds']),history_cache_reused=cache_reused,
            validation_enters_joint_prefit=True,validation_used_for_action_planning=False,
            global_optimality_certified=False,truth_used_in_fit=False),
        cost=dict(seconds=perf_counter()-began,history_preparation_seconds=history.seconds,
            history_cost_included_this_call=not cache_reused,fixed_future_seconds=fixed_seconds,
            joint_post_seconds=joint_post['seconds']))


__all__=['compare_fixed_and_joint','prepare_refit_history','concat_models','RefitHistory']
