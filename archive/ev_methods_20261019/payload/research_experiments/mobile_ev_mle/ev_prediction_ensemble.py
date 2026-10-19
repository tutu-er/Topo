"""History-only AC explanations and small counterfactual forecast ensembles.

Fixed-line alternatives profile all historical P/Q and the single-interface
source using the complete AC likelihood. Forecasts are three declared stress
points, not posterior draws or a statistical coverage set. Each future is
jointly fitted with the SAME history separately; alternative futures are never
concatenated. No simulation path, truth, or future measurement noise is input.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from inspect import signature
import json
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from . import likelihood as lk
from . import gradient_price_study as io
from .symmetric_ac_local import mean_and_jacobian,whiten_observations,historical_profile_geometry

CHANNELS=('p','q','v','master_p','master_q','source_p','source_q')
DEFAULTS=signature(lk.fit_candidate).parameters


def _serial(value):
    if isinstance(value,np.ndarray):return _serial(value.tolist())
    if isinstance(value,np.generic):return _serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:_serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_serial(v) for v in value]
    return value


def _observed_channels(history):
    return tuple(k for k in CHANNELS if getattr(history,k) is not None)


def _validate_history(history,hm):
    if not isinstance(history,lk.Observations) or not isinstance(hm,lk.MeasurementModel):
        raise TypeError('only historical Observations and MeasurementModel are accepted')
    if hm.fit_bias or hm.calibration is not None:
        raise ValueError('independently calibrated fixed biases are required')
    channels=_observed_channels(history)
    if not (all(k in channels for k in ('master_p','master_q')) or
            all(k in channels for k in ('source_p','source_q'))):
        raise ValueError('free single-interface source requires master or independent source P/Q')
    if np.any(np.asarray(history.root_v)<=0):
        raise ValueError('known positive root voltage is required')
    for k in channels:
        sigma=np.broadcast_to(np.asarray(getattr(hm,k+'_sigma'),float),np.asarray(getattr(history,k)).shape)
        if not np.isfinite(sigma).all() or np.any(sigma<=0):
            raise ValueError('positive finite historical sigma required: '+k)
    return channels


def _bounds(value,shape,name):
    lo=np.broadcast_to(np.asarray(value[0],float),shape).copy()
    hi=np.broadcast_to(np.asarray(value[1],float),shape).copy()
    if not np.isfinite([lo,hi]).all() or np.any(lo>=hi):
        raise ValueError('ordered finite bounds required: '+name)
    return lo,hi


def _boxes(candidate,history,options):
    shapes=dict(r=(candidate.m,),x=(candidate.m,),p=(history.t,history.n),
                q=(history.t,history.n),source=(history.t,2))
    boxes={k:_bounds(options.get(k+'_bounds',DEFAULTS[k+'_bounds'].default),s,k)
           for k,s in shapes.items()}
    if np.any(boxes['r'][0]<0) or np.any(boxes['x'][0]<0):
        raise ValueError('nonnegative line bounds are required')
    return boxes


def _state(fit):
    return np.column_stack((fit.p,fit.q,fit.source_p,fit.source_q))


def _state_bounds(boxes):
    return (np.column_stack((boxes['p'][0],boxes['q'][0],boxes['source'][0])).ravel(),
            np.column_stack((boxes['p'][1],boxes['q'][1],boxes['source'][1])).ravel())


def _feasible_point(fit,boxes,*,atol=1e-9):
    values=dict(r=np.asarray(fit.r),x=np.asarray(fit.x),p=np.asarray(fit.p),
                q=np.asarray(fit.q),source=np.column_stack((fit.source_p,fit.source_q)))
    active=[]
    for k,point in values.items():
        lo,hi=boxes[k]
        if point.shape!=lo.shape or not np.isfinite(point).all() or np.any(point<lo-atol) or np.any(point>hi+atol):
            raise ValueError('historical explanation outside original '+k+' box')
        tol=1e-7*np.maximum(hi-lo,1e-6)
        for side,bound in (('lower',lo),('upper',hi)):
            active.extend(dict(parameter=k,index=i.tolist(),side=side,value=float(point[tuple(i)]),
                bound=float(bound[tuple(i)])) for i in np.argwhere(abs(point-bound)<=tol))
    return active


def _geometry(candidate,fit,history,hm,channels):
    return mean_and_jacobian(candidate,fit.r,fit.x,fit.p,fit.q,fit.source_p,fit.source_q,
        history.root_v,hm,channels=channels)


def _fit_record(candidate,fit):
    result=io.fit_record(candidate,fit)
    return _serial(result)


def _profile_fixed_lines(candidate,history,hm,initial,r,x,boxes,*,max_nfev,tol,voltage_bounds,call_counter=None):
    """Complete nonlinear state profile at EXACTLY fixed R/X, without tiny boxes."""
    channels=_validate_history(history,hm);observed=whiten_observations(history,hm,channels)
    lo,hi=_state_bounds(boxes);n=history.n;t=history.t
    start=np.asarray(initial,float).ravel()
    if start.shape!=lo.shape or np.any(start<lo-1e-9) or np.any(start>hi+1e-9):
        raise ValueError('fixed-line state warm start is not in the original box')
    start=np.clip(start,lo,hi)
    scale=np.column_stack((np.broadcast_to(hm.p_sigma,(t,n)),
        np.broadcast_to(hm.q_sigma,(t,n)),np.full((t,2),.01))).ravel()
    cache={};calls=0
    def evaluate(v):
        nonlocal calls
        if 'x' not in cache or not np.array_equal(cache['x'],v):
            w=v.reshape(t,2*n+2);calls+=1
            if call_counter is not None:call_counter['ac_calls']+=1
            g=mean_and_jacobian(candidate,r,x,w[:,:n],w[:,n:2*n],w[:,2*n],w[:,2*n+1],
                history.root_v,hm,channels=channels)
            cache.clear();cache.update(x=v.copy(),geometry=g,residual=g['mean']-observed)
        return cache['residual']
    def jacobian(v):
        evaluate(v);return cache['geometry']['state_jacobian']
    result=least_squares(evaluate,start,jac=jacobian,bounds=(lo,hi),method='trf',
        tr_solver='exact',x_scale=scale,max_nfev=max_nfev,ftol=tol,xtol=tol,gtol=tol)
    residual=evaluate(result.x);g=cache['geometry'];phy=g['physical'];voltage=abs(phy['voltage'])
    physical=bool(np.all(voltage>=voltage_bounds[0]) and np.all(voltage<=voltage_bounds[1]))
    w=result.x.reshape(t,2*n+2);loss=float(residual@residual)
    norm=sum(float(np.log(2*np.pi*np.broadcast_to(getattr(hm,k+'_sigma'),
        np.asarray(getattr(history,k)).shape)**2).sum()) for k in channels)
    predicted=dict(phy)
    for k in ('p','q','v'):predicted[k]=np.asarray(phy[k])+np.asarray(getattr(hm,'bias_'+k))
    fit=lk.FitResult(candidate=candidate.name,r=np.asarray(r).copy(),x=np.asarray(x).copy(),
        p=w[:,:n].copy(),q=w[:,n:2*n].copy(),source_p=w[:,2*n].copy(),source_q=w[:,2*n+1].copy(),
        bias_p=np.broadcast_to(hm.bias_p,(n,)).copy(),bias_q=np.broadcast_to(hm.bias_q,(n,)).copy(),
        bias_v=np.broadcast_to(hm.bias_v,(n,)).copy(),loss=loss,loglik=-.5*(loss+norm),
        success=bool(result.success and physical and np.isfinite(loss)),status=int(result.status),
        message=str(result.message),nfev=int(result.nfev),optimality=float(result.optimality),predicted=predicted,
        diagnostics=dict(objective='complete_known_diagonal_Gaussian_fixed_lines_H_profile',
            fixed_lines=True,global_optimality_certified=False,source_node=1,source_spatial_location_shared=True,
            ac_evaluations=calls,ac_equation_residual=float(phy['ac_equation_residual']),
            power_balance_residual=float(phy['power_balance_residual']),physical_voltage_feasible=physical,
            min_voltage=float(voltage.min()),max_voltage=float(voltage.max()),bias_policy='known_calibrated',
            root_voltage_known=True))
    return fit,dict(ac_calls=calls,fit_calls=1,solver_success=bool(result.success),
        physical_voltage_feasible=physical,loss_full_SSE=loss,optimality=float(result.optimality))


def build_history_explanations(history,candidates,hfits,hm,*,fit_options=None,max_explanations=3,
        history_gate_half_sse=2.,radius_fraction=.15,max_backtracks=2,rcond=1e-10,
        distinct_scaled_distance=1e-3):
    """Return each required center plus at most two H-supported flat alternatives.

    The smallest scaled, whitened profile-information direction is only a
    proposal. Actual bounded nonlinear state profiling and AC checks decide
    retention. The gate is a computational loss budget, not a confidence set.
    """
    began=perf_counter();channels=_validate_history(history,hm);candidates=tuple(candidates)
    if not candidates or len({c.name for c in candidates})!=len(candidates):
        raise ValueError('unique required candidates are needed')
    if max_explanations not in (1,2,3) or history_gate_half_sse<0 or radius_fraction<=0 or max_backtracks<0:
        raise ValueError('use 1..3 explanations, nonnegative gate/backtracks, positive radius')
    options=dict(fit_options or {})
    if not options.get('estimate_source',True):raise ValueError('this module profiles the anchored single-interface source')
    max_nfev=int(options.get('max_nfev',150));tol=float(options.get('tol',1e-9))
    voltage_bounds=options.get('voltage_bounds',DEFAULTS['voltage_bounds'].default)
    answer=dict(success=True,explanations={},records={},diagnostics=dict(uses_truth=False,uses_future_noise=False,
        historical_support_is_confidence_set=False,source_interface=1,shared_lines_H_N=True,
        original_parameter_box_shrunk=False,fit_calls=0,ac_calls=0,failures={},candidate_diagnostics={}))
    diag=answer['diagnostics'];fits_by_name=dict(hfits)
    for candidate in candidates:
        name=candidate.name;attempts=[];diag['candidate_diagnostics'][name]=dict(attempts=attempts)
        bank=[];answer['explanations'][name]=bank
        try:
            fit=fits_by_name.get(name)
            if fit is None or not fit.success or not np.isfinite(fit.loss):
                raise ValueError('successful historical fit is required')
            boxes=_boxes(candidate,history,options);active=_feasible_point(fit,boxes)
            diag['ac_calls']+=1;g=_geometry(candidate,fit,history,hm,channels)
            observed=whiten_observations(history,hm,channels);loss=float(np.sum((g['mean']-observed)**2))
            if not np.isclose(loss,fit.loss,atol=1e-6,rtol=1e-6):
                raise ValueError('historical fit SSE does not match the complete AC meter model')
            v=abs(g['physical']['voltage'])
            if np.any(v<voltage_bounds[0]) or np.any(v>voltage_bounds[1]):
                raise ValueError('historical center outside declared voltage range')
            bank.append(dict(id=name+'__center',candidate=name,kind='attained_local_H_fit',fit=fit,
                history_gap_half_sse=0.,active_bounds=active))
            geometry=historical_profile_geometry(g['line_jacobian'],g['state_jacobian'],rcond=rcond)
            spans=np.r_[boxes['r'][1]-boxes['r'][0],boxes['x'][1]-boxes['x'][0]]
            projected=g['line_jacobian']+g['state_jacobian']@geometry['state_compensation']
            _,singular,right=np.linalg.svd(projected*spans[None,:],full_matrices=False)
            direction=spans*right[-1]
            direction/=max(np.max(abs(direction)/spans),1e-300)
            delta_state=geometry['state_compensation']@direction
            center=np.r_[fit.r,fit.x];w0=_state(fit).ravel();wlo,whi=_state_bounds(boxes)
            line_lo=np.r_[boxes['r'][0],boxes['x'][0]];line_hi=np.r_[boxes['r'][1],boxes['x'][1]]
            joint=np.r_[center,w0];lo=np.r_[line_lo,wlo];hi=np.r_[line_hi,whi]
            diag['candidate_diagnostics'][name].update(scaled_singular_values=singular.tolist(),
                line_direction=direction.tolist(),state_compensation_direction=delta_state.tolist(),
                active_bounds=active,state_rank=geometry['state_rank'],
                direction_meaning='smallest scaled local whitened profile direction; proposal, not coverage')
            for sign in (1.,-1.):
                if len(bank)>=max_explanations:break
                raw_line=sign*direction;raw_state=sign*delta_state
                raw_ray=np.r_[raw_line,raw_state]
                raw_lengths=np.r_[(hi[raw_ray>0]-joint[raw_ray>0])/raw_ray[raw_ray>0],
                                  (lo[raw_ray<0]-joint[raw_ray<0])/raw_ray[raw_ray<0]]
                raw_limit=max(0.,float(raw_lengths.min())) if len(raw_lengths) else 0.
                if raw_limit<=1e-10:
                    attempts.append(dict(sign=sign,retained=False,reason='active_boundary_blocks_original_ray',
                        max_feasible_step=raw_limit,original_smallest_profile_direction=True))
                # Projection is a NEW computational direction. Keep its
                # original blocked attempt and recompute state compensation;
                # the nonlinear profile and actual H gate decide retention.
                line_ray=raw_line.copy()
                line_tol=1e-7*np.maximum(spans,1e-6)
                outward=((center<=line_lo+line_tol)&(line_ray<0))|((center>=line_hi-line_tol)&(line_ray>0))
                line_ray[outward]=0.
                was_projected=bool(np.any(outward))
                size=float(np.max(abs(line_ray)/spans))
                if size<=1e-12:
                    attempts.append(dict(sign=sign,retained=False,reason='zero_projected_line_direction',
                        original_smallest_profile_direction=False,projected_line_coordinates=np.flatnonzero(outward).tolist()))
                    continue
                line_ray/=size;state_ray=geometry['state_compensation']@line_ray
                state_tol=1e-7*np.maximum(whi-wlo,1e-6)
                state_outward=((w0<=wlo+state_tol)&(state_ray<0))|((w0>=whi-state_tol)&(state_ray>0))
                state_ray[state_outward]=0.
                ray=np.r_[line_ray,state_ray]
                lengths=np.r_[(hi[ray>0]-joint[ray>0])/ray[ray>0],
                              (lo[ray<0]-joint[ray<0])/ray[ray<0]]
                limit=max(0.,float(lengths.min())) if len(lengths) else 0.
                radius=min(radius_fraction,.98*limit)
                if radius<=1e-10:
                    attempts.append(dict(sign=sign,retained=False,reason='active_boundary_blocks_projected_ray',max_feasible_step=limit))
                    continue
                retained=False
                for backtrack in range(int(max_backtracks)+1):
                    step=radius/(2**backtrack);theta=center+step*line_ray
                    initial=(w0+step*state_ray).reshape(history.t,-1)
                    attempt=dict(sign=sign,backtrack=backtrack,step_fraction=step,max_feasible_step=limit,
                        retained=False,fixed_lines=True,proposed_r=theta[:candidate.m].tolist(),
                        proposed_x=theta[candidate.m:].tolist(),direction_was_line_tangent_projected=was_projected,
                        original_smallest_profile_direction=not was_projected and not np.any(state_outward),
                        projected_line_coordinates=np.flatnonzero(outward).tolist(),
                        projected_state_coordinates=np.flatnonzero(state_outward).tolist(),
                        actual_line_direction=line_ray.tolist(),actual_state_warm_direction=state_ray.tolist(),
                        projected_whitened_profile_curvature=float(line_ray@geometry['precision']@line_ray))
                    attempts.append(attempt)
                    before_ac=diag['ac_calls']
                    try:
                        diag['fit_calls']+=1
                        alternative,cost=_profile_fixed_lines(candidate,history,hm,initial,theta[:candidate.m],
                            theta[candidate.m:],boxes,max_nfev=max_nfev,tol=tol,voltage_bounds=voltage_bounds,
                            call_counter=diag)
                        attempt['profile']=cost
                        attempt['fit']=_fit_record(candidate,alternative)
                        gap=.5*(alternative.loss-fit.loss)
                        distance=min(float(np.max(abs(np.r_[alternative.r-b['fit'].r,alternative.x-b['fit'].x])/spans)) for b in bank)
                        attempt.update(history_gap_half_sse=float(gap),distinct_scaled_distance=distance)
                        legal=_feasible_point(alternative,boxes)
                        supported=bool(alternative.success and gap<=history_gate_half_sse and distance>=distinct_scaled_distance)
                        if supported:
                            bank.append(dict(id=name+'__flat_'+('plus' if sign>0 else 'minus'),candidate=name,
                                kind='fixed_line_local_AC_H_profile',fit=alternative,history_gap_half_sse=float(gap),
                                active_bounds=legal,line_step_fraction=float(sign*step)))
                            attempt.update(retained=True,reason='legal_AC_history_supported_profile');retained=True;break
                        attempt['reason']='profile_optimization_or_voltage_failure' if not alternative.success else 'duplicate_explanation' if distance<distinct_scaled_distance else 'outside_history_loss_gate'
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                        attempt.update(reason='profile_exception',error=type(error).__name__+':'+str(error))
                    finally:attempt['ac_calls_spent']=diag['ac_calls']-before_ac
                if not retained:diag['candidate_diagnostics'][name]['unretained_sign_'+str(sign)]=True
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            answer['success']=False;diag['failures'][name]=type(error).__name__+':'+str(error)
        answer['records'][name]=[dict(**{k:_serial(v) for k,v in b.items() if k!='fit'},
            fit=_fit_record(candidate,b['fit'])) for b in bank]
    diag.update(seconds=perf_counter()-began,required_candidates=len(candidates),
        retained_explanations=sum(map(len,answer['explanations'].values())),history_gate_half_sse=history_gate_half_sse,
        radius_fraction=radius_fraction,max_backtracks=int(max_backtracks))
    diag['better_H_attainment_found']={name:min(x['fit'].loss for x in bank)-hfits[name].loss
        for name,bank in answer['explanations'].items() if bank and min(x['fit'].loss for x in bank)<hfits[name].loss-1e-6}
    return answer


def build_forecast_scenarios(history,hm,hfits,horizon,*,base_kw=20.,tracking_fraction=.05,
        load_floor_pu=.002,source_floor_pu=.005,shock_shape=None,mode_scale=1.):
    """Center and +/- one declared H-derived load/source/tracking stress mode.

    Component floors and tracking fraction are planning assumptions. Three
    points do not cover all directions or claim predictive probability.
    """
    _validate_history(history,hm)
    if (not np.isfinite([base_kw,tracking_fraction,load_floor_pu,source_floor_pu,mode_scale]).all()
            or int(horizon)!=horizon or horizon<1 or base_kw<=0 or not 0<=tracking_fraction<1 or min(load_floor_pu,source_floor_pu,mode_scale)<0):
        raise ValueError('positive horizon/base and declared nonnegative stress scales required')
    horizon=int(horizon);n=history.n
    shape=np.ones(horizon) if shock_shape is None else np.asarray(shock_shape,float)
    if shape.shape!=(horizon,) or not np.isfinite(shape).all() or np.max(abs(shape))>1+1e-12:
        raise ValueError('shock_shape must have horizon entries in [-1,1]')
    p=np.asarray(history.p)-np.broadcast_to(hm.bias_p,history.p.shape)
    q=np.asarray(history.q)-np.broadcast_to(hm.bias_q,history.q.shape)
    background=dict(p=np.tile(p.mean(axis=0),(horizon,1)),q=np.tile(q.mean(axis=0),(horizon,1)),
        root_v=np.full(horizon,np.mean(history.root_v)))
    offsets={};mode_records={}
    for name,fit in hfits.items():
        if not fit.success or np.shape(fit.source_p)!=(history.t,) or np.shape(fit.source_q)!=(history.t,):
            raise ValueError('all required historical source explanations must succeed')
        values=np.column_stack((p,q,fit.source_p,fit.source_q));centered=values-values.mean(axis=0)
        # One principal mode in physical pu units, then explicit component
        # floors prevent declaring zero uncertainty from a short flat history.
        _,singular,right=np.linalg.svd(centered,full_matrices=False)
        raw=(singular[0]/np.sqrt(max(history.t-1,1)))*right[0] if len(singular) else np.zeros(2*n+2)
        if np.sum(raw[:n])<0:raw=-raw
        floors=np.r_[np.full(2*n,load_floor_pu),np.full(2,source_floor_pu)]
        signs=np.where(raw<0,-1.,1.)
        mode=mode_scale*signs*np.maximum(abs(raw),floors)
        offsets[name]=dict(p=shape[:,None]*mode[:n],q=shape[:,None]*mode[n:2*n],
            source_p=shape*mode[2*n],source_q=shape*mode[2*n+1])
        mode_records[name]=dict(raw_history_mode_pu=raw.tolist(),declared_mode_pu=mode.tolist(),
            component_floor_pu=floors.tolist(),history_count=history.t)
    scenarios=[]
    for label,sign,weight in (('center',0.,.5),('minus',-1.,.25),('plus',1.,.25)):
        scenarios.append(dict(id=label,sign=sign,weight=weight,background=background,
            offsets=offsets,control_gain=np.full((horizon,n),1.+sign*tracking_fraction),
            source_interface=1,EV_in_total_PQ_once=True))
    return dict(scenarios=scenarios,diagnostics=dict(uses_truth=False,uses_future_noise=False,random_draws=0,
        finite_stress_scenarios=True,predictive_distribution_calibrated=False,statistical_coverage_claimed=False,
        scenario_weights_meaning='declared planning weights, not posterior or coverage probabilities',
        mode_method='dominant joint H variation with declared component floors',mode_records=mode_records,
        shock_shape=shape.tolist(),tracking_fraction=tracking_fraction,base_kw=base_kw,
        temporal_forecast_scope='declared shape; constant default does not identify the best actual time'))


def _explanations(value):
    if isinstance(value,dict) and 'explanations' in value:
        if not value['success']:raise ValueError('required historical explanation failure')
        return value['explanations']
    return value


def _scenarios(value):
    result=value['scenarios'] if isinstance(value,dict) and 'scenarios' in value else value
    result=tuple(result)
    if not result or len({s['id'] for s in result})!=len(result):raise ValueError('unique declared scenarios required')
    weights=np.asarray([s['weight'] for s in result],float)
    if not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1.):
        raise ValueError('scenario planning weights must be nonnegative and sum to one')
    return result


def _bundles(candidates,explanations,scenarios,axes,reference_kw,base_kw):
    banks=_explanations(explanations);scenes=_scenarios(scenarios);axes=tuple(axes);u=np.asarray(reference_kw,float)
    if not axes or len(set(axes))!=len(axes) or any(int(a)!=a or a<0 or a>2 for a in axes):
        raise ValueError('distinct explanation indices from 0..2 required')
    if u.ndim!=2 or not np.isfinite(u).all() or base_kw<=0:raise ValueError('finite reference power matrix and positive base required')
    bundles=[]
    for axis in axes:
        chosen={};fallback=[]
        for candidate in candidates:
            bank=banks.get(candidate.name,[])
            if not bank:raise ValueError('required candidate has no historical explanation: '+candidate.name)
            index=int(axis) if axis<len(bank) else 0
            chosen[candidate.name]=bank[index]
            if index!=axis:fallback.append(candidate.name)
        for scene in scenes:
            identifier='axis_'+str(axis)+'__'+scene['id'];overrides={};forecast_background={}
            gain=np.broadcast_to(np.asarray(scene['control_gain'],float),u.shape).copy()
            if not np.isfinite(gain).all() or np.any(gain<=0):raise ValueError('positive finite tracking gain required')
            for candidate in candidates:
                name=candidate.name;fit=chosen[name]['fit'];offset=scene['offsets'][name];sign=float(scene['sign'])
                background=dict(p=np.asarray(scene['background']['p'])+sign*np.asarray(offset['p']),
                    q=np.asarray(scene['background']['q'])+sign*np.asarray(offset['q']),
                    source_p=np.full(len(u),np.mean(fit.source_p))+sign*np.asarray(offset['source_p']),
                    source_q=np.full(len(u),np.mean(fit.source_q))+sign*np.asarray(offset['source_q']),
                    root_v=np.asarray(scene['background']['root_v']).copy())
                forecast_background[name]=background
                overrides[name]=dict(background,p=background['p']+gain*u/base_kw,control_gain=gain)
            bundles.append(dict(id=identifier,axis=int(axis),scenario_id=scene['id'],
                weight=float(scene['weight'])/len(axes),fits={n:b['fit'] for n,b in chosen.items()},
                explanation_ids={n:b['id'] for n,b in chosen.items()},missing_flat_fallback_candidates=fallback,
                forecast_overrides=overrides,forecast_background=forecast_background,control_gain=gain))
    # Exact de-duplication only: same chosen explanation IDs and exactly the
    # same physical background/gain. Aliases retain both missing-flat records;
    # planning weights are added, so the weighted objective is unchanged.
    unique={};result=[]
    for bundle in bundles:
        payload=dict(explanation_ids=bundle['explanation_ids'],
            forecast_background=bundle['forecast_background'],control_gain=bundle['control_gain'])
        key=hashlib.sha256(json.dumps(_serial(payload),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        alias={k:bundle[k] for k in ('id','axis','scenario_id','weight','missing_flat_fallback_candidates')}
        if key in unique:
            previous=unique[key];previous['weight']+=bundle['weight']
            previous['aliases'].append(bundle['id']);previous['alias_records'].append(alias)
        else:
            bundle.update(exact_content_sha256=key,aliases=[bundle['id']],alias_records=[alias])
            unique[key]=bundle;result.append(bundle)
    return result


def build_ensemble_contexts(history,candidates,hfits,hm,fm,reference_kw,lower,upper,*,base_kw=20.,
        line_bounds=None,state_bounds=None,explanations,scenarios,ensemble_axes=(0,1),step_kw=.001):
    """Build distinct original-domain proxies, one per bundle and future scenario."""
    from .ev_original_ac_check import build_multisite_context
    candidates=tuple(candidates);names=[c.name for c in candidates];pairs=[(a,b) for a in names for b in names if a!=b]
    out=dict(success=False,models=[],contexts=[],weights={},diagnostics=dict(ac_calls=0,failures=[],
        uses_truth=False,uses_future_noise=False,same_action_across_contexts=True,
        alternative_futures_stacked=False,source_interface=1,history_baseline='original attained H fit per candidate'))
    if len(pairs)==0:raise ValueError('at least two required candidates are needed')
    bundles=_bundles(candidates,explanations,scenarios,ensemble_axes,reference_kw,base_kw)
    for bundle in bundles:
        built=build_multisite_context(history,candidates,bundle['fits'],hm,fm,reference_kw,lower,upper,
            base_kw,line_bounds=line_bounds,state_bounds=state_bounds,step_kw=step_kw,
            forecast_overrides=bundle['forecast_overrides'])
        out['diagnostics']['ac_calls']+=built['diagnostics']['ac_calls']
        context={k:v for k,v in bundle.items() if k!='forecast_overrides'}
        context.update(success=built['success'],diagnostics=built['diagnostics'],model_names=[])
        out['contexts'].append(context)
        if not built['success']:
            out['diagnostics']['failures'].append(dict(ensemble_id=bundle['id'],failure=built['diagnostics']['failure']))
            continue
        for model in built['models']:
            base_pair=model['name'];a,b=model['a'],model['b'];model['name']=base_pair+'__'+bundle['id']
            model.update(ensemble_id=bundle['id'],base_pair=base_pair,ensemble_weight=bundle['weight']/len(pairs),
                task_weight=bundle['weight']/len(pairs),
                expansion_reference_history_cost=float(bundle['fits'][a].loss),
                expansion_competitor_history_cost=float(bundle['fits'][b].loss),
                reference_history_cost=float(hfits[a].loss),competitor_history_cost=float(hfits[b].loss),
                alternative_futures_stacked=False)
            out['models'].append(model);context['model_names'].append(model['name']);out['weights'][model['name']]=model['ensemble_weight']
    out['success']=bool(not out['diagnostics']['failures'] and len(out['models'])==len(bundles)*len(pairs))
    out['diagnostics'].update(contexts=len(bundles),base_pairs=len(pairs),expanded_pairs=len(out['models']),
        total_weight=float(sum(out['weights'].values())),missing_flat_fallbacks=[
            dict(id=b['id'],candidates=b['missing_flat_fallback_candidates']) for b in bundles if b['missing_flat_fallback_candidates']])
    out['context_records']=[_serial({k:v for k,v in b.items() if k!='fits'}) for b in out['contexts']]
    return out


def evaluate_ensemble_action(history,candidates,hfits,hm,fm,jm,power_kw,*,explanations,scenarios,
        ensemble_axes=(0,1),base_kw=20.,fit_options=None,seed=0,contexts=None):
    """Complete AC mean evaluation, separately refitting H + ONE future each time."""
    began=perf_counter();_validate_history(history,hm);candidates=tuple(candidates);power=np.asarray(power_kw,float)
    if power.ndim!=2 or power.shape[1]!=history.n or not np.isfinite(power).all() or base_kw<=0:
        raise ValueError('finite time-by-terminal command and positive power base required')
    from .gradient_price import _models_for_future
    _,expected_joint=_models_for_future(history,hm,len(power),fm)
    if jm.fit_bias or jm.calibration is not None:raise ValueError('joint meter calibration must remain fixed')
    for k in _observed_channels(history):
        shape=(history.t+len(power),history.n) if k in ('p','q','v') else (history.t+len(power),)
        if not np.array_equal(np.broadcast_to(getattr(jm,k+'_sigma'),shape),
                np.broadcast_to(getattr(expected_joint,k+'_sigma'),shape)):
            raise ValueError('joint meter must preserve the same H and one future: '+k)
    bundles=(_bundles(candidates,explanations,scenarios,ensemble_axes,power,base_kw)
             if contexts is None else contexts)
    names=[c.name for c in candidates];ordered=[(a,b) for a in names for b in names if a!=b]
    out=dict(success=False,power_kw=power,records=[],pairs=[],failures=[],fit_calls=0,fit_ac_calls=0,forecast_calls=0,
        mean_added_score=None,worst_added_score=None,uses_truth=False,uses_future_noise=False,
        future_added_noise=False,alternative_futures_stacked=False,same_action_across_contexts=True,
        global_AC_certificate=False,source_interface=1)
    for bundle_index,bundle in enumerate(bundles):
        gain=np.broadcast_to(np.asarray(bundle['control_gain']),power.shape);allfits={}
        for source_index,reference in enumerate(candidates):
            a=reference.name;fit=bundle['fits'][a];background=bundle['forecast_background'][a]
            forecast={k:np.asarray(v).copy() for k,v in background.items()}
            if any(forecast[k].shape!=(power.shape if k in ('p','q') else (len(power),))
                   for k in ('p','q','source_p','source_q','root_v')):
                raise ValueError('saved forecast horizon/site shape differs from command')
            forecast['p']=forecast['p']+gain*power/base_kw
            record=dict(ensemble_id=bundle['id'],a=a,explanation_id=bundle['explanation_ids'][a],
                forecast_state=forecast,control_gain=gain,success=False,fits=[],failures={})
            out['records'].append(record)
            try:
                out['forecast_calls']+=1
                physical=lk.ac_predict(reference,fit.r,fit.x,forecast['p'],forecast['q'],
                    forecast['source_p'],forecast['source_q'],forecast['root_v'])
                options=fit_options or {}
                for k in ('p','q','source'):
                    value=np.column_stack((forecast['source_p'],forecast['source_q'])) if k=='source' else forecast[k]
                    lo,hi=_bounds(options.get(k+'_bounds',DEFAULTS[k+'_bounds'].default),value.shape,k)
                    if np.any(value<lo-1e-9) or np.any(value>hi+1e-9):raise ValueError('forecast outside declared original '+k+' box')
                vlo,vhi=options.get('voltage_bounds',DEFAULTS['voltage_bounds'].default)
                if np.any(abs(physical['voltage'])<vlo) or np.any(abs(physical['voltage'])>vhi):
                    raise ValueError('forecast outside declared AC voltage range')
                values={}
                for k in CHANNELS:
                    if getattr(history,k) is None:values[k]=None
                    else:values[k]=np.asarray(physical[k])+(getattr(fm,'bias_'+k) if k in ('p','q','v') else 0.)
                future=lk.Observations(**values,root_v=forecast['root_v'])
                # Exactly H + this future. Independent nuisance/line refit for
                # every mutually exclusive future; no stacking across scenes.
                fitted,failed,cost=io.fit_all_candidates(candidates,lk.concat_observations(history,future),jm,
                    initial=hfits,seed=int(seed)+1009*bundle_index+97*source_index,fit_options=fit_options)
                out['fit_calls']+=cost['fit_calls'];out['fit_ac_calls']+=cost['ac_calls']
                record.update(predicted_observations=io.observations_record(future),
                    fits=[io.fit_record(c,fitted[c.name]) for c in candidates if c.name in fitted],
                    failures=failed,cost=cost,success=not failed and len(fitted)==len(candidates),
                    joint_observation_count=history.t+len(power),history_observation_count=history.t,
                    future_observation_count=len(power))
                allfits[a]=fitted
                if not record['success']:out['failures'].append(dict(ensemble_id=bundle['id'],a=a,kind='required_joint_fit_failure',failures=failed))
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                record['failures']['__forecast_or_joint__']=type(error).__name__+':'+str(error)
                out['failures'].append(dict(ensemble_id=bundle['id'],a=a,kind='forecast_or_joint_exception',error=str(error)));allfits[a]={}
        for a,b in ordered:
            own=allfits.get(a,{}).get(a);other=allfits.get(a,{}).get(b)
            success=bool(own is not None and other is not None and own.success and other.success)
            name=a+'__to__'+b+'__'+bundle['id'];weight=float(bundle['weight'])/len(ordered)
            row=dict(name=name,a=a,b=b,ensemble_id=bundle['id'],weight=weight,success=success,added=None,joint=None,self_relative=None)
            if success:row.update(added=.5*(other.loss-hfits[b].loss),joint=.5*(other.loss-hfits[a].loss),
                self_relative=.5*(other.loss-own.loss),joint_loss=float(other.loss),self_loss=float(own.loss))
            out['pairs'].append(row)
    out['success']=bool(not out['failures'] and out['pairs'] and all(r['success'] for r in out['pairs']))
    if out['success']:
        weights=np.array([r['weight'] for r in out['pairs']]);scores=np.array([r['added'] for r in out['pairs']])
        if not np.isclose(weights.sum(),1.):raise ValueError('expanded task weights do not sum to one')
        out.update(mean_added_score=float(weights@scores),worst_added_score=float(scores.min()),score=float(weights@scores))
    out.update(ac_calls=out['forecast_calls']+out['fit_ac_calls'],seconds=perf_counter()-began,
        required_fit_states=len(bundles)*len(candidates)**2,returned_fit_states=sum(len(r['fits']) for r in out['records']),
        score_meaning='declared ensemble-weighted prediction distance; not topology probability')
    return out


__all__=['build_history_explanations','build_forecast_scenarios','build_ensemble_contexts','evaluate_ensemble_action']
