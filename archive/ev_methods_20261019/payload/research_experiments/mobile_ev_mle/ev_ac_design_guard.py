"""H-only full-AC predicted-action check for a continuous MILP proposal.

The check compares attained local joint-fit scores for a proposed curve and
the same-service baseline. All reference/candidate fits are retained. This
is a model check and acceptance rule, not a global AC distance certificate
or a statistical guarantee of future topology discrimination.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from . import likelihood as lk
from . import gradient_price_study as io
from .symmetric_ac_local import baseline_from_history


def _empty_result(u, site, names):
    return dict(success=False, u=np.asarray(u, float).copy(), site=int(site),
        score=None, added_score=None, joint_score=None, self_relative_score=None,
        pairs=[], records=[], failures=[], required_candidates=names,
        required_ordered_pairs=len(names)*(len(names)-1), required_fit_states=len(names)**2,
        returned_fit_states=0, successful_fit_states=0,
        fit_calls=0, fit_ac_calls=0, forecast_calls=0, ac_calls=0,
        uses_truth=False, uses_future_noise=False, future_added_noise=False,
        source_interface_shared=True, future_total_pq_includes_EV_once=True,
        global_design_optimality_certified=False,
        score_kind='minimum attained local AC joint SSE increase relative to competitor H cost, divided by two',
        scope='H-only full-AC means and attained local joint fits; no global AC distance or future benefit certificate')


def evaluate_predicted_action(history, candidates, hfits, hm, fm, jm, u, site=0,
                              base_kw=20., fit_options=None, seed=0):
    """Forecast every reference, then fit every candidate jointly to H+forecast.

    Future terminal backgrounds are calibrated H meter means; each source
    forecast is the reference's H fitted source mean at the shared interface.
    The same line parameters explain H and N within each joint fit. Future
    observations contain no added noise. All required failures are preserved.
    """
    began=perf_counter(); candidates=tuple(candidates); u=np.asarray(u, float)
    if (u.ndim != 1 or not len(u) or not np.isfinite(u).all()
            or not isinstance(site,(int,np.integer)) or not 0 <= site < history.n
            or not np.isfinite(base_kw) or base_kw <= 0
            or not isinstance(seed,(int,np.integer))):
        raise ValueError('finite nonempty control, valid site/base and integer seed required')
    names=[c.name for c in candidates]
    if len(names)<2 or len(set(names))!=len(names) or any(c.n!=history.n for c in candidates):
        raise ValueError('at least two unique candidates matching history terminals required')
    out=_empty_result(u,site,names)
    options=dict(fit_options or {})
    out['fit_options']=options; out['seed']=int(seed); out['base_kw']=float(base_kw)
    valid_history=True
    for candidate in candidates:
        fit=hfits.get(candidate.name)
        if fit is None or not fit.success or not np.isfinite(fit.loss):
            out['failures'].append(dict(kind='required_history_fit_failure',candidate=candidate.name))
            valid_history=False
    if not valid_history:
        out['seconds']=perf_counter()-began
        return out
    try:
        baseline=baseline_from_history(history,hm,len(u))
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        out['failures'].append(dict(kind='historical_forecast_failure',message=str(error)))
        out['seconds']=perf_counter()-began
        return out
    historical_losses={name:float(hfits[name].loss) for name in names}
    out['historical_costs']=historical_losses
    out['common_background_forecast']=baseline
    all_fits={}
    for index,reference in enumerate(candidates):
        a=reference.name; center=hfits[a]
        record=dict(source=a,reference=a,u=u.copy(),site=int(site),
            predicted_observations=None,fits=[],fit_failures={},
            cost=dict(fit_calls=0,ac_calls=0,seconds=0.),forecast_calls=0,
            success=False,seed=int(seed)+1009*index)
        out['records'].append(record)
        p=baseline['p'].copy(); p[:,site]+=u/base_kw
        sp=np.full(len(u),float(np.mean(center.source_p)))
        sq=np.full(len(u),float(np.mean(center.source_q)))
        record['forecast_state']=dict(p=p,q=baseline['q'].copy(),source_p=sp,source_q=sq,
            root_v=baseline['root_v'].copy())
        try:
            out['forecast_calls']+=1; record['forecast_calls']=1
            physical=lk.ac_predict(reference,center.r,center.x,p,baseline['q'],sp,sq,baseline['root_v'])
            values={}
            for name in ('p','q','v','master_p','master_q','source_p','source_q'):
                if name not in ('p','q','v') and getattr(history,name) is None:
                    values[name]=None;continue
                value=np.asarray(physical[name],float).copy()
                if name in ('p','q','v'):
                    value+=np.asarray(getattr(fm,'bias_'+name),float)
                values[name]=value
            predicted=lk.Observations(root_v=baseline['root_v'],**values)
            record['predicted_observations']=io.observations_record(predicted)
            record['predicted_voltage_range']=[float(np.min(physical['v'])),float(np.max(physical['v']))]
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            message=type(error).__name__+':'+str(error)
            record['fit_failures']['__forecast__']=message
            out['failures'].append(dict(kind='reference_forecast_failure',source=a,message=message))
            all_fits[a]={};continue
        try:
            fitted,failed,cost=io.fit_all_candidates(candidates,lk.concat_observations(history,predicted),jm,
                initial=hfits,seed=record['seed'],fit_options=options)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            # Ordinary per-candidate fit failures are already counted and
            # returned by io. A wrapper/configuration failure stays explicit.
            fitted,failed,cost={}, {'__joint_fit__':type(error).__name__+':'+str(error)}, record['cost']
        record['cost']=cost;record['fit_failures']=dict(failed)
        out['fit_calls']+=int(cost['fit_calls']);out['fit_ac_calls']+=int(cost['ac_calls'])
        out['returned_fit_states']+=len(fitted)
        clean={}
        for candidate in candidates:
            b=candidate.name; fit=fitted.get(b)
            if fit is None:
                record['fit_failures'].setdefault(b,'required_joint_fit_missing')
                continue
            record['fits'].append(io.fit_record(candidate,fit))
            if fit.success and np.isfinite(fit.loss):
                clean[b]=fit;out['successful_fit_states']+=1
            else:
                record['fit_failures'].setdefault(b,'required_joint_fit_nonconverged')
        record['success']=bool(not record['fit_failures'] and len(clean)==len(candidates))
        if not record['success']:
            out['failures'].append(dict(kind='required_joint_fit_failure',source=a,
                candidates=record['fit_failures'].copy()))
        all_fits[a]=clean
    for reference in candidates:
        a=reference.name; own=all_fits.get(a,{}).get(a)
        if own is not None:
            record=next(r for r in out['records'] if r['source']==a)
            record['self_joint_loss']=float(own.loss)
            record['self_history_difference']=float(own.loss)-historical_losses[a]
        for competitor in candidates:
            b=competitor.name
            if a==b:continue
            fitted=all_fits.get(a,{}).get(b)
            row=dict(a=a,b=b,success=fitted is not None and own is not None,
                historical_a=historical_losses[a],historical_b=historical_losses[b],
                joint_loss=None if fitted is None else float(fitted.loss),
                self_joint_loss=None if own is None else float(own.loss),
                added=None,joint=None,self_relative=None)
            if fitted is not None:
                row['added']=.5*(float(fitted.loss)-historical_losses[b])
                row['joint']=.5*(float(fitted.loss)-historical_losses[a])
            if fitted is not None and own is not None:
                row['self_relative']=.5*(float(fitted.loss)-float(own.loss))
            out['pairs'].append(row)
    out['success']=bool(not out['failures'] and len(out['pairs'])==out['required_ordered_pairs']
                        and all(row['success'] for row in out['pairs']))
    if out['success']:
        out['added_score']=float(min(row['added'] for row in out['pairs']))
        out['joint_score']=float(min(row['joint'] for row in out['pairs']))
        out['self_relative_score']=float(min(row['self_relative'] for row in out['pairs']))
        out['score']=out['added_score']
    out['ac_calls']=out['fit_ac_calls']+out['forecast_calls']
    out['seconds']=perf_counter()-began
    return out


def guard_predicted_actions(history,candidates,hfits,hm,fm,jm,baseline_u,candidate_u,
                            site=0,base_kw=20.,fit_options=None,seed=0,gain_tolerance=1e-8):
    """Accept only a successful proposal with improved attained predicted score.

    Exactly equal injection vectors reuse one full predicted-action result.
    This cache is computational reuse, not an independent history replicate.
    A caller must separately check operation feasibility before execution.
    """
    began=perf_counter()
    baseline_u,candidate_u=np.asarray(baseline_u,float),np.asarray(candidate_u,float)
    if baseline_u.shape!=candidate_u.shape or not np.isfinite(gain_tolerance) or gain_tolerance<0:
        raise ValueError('same-shaped controls and nonnegative finite gain tolerance required')
    kwargs=dict(site=site,base_kw=base_kw,fit_options=fit_options,seed=seed)
    baseline=evaluate_predicted_action(history,candidates,hfits,hm,fm,jm,baseline_u,**kwargs)
    same=bool(np.array_equal(baseline_u,candidate_u))
    candidate=baseline if same else evaluate_predicted_action(history,candidates,hfits,hm,fm,jm,candidate_u,**kwargs)
    complete=baseline['success'] and candidate['success']
    gain=float(candidate['score']-baseline['score']) if complete else None
    accepted=bool(complete and gain>gain_tolerance)
    if not baseline['success']:reason='baseline_fit_or_forecast_failure'
    elif not candidate['success']:reason='candidate_fit_or_forecast_failure'
    elif same:reason='identical_action_cached'
    elif accepted:reason='attained_local_predicted_score_improved'
    else:reason='attained_local_predicted_score_not_improved'
    unique=[baseline] if same else [baseline,candidate]
    totals={k:sum(r[k] for r in unique) for k in ('fit_calls','fit_ac_calls','forecast_calls','ac_calls',
                'returned_fit_states','successful_fit_states')}
    return dict(selected_u=(candidate_u if accepted else baseline_u).copy(),accepted=accepted,
        baseline=baseline,candidate=candidate,baseline_score=baseline['score'],candidate_score=candidate['score'],
        score_gain=gain,gain_tolerance=float(gain_tolerance),reason=reason,cache_hit=same,
        evaluated_actions=len(unique),same_history_shared=True,totals=totals,**totals,
        seconds=perf_counter()-began,uses_truth=False,uses_future_noise=False,
        candidate_search='continuous MILP proposal; AC checks proposal and same-service baseline',
        acceptance_scope='attained local H-only predicted AC scores; no global improvement or statistical risk guarantee')
