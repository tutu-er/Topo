"""One observed-history AC objective, bounded MILP proposals and parent guards.

The full AC objective and all reference vectors are common to the four domains.
The MILP is a proposal mechanism. Its bounds never certify the nonlinear AC
objective. No simulator, truth, future observation or regime enters this file.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
from time import perf_counter
import numpy as np

from . import ev_fixed_train_design as fixed
from . import ev_mobile_v2_design as complete
from . import ev_mobile_v3_ac as geometry
from . import ev_mobile_v4_bounded as bounded
from . import ev_mobile_v4_master as master
from . import ev_mobile_v4_domain as nested
from .ev_mobile_v3_domain import tighten_operation_bounds, factor_bound_provider
from . import gradient_price_study as io
from .information_context import meter_to_record


def build_context(observed, meter, bank, fits, future_meter, forecast, *,
                  base_kw, source_p_bounds, source_q_bounds):
    """Use D as the entire observed block; no legacy held-out-data aliasing."""
    bank = tuple(bank); p, q = np.asarray(forecast['p']), np.asarray(forecast['q'])
    if (p.ndim != 2 or p.shape != q.shape or p.shape[1] != observed.n
            or not np.isfinite(p).all() or not np.isfinite(q).all()):
        raise ValueError('forecast time/site dimensions disagree')
    metadata = forecast['metadata']
    if (metadata.get('uses_truth') is not False or metadata.get('uses_future_noise') is not False
            or metadata.get('observed_history_only') is not True
            or int(metadata.get('observed_count', -1)) != observed.t):
        raise ValueError('forecast must use precisely the supplied observed D')
    if set(fits) != {b.name for b in bank} or set(forecast['source_by_candidate']) != set(fits):
        raise ValueError('every candidate must retain one reference and source forecast')
    lines, forecasts = {}, {}
    for b in bank:
        fit = fits[b.name]
        if not fit.success or any(np.asarray(getattr(fit,k)).shape!=(observed.t,)
                or not np.isfinite(getattr(fit,k)).all() for k in ('source_p','source_q')):
            raise ValueError('incomplete observed-history reference fit')
        r, x = np.asarray(fit.r).copy(), np.asarray(fit.x).copy()
        r.setflags(write=False); x.setflags(write=False)
        lines[b.name] = dict(r=r, x=x, sha256=fixed._line_id(r, x))
        source = forecast['source_by_candidate'][b.name]
        for key, bounds in (('source_p', source_p_bounds), ('source_q', source_q_bounds)):
            values = np.asarray(source[key], float)
            if values.shape != (len(p),) or not np.isfinite(values).all() or np.any(values < bounds[0]) or np.any(values > bounds[1]):
                raise ValueError('candidate source forecast outside the common physical interface domain')
        forecasts[b.name] = {key: np.asarray(source[key]).copy() for key in ('source_p','source_q')}
    return dict(success=True, bank=bank, train_fits=fits, lines=lines, forecasts=forecasts,
        background=dict(p=p.copy(), q=q.copy()), root_v=np.asarray(forecast['root_v']).copy(),
        future_meter=future_meter, control_shape=p.shape, base_kw=float(base_kw),
        source_p_bounds=tuple(source_p_bounds), source_q_bounds=tuple(source_q_bounds),
        _rebuild_inputs=dict(training=observed, train_meter=meter),
        information_scope='entire already observed D=H plus four further observations',
        forecast_metadata=io.serial(metadata), reference_count_per_candidate=1)


class CommonACObjective:
    def __init__(self, context, fit_options, *, seed):
        self.context = context
        self.engine = complete.JointPredictiveEngine(context, fit_options=fit_options, seed=seed)
        identity = dict(observed=io.observations_record(self.engine.training),
            meter=meter_to_record(self.engine.tm), lines=context['lines'], background=context['background'],
            sources=context['forecasts'], future_meter=meter_to_record(context['future_meter']),
            future_root=context['root_v'],source_interface=1,
            bank=[dict(name=b.name,n=b.n,supports=b.supports) for b in context['bank']],
            options=fit_options, seed=seed, aggregation='mean nearest signed joint half-SSE')
        self.objective_id = io.identity(io.serial(identity), {})
        self.cache = nested.ACScoreCache(self.objective_id, self.evaluate)

    def evaluate(self, power):
        result = dict(self.engine.evaluate(power))
        result.update(objective_id=self.objective_id, uses_validation=True,
            uses_observed_D=True, history_scope='same complete D as actual evaluation',
            nearest=result.get('aggregation', {}).get('nearest', []))
        return result


def reference_floor(parent, proposed, *, tolerance=1e-7, expected_references=None):
    """A deterministic acceptance test; no confidence/probability claim."""
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('finite nonnegative reference tolerance required')
    expected = (list(expected_references) if expected_references is not None
                else [r['a'] for r in parent.get('references',[])])
    if len(expected)<2 or len(expected)!=len(set(expected)):
        raise ValueError('explicit complete distinct reference names required')
    def extract(profile):
        rows=profile.get('nearest',[])
        if len(rows)!=len(expected) or {r['a'] for r in rows}!=set(expected):return None
        values={r['a']:float(r['half_gap']) for r in rows}
        return values if len(values)==len(rows) and np.isfinite(list(values.values())).all() else None
    before,after=extract(parent),extract(proposed)
    valid=bool(parent.get('success') and proposed.get('success') and before is not None and after is not None)
    slacks={a:after[a]-before[a] for a in expected} if valid else {a:None for a in expected}
    violated=[a for a in expected if not valid or slacks[a]<-tolerance*(1+abs(before[a]))]
    return dict(success=valid and not violated, complete=valid, slacks=slacks,
                violated=violated, tolerance=tolerance, floor=before,expected_references=expected)


def aligned_pairs(context, power, full_profile, fit_options, *, qp_seconds=3.,
                  parent_relative_gap=1e-5):
    """Fit physical QPs, then align each pair to the full AC value/price.

    No own-QP is needed: its attained parent value/derivative is contained in
    the full signed gap used by the affine correction. Bounds remain bounds on
    the corrected model with the saved numerical constants.
    """
    if not full_profile.get('success'):
        raise ValueError('complete parent AC score required')
    data = geometry.geometry_inputs(context, power)
    labels=[b.name for b in context['bank']]
    expected={(a,b) for a in labels for b in labels if a!=b}
    for rows in (data['pairs'],full_profile['pairs']):
        actual=[(r['a'],r['b']) for r in rows]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('every directed pair must occur exactly once')
    ac = {(r['a'],r['b']):r for r in full_profile['pairs']}
    domains = {}
    for b in context['bank']:
        fit = context['train_fits'][b.name]; fs = context['forecasts'][b.name]
        source = np.vstack((np.c_[fit.source_p,fit.source_q], np.c_[fs['source_p'],fs['source_q']]))
        domains[b.name] = bounded.physical_delta_domain(fit.r, fit.x, source,
            r_bounds=fit_options['r_bounds'], x_bounds=fit_options['x_bounds'],
            source_bounds=(np.array([fit_options['source_p_bounds'][0],fit_options['source_q_bounds'][0]]),
                           np.array([fit_options['source_p_bounds'][1],fit_options['source_q_bounds'][1]])),
            ratio_bounds=fit_options['ratio_bounds'])
    pairs, evidence = [], []
    for raw in data['pairs']:
        key = raw['a'], raw['b']; target = ac[key]
        if not target.get('success'):
            raise ValueError('failed directed parent explanation retained')
        pair = bounded.build_pair(*key, raw['residual'], raw['B'], raw['J'], power, domains[key[1]])
        solution = bounded.solve_pair(pair, power, backend='gurobi', seconds=qp_seconds)
        if (not np.isfinite(solution['inner_gap']) or solution['inner_gap']<0
                or solution['inner_gap'] > parent_relative_gap*(1+abs(solution['primal_upper']))):
            raise ValueError('parent QP primal/dual gap too wide for first-order alignment')
        alpha = float(target['half_gap']-solution['primal_upper'])
        beta = np.asarray(target['gradient_per_kw']).ravel()-solution['raw_price']
        pair = replace(pair, affine_offset=alpha, affine_control=beta.copy(),
                       affine_reference=np.asarray(power).ravel().copy())
        pairs.append(pair)
        evidence.append(dict(a=key[0],b=key[1],full_AC_half_gap=target['half_gap'],
            full_AC_price=target['gradient_per_kw'],alpha=alpha,beta=beta,
            parent_QP=solution, parent_value_alignment_tolerance=solution['inner_gap'],
            price_alignment_error_bound=solution['price_error_norm_bound'],
            input_sha256={k:sha256(np.asarray(raw[k],dtype='<f8').tobytes()).hexdigest()
                          for k in ('residual','B','J')}))
    return pairs, dict(pairs=evidence,geometry=data['diagnostics'],
        historical_scope=context['information_scope'], shared_lines=True,
        original_physical_domain=True, exact_AC_bound=False)


def run_family(context, domains, square_plans, square_costs, fit_options, *,
               guard, seed, total_seconds=300., solver_seconds=3., segments=4,
               max_exchange=2, qp_seconds=3.):
    """Run the whole four-domain policy with one cold engine and total budget."""
    began=perf_counter(); objective=CommonACObjective(context,fit_options,seed=seed)
    plans={}; logs={}; used_square=set(); prepaid=0.
    domain_began=began;domain_prepaid=0.
    def remaining():
        total=total_seconds-(perf_counter()-began)-prepaid
        quota=total_seconds/len(nested.DOMAIN_FLAGS)-(perf_counter()-domain_began)-(prepaid-domain_prepaid)
        return min(total,quota)
    for name in nested.DOMAIN_FLAGS:
        start=perf_counter(); log=dict(domain=name,guard=bool(guard),steps=[],initials=[],
            planning_failure=None, budget_seconds=total_seconds,
            domain_soft_quota_seconds=total_seconds/len(nested.DOMAIN_FLAGS))
        domain_began=start;domain_prepaid=prepaid
        logs[name]=log
        parents=[plans[p] for p in nested.PARENTS[name] if plans.get(p) is not None]
        lifted=[nested.inherit_plan(p,domains,name,objective.cache) for p in parents]
        current=max(lifted,key=lambda p:p.score) if lifted else None
        log['direct_parents']=[dict(domain=p.domain,score=p.score,action_key=p.action_key) for p in parents]
        try:
            # One physical square initializer per domain, billed to this family.
            if remaining()>0:
                if name not in used_square:
                    prepaid+=float(square_costs[name]);used_square.add(name)
                candidate=nested.score_operation(name,domains[name],square_plans[name],objective.cache,origin='square_initializer')
                floor=None if current is None else reference_floor(current.score_record,candidate.score_record,
                    expected_references=[b.name for b in context['bank']])
                take=current is None or (candidate.score>current.score+1e-9+1e-10*abs(current.score) and (not guard or floor['success']))
                log['initials'].append(dict(origin=candidate.origin,score=candidate.score,
                    action_key=candidate.action_key,reference_floor=floor,accepted=take))
                if take:current=candidate
            if current is None:
                log['planning_failure']='no complete common AC initializer';plans[name]=None;continue
            log['parent_score']=current.score;log['parent_action']=current.action_key
            if remaining()>0:
                u=domains[name].decode(current.x)['u_kw']
                pairs,details=aligned_pairs(context,u,current.score_record,fit_options,qp_seconds=qp_seconds)
                step=dict(alignment=details,accepted=False)
                log['steps'].append(step)
                left=remaining()
                if left>0:
                    operation=tighten_operation_bounds(domains[name])
                    solved=master.solve_bounded_milp(pairs,operation,
                        initial_operations={'parent':current.x},segments=segments,max_iterations=max_exchange,
                        solver_seconds=solver_seconds,total_seconds=left,qp_backend='gurobi',qp_seconds=qp_seconds,
                        factor_bounds_provider=factor_bound_provider(domains[name],operation))
                    step['master']=solved
                    proposals=solved.get('proposals',[])
                    proposal=max(proposals,key=lambda r:r['score_lower']) if proposals else None
                    if proposal is not None and remaining()>0:
                        candidate=nested.score_operation(name,domains[name],proposal['operation_x'],objective.cache,origin='aligned_bounded_MILP')
                        floor=reference_floor(current.score_record,candidate.score_record,
                            expected_references=[b.name for b in context['bank']])
                        improves=candidate.score>current.score+1e-9+1e-10*abs(current.score)
                        step.update(proposed_action=candidate.action_key,AC_score=candidate.score,
                            previous_AC_score=current.score,reference_floor=floor,mean_improves=improves,
                            raw_proxy_lower=proposal['score_lower'],raw_proxy_upper=proposal['score_upper'])
                        if improves and (not guard or floor['success']):
                            current=candidate;step['accepted']=True
                        else:step['rejection_reason']='reference_sacrifice' if improves else 'full_AC_mean_not_improved'
                    elif proposal is not None:step['failure']='budget before full AC proposal'
                else:step['failure']='budget after alignment'
            else:log['planning_failure']='total family budget before proposal; parent kept'
        except (ValueError,RuntimeError,AssertionError,FloatingPointError,np.linalg.LinAlgError) as exc:
            log['planning_failure']=type(exc).__name__+': '+str(exc)
        plans[name]=current
        if current is not None:
            parent_max=max((p.score for p in parents),default=None)
            if parent_max is not None and current.score<parent_max-1e-9:
                raise AssertionError('same planning score decreased after inheriting a legal parent')
            log.update(selected_score=current.score,selected_action=current.action_key,
                selected_origin=current.origin,parent_best_score=parent_max,
                scalar_monotonicity_verified=True,selected_reference_gaps=current.score_record['nearest'])
        log.update(seconds=perf_counter()-start,remaining_seconds=remaining())
    return dict(plans=plans,logs=logs,objective_id=objective.objective_id,
        cache=objective.cache.records(),engine_counters=objective.engine.counters(),
        actual_seconds=perf_counter()-began,square_dependency_seconds=prepaid,
        standalone_planning_seconds=perf_counter()-began+prepaid,
        budget_seconds=total_seconds,guard=bool(guard),uses_truth=False,uses_future_noise=False)
