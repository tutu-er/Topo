"""Source-profile gradient MILPs with complete AC acceptance.

Every topology retains its one training R/X vector.  A linear MILP chooses
continuous powers and the battery's discrete path; complete curves/routes are
never enumerated.  Local source fits, MILP bounds, and nonlinear acceptance
scores have separate records.  No validation/truth/noise input is accepted.
"""
from __future__ import annotations

from time import perf_counter
from hashlib import sha256

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from .ev_fixed_gradient import profile_pair
from .ev_fixed_response import FixedResponse
from . import ev_fixed_train_design as fixed
from .ev_original_domain_milp import _solve_mip


DEFAULT_OPTIONS = dict(max_starts=2, max_iterations=2,
    trust_kw=2*fixed.CONFIG['command_power_kw'],
    solver_seconds=2., total_seconds=35., backend='gurobi',
    acceptance_atol=1e-9, acceptance_rtol=1e-10, gradient_zero_tol=1e-12)


def source_pair_seed(seed, pair_name):
    """A retained pair keeps its source restarts when other models are removed."""
    return (int(seed)+int.from_bytes(sha256(pair_name.encode()).digest()[:4],'little'))%(2**32)


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _aggregate(pairs, aggregation):
    """Retain all pairs; provide all active nearest branches for tie handling."""
    if not pairs or not all(p.get('success') for p in pairs):
        return dict(success=False, score=None, failure='a required source profile failed')
    if not all(np.isfinite(p['half_SSE']) and np.isfinite(p['gradient_per_kw']).all()
               for p in pairs):
        return dict(success=False, score=None, failure='a required profile value/gradient is nonfinite')
    if aggregation not in ('nearest', 'mean', 'binarymean'):
        raise ValueError('aggregation must be nearest, mean, or binarymean')
    groups = sorted({p['a'] for p in pairs})
    if aggregation == 'binarymean' and len(groups) != 2:
        raise ValueError('binarymean requires exactly two topologies')
    if aggregation in ('mean', 'binarymean'):
        grad = np.mean([p['gradient_per_kw'] for p in pairs], axis=0)
        return dict(success=True, score=float(np.mean([p['half_SSE'] for p in pairs])),
            gradient_per_kw=grad, active_groups=[[dict(a='all', b='mean',
                gradient_per_kw=grad)]], nearest=[], aggregation=aggregation)
    active_groups, nearest, scores = [], [], []
    for a in groups:
        rows = [p for p in pairs if p['a'] == a]
        score = min(p['half_SSE'] for p in rows)
        active = [p for p in rows if np.isclose(p['half_SSE'], score,
            atol=1e-10, rtol=1e-10)]
        active_groups.append(active)
        nearest.append(dict(a=a, value=float(score), active=[p['b'] for p in active]))
        scores.append(score)
    return dict(success=True, score=float(np.mean(scores)), active_groups=active_groups,
        # This average is an ordinary gradient only if active branch gradients
        # agree. The MILP below uses every tied branch, not this summary vector.
        gradient_per_kw=np.mean([np.mean([p['gradient_per_kw'] for p in active],
            axis=0) for active in active_groups], axis=0), nearest=nearest,
        aggregation=aggregation,
        derivative_scope='directional derivative is mean of active-branch minima')


class ProfileEngine:
    """Cached all-pair source profiles and gradients at immutable training lines."""
    def __init__(self, context, *, fit_options=None, seed=0, cache=None):
        if not context.get('success'):
            raise ValueError('a successful fixed-training context is required')
        self.context, self.seed = context, int(seed)
        self.cache = {} if cache is None else cache
        self.fit_options = dict(multistart=2, max_nfev=250, tol=1e-10)
        supplied = dict(fit_options or {})
        for key in ('source_p_bounds', 'source_q_bounds'):
            if key in supplied and tuple(supplied.pop(key)) != tuple(context[key]):
                raise ValueError('source bounds must equal the fixed context')
        voltage = supplied.pop('voltage_bounds', (.85, 1.15))
        if tuple(voltage) != (.85, 1.15):
            raise ValueError('gradient profile voltage audit is fixed at [.85,1.15]')
        if set(supplied)-set(self.fit_options):
            raise ValueError('unsupported profile options '+repr(sorted(set(supplied)-set(self.fit_options))))
        self.fit_options.update(supplied)
        self.response = {}
        self.profile_evaluations = self.cache_hits = 0
        self.pair_fit_calls = 0
        self.source_solver_calls = self.source_total_nfev = 0
        self.source_kkt = {}
        self.actual_seconds = 0.
        self.events = []
        # This signature includes all physical/planning inputs and fit settings,
        # so a shared cache cannot accidentally reuse another trained history.
        import json
        digest = sha256()
        for candidate in context['bank']:
            digest.update(candidate.name.encode())
            digest.update(repr(candidate.supports).encode())
            digest.update(repr(candidate.parents).encode())
            for arr in (context['lines'][candidate.name]['r'],
                        context['lines'][candidate.name]['x'],
                        context['forecasts'][candidate.name]['source_p'],
                        context['forecasts'][candidate.name]['source_q']):
                digest.update(np.asarray(arr, dtype='<f8').tobytes())
        T, n = context['control_shape']
        meter = context['future_meter']
        self.sigma = np.column_stack((fixed._broadcast(meter.v_sigma, (T,n), 'V sigma'),
            fixed._broadcast(meter.master_p_sigma, (T,), 'root P sigma'),
            fixed._broadcast(meter.master_q_sigma, (T,), 'root Q sigma')))
        for arr in (context['background']['p'], context['background']['q'],
                    context['root_v'], context['source_lower'], context['source_upper'], self.sigma):
            digest.update(np.asarray(arr, dtype='<f8').tobytes())
        digest.update(json.dumps(self.fit_options, sort_keys=True).encode())
        digest.update(repr((context['base_kw'], self.seed)).encode())
        self.signature = digest.hexdigest()
        self.source_bounds = np.array([context['source_lower'], context['source_upper']]).reshape(2,T,2)

    def _response(self, method, name, role):
        key = method, name, role
        if key not in self.response:
            candidate = next(c for c in self.context['bank'] if c.name == name)
            line = self.context['lines'][name]
            self.response[key] = FixedResponse(candidate, line['r'], line['x'], method=method)
        return self.response[key]

    def counters(self):
        total = dict(ac_value_calls=0, ac_jacobian_calls=0, evaluate_calls=0,
            approximate_value_calls=0, approximate_jacobian_calls=0, cache_hits=0)
        by_method = {}
        for (method, name, role),response in self.response.items():
            for key, value in response.counters.items():
                total[key] = total.get(key, 0)+value
                by_method.setdefault(method, {})[key] = by_method.setdefault(method, {}).get(key,0)+value
        return dict(response_calls=total, profile_evaluations=self.profile_evaluations,
            profile_cache_hits=self.cache_hits, pair_fit_calls=self.pair_fit_calls,
            source_solver_calls=self.source_solver_calls,source_total_nfev=self.source_total_nfev,
            actual_profile_seconds=self.actual_seconds,response_calls_by_method=by_method,
            cache_payment_log=list(self.events),source_kkt_by_method={method:dict(
                successful_fits=len(values),scaled_max=max(values,default=None),
                scaled_count_gt_1e_minus_5=sum(v>1e-5 for v in values))
                for method,values in self.source_kkt.items()})

    def evaluate(self, power_kw, method='ac', *, deadline=None):
        if method not in ('ac', 'i2'):
            raise ValueError('gradient method must be ac or i2')
        u = np.asarray(power_kw, float)
        if u.shape != self.context['control_shape'] or not np.isfinite(u).all():
            raise ValueError('finite matching time-by-terminal power required')
        if any(fixed._line_id(line['r'],line['x'])!=line['sha256']
               for line in self.context['lines'].values()):
            raise ValueError('fixed training line snapshot changed before source profile')
        # Exact input bytes are retained: nearby gradients must never alias a
        # rounded control cache entry during finite-difference verification.
        key = self.signature, method, u.astype('<f8').tobytes()
        if key in self.cache:
            self.cache_hits += 1
            self.events.append(dict(method=method,cache_hit=True,actual_seconds=0.,
                paid_pair_fits=0,source_profile_count=len(self.cache[key]['pairs'])))
            return self.cache[key]
        began = perf_counter()
        context = self.context
        before = {name:fixed._line_id(v['r'], v['x']) for name,v in context['lines'].items()}
        pairs = []; paid_pair_fits = 0; deadline_reached = False
        for index, model in enumerate(context['models']):
            a, b = model['a'], model['b']
            sr = context['forecasts'][a]
            si = context['forecasts'][b]
            attempted = deadline is None or perf_counter() < deadline
            try:
                if not attempted:
                    deadline_reached = True
                    raise RuntimeError('source_profile_not_attempted: planning time budget exhausted')
                pair = profile_pair(self._response(method,a,'reference'),
                    self._response(method,b,'competitor'), context['background']['p'],
                    context['background']['q'], np.column_stack((sr['source_p'],sr['source_q'])),
                    np.column_stack((si['source_p'],si['source_q'])), context['root_v'],
                    u, self.sigma, base_kw=context['base_kw'], source_bounds=self.source_bounds,
                    seed=source_pair_seed(self.seed,model['name']), **self.fit_options)
                if pair.get('reference_values') is not None:
                    voltage = pair['reference_values'][:, :u.shape[1]]
                    if voltage.min() < .85 or voltage.max() > 1.15:
                        pair.update(success=False, failure='reference voltage outside audit bounds')
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                pair = dict(success=False, failure=type(exc).__name__+': '+str(exc))
            self.pair_fit_calls += int(attempted); paid_pair_fits += int(attempted)
            self.source_solver_calls += len(pair.get('attempts',[]))
            self.source_total_nfev += int(pair.get('total_nfev',0))
            if pair.get('success') and 'scaled_source_kkt_violation' in pair:
                self.source_kkt.setdefault(method,[]).append(float(pair['scaled_source_kkt_violation']))
            pair.update(a=a,b=b,name=model['name'],line_parameters_fixed=True,
                source_interface=1,attempted=attempted)
            pairs.append(pair)
        unchanged = all(before[name] == line['sha256'] == fixed._line_id(line['r'],line['x'])
            for name,line in context['lines'].items())
        record = dict(success=bool(unchanged and len(pairs)==len(context['models'])
            and all(p.get('success') for p in pairs)), pairs=pairs, power_kw=u.copy(),
            expected_pair_count=len(context['models']), method=method,
            line_parameters_unchanged=unchanged, seconds=perf_counter()-began,
            deadline_reached=deadline_reached,paid_pair_fits=paid_pair_fits,
            uses_validation=False, uses_truth=False, uses_future_noise=False,
            global_AC_certificate=False, source_fit_scope='attained local source profiles')
        self.profile_evaluations += 1
        self.actual_seconds += record['seconds']
        if not deadline_reached:self.cache[key] = record
        self.events.append(dict(method=method,cache_hit=False,actual_seconds=record['seconds'],
            paid_pair_fits=paid_pair_fits,source_profile_count=len(pairs),
            deadline_reached=deadline_reached))
        return record


def solve_gradient_step(domain, power_kw, active_groups, *, trust_kw=6.,
                        solver_seconds=2., backend='gurobi', start=None):
    """Maximize the first directional derivative in the operation MILP.

    A nearest-rival tie uses one auxiliary scalar per reference with
    eta[a] <= gradient[a,b] @ (v-u), for every tied rival b.  With unique
    branches (including two models), this is simply max gradient @ (v-u).
    Reported solver bounds apply to this linear step alone.
    """
    began = perf_counter()
    u = np.asarray(power_kw, float)
    if u.shape != domain.indices.u.shape or not np.isfinite(u).all():
        raise ValueError('power shape must match the operation domain')
    if not np.isfinite(trust_kw) or trust_kw <= 0:
        raise ValueError('positive finite trust_kw required')
    if not active_groups or any(not group for group in active_groups):
        raise ValueError('nonempty active branch groups required')
    op = domain.as_dict(); U = np.asarray(op['u_map'], float)
    N, G = len(op['lower']), len(active_groups)
    lo = np.maximum(op['control_lower'], u.ravel()-trust_kw)
    hi = np.minimum(op['control_upper'], u.ravel()+trust_kw)
    all_grad = [np.asarray(pair['gradient_per_kw'],float).ravel()
        for group in active_groups for pair in group]
    if any(g.shape != u.ravel().shape or not np.isfinite(g).all() for g in all_grad):
        raise ValueError('finite matching pair gradients required')
    bound = max(1., max(float(np.sum(abs(g)*np.maximum(abs(lo-u.ravel()),abs(hi-u.ravel()))))
        for g in all_grad))
    def lift(A):
        from scipy.sparse import hstack, csr_matrix
        return hstack((csr_matrix(A),csr_matrix((A.shape[0],G))),format='csr')
    constraints = [LinearConstraint(lift(c.A),c.lb,c.ub) for c in op['constraints']]
    constraints.append(LinearConstraint(lift(U),lo,hi))
    for group_index, group in enumerate(active_groups):
        for pair in group:
            gradient = np.asarray(pair['gradient_per_kw'],float).ravel()
            row = np.zeros(N+G)
            row[:N] = -(gradient@U)
            row[N+group_index] = 1.
            constraints.append(LinearConstraint(row,-np.inf,-float(gradient@u.ravel())))
    objective = np.zeros(N+G); objective[N:] = -1./G
    fullstart = None if start is None else np.r_[np.asarray(start,float),np.zeros(G)]
    result = _solve_mip(objective,np.r_[op['integrality'],np.zeros(G,int)],
        Bounds(np.r_[op['lower'],np.full(G,-bound)],np.r_[op['upper'],np.full(G,bound)]),
        constraints,time_limit=solver_seconds,backend=backend,start=fullstart)
    x = None if result.x is None else np.asarray(result.x[:N],float)
    audit = domain.check(x) if x is not None else dict(feasible=False,issues=['no_incumbent'])
    proposal = None if x is None else U@x
    trust_violation = None if proposal is None else float(max(0.,
        np.max(lo-proposal,initial=0.),np.max(proposal-hi,initial=0.)))
    attained_gain = None if proposal is None else float(np.mean([
        min(float(np.asarray(pair['gradient_per_kw']).ravel()@(proposal-u.ravel()))
            for pair in group) for group in active_groups]))
    predicted = None if result.fun is None else -float(result.fun)
    dual = _finite(getattr(result,'mip_dual_bound',None))
    record = dict(success=bool(x is not None and audit['feasible'] and trust_violation<=1e-7), operation=audit,
        x=x,power_kw=None if x is None else (U@x).reshape(u.shape),
        predicted_directional_gain=predicted,linear_best_bound=None if dual is None else -dual,
        attained_directional_gain=attained_gain,trust_violation=trust_violation,
        linear_mip_gap=_finite(getattr(result,'mip_gap',None)),status=int(result.status),
        message=str(result.message),seconds=perf_counter()-began,backend=backend,
        bound_scope='linearized directional-gain MILP only; full AC global gap unavailable',
        full_AC_global_gap=None,trust_kw=float(trust_kw),active_group_count=G,
        active_branch_count=sum(map(len,active_groups)))
    return record


def select_gradient_controls(context, domain, initial_powers, *, method='ac',
        aggregation='nearest', fit_options=None, options=None, seed=0, cache=None):
    """Retain the best feasible seeded or complete-AC accepted gradient action.

    ``initial_powers`` maps public deterministic baseline or inherited-plan names
    to powers. Sorting these few starting points is initialization, never the
    selector's action domain. Every iterative proposal is a linear MILP over
    the full operation domain. The time budget is checked before each source
    profile, which can overrun a deadline by one source-fit call.  If no full
    score completes, a feasible baseline is separately retained as an unscored
    fallback; the incomplete method is still reported as a failure.
    """
    began = perf_counter()
    config = dict(DEFAULT_OPTIONS); config.update(options or {})
    if set(config)-set(DEFAULT_OPTIONS):
        raise ValueError('unsupported gradient options '+repr(sorted(set(config)-set(DEFAULT_OPTIONS))))
    if method not in ('ac','i2'):
        raise ValueError('method must be ac or i2')
    if aggregation not in ('nearest','mean','binarymean'):
        raise ValueError('unknown aggregation')
    for key in ('max_starts','max_iterations'):
        if int(config[key]) != config[key] or config[key] < 0:
            raise ValueError(key+' must be a nonnegative integer')
    for key in ('trust_kw','solver_seconds','total_seconds'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key+' must be positive and finite')
    for key in ('acceptance_atol','acceptance_rtol','gradient_zero_tol'):
        if not np.isfinite(config[key]) or config[key]<0:
            raise ValueError(key+' must be finite and nonnegative')
    if not initial_powers:
        raise ValueError('at least one public feasible starting plan is required')
    engine = ProfileEngine(context,fit_options=fit_options,seed=seed,cache=cache)
    deadline=began+config['total_seconds']
    record = dict(method=method,aggregation=aggregation,configuration=config,
        initial_candidates=[],iterations=[],failures=[],all_AC_tasks_required=True,
        uses_validation=False,uses_truth=False,uses_future_noise=False,
        random_used_by_primary=False,
        line_parameters_fixed=True,one_trained_parameter_vector=True,
        random_role='no random action proposals; deterministic source-fit restart seed only',
        expected_pair_count=len(context['models']),global_AC_gap=None,
        initialization_scope='few deterministic baselines/inherited feasible plans; no route library',
        initialization_provenance='driver supplies public deterministic baselines and inherited plans',
        budget_scope='all profiles including the first seed are budgeted; each source fit is atomic')
    initial, seen = [], set();seed_profile_attempted=False;fallback=None
    for origin,power in initial_powers.items():
        if power is None: continue
        # Once an operation has been checked, preserve it even if its full AC
        # score cannot finish within the budget. Every source fit is charged.
        if seed_profile_attempted and perf_counter()>=deadline:
            record['initial_candidates'].append(dict(origin=origin,success=False,
                attempted=False,failure='initial_plan_not_attempted: time budget exhausted'))
            continue
        u = np.asarray(power,float)
        if u.shape != context['control_shape'] or not np.isfinite(u).all():
            record['initial_candidates'].append(dict(origin=origin,success=False,failure='invalid power shape'))
            continue
        key = fixed._control_key(u)
        if key in seen: continue
        seen.add(key)
        operation = fixed._realize(domain,u)
        row = dict(origin=origin,operation=operation['audit'],success=False)
        if operation['success']:
            if fallback is None:
                fallback=dict(origin=origin,power=u.copy(),x=operation['x'],operation=operation['audit'])
            profile = engine.evaluate(u,'ac',deadline=deadline)
            seed_profile_attempted=True
            value = _aggregate(profile['pairs'],aggregation) if profile['success'] else dict(success=False,score=None)
            row.update(success=value['success'],score=value['score'],profile=profile)
            if value['success']:
                initial.append(dict(origin=origin,power=u.copy(),x=operation['x'],
                    score=value['score'],profile=profile))
        record['initial_candidates'].append(row)
    if not initial:
        record.update(success=False,seconds=perf_counter()-began,counters=engine.counters(),
            failure='no feasible complete-AC starting plan',selected_origin=None,selected_score=None,
            fallback_power=None if fallback is None else fallback['power'],
            fallback_operation=None if fallback is None else fallback['operation'],
            fallback_origin=None if fallback is None else fallback['origin'],
            fallback_scope='feasible operation retained; incomplete AC score is not an optimization result')
        return None,record
    record['initialization_seconds']=perf_counter()-began
    initial.sort(key=lambda row:-row['score'])
    best=initial[0].copy();record['initial_best_score']=best['score']
    started=0
    for start_index,start in enumerate(initial[:int(config['max_starts'])]):
        if perf_counter()-began >= config['total_seconds']:break
        started+=1;current=start.copy();trust=float(config['trust_kw'])
        for iteration in range(int(config['max_iterations'])):
            if perf_counter()-began >= config['total_seconds']:break
            row=dict(start_index=start_index,start_origin=start['origin'],iteration=iteration,
                before_score=current['score'],gradient_method=method,accepted=False)
            profile = engine.evaluate(current['power'],method,deadline=deadline)
            row['gradient_profile']=profile
            gradient = _aggregate(profile['pairs'],aggregation) if profile['success'] else dict(success=False)
            if not gradient['success']:
                row['failure']='required gradient source fit failed'
                record['iterations'].append(row);break
            row.update(gradient=gradient,trust_kw=trust)
            largest = max(float(np.max(abs(p['gradient_per_kw']),initial=0.))
                for group in gradient['active_groups'] for p in group)
            if largest <= config['gradient_zero_tol']:
                row['stop_reason']='zero_gradient; deterministic seeded plan retained'
                record['iterations'].append(row);break
            remaining=config['total_seconds']-(perf_counter()-began)
            if remaining<=0:
                row['stop_reason']='time_budget_before_MILP';record['iterations'].append(row);break
            try:
                step=solve_gradient_step(domain,current['power'],gradient['active_groups'],
                    trust_kw=trust,solver_seconds=min(float(config['solver_seconds']),remaining),
                    backend=config['backend'],start=current['x'])
                row['MILP']=step
                if not step['success']:
                    row['failure']='linear MILP has no feasible incumbent'
                elif np.allclose(step['power_kw'],current['power'],rtol=0,atol=1e-9):
                    row['stop_reason']='unchanged_MILP_action'
                elif perf_counter()>=deadline:
                    row['stop_reason']='time_budget_before_complete_AC_acceptance'
                else:
                    trial=engine.evaluate(step['power_kw'],'ac',deadline=deadline)
                    value=_aggregate(trial['pairs'],aggregation) if trial['success'] else dict(success=False,score=None)
                    row.update(candidate_profile=trial,candidate_score=value['score'])
                    tolerance=config['acceptance_atol']+config['acceptance_rtol']*abs(current['score'])
                    row['acceptance_tolerance']=float(tolerance)
                    if value['success'] and value['score']>current['score']+tolerance:
                        row['accepted']=True
                        current=dict(origin='gradient_MILP',power=step['power_kw'].copy(),x=step['x'],
                            score=value['score'],profile=trial)
                        if current['score']>best['score']:best=current.copy()
                    else:
                        row['rejection_reason']='full_AC_score_not_improved' if value['success'] else 'required_AC_source_fit_failed'
                        trust*=.5
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                row['failure']=type(exc).__name__+': '+str(exc)
            row['after_score']=current['score'];record['iterations'].append(row)
            if row.get('stop_reason'):break
    record.update(success=True,selected_origin=best['origin'],selected_score=best['score'],
        selected_profile=best['profile'],selected_operation=domain.check(best['x']),
        selected_operation_x=best['x'],inheritance_protected=best['score']>=record['initial_best_score']-1e-10,
        selected_is_new_MILP_proposal=best['origin']=='gradient_MILP',
        starts_completed=started,seconds=perf_counter()-began,counters=engine.counters())
    record['time_budget_exceeded']=record['seconds']>config['total_seconds']
    record['optimization_seconds']=record['seconds']-record['initialization_seconds']
    record['linear_MILP_seconds']=float(sum(row.get('MILP',{}).get('seconds',0.)
        for row in record['iterations']))
    record['source_profile_seconds']=record['counters'].get('actual_profile_seconds',0.)
    return best['power'].copy(),record
