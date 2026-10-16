"""Mobile EV MILPs with capped tasks or shared-line AC gradient acceptance.

Every reference keeps one training line vector.  In the joint predictive
engine each competing model may refit one line vector across the real noisy
training history and the reference's noiseless predicted future.  The local
fits, envelope derivatives and operation-MILP bounds have separate scopes.
No validation, simulator truth or future noise is an argument of this module.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csr_matrix, hstack

from . import likelihood as lk
from . import ev_fixed_train_design as fixed
from . import ev_gradient_flow_design as gradient
from . import ev_systematic_design as scalar
from . import ev_systematic_refit as refit
from .ev_rx_fit import fit_rx_inputs
from .ev_original_domain_milp import _solve_mip, evaluate_original_pair


# q=||mu_a-mu_b||^2/2; ideal equal-covariance Gaussian error is
# Phi(-sqrt(q/2)).  This is a planning scale, not achieved AC test power.
TASK_CAP = 2.*1.6448536269514722**2
METHODS=('none','constant','highlow','square_fixed','square_mobile',
    'scalar_fixed','scalar_mobile','cap_fixed','cap_mobile','jointgrad_fixed','jointgrad_mobile')


def _name_seed(seed, name):
    return (int(seed)+int.from_bytes(sha256(name.encode()).digest()[:4], 'little')) % 2**32


def capped_value(values, cap=TASK_CAP):
    values=np.asarray(values,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty pair values required')
    if not np.isfinite(cap) or cap<=0:raise ValueError('positive task cap required')
    return float(np.mean(np.minimum(values,cap)))


def _joint_aggregate(pairs, names):
    """Signed joint total-loss gaps, retaining every reference and rival."""
    expected={(a,b) for a in names for b in names if a!=b}
    if len(pairs)!=len(expected) or {(p['a'],p['b']) for p in pairs}!=expected:
        return dict(success=False,failure='all directed pairs required')
    if not all(p.get('success') for p in pairs):
        return dict(success=False,failure='a required joint predictive fit failed')
    nearest=[];active_groups=[]
    for a in names:
        rows=[p for p in pairs if p['a']==a];value=min(p['half_gap'] for p in rows)
        active=[p for p in rows if np.isclose(p['half_gap'],value,atol=1e-9,rtol=1e-9)]
        nearest.append(dict(a=a,half_gap=float(value),active=[p['b'] for p in active]))
        active_groups.append(active)
    return dict(success=True,score=float(np.mean([p['half_gap'] for p in nearest])),
        nearest=nearest,active_groups=active_groups,
        gradient_per_kw=np.mean([np.mean([p['gradient_per_kw'] for p in group],axis=0)
            for group in active_groups],axis=0),
        derivative_scope='local envelope derivative; tied branches enter the MILP separately')


class JointPredictiveEngine:
    """Complete-AC K-by-K local fits using training observations only.

    A model a supplies mu_a(u) from its sole training parameter vector.  For
    each b (including b=a), fit S_H,b+S_E,b(mu_a(u)) with one shared line vector.
    The signed task is half of L_ab-L_aa.  It incorporates the existing noisy
    history once.  No fitted predictive vector replaces a's training vector.
    """
    def __init__(self, context, *, fit_options=None, seed=0):
        if not context.get('success') or '_rebuild_inputs' not in context:
            raise ValueError('successful rebuildable training-only context required')
        args=context['_rebuild_inputs'];self.training=args['training'];self.tm=args['train_meter']
        self.context=context;self.seed=int(seed);self.cache={};self.evaluations=0;self.cache_hits=0
        self.fits=0;self.fixed_source_fits=0;self.total_seconds=0.;self.events=[]
        self.options=refit._options(fit_options)
        for key in ('source_p_bounds','source_q_bounds'):
            if tuple(self.options[key])!=tuple(context[key]):
                raise ValueError('joint predictive source domain must match context')
        self.names=[c.name for c in context['bank']]
        if len(set(self.names))!=len(self.names) or len(self.names)<2:
            raise ValueError('distinct candidate bank required')
        self.training_state={name:refit._state(context['train_fits'][name]) for name in self.names}
        for candidate in context['bank']:
            score=refit._score(candidate,self.training,self.tm,self.training_state[candidate.name],
                self.options,(self.training.t,),('training',))
            if not score['usable']:raise ValueError('training state outside joint predictive domain')
        self.signature=sha256(repr((self.names,self.seed,self.options)).encode()).hexdigest()

    def counters(self):
        return dict(profile_evaluations=self.evaluations,cache_hits=self.cache_hits,
            joint_fit_calls=self.fits,fixed_source_witness_fit_calls=self.fixed_source_fits,
            actual_profile_seconds=self.total_seconds,
            payment_log=list(self.events))

    def evaluate(self, power_kw, *, deadline=None):
        began=perf_counter();ctx=self.context;u=np.asarray(power_kw,float)
        if u.shape!=ctx['control_shape'] or not np.isfinite(u).all():
            raise ValueError('finite matching power required')
        if any(fixed._line_id(v['r'],v['x'])!=v['sha256'] for v in ctx['lines'].values()):
            raise ValueError('immutable training line snapshot changed')
        key=u.astype('<f8').tobytes()
        if key in self.cache:self.cache_hits+=1;return self.cache[key]
        T,n=u.shape;total=dict(p=ctx['background']['p']+u/ctx['base_kw'],q=ctx['background']['q'])
        columns=np.array([t*(2*n+2)+j for t in range(T) for j in range(n)])
        model=refit.concat_models((self.training,lk.Observations(p=total['p'],q=total['q'],
            v=np.ones((T,n)),root_v=ctx['root_v'],master_p=np.zeros(T),master_q=np.zeros(T))),
            (self.tm,ctx['future_meter']))
        references=[];pairs=[];fits_before=self.fits;fixed_before=self.fixed_source_fits;budget_failure=False
        for a in ctx['bank']:
            row=dict(a=a.name,success=False,fits={})
            try:
                source=ctx['forecasts'][a.name];line=ctx['lines'][a.name]
                source_vec=fixed._source_vector(source['source_p'],source['source_q'])
                ga=fixed._geometry(a,line,total,ctx['root_v'],ctx['future_meter'],source_vec)
                physical=ga['physical'];meter=ctx['future_meter']
                if not (np.min(physical['v'])>=self.options['voltage_bounds'][0]
                        and np.max(physical['v'])<=self.options['voltage_bounds'][1]):
                    raise ValueError('reference voltage outside declared physical range')
                future=lk.Observations(p=total['p']+fixed._broadcast(meter.bias_p,(T,n),'P bias'),
                    q=total['q']+fixed._broadcast(meter.bias_q,(T,n),'Q bias'),
                    v=physical['v']+fixed._broadcast(meter.bias_v,(T,n),'V bias'),
                    root_v=ctx['root_v'],master_p=physical['master_p'],master_q=physical['master_q'])
                obs=lk.concat_observations(self.training,future)
                Ja=ga['state_jacobian'][:,columns]/ctx['base_kw']
                for b in ctx['bank']:
                    if deadline is not None and perf_counter()>=deadline:
                        budget_failure=True
                        row['fits'][b.name]=dict(success=False,failure='joint predictive budget exhausted',attempted=False)
                        continue
                    start=self.training_state[b.name];fs=ctx['forecasts'][b.name]
                    fixed_fit=fit_rx_inputs(b,future,meter,initial=dict(r=start['r'],x=start['x'],
                        source_p=fs['source_p'],source_q=fs['source_q']),fixed_r=start['r'],fixed_x=start['x'],
                        seed=_name_seed(self.seed,a.name+'__'+b.name),**self.options)
                    self.fixed_source_fits+=1
                    witness=refit._join_state(start['r'],start['x'],(start,fixed_fit))
                    fitted,state=refit._stage(b,obs,model,self.options,[('training_theta_source_fit',witness)],
                        (self.training.t,T),('training','future'),
                        _name_seed(self.seed,a.name+'__'+b.name+'joint'),None)
                    self.fits+=1
                    residual=np.asarray(fitted['blocks']['future']['residual'],float)
                    r=np.r_[residual[:,:n].ravel(),residual[:,n],residual[:,n+1]]
                    sp=np.asarray(state['source_p'])[-T:];sq=np.asarray(state['source_q'])[-T:]
                    gb=fixed._geometry(b,dict(r=state['r'],x=state['x']),total,ctx['root_v'],meter,
                        fixed._source_vector(sp,sq))
                    Jb=gb['state_jacobian'][:,columns]/ctx['base_kw']
                    price=((Jb-Ja).T@r).reshape(T,n)
                    row['fits'][b.name]=dict(fitted,attempted=True,fixed_source_success=bool(fixed_fit.success),
                        gradient_per_kw=price,gradient_source='physical AC at the attained shared-line fit',
                        gradient_reliable=bool(fitted['success'] and fixed_fit.success),
                        future_reference_derivative_subtracted=True)
                own=row['fits'][a.name]
                for b in ctx['bank']:
                    if a.name==b.name:continue
                    rival=row['fits'][b.name];success=bool(own.get('success') and rival.get('success')
                        and own.get('gradient_reliable') and rival.get('gradient_reliable'))
                    pair=dict(a=a.name,b=b.name,name=a.name+'__to__'+b.name,success=success)
                    if 'loss' in own and 'loss' in rival:
                        pair.update(half_gap=.5*(rival['loss']-own['loss']),
                            gradient_per_kw=np.asarray(rival['gradient_per_kw'])-np.asarray(own['gradient_per_kw']),
                            own_loss=own['loss'],rival_loss=rival['loss'])
                    else:pair['failure']='required reference/rival local fit unavailable'
                    pairs.append(pair)
                row['success']=all(f.get('success') and f.get('gradient_reliable') for f in row['fits'].values())
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                row['failure']=type(exc).__name__+': '+str(exc)
                for b in ctx['bank']:
                    row['fits'].setdefault(b.name,dict(success=False,attempted=False,failure=row['failure']))
                    if b.name!=a.name and not any(p['a']==a.name and p['b']==b.name for p in pairs):
                        pairs.append(dict(a=a.name,b=b.name,name=a.name+'__to__'+b.name,
                            success=False,failure=row['failure']))
            references.append(row)
        aggregate=_joint_aggregate(pairs,self.names)
        seconds=perf_counter()-began;self.evaluations+=1;self.total_seconds+=seconds
        record=dict(success=bool(aggregate.get('success') and all(r['success'] for r in references)),
            references=references,pairs=pairs,aggregation=aggregate,score=aggregate.get('score'),
            power_kw=u.copy(),seconds=seconds,joint_fit_calls=self.fits-fits_before,
            fixed_source_witness_fit_calls=self.fixed_source_fits-fixed_before,
            line_parameters_fixed_for_references=True,one_trained_parameter_vector=True,
            rival_line_parameters_shared_history_and_prediction=True,source_interface=1,
            uses_truth=False,uses_validation=False,uses_future_noise=False,
            uses_training_history=True,full_AC_global_gap=None,global_optimality_certified=False,
            deadline_reached=budget_failure,score_scope='attained signed predictive joint total half-SSE gaps')
        self.events.append(dict(seconds=seconds,joint_fit_calls=record['joint_fit_calls'],success=record['success']))
        if not budget_failure:self.cache[key]=record
        return record


def solve_capped_milp(context,domain,*,tables,cap=TASK_CAP,initial_operation=None,
        solver_seconds=2.,backend='gurobi',mip_rel_gap=1e-8):
    """Reuse scalar chords; every task has a fixed cap, no new binary vars.

    The implementation reuses the complete operation/chord formulation by
    representing each pair as a separate reference for its sole rival.  The
    actual pair matrices remain unchanged; a small lifted model is built here
    so cap bounds can be imposed without changing the frozen implementation.
    """
    began=perf_counter();scalar._groups(context)
    if not np.isfinite(cap) or cap<=0:raise ValueError('positive task cap required')
    from . import ev_scalar_distance_milp as kernel
    if not tables.get('success') or tables.get('problem_id')!=kernel.problem_identity(context,domain):
        raise ValueError('tables differ from context/operations')
    P=len(context['pairs']);T,n=context['control_shape'];op=domain.as_dict();N=len(op['lower'])
    size=N;indices={};binary=continuous=0
    if tables['grid_points']!=3:
        for t in range(T):
            for j in range(n):
                count=len(tables['grids'][t][j]);lam=np.arange(size,size+count);size+=count
                seg=np.arange(size,size+max(0,count-1));size+=len(seg)
                indices[t,j]=(lam,seg);continuous+=len(lam);binary+=len(seg)
    eta=np.arange(size,size+P);size+=P;continuous+=P;width=size-N
    lower=np.r_[op['lower'],np.zeros(width)];upper=np.r_[op['upper'],np.ones(width)]
    integer=np.r_[op['integrality'],np.zeros(width,int)]
    constraints=[LinearConstraint(hstack((csr_matrix(c.A),csr_matrix((c.A.shape[0],width))),format='csr'),c.lb,c.ub)
        for c in op['constraints']]
    expressions=np.zeros((P,size));constants=np.zeros(P)
    if tables['grid_points']!=3:
        for (t,j),(lam,seg) in indices.items():
            grid=tables['grids'][t][j];y=int(domain.indices.connected_arc[t,j]);integer[seg]=1
            row=np.zeros(size);row[lam]=1;row[y]-=1;constraints.append(LinearConstraint(row,0,0))
            row=np.zeros(size);row[domain.indices.u[t,j]]=1;row[lam]-=grid
            constraints.append(LinearConstraint(row,0,0))
            row=np.zeros(size);row[seg]=1;row[y]-=1;constraints.append(LinearConstraint(row,0,0))
            for k,index in enumerate(lam):
                row=np.zeros(size);row[index]=1
                if k:row[seg[k-1]]-=1
                if k<len(seg):row[seg[k]]-=1
                constraints.append(LinearConstraint(row,-np.inf,0))
    for k in range(P):
        constants[k]=sum(row['primal_upper'] for row in tables['background'][k])
        for t in range(T):
            q0=tables['background'][k][t]['primal_upper']
            for j in range(n):
                node=tables['tables'][k][t][j];g=node['grid'];values=node['primal_upper']
                if tables['grid_points']==3:
                    if g[-1]>0:expressions[k,domain.indices.charge[t,j]]+=(values[-1]-q0)/g[-1]
                    if g[0]<0:expressions[k,domain.indices.discharge[t,j]]+=(values[0]-q0)/(-g[0])
                else:
                    lam,_=indices[t,j];expressions[k,lam]+=values
                    expressions[k,int(domain.indices.connected_arc[t,j])]-=q0
        upper[eta[k]]=cap
        row=-expressions[k].copy();row[eta[k]]+=1
        constraints.append(LinearConstraint(row,-np.inf,constants[k]))
    objective=np.zeros(size);objective[eta]=-1/P
    # The initial operation is retained by the policy.  A partial inaccurate
    # SOS2 warm start is less useful than the solver's own feasible construction.
    solve_began=perf_counter()
    result=_solve_mip(objective,integer,Bounds(lower,upper),constraints,time_limit=solver_seconds,
        mip_rel_gap=mip_rel_gap,backend=backend)
    solving_seconds=perf_counter()-solve_began
    x=None if result.x is None else np.asarray(result.x[:N]);audit=domain.check(x)
    power=None if x is None else np.asarray(op['u_map']@x).reshape(T,n)
    qps=[] if not audit['feasible'] else [evaluate_original_pair(p,power.ravel()) for p in context['pairs']]
    lb=None if not qps else capped_value([.5*q['tangent_lower_numeric'] for q in qps],cap)
    ub=getattr(result,'mip_dual_bound',None);ub=None if ub is None or not np.isfinite(ub) else -float(ub)
    consistent=bool(lb is None or ub is None or lb<=ub+1e-8*(1+abs(ub)))
    return dict(success=bool(audit['feasible'] and qps and all(q['success'] for q in qps) and consistent),
        power_kw=power,operation_x=x,operation=audit,status=int(result.status),message=str(result.message),
        solver_proxy_incumbent=None if result.fun is None else -float(result.fun),solver_proxy_best_bound=ub,
        frozen_objective_lower=lb,frozen_objective_upper=ub,bound_consistent=consistent,
        frozen_absolute_gap=None if lb is None or ub is None else max(0.,ub-lb),
        action_QP=qps,cap=cap,added_binary_count=binary,added_continuous_count=continuous,
        pair_count=P,seconds=perf_counter()-began,MILP_seconds=solving_seconds,all_candidates_retained=True,
        solver_mip_gap=getattr(result,'mip_gap',None),
        original_binary_count=int(np.count_nonzero(op['integrality'])),
        capped_chord_error_upper=float(np.mean(np.minimum(cap,tables['pair_error_upper']))),
        bound_scope='frozen affine source-profile capped task objective only',full_AC_global_gap=None)


def cap_policy(context,domain,initials,*,cap=TASK_CAP,grids=(3,9),total_seconds=15.,
        profile_options=None,seed=0,solver_seconds=2.,backend='gurobi',profile_cache=None,prebuilt_tables=None):
    began=perf_counter();deadline=began+total_seconds
    engine=gradient.ProfileEngine(context,fit_options=profile_options,seed=seed,cache=profile_cache)
    record=dict(success=False,initial_candidates=[],grids=[],failures=[],cap=cap,
        objective='mean capped absolute directed task half-SSE',uses_truth=False,
        uses_validation=False,uses_future_noise=False,line_parameters_fixed=True)
    best=None;seen=set()
    for origin,power in initials.items():
        u=np.asarray(power,float);key=fixed._control_key(u)
        if key in seen:continue
        seen.add(key);operation=scalar._realize(domain,u,seconds=solver_seconds,backend=backend)
        row=dict(origin=origin,operation=operation['audit'],success=False)
        if operation['success']:
            profile=engine.evaluate(u,deadline=deadline)
            if profile.get('success'):
                score=capped_value([p['half_SSE'] for p in profile['pairs']],cap)
                row.update(success=True,score=score,profile=profile)
                if best is None or score>best['score']+1e-10:
                    best=dict(origin=origin,power=u.copy(),x=operation['x'],profile=profile,score=score)
        record['initial_candidates'].append(row)
    if best is None:
        record.update(failure='no fully checked initial',seconds=perf_counter()-began)
        return None,record
    initial_score=best['score'];table_cache={}
    for points in grids:
        if perf_counter()>=deadline:record['failures'].append('budget before grid');break
        table=(prebuilt_tables or {}).get(points)
        inherited=table is not None and table.get('success')
        if not inherited:
            table=scalar.build_systematic_tables(context,domain,grid_points=points,cache=table_cache,deadline=deadline)
        row=dict(grid_points=points,table=table,table_inherited_from_scalar_parent=inherited,accepted=False)
        if table['success'] and perf_counter()<deadline:
            solved=solve_capped_milp(context,domain,tables=table,cap=cap,
                initial_operation=best['x'],solver_seconds=min(solver_seconds,max(.001,deadline-perf_counter())),backend=backend)
            row['raw']=solved
            if solved['success'] and perf_counter()<deadline:
                profile=engine.evaluate(solved['power_kw'],deadline=deadline)
                if profile.get('success'):
                    score=capped_value([p['half_SSE'] for p in profile['pairs']],cap)
                    row.update(raw_AC_score=score,raw_AC_profile=profile)
                    if score>best['score']+1e-9:
                        best=dict(origin='capped_MILP_'+str(points),power=solved['power_kw'].copy(),
                            x=solved['operation_x'],profile=profile,score=score);row['accepted']=True
        else:row['failure']=table.get('failure','budget before solve')
        record['grids'].append(row)
    record.update(success=True,selected_score=best['score'],selected_origin=best['origin'],
        selected_profile=best['profile'],selected_operation=domain.check(best['x']),selected_operation_x=best['x'],
        initial_best_score=initial_score,display_gain_from_best_parent=best['score']-initial_score,
        inheritance_protected=best['score']>=initial_score-1e-10,counters=engine.counters(),
        seconds=perf_counter()-began,full_AC_global_gap=None)
    return best['power'],record


def joint_gradient_policy(context,domain,initials,*,joint_options=None,total_seconds=15.,
        seed=0,solver_seconds=2.,backend='gurobi',max_starts=2,max_iterations=2,trust_kw=None,engine=None):
    """Full AC envelope prices choose power and path in an operation MILP."""
    began=perf_counter();deadline=began+total_seconds
    if engine is None:engine=JointPredictiveEngine(context,fit_options=joint_options,seed=seed)
    elif engine.context is not context or engine.options!=refit._options(joint_options) or engine.seed!=int(seed):
        raise ValueError('shared predictive engine differs from context/options/seed')
    counters_before=engine.counters()
    def counter_delta():
        after=engine.counters()
        return {k:(after[k][len(counters_before[k]):] if isinstance(after[k],list)
            else after[k]-counters_before[k]) for k in after}
    record=dict(success=False,initial_candidates=[],starts=[],failures=[],
        uses_truth=False,uses_validation=False,uses_future_noise=False,
        line_parameters_fixed_for_reference=True,rival_lines_refitted=True,
        objective='mean reference nearest signed shared-line predictive total half-SSE gap',
        global_optimality_certified=False,full_AC_global_gap=None)
    checked=[];seen=set()
    for origin,power in initials.items():
        u=np.asarray(power,float);key=u.astype('<f8').tobytes()
        if key in seen:continue
        canonical=next((row for row in checked if np.max(abs(row['power']-u))<=1e-10),None)
        if canonical is not None:
            record['initial_candidates'].append(dict(origin=origin,success=True,near_duplicate=True,
                canonical_origin=canonical['origin'],canonical_power_kw=canonical['power'].copy(),
                score=canonical['score'],new_curve_not_scored=True))
            continue
        seen.add(key)
        if perf_counter()>=deadline:
            record['initial_candidates'].append(dict(origin=origin,success=False,failure='initial budget exhausted'));continue
        op=scalar._realize(domain,u,seconds=min(solver_seconds,max(.001,deadline-perf_counter())),backend=backend)
        row=dict(origin=origin,success=False,operation=op['audit'])
        if op['success']:
            profile=engine.evaluate(u,deadline=deadline);row['profile']=profile
            if profile['success']:
                row.update(success=True,score=profile['score'])
                checked.append(dict(origin=origin,power=u.copy(),x=op['x'],profile=profile,score=profile['score']))
        record['initial_candidates'].append(row)
    if not checked:
        record.update(failure='no complete shared-line predictive initial score',seconds=perf_counter()-began,
            counters=counter_delta());return None,record
    checked.sort(key=lambda p:-p['score']);best=checked[0];initial_score=best['score']
    for initial in checked[:max_starts]:
        current=initial;start_log=dict(origin=initial['origin'],initial_score=initial['score'],iterations=[])
        for iteration in range(max_iterations):
            if perf_counter()>=deadline:
                start_log['failure']='budget before next joint gradient';break
            aggregation=current['profile']['aggregation'];groups=aggregation['active_groups']
            norm=max(float(np.max(abs(p['gradient_per_kw']))) for group in groups for p in group)
            if norm<1e-12:start_log['failure']='zero local price; inherited action retained';break
            solved=gradient.solve_gradient_step(domain,current['power'],groups,
                trust_kw=2*fixed.CONFIG['command_power_kw'] if trust_kw is None else trust_kw,
                solver_seconds=min(solver_seconds,max(.001,deadline-perf_counter())),backend=backend,start=current['x'])
            row=dict(iteration=iteration,raw=solved,before_score=current['score'],accepted=False,
                gradient_max_abs=norm,linear_bound_scope='one local operation-price step only')
            if not solved['success']:
                row['failure']='no feasible gradient MILP proposal';start_log['iterations'].append(row);break
            change=float(np.max(abs(np.asarray(solved['power_kw'])-current['power'])))
            row['proposal_max_delta_kw']=change
            first_order_bound=solved.get('linear_best_bound')
            stationary=bool(first_order_bound is not None and first_order_bound<=1e-10*(1+abs(current['score'])))
            if change<=1e-10 or stationary:
                row.update(failure=('zero power step' if change<=1e-10 else 'no certified positive first-order step')+
                        '; current complete AC score retained',
                    repeated_AC_score_avoided=True,selected_score=current['score'])
                start_log['iterations'].append(row);break
            profile=engine.evaluate(solved['power_kw'],deadline=deadline);row['raw_AC_joint_profile']=profile
            if profile['success'] and profile['score']>current['score']+1e-9+1e-10*abs(current['score']):
                current=dict(origin='joint_gradient_MILP',power=solved['power_kw'].copy(),x=solved['x'],
                    profile=profile,score=profile['score']);row['accepted']=True
                if current['score']>best['score']+1e-10:best=current
            else:row['failure']='joint predictive score failed or did not improve'
            row['selected_score']=current['score'];start_log['iterations'].append(row)
            if not row['accepted']:break
        record['starts'].append(start_log)
    record.update(success=True,selected_score=best['score'],selected_origin=best['origin'],
        selected_profile=best['profile'],selected_operation=domain.check(best['x']),selected_operation_x=best['x'],
        initial_best_score=initial_score,inheritance_protected=best['score']>=initial_score-1e-10,
        counters=counter_delta(),seconds=perf_counter()-began,
        time_budget_exceeded=perf_counter()>deadline,max_starts=max_starts,max_iterations=max_iterations)
    attempts=[item for start in record['starts'] for item in start['iterations']]
    record.update(gradient_MILP_calls=len(attempts),gradient_MILP_accepted=sum(item['accepted'] for item in attempts),
        completed_requested_gradient_iterations=bool(len(record['starts'])==min(max_starts,len(checked))
            and all(len(start['iterations'])==max_iterations for start in record['starts'])))
    return best['power'],record


def audit_envelope_gradient(engine,power_kw,direction,*,step=1e-3):
    """Reprofile the same local fitting procedure for an independent price check."""
    u=np.asarray(power_kw,float);direction=np.asarray(direction,float)
    if direction.shape!=u.shape or not np.isfinite(direction).all() or step<=0:
        raise ValueError('finite matching direction and positive step required')
    central=engine.evaluate(u);plus=engine.evaluate(u+step*direction);minus=engine.evaluate(u-step*direction)
    if not all(p['success'] for p in (central,plus,minus)):
        return dict(success=False,profiles=[central,plus,minus],failure='a required reprofile failed')
    by_plus={(p['a'],p['b']):p for p in plus['pairs']};by_minus={(p['a'],p['b']):p for p in minus['pairs']}
    pairs=[]
    for p in central['pairs']:
        key=p['a'],p['b'];analytic=float(np.sum(direction*p['gradient_per_kw']))
        finite=(by_plus[key]['half_gap']-by_minus[key]['half_gap'])/(2*step)
        pairs.append(dict(a=key[0],b=key[1],analytic=analytic,finite_difference=finite,
            absolute_error=abs(analytic-finite),scaled_error=abs(analytic-finite)/(1+abs(finite))))
    return dict(success=True,step=step,pairs=pairs,max_scaled_error=max(p['scaled_error'] for p in pairs),
        aggregate_central_difference=(plus['score']-minus['score'])/(2*step),
        aggregate_analytic_mean=float(np.sum(direction*central['aggregation']['gradient_per_kw'])),
        active_constraint_counts=[{name:fit.get('optimizer',{}).get('diagnostics',{}).get('active_bound_count')
            for name,fit in row['fits'].items()} for row in central['references']],
        scope='reprofile finite difference of achieved local fits; branch changes are retained')


def select_plans(training,bank,fits,tm,fm,forecast,*,domains,base_kw,seed,
        training_options=None,source_options=None,profile_options=None,total_seconds=15.,
        solver_seconds=2.,grids=(3,9),cap=TASK_CAP,backend='gurobi'):
    """Eleven matched methods; callers supply only the public operation domains.

    ``domains`` maps fixed/mobile to MobileServiceDomain.  Every standalone
    planning cost includes its ancestor plans and one shared context build.
    Timings are recorded separately from the scientific predictive scores.
    """
    from copy import deepcopy
    from . import ev_gradient_flow_study as public
    from .ev_rx_precision import solve_squared_precision
    began=perf_counter();T,n=np.asarray(forecast['p']).shape
    if set(domains)!={'fixed','mobile'} or any(d.indices.u.shape!=(T,n) for d in domains.values()):
        raise ValueError('matching fixed and mobile operation domains required')
    sources=dict(source_options or {});profile=dict(profile_options or {})
    joint=dict(training_options or {});joint.update(sources)
    joint['multistart']=1
    curves=public.public_curves(T,n);plans={k:curves[k].copy() for k in ('none','constant','highlow')}
    metadata={k:dict(success=True,planning_actual_seconds=0.,planning_seconds=0.,dependencies=[]) for k in plans}
    exchange=deepcopy(fixed.DEFAULT_EXCHANGE);exchange.update(total_seconds=total_seconds,
        solver_seconds=solver_seconds,max_iterations=3)
    for domain_name in ('fixed','mobile'):
        name='square_'+domain_name;started=perf_counter();deps=[] if domain_name=='fixed' else ['square_fixed']
        dependency=sum(metadata[k]['planning_actual_seconds'] for k in deps)
        initial={k:curves[k] for k in ('constant','highlow')};initial['public_square']=curves['analytic_square']
        if deps:initial['fixed_parent']=plans['square_fixed']
        try:
            row=solve_squared_precision(domains[domain_name],fm,initial,
                exchange_options={**exchange,'total_seconds':max(.01,total_seconds-dependency)})
            u=row.get('power_kw');failure=None if u is not None and row.get('success') else 'square incumbent unavailable'
            # Preserve the old public tie rule for the proved uniform T=6 case.
            if T==6 and np.all(np.asarray(fm.v_sigma)==np.asarray(fm.v_sigma).ravel()[0]) and u is not None:
                canonical=curves['analytic_square'];weights=row['weights']
                if scalar._realize(domains[domain_name],canonical,seconds=solver_seconds,backend=backend)['success']:
                    if np.sum(weights*canonical**2)>=np.sum(weights*np.asarray(u)**2)-1e-7:
                        u=canonical.copy()
            plans[name]=np.asarray(u).copy() if u is not None else curves['analytic_square'].copy()
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failure=type(exc).__name__+': '+str(exc);row=dict(failure=failure);plans[name]=curves['analytic_square'].copy()
        own=perf_counter()-started;metadata[name]=dict(success=failure is None,failure=failure,raw=row,
            dependencies=deps,planning_actual_seconds=own,planning_seconds=own+dependency)
    started=perf_counter();context=fixed.build_fixed_contexts(training,bank,fits,tm,fm,
        -fixed.CONFIG['command_power_kw'],fixed.CONFIG['command_power_kw'],plans['square_fixed'],
        forecast=forecast,base_kw=base_kw,**sources)
    context_seconds=perf_counter()-started
    joint_engine=None;source_cache={}
    for name in ('scalar_fixed','scalar_mobile','cap_fixed','cap_mobile','jointgrad_fixed','jointgrad_mobile'):
        started=perf_counter();mobile=name.endswith('mobile');domain=domains['mobile' if mobile else 'fixed']
        initial={k:plans[k] for k in ('square_fixed','constant','highlow')};deps=['square_fixed']
        if mobile:initial['square_mobile']=plans['square_mobile'];deps.append('square_mobile')
        if name.startswith('scalar'):
            if mobile:initial['fixed_parent']=plans['scalar_fixed'];deps.append('scalar_fixed')
        elif name.startswith('cap'):
            initial['scalar_parent']=plans['scalar_mobile' if mobile else 'scalar_fixed']
            deps.append('scalar_mobile' if mobile else 'scalar_fixed')
            if mobile:initial['fixed_parent']=plans['cap_fixed'];deps.append('cap_fixed')
        else:
            initial={k:plans[k] for k in ('square_fixed',)}
            if mobile:initial['square_mobile']=plans['square_mobile']
            initial['scalar_jump']=plans['scalar_mobile' if mobile else 'scalar_fixed']
            deps.append('scalar_mobile' if mobile else 'scalar_fixed')
            if mobile:initial['fixed_parent']=plans['jointgrad_fixed'];deps.append('jointgrad_fixed')
        # Charge each ancestor once even when multiple parents share it.
        ancestors=set()
        def visit(key):
            if key in ancestors:return
            ancestors.add(key)
            for parent in metadata[key]['dependencies']:visit(parent)
        for key in deps:visit(key)
        dependency=sum(metadata[k]['planning_actual_seconds'] for k in ancestors)+context_seconds
        seconds=max(.01,total_seconds-dependency)
        try:
            if not context.get('success'):raise RuntimeError('training-only context failed')
            if name.startswith('scalar'):
                u,row=scalar.scalar_policy(context,domain,initial,grids=grids,total_seconds=seconds,
                    fit_options={**profile,**sources},seed=seed,solver_seconds=solver_seconds,backend=backend,
                    cache=source_cache)
            elif name.startswith('cap'):
                parent_tables={item['grid_points']:item['table'] for item in
                    metadata['scalar_mobile' if mobile else 'scalar_fixed']['raw'].get('grids',[])
                    if item.get('table',{}).get('success')}
                u,row=cap_policy(context,domain,initial,cap=cap,grids=grids,total_seconds=seconds,
                    profile_options={**profile,**sources},seed=seed,solver_seconds=solver_seconds,backend=backend,
                    profile_cache=source_cache,prebuilt_tables=parent_tables)
            else:
                if joint_engine is None:joint_engine=JointPredictiveEngine(context,fit_options=joint,seed=seed)
                u,row=joint_gradient_policy(context,domain,initial,joint_options=joint,total_seconds=seconds,
                    seed=seed,solver_seconds=solver_seconds,backend=backend,max_starts=2,max_iterations=2,
                    engine=joint_engine)
            failure=None if u is not None and row.get('success') else 'no complete checked optimized plan'
            plans[name]=np.asarray(u).copy() if u is not None else plans['square_mobile' if mobile else 'square_fixed'].copy()
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failure=type(exc).__name__+': '+str(exc);row=dict(failure=failure)
            plans[name]=plans['square_mobile' if mobile else 'square_fixed'].copy()
        own=perf_counter()-started;metadata[name]=dict(success=failure is None,failure=failure,raw=row,
            dependencies=deps,standalone_ancestors=sorted(ancestors),planning_actual_seconds=own,
            planning_seconds=own+dependency,soft_budget_overrun_seconds=max(0.,own+dependency-total_seconds))
    return plans,dict(methods=metadata,context=context,context_seconds=context_seconds,
        seconds=perf_counter()-began,uses_truth=False,uses_validation=False,uses_future_noise=False,
        source_profile_options={**profile,**sources},joint_predictive_options=joint,
        joint_predictive_multistart=1,total_seconds=total_seconds,solver_seconds=solver_seconds,task_cap=cap),domains
