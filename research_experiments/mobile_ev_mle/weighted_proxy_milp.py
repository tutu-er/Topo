"""Expected-scenario, max-min frozen-proxy MILPs; research prototype only.

Objective: max_u min_h sum_omega w[omega] q[h,omega](u). Weights are
predeclared, nonnegative and sum to one. This is not EIG or an error probability.
Nuisance QPs are separate hypothetical-world refits, never extra H samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import hashlib
import inspect
import numpy as np

from research_experiments.mobile_ev_mle.ev_information_milp import (
    FrozenPair,solve_pair,support_cut,affine_range,control_box,proxy_box_upper,_secant_sum)
from research_experiments.mobile_ev_mle.ev_operations_milp import (
    EVConfig,operation_model,decode_plan,check_plan)


@dataclass(frozen=True)
class SquaredCut:
    intercept: float
    slope: np.ndarray
    direction: np.ndarray
    amplitude: float
    pair_fingerprint: str

    def value(self,u):return float(self.intercept+self.slope@np.asarray(u,float).ravel())


def bank(groups,weights,dimension=None):
    """Validate every declared competitor/scenario, including zero mass."""
    if not groups or not weights:raise ValueError('nonempty competitor/scenario bank required')
    if any(not isinstance(key,str) or not key for key in (*weights,*groups)):
        raise ValueError('nonempty string competitor/scenario labels required')
    scenarios=tuple(sorted(weights));models=tuple(sorted(groups))
    w=np.asarray([weights[key] for key in scenarios],float)
    if not np.isfinite(w).all() or np.any(w<0) or abs(float(w.sum())-1)>1e-12:
        raise ValueError('predeclared nonnegative weights must sum to one; no implicit normalization')
    result={}
    for h in models:
        if set(groups[h])!=set(scenarios):
            raise ValueError('all declared scenarios required for every competitor')
        for scenario in scenarios:
            pair=groups[h][scenario]
            if not isinstance(pair,FrozenPair):raise ValueError('FrozenPair values required')
            if dimension is not None and pair.control_dimension!=dimension:
                raise ValueError('control dimensions differ from operation model')
            result[h,scenario]=pair
    return models,scenarios,{key:float(weights[key]) for key in scenarios},result


def squared_support_cut(pair,direction,amplitude,lower=None,upper=None):
    """d=2*a*ell-a^2 <= q, with a>=0 and box-min support ell<=sqrt(q)."""
    a=float(amplitude)
    if not np.isfinite(a) or a<0:raise ValueError('finite nonnegative amplitude required')
    ell=support_cut(pair,direction,lower,upper)
    slope=2*a*ell.slope
    direction=np.array(ell.direction,copy=True);slope=np.array(slope,copy=True)
    direction.setflags(write=False);slope.setflags(write=False)
    return SquaredCut(2*a*ell.intercept-a*a,slope,direction,a,pair.fingerprint())


def cut_from_qp(pair,u,answer=None,lower=None,upper=None):
    answer=solve_pair(pair,u) if answer is None else answer
    residual=np.asarray(answer['residual'],float);a=float(np.linalg.norm(residual))
    direction=np.zeros_like(residual) if a==0 else residual/a
    return squared_support_cut(pair,direction,a,lower,upper)


def _verify_cut(pair,cut,lower,upper):
    """Recompute support to reject an invalid or differently bound cut."""
    if not isinstance(cut,SquaredCut) or cut.pair_fingerprint!=pair.fingerprint():
        raise ValueError('cut belongs to a different frozen pair')
    expected=squared_support_cut(pair,cut.direction,cut.amplitude,lower,upper)
    if (not np.isfinite(cut.intercept) or not np.isfinite(cut.slope).all() or
        cut.slope.shape!=expected.slope.shape or
        not np.allclose(cut.slope,expected.slope,rtol=0,atol=1e-12) or
        abs(cut.intercept-expected.intercept)>1e-12):
        raise ValueError('invalid support lower bound; regenerate from the declared pair/box')


def _witness_bounds(pairs,lower,upper,witnesses=None):
    result={}
    for key,pair in pairs.items():
        zs=[np.clip(np.zeros(len(pair.lower)),pair.lower,pair.upper)] if witnesses is None else witnesses[key]
        if not zs:raise ValueError('at least one feasible fixed witness for every pair required')
        result[key]=proxy_box_upper([pair],lower,upper,[zs])
    return result


def _common(config,groups,weights,witnesses=None):
    models,scenarios,w,pairs=bank(groups,weights,config.n_slots*config.n_sites)
    lower,upper=control_box(config)
    bars=_witness_bounds(pairs,lower,upper,witnesses)
    objective_bar=min(sum(w[omega]*bars[h,omega] for omega in scenarios) for h in models)
    return models,scenarios,w,pairs,lower,upper,bars,objective_bar


def _record(model,indices,result,scale,config):
    plan=None if result.x is None else decode_plan(result,indices)
    if plan is not None and not check_plan(plan,config,tolerance=2e-6)['feasible']:
        raise RuntimeError('MILP incumbent failed independent operation checks')
    return dict(result=result,plan=plan,scale=scale,
        incumbent=None if result.objective_value is None else float(result.objective_value*scale),
        master_best_bound=None if result.objective_bound is None else float(result.objective_bound*scale),
        variables=model.n_variables,rows=len(model.rows))


def expected_lower_master(config,groups,weights,cuts,*,witnesses=None,time_limit=5,mip_rel_gap=1e-9):
    models,scenarios,w,pairs,lower,upper,bars,bar=_common(config,groups,weights,witnesses)
    model,indices=operation_model(config)
    scale=max(1.,max(bars.values()));T=int(model.add_variables('expected_lower_score',(),0,bar/scale))
    big_m=[];values={}
    for index,(key,pair) in enumerate(pairs.items()):
        supplied=list(cuts.get(key,()))
        # Zero ensures feasibility even when all nonzero direction cuts are negative.
        zero=squared_support_cut(pair,np.zeros(len(pair.c)+len(pair.F)),0,lower,upper)
        group=[zero]+supplied
        S=int(model.add_variables('lower_pair_'+str(index),(),0,bars[key]/scale));values[key]=S
        binaries=model.add_variables('choose_cut_'+str(index),len(group),0,1,integer=True)
        model.add_row({int(j):1 for j in binaries},1,1)
        for j,cut in enumerate(group):
            _verify_cut(pair,cut,lower,upper)
            minimum,_=affine_range(cut.intercept,cut.slope,lower,upper)
            M=max(0.,bars[key]-minimum)+1e-10*max(1.,bars[key],abs(minimum))
            row={S:1,int(binaries[j]):M/scale}
            row.update({int(i):-float(v)/scale for i,v in zip(indices.u_index.ravel(),cut.slope) if v})
            model.add_row(row,upper=(cut.intercept+M)/scale);big_m.append(M)
    for h in models:
        row={T:1};row.update({values[h,omega]:-w[omega] for omega in scenarios if w[omega]})
        model.add_row(row,upper=0)
    solved=model.solve({T:1},time_limit=time_limit,mip_rel_gap=mip_rel_gap)
    return dict(_record(model,indices,solved,scale,config),big_m_max=max(big_m,default=0),pair_qbar=bars,
        semantics='incumbent is a full frozen expected-proxy lower bound; master best-bound only bounds cut model',
        full_proxy_upper_bound=None,enumerated_complete_actions=0)


def expected_upper_master(config,groups,weights,witnesses,*,segments=4,time_limit=5,mip_rel_gap=1e-9):
    if not isinstance(segments,int) or segments<1:raise ValueError('positive integer segment count required')
    models,scenarios,w,pairs,lower,upper,bars,bar=_common(config,groups,weights,witnesses)
    model,indices=operation_model(config)
    scale=max(1.,max(bars.values()));T=int(model.add_variables('expected_upper_score',(),0,bar/scale))
    values={};errors=[]
    for index,(key,pair) in enumerate(pairs.items()):
        S=int(model.add_variables('upper_pair_'+str(index),(),0,bars[key]/scale));values[key]=S
        for j,z in enumerate(witnesses[key]):
            z=np.asarray(z,float)
            terms,const,error,_=_secant_sum(model,indices.u_index.ravel(),
                pair.c+pair.A@z,pair.B,float(np.sum((pair.F@z)**2)),segments=segments,
                tag=f'weighted_upper_{index}_{j}',scale=scale)
            row={S:1};row.update({i:-v for i,v in terms.items()})
            model.add_row(row,upper=const)
            errors.append(dict(competitor=key[0],scenario=key[1],witness=j,square_secant_error_upper=error))
    for h in models:
        row={T:1};row.update({values[h,omega]:-w[omega] for omega in scenarios if w[omega]})
        model.add_row(row,upper=0)
    solved=model.solve({T:1},time_limit=time_limit,mip_rel_gap=mip_rel_gap)
    record=_record(model,indices,solved,scale,config);bound=record['master_best_bound']
    valid_bound=bar if bound is None or not np.isfinite(bound) else min(bar,bound)
    return dict(record,full_proxy_upper_bound=float(valid_bound),outer_box_upper=float(bar),pair_qbar=bars,
        secant_errors=errors,semantics='global master best-bound upper-bounds full frozen expected proxy; incumbent does not',
        enumerated_complete_actions=0)


def evaluate_expected(groups,weights,u):
    models,scenarios,w,pairs=bank(groups,weights)
    answers={key:solve_pair(pair,u) for key,pair in pairs.items()}
    if any(not answer['success'] for answer in answers.values()):
        raise RuntimeError('required nuisance QP failed; no competitor/scenario removed')
    if any(answer['q_lower']>answer['q_upper'] for answer in answers.values()):
        raise RuntimeError('invalid QP lower/upper interval')
    lo={h:sum(w[omega]*answers[h,omega]['q_lower'] for omega in scenarios) for h in models}
    hi={h:sum(w[omega]*answers[h,omega]['q_upper'] for omega in scenarios) for h in models}
    return dict(score_lower=min(lo.values()),score_upper=min(hi.values()),
        competitor_lower=lo,competitor_upper=hi,answers=answers,
        objective='min_h sum_omega w_omega q_homega',is_EIG=False,topology_probability=False)


def optimize_expected(config,groups,weights,*,initial_plans=(),max_iterations=6,segments=2,
                      max_segments=16,solver_seconds=5,total_seconds=30,absolute_gap=1e-6,relative_gap=1e-6):
    """Generic frozen-proxy action generation via operations MILPs, no routes library."""
    if not isinstance(max_iterations,int) or max_iterations<1:raise ValueError('positive integer iterations required')
    _,_,_,pairs,lo,hi,_,UB=_common(config,groups,weights)
    cuts={key:[] for key in pairs};witnesses={key:[np.clip(np.zeros(len(p.lower)),p.lower,p.upper)] for key,p in pairs.items()}
    began=perf_counter();records=[];masters=[];cache={};best=None;LB=0.;qp_calls=0;failure=None
    def evaluate(plan,origin):
        nonlocal best,LB,qp_calls
        check=check_plan(plan,config,tolerance=2e-6)
        if not check['feasible']:raise ValueError('declared control plan is not operationally feasible')
        u=plan.u_kw;key=np.ascontiguousarray(u).tobytes()
        if key in cache:return
        answer=evaluate_expected(groups,weights,u);qp_calls+=len(pairs);cache[key]=True
        for pair_key,sol in answer['answers'].items():
            cuts[pair_key].append(cut_from_qp(pairs[pair_key],u,sol,lo,hi))
            witnesses[pair_key].append(sol['z'].copy())
        row=dict(origin=origin,u_kw=u.copy(),lower=answer['score_lower'],upper=answer['score_upper'],
            competitor_lower=answer['competitor_lower'],competitor_upper=answer['competitor_upper'],
            operation_max_violation=check['max_violation'],operation_check_tolerance=2e-6,
            QP_evaluated_at_raw_numerical_control=True)
        records.append(row)
        if best is None or row['lower']>LB:best=dict(row,plan=plan);LB=row['lower']
    model,indices=operation_model(config)
    seed=model.solve(time_limit=solver_seconds)
    if seed.x is None:raise RuntimeError('operations MILP could not produce a feasible seed')
    try:
        evaluate(decode_plan(seed,indices),'operations_MILP_seed')
        for plan in initial_plans:evaluate(plan,'declared_initial_plan')
        for iteration in range(max_iterations):
            if perf_counter()-began>=total_seconds:break
            lower=expected_lower_master(config,groups,weights,cuts,witnesses=witnesses,time_limit=solver_seconds)
            upper=expected_upper_master(config,groups,weights,witnesses,segments=segments,time_limit=solver_seconds)
            masters.extend([dict(kind=kind,status=int(master['result'].status),success=bool(master['result'].success),
                incumbent=master['incumbent'],master_best_bound=master['master_best_bound'],
                full_proxy_upper_bound=master.get('full_proxy_upper_bound'),variables=master['variables'],rows=master['rows'])
                for kind,master in [('lower',lower),('upper',upper)]])
            if lower['plan'] is not None:evaluate(lower['plan'],'lower_master_'+str(iteration))
            if upper['plan'] is not None:evaluate(upper['plan'],'upper_master_'+str(iteration))
            UB=min(UB,upper['full_proxy_upper_bound'])
            if LB>UB:raise RuntimeError('invalid expected-proxy design bounds; keep raw controls and numerical violations')
            if UB-LB<=absolute_gap+relative_gap*max(1.,abs(UB)):break
            segments=min(max_segments,2*segments)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
        failure=str(error)
    valid=LB<=UB
    met=failure is None and valid and UB-LB<=absolute_gap+relative_gap*max(1.,abs(UB))
    return dict(status='proxy_gap_met' if met else 'required_failure' if failure else 'budget_or_iteration_limit',
        failure=failure,plan=None if best is None else best['plan'],score_lower=LB,score_upper=UB,
        proxy_gap=UB-LB if valid else None,raw_proxy_gap=UB-LB,bounds_valid=valid,
        qp_calls=qp_calls,records=records,master_records=masters,
        seconds=perf_counter()-began,enumerated_complete_actions=0,
        scope='predeclared finite affine scenario proxy only',is_EIG=False,is_formal_policy=False,
        nonlinear_AC_certificate=False,weights_calibrated=False)


def source_hashes():
    files={Path(__file__),Path(inspect.getfile(FrozenPair)),Path(inspect.getfile(EVConfig))}
    files.add(Path(inspect.getfile(FrozenPair)).with_name('history_qp_acceleration.py'))
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
