"""Exact finite-scenario paired prediction-distance-increase price planning.

All topology parameters are fixed by the caller before this module is used.
For each complete random scenario the two-model prediction difference is
``base + price_coeff @ c``. The event concerns increase in its normalized L1
norm relative to the same zero-DR background. No voltage target, weight fit,
topology search, or recovery probability is used here.
"""
from __future__ import annotations

import math
import time
import numpy as np

MICRO = 1.e6
POSITIVE_GUARD_MICRO = 1.e-6
LEX_TOLERANCE_MICRO = 1.e-6


def _finite_attr(model, name):
    try:
        value=float(model.getAttr(name))
        return value if math.isfinite(value) and abs(value)<1.e80 else None
    except Exception:
        return None


def _receipt(model, sense):
    incumbent=_finite_attr(model,'ObjVal') if model.SolCount else None
    bound=_finite_attr(model,'ObjBound')
    return dict(status_code=int(model.Status),objective_sense=sense,
        objective_value=incumbent,best_bound=bound,
        absolute_gap=None if incumbent is None or bound is None else abs(incumbent-bound),
        solver_reported_optimal=bool(model.Status==2),solution_count=int(model.SolCount),
        runtime=_finite_attr(model,'Runtime'),node_count=_finite_attr(model,'NodeCount'))


def _check_original(problem, amplitudes):
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import evaluate_plan
    full=evaluate_plan(problem,amplitudes,tolerance=1.e-7)
    names=('numerical_feasible','maximum_violation','violations','expected_cost',
        'hard_budget_bound','exact_box_worst_cost','exclusive_group_positive_counts',
        'actual_amplitude_lower','actual_amplitude_upper')
    return {name:full[name] for name in names}


def _scenarios(scenarios, n_contracts):
    if not scenarios:
        raise ValueError('at least one complete random scenario is required')
    out=[]
    for scenario in scenarios:
        base=np.asarray(scenario['base'],float)
        coefficient=np.asarray(scenario['price_coeff'],float)
        normalizer=float(scenario['normalizer'])
        if (base.ndim!=1 or coefficient.shape!=(len(base),n_contracts)
                or not np.isfinite(base).all() or not np.isfinite(coefficient).all()
                or not math.isfinite(normalizer) or normalizer<=0):
            raise ValueError('finite base[M], price_coeff[M,J], positive normalizer required')
        out.append(dict(base=base,price_coeff=coefficient,normalizer=normalizer))
    return out


def evaluate_distance(scenarios, prices, probability=.9):
    """Recompute the exact paired gains directly, without solver variables."""
    prices=np.asarray(prices,float)
    probability=float(probability)
    if prices.ndim!=1 or not np.isfinite(prices).all() or not 0<probability<=1:
        raise ValueError('finite price vector and probability in (0,1] required')
    scenarios=_scenarios(scenarios,len(prices))
    base_distance=[];controlled_distance=[];gains=[]
    for scenario in scenarios:
        base=scenario['base'];delta=scenario['price_coeff']@prices
        # Coordinate differences avoid subtracting large unchanged totals.
        gains.append(float((np.abs(base+delta)-np.abs(base)).sum()/scenario['normalizer']))
        base_distance.append(float(np.abs(base).sum()/scenario['normalizer']))
        controlled_distance.append(float(np.abs(base+delta).sum()/scenario['normalizer']))
    required=max(1,int(math.ceil(probability*len(scenarios)-1.e-12)))
    quantile=sorted((MICRO*v for v in gains),reverse=True)[required-1]
    passes=[bool(MICRO*v>POSITIVE_GUARD_MICRO) for v in gains]
    return dict(scenario_gain_native=gains,scenario_base_distance_native=base_distance,
        scenario_prediction_distance_native=controlled_distance,quantile_micro=quantile,
        gamma=quantile,scenario_positive_passes=passes,positive_pass_count=sum(passes),
        scenario_count=len(scenarios),required_scenarios=required,
        sample_requirement_met=bool(sum(passes)>=required))


def _bounds(base, coefficient, caps, groups, beta, budget):
    positive=np.maximum(coefficient,0.);negative=np.maximum(-coefficient,0.)
    box_up=sum(float(np.max(positive[group]*caps[group],initial=0.)) for group in groups)
    box_down=sum(float(np.max(negative[group]*caps[group],initial=0.)) for group in groups)
    live=caps>0
    ell_up=math.sqrt(max(0.,budget))*float(np.linalg.norm(positive[live]/np.sqrt(beta[live])))
    ell_down=math.sqrt(max(0.,budget))*float(np.linalg.norm(negative[live]/np.sqrt(beta[live])))
    pad=1.e-10+64*np.finfo(float).eps*(1+abs(base)+box_up+box_down)
    return float(base-min(box_down,ell_down)-pad),float(base+min(box_up,ell_up)+pad)


def solve_distance(problem, scenarios, seconds=60, probability=.9, warm_start_prices=None):
    """Optimize continuous prices and mutually exclusive module types.

    Stage one maximizes the supplied-scenario lower90% paired-gain quantile;
    stage two minimizes original expected payment within1e-6 micro-pu2 of it.
    This is a finite-scenario result, not an actual-probability certificate.
    Optional warm starts are full original-domain price vectors. They only
    provide verified incumbents; they do not restrict the continuous domain.
    """
    seconds=float(seconds);probability=float(probability)
    if not np.isfinite((seconds,probability)).all() or seconds<=0 or not 0<probability<=1:
        raise ValueError('positive time and probability in (0,1] required')
    n=int(problem.n_contracts)
    scenarios=_scenarios(scenarios,n)
    started=time.monotonic()
    required=max(1,int(math.ceil(probability*len(scenarios)-1.e-12)))
    result=dict(status='not_solved',has_incumbent=False,feasible_incumbent=False,
        prices=None,amplitudes=None,primary=None,secondary=None,
        scenario_count=len(scenarios),required_scenarios=required,probability=probability,
        positive_guard_micro=POSITIVE_GUARD_MICRO,
        objective_units='micro-pu2 paired frozen-model prediction-distance increase',
        probability_scope='supplied-scenario excitation rate only; not topology recovery or field probability',
        objective_formula='(||base+price_coeff@prices||1-||base||1)/normalizer',
        no_parameter_refit=True,no_voltage_target=True)
    zero=np.zeros(n)
    zero_check=_check_original(problem,zero)
    result['zero_plan_evaluation']=zero_check
    if not zero_check['numerical_feasible']:
        result.update(status='zero_plan_infeasible',solver_calls=0,elapsed_seconds=time.monotonic()-started)
        return result
    lam=np.asarray(problem.elasticities,float)
    caps=np.asarray(problem.amplitude_upper,float)/lam
    if problem.budget==0:
        caps[:]=0.
    beta=np.asarray(problem.hard_cost_coefficients,float)*lam**2
    expected_beta=np.asarray(problem.expected_cost_coefficients,float)*lam**2
    groups=[list(group) for group in problem.exclusive_groups]
    used={j for group in groups for j in group}
    groups.extend([[j] for j in range(n) if j not in used])
    best_start=dict(prices=zero.copy(),evaluation=evaluate_distance(scenarios,zero,probability),
                    original_evaluation=zero_check,index=None)
    provided=0;verified=0;rejections=[]
    if warm_start_prices is not None:
        if isinstance(warm_start_prices,np.ndarray) and warm_start_prices.shape==(n,):
            warm_start_prices=[warm_start_prices]
        for index,candidate in enumerate(warm_start_prices):
            provided+=1
            try:
                candidate=np.asarray(candidate,float)
                if candidate.shape!=(n,) or not np.isfinite(candidate).all() or np.any(candidate<0) or np.any(candidate>caps+1.e-12):
                    raise ValueError('warm quote violates original price vector/caps')
                candidate=np.minimum(candidate,caps)
                amplitude=lam*candidate
                if any(np.count_nonzero(amplitude[list(group)]>0)>1 for group in problem.exclusive_groups):
                    raise ValueError('warm quote activates multiple original types on one resource')
                original=_check_original(problem,amplitude)
                if not original['numerical_feasible']:
                    raise ValueError('warm quote fails original all-support device/budget checks')
                evaluation=evaluate_distance(scenarios,candidate,probability)
                verified+=1
                current=evaluation['quantile_micro'];incumbent=best_start['evaluation']['quantile_micro']
                better=current>incumbent+1.e-8
                cheaper=abs(current-incumbent)<=1.e-8 and original['expected_cost']<best_start['original_evaluation']['expected_cost']
                if better or cheaper:
                    best_start=dict(prices=candidate.copy(),evaluation=evaluation,original_evaluation=original,index=index)
            except Exception as error:
                rejections.append(dict(index=index,error_type=type(error).__name__,message=str(error)))
    result.update(provided_warm_start_count=provided,verified_warm_start_count=verified,
        warm_start_rejections=rejections,warm_start_best_index=best_start['index'],
        warm_start_best_prices=best_start['prices'].tolist(),
        warm_start_best_quantile_micro=best_start['evaluation']['quantile_micro'],
        warm_start_best_expected_cost=best_start['original_evaluation']['expected_cost'],
        warm_start_scope='verified computational incumbents only; original continuous optimization domain unchanged')

    import gurobipy as gp
    from gurobipy import GRB
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    model=gp.Model('paired_frozen_prediction_distance',env=env)
    compilation=dict(constant_coordinates_skipped=0,sign_stable_coordinates=0,
        exact_abs_coordinates=0,duplicate_device_rows=0,zero_device_rows=0,
        robust_device_abs=0)
    zero_starts=[];warm_records=[]
    try:
        for key,value in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=0,NumericFocus=2,
                FeasibilityTol=1.e-9,OptimalityTol=1.e-9,BarQCPConvTol=1.e-8,
                MIPGap=1.e-8,MIPGapAbs=1.e-7,DualReductions=0).items():
            model.setParam(key,value)
        price=model.addVars(n,lb=0.,ub={j:float(caps[j]) for j in range(n)},name='price')
        activate=model.addVars(n,vtype=GRB.BINARY,name='module_type')
        for j in range(n):
            model.addConstr(price[j]<=float(caps[j])*activate[j])
            if caps[j]==0:activate[j].UB=0.
        for gi,group in enumerate(groups):
            model.addConstr(gp.quicksum(activate[j] for j in group)<=1,name=f'one_type_{gi}')
        model.addQConstr(gp.quicksum(float(beta[j])*price[j]*price[j] for j in range(n))<=float(problem.budget),name='original_hard_budget')
        expected_fee=gp.quicksum(float(expected_beta[j])*price[j]*price[j] for j in range(n))

        signatures=set()
        for i,(row,rhs) in enumerate(zip(problem.device_matrix,problem.device_rhs)):
            signature=(np.asarray(row).tobytes(),float(rhs))
            if signature in signatures:
                compilation['duplicate_device_rows']+=1;continue
            signatures.add(signature)
            linear=np.asarray(row)*lam
            if not np.any(linear*caps!=0):
                if float(rhs)<-1.e-12:raise AssertionError('zero-domain check and device row disagree')
                compilation['zero_device_rows']+=1;continue
            expression=gp.quicksum(float(linear[j])*price[j] for j in range(n) if linear[j]!=0)
            for k in range(problem.n_factors):
                coefficient=linear*np.asarray(problem.response_factors[:,k])
                if not np.any(coefficient*caps!=0):continue
                factor=gp.quicksum(float(coefficient[j])*price[j] for j in range(n) if coefficient[j]!=0)
                absolute=model.addVar(lb=0.,name=f'device_abs_{i}_{k}')
                model.addConstr(absolute>=factor);model.addConstr(absolute>=-factor)
                expression+=absolute;zero_starts.append((absolute,0.))
                warm_records.append((absolute,'abs_linear',coefficient.copy()))
                compilation['robust_device_abs']+=1
            model.addConstr(expression<=float(rhs),name=f'original_device_{i}')

        scenario_vars=[]
        for s,scenario in enumerate(scenarios):
            expression=gp.LinExpr(0.);lower=0.;upper=0.
            normalizer=scenario['normalizer']
            for i,(base,coefficient) in enumerate(zip(scenario['base']*MICRO,scenario['price_coeff']*MICRO)):
                if not np.any(coefficient*caps!=0):
                    compilation['constant_coordinates_skipped']+=1;continue
                lo,hi=_bounds(float(base),coefficient,caps,groups,beta,float(problem.budget))
                minimum=0. if lo<=0<=hi else min(abs(lo),abs(hi))
                maximum=max(abs(lo),abs(hi))
                lower+=(minimum-abs(float(base)))/normalizer
                upper+=(maximum-abs(float(base)))/normalizer
                linear=gp.quicksum(float(coefficient[j])*price[j] for j in range(n) if coefficient[j]!=0)
                # Structural nonnegative/nonpositive tests avoid fake crossings
                # introduced solely by outward numerical interval rounding.
                live=coefficient[caps>0]
                positive=(base>=0 and np.all(live>=0)) or lo>=0
                negative=(base<=0 and np.all(live<=0)) or hi<=0
                if positive:
                    expression+=linear/normalizer
                    compilation['sign_stable_coordinates']+=1
                elif negative:
                    expression-=linear/normalizer
                    compilation['sign_stable_coordinates']+=1
                else:
                    residual=model.addVar(lb=lo,ub=hi,name=f'prediction_difference_{s}_{i}')
                    absolute=model.addVar(lb=0.,ub=maximum,name=f'exact_absolute_{s}_{i}')
                    model.addConstr(residual==float(base)+linear)
                    model.addGenConstrAbs(absolute,residual)
                    expression+=(absolute-abs(float(base)))/normalizer
                    zero_starts.extend([(residual,float(base)),(absolute,abs(float(base)))])
                    warm_records.extend([(residual,'affine',(float(base),coefficient.copy())),
                                         (absolute,'abs_affine',(float(base),coefficient.copy()))])
                    compilation['exact_abs_coordinates']+=1
            # Zero price is feasible and its paired gain is identically zero.
            lower=min(lower,0.);upper=max(upper,0.)
            gain=model.addVar(lb=lower,ub=upper,name=f'paired_gain_micro_{s}')
            model.addConstr(gain==expression)
            zero_starts.append((gain,0.))
            warm_records.append((gain,'gain',s))
            scenario_vars.append(dict(var=gain,lower=lower,upper=upper))
        gamma_upper=sorted((s['upper'] for s in scenario_vars),reverse=True)[required-1]
        gamma=model.addVar(lb=0.,ub=max(0.,gamma_upper),name='quantile_micro')
        flags=model.addVars(len(scenarios),vtype=GRB.BINARY,name='scenario_selected')
        for s,record in enumerate(scenario_vars):
            big_m=max(0.,gamma_upper-record['lower'])+1.e-8
            model.addConstr(record['var']>=gamma-big_m*(1-flags[s]),name=f'quantile_event_{s}')
        model.addConstr(gp.quicksum(flags[s] for s in range(len(scenarios)))>=required)
        starts=[best_start]
        if np.any(best_start['prices']!=0):
            starts.append(dict(prices=zero.copy(),evaluation=evaluate_distance(scenarios,zero,probability),
                original_evaluation=zero_check,index=None))
        model.NumStart=len(starts)
        for start_number,point in enumerate(starts):
            model.Params.StartNumber=start_number
            c_start=point['prices'];evaluation=point['evaluation']
            for j in range(n):
                price[j].Start=float(c_start[j]);activate[j].Start=1. if c_start[j]>0 else 0.
            for var,value in zero_starts:var.Start=float(value)
            for var,kind,data in warm_records:
                if kind=='abs_linear':value=abs(float(data@c_start))
                elif kind=='affine':value=float(data[0]+data[1]@c_start)
                elif kind=='abs_affine':value=abs(float(data[0]+data[1]@c_start))
                elif kind=='gain':value=float(evaluation['scenario_gain_native'][data]*MICRO)
                else:raise AssertionError('unknown warm-start variable role')
                var.Start=value
            start_gamma=max(0.,min(gamma_upper,evaluation['quantile_micro']-5.e-7))
            gamma.Start=float(start_gamma)
            selected=sorted(range(len(scenarios)),key=lambda s:(-evaluation['scenario_gain_native'][s],s))[:required]
            for s in range(len(scenarios)):flags[s].Start=1. if s in selected else 0.
        model.Params.StartNumber=0
        result['complete_zero_start_supplied']=True
        result['complete_verified_nonzero_start_supplied']=bool(np.any(best_start['prices']!=0))
        result['mip_start_count']=len(starts)
        result['zero_start_variable_count']=2*n+len(zero_starts)+1+len(scenarios)
        model.setObjective(gamma,GRB.MAXIMIZE)
        model.Params.TimeLimit=max(.001,seconds-(time.monotonic()-started))
        model.optimize();result['primary']=_receipt(model,'maximize_paired_distance_quantile_micro')
        result['solver_calls']=1;result['compilation']=compilation
        result['model_size']=dict(variables=int(model.NumVars),binary_variables=int(model.NumBinVars),
            linear_constraints=int(model.NumConstrs),quadratic_constraints=int(model.NumQConstrs),general_constraints=int(model.NumGenConstrs))
        if not model.SolCount:
            result.update(status='no_primary_incumbent',elapsed_seconds=time.monotonic()-started)
            return result

        def extract():
            raw=np.array([price[j].X for j in range(n)],float)
            chosen=raw.copy();chosen[np.abs(chosen)<=1.e-10]=0.
            close=np.abs(chosen-caps)<=1.e-10;chosen[close]=caps[close]
            amplitude=lam*chosen
            check=_check_original(problem,amplitude)
            prediction=evaluate_distance(scenarios,chosen,probability)
            solver_gamma=float(gamma.X)
            feasible=bool(check['numerical_feasible'] and solver_gamma<=prediction['quantile_micro']+1.e-5)
            return dict(prices=chosen.tolist(),amplitudes=amplitude.tolist(),**prediction,
                optimizer_gamma_micro=solver_gamma,original_evaluation=check,
                expected_cost=check['expected_cost'],hard_budget_bound=check['hard_budget_bound'],
                maximum_violation=check['maximum_violation'],violations=check['violations'],
                feasible_incumbent=feasible,numerical_price_snap_max=float(np.max(np.abs(chosen-raw),initial=0.)))

        chosen=extract();chosen_stage='primary'
        result['primary_recomputed_quantile_micro']=chosen['quantile_micro']
        if model.Status==GRB.OPTIMAL and chosen['feasible_incumbent'] and seconds-(time.monotonic()-started)>1.:
            optimum=min(float(gamma.X),chosen['quantile_micro'])
            floor=max(0.,optimum-LEX_TOLERANCE_MICRO)
            if optimum>POSITIVE_GUARD_MICRO:
                floor=min(optimum,max(floor,POSITIVE_GUARD_MICRO+1.e-9))
            model.addConstr(gamma>=float(floor),name='preserve_primary_quantile')
            model.setObjective(expected_fee,GRB.MINIMIZE)
            model.Params.TimeLimit=max(.001,seconds-(time.monotonic()-started))
            model.optimize();result['secondary']=_receipt(model,'minimize_original_expected_payment')
            result['solver_calls']+=1;result['secondary_quantile_floor_micro']=float(floor)
            if model.SolCount:
                candidate=extract()
                if candidate['feasible_incumbent'] and candidate['quantile_micro']>=floor-1.e-5:
                    chosen=candidate;chosen_stage='secondary'
                else:
                    result['secondary_rejected']=dict(reason='physical or paired-quantile replay failed',
                        quantile_micro=candidate['quantile_micro'],original_evaluation=candidate['original_evaluation'])
        else:
            result['secondary_skipped_reason']='primary not optimal/feasible or shared time exhausted'
        result.update(chosen,has_incumbent=True,chosen_stage=chosen_stage,
            status='feasible' if chosen['feasible_incumbent'] else 'incumbent_failed_independent_check',
            elapsed_seconds=time.monotonic()-started)
        result['conditional_finite_scenario_optimality']=bool(result['primary']['solver_reported_optimal'] and chosen['feasible_incumbent'])
        result['optimality_scope']='primary supplied-scenario quantile only; secondary fee optimum separately recorded'
        return result
    finally:
        model.dispose();env.dispose()
