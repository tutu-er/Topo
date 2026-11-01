"""Exact finite-scenario continuous validation-price search for fixed train fits.

The caller fixes all training prices/weights.  Validation residuals must then be
affine in the eight validation prices.  Both sides of the validation comparison
use exact absolute values and exact finite-family minima.  The model preserves
the original physical price domain; it never refits weights on validation.

This solver's 90% event is acceptance plus validation exclusion in the supplied
scenario table.  It is not a statistical coverage or topology recovery claim.
"""
from __future__ import annotations

import math
import time
import numpy as np

MICRO = 1.e6
BAND_EPSILON = 1.e-12
POSITIVE_GUARD_MICRO = 1.e-6
LEX_TOLERANCE_MICRO = 1.e-6


def _finite_attr(model, name):
    try:
        x = float(model.getAttr(name))
        return x if math.isfinite(x) and abs(x) < 1.e80 else None
    except Exception:
        return None


def _receipt(model, sense):
    incumbent = _finite_attr(model, 'ObjVal') if model.SolCount else None
    bound = _finite_attr(model, 'ObjBound')
    return dict(status_code=int(model.Status), objective_sense=sense,
        objective_value=incumbent, best_bound=bound,
        absolute_gap=None if incumbent is None or bound is None else abs(incumbent-bound),
        solver_reported_optimal=bool(model.Status == 2), solution_count=int(model.SolCount),
        runtime=_finite_attr(model, 'Runtime'), node_count=_finite_attr(model, 'NodeCount'))


def _compact_evaluation(problem, amplitudes):
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import evaluate_plan
    full = evaluate_plan(problem, amplitudes, tolerance=1.e-7)
    names = ('numerical_feasible', 'maximum_violation', 'violations', 'expected_cost',
        'hard_budget_bound', 'exact_box_worst_cost', 'exclusive_group_positive_counts',
        'actual_amplitude_lower', 'actual_amplitude_upper')
    return {k: full[k] for k in names}


def _tables(tables):
    if not tables:
        raise ValueError('at least one complete joint scenario is required')
    result = []
    for s, table in enumerate(tables):
        n = table['validation_size']
        if isinstance(n, (bool, np.bool_)) or int(n) != n or n <= 0:
            raise ValueError('validation_size must be a positive integer')
        gain = float(table['acceptance_margin'])
        if not np.isfinite(gain):
            raise ValueError('finite original training acceptance margin required')
        families, ids = [], set()
        for family in table['families']:
            identity = int(family['family_id'])
            if identity in ids:
                raise ValueError('family IDs must be distinct within a scenario')
            ids.add(identity)
            base = np.asarray(family['residual_base'], float)
            coefficient = np.asarray(family['residual_price'], float)
            constant = float(family['constant_loss'])
            if (base.ndim != 1 or coefficient.shape != (len(base), 8)
                    or not np.isfinite(base).all() or not np.isfinite(coefficient).all()
                    or not math.isfinite(constant) or constant < 0 or len(base) > n):
                raise ValueError('finite residual[M], residual_price[M,8], nonnegative normalized constant required')
            families.append(dict(family_id=identity, has_target=bool(family['has_target']),
                residual_base=base, residual_price=coefficient, constant_loss=constant))
        if not families:
            raise ValueError('each scenario must retain at least one complete historical family')
        available=bool(any(f['has_target'] for f in families) and any(not f['has_target'] for f in families))
        result.append(dict(validation_size=int(n), acceptance_margin=gain, families=families,
                           comparison_available=available))
    return result


def _price_groups(problem, fixed, caps):
    """Return disjoint groups on validation columns, including ungrouped singles."""
    groups, used = [], set()
    for original in problem.exclusive_groups:
        free = [int(j)-8 for j in original if j >= 8]
        fixed_count = sum(fixed[j] > 0 for j in original if j < 8)
        if fixed_count > 1:
            raise ValueError('fixed training amplitudes violate original type exclusivity')
        if fixed_count:
            caps[free] = 0.
        if free:
            groups.append(free); used.update(free)
    groups.extend([[j] for j in range(8) if j not in used])
    return groups


def _row_bounds(base, coefficients, caps, groups, beta, remaining):
    """Safe affine residual bounds using box/group and remaining quadratic budget."""
    positive = np.maximum(coefficients, 0.)
    negative = np.maximum(-coefficients, 0.)
    box_up = sum(float(np.max(positive[g]*caps[g], initial=0.)) for g in groups)
    box_down = sum(float(np.max(negative[g]*caps[g], initial=0.)) for g in groups)
    live = caps > 0
    ell_up = math.sqrt(max(0., remaining))*float(np.linalg.norm(positive[live]/np.sqrt(beta[live])))
    ell_down = math.sqrt(max(0., remaining))*float(np.linalg.norm(negative[live]/np.sqrt(beta[live])))
    up, down = min(box_up, ell_up), min(box_down, ell_down)
    # Outward rounding only; this does not tighten the physical price domain.
    pad = 1.e-10 + 64*np.finfo(float).eps*(1+abs(base)+box_up+box_down)
    return float(base-down-pad), float(base+up+pad)


def evaluate_affine(tables, prices, probability=.9):
    """Independent arithmetic from residual tables, without any solver variable."""
    tables = _tables(tables)
    prices = np.asarray(prices, float)
    if prices.shape == (16,):
        prices = prices[8:]
    if prices.shape != (8,) or not np.isfinite(prices).all():
        raise ValueError('eight validation prices or sixteen full prices required')
    k = int(math.ceil(float(probability)*len(tables)-1.e-12))
    margins, family_losses, acceptance, training_acceptance, available = [], [], [], [], []
    for table in tables:
        losses = [f['constant_loss'] + float(np.abs(f['residual_base']+f['residual_price']@prices).sum())/table['validation_size'] for f in table['families']]
        yes = [v for v,f in zip(losses, table['families']) if f['has_target']]
        no = [v for v,f in zip(losses, table['families']) if not f['has_target']]
        margins.append(min(no)-1.03*min(yes)-BAND_EPSILON if no and yes else None)
        family_losses.append(losses)
        training_acceptance.append(table['acceptance_margin'] > BAND_EPSILON)
        available.append(table['comparison_available'])
        acceptance.append(bool(training_acceptance[-1] and available[-1]))
    eligible = sorted((m*MICRO for m,a in zip(margins,acceptance) if a), reverse=True)
    quantile = eligible[k-1] if len(eligible) >= k else None
    passes = [bool(a and m is not None and m*MICRO > POSITIVE_GUARD_MICRO) for a,m in zip(acceptance,margins)]
    return dict(scenario_margins=margins, scenario_family_losses=family_losses,
        acceptance_passes=acceptance, training_acceptance_passes=training_acceptance,
        comparison_available=available, excitation_passes=passes,
        quantile_micro=quantile, gamma=quantile, required_scenarios=k,
        sample_requirement_met=bool(sum(passes) >= k),
        scenario_count=len(tables), excitation_pass_count=sum(passes))


def solve_affine(problem, tables, train_amplitudes16, seconds=30, probability=.9):
    """Maximize sample exclusion quantile, then minimize fee, exact on its table.

    Input residual coefficients multiply validation PRICES, not amplitudes.
    Training amplitudes0..7 are held exactly fixed; input8..15 must be zero.
    Original device/price/amplitude/type/budget constraints all remain active.
    """
    if problem.n_contracts != 16:
        raise ValueError('this isolated case solver requires sixteen original contracts')
    seconds, probability = float(seconds), float(probability)
    if not np.isfinite((seconds,probability)).all() or seconds <= 0 or not 0 < probability <= 1:
        raise ValueError('positive time and probability in (0,1] required')
    tables = _tables(tables)
    fixed = np.asarray(train_amplitudes16, float).copy()
    if fixed.shape != (16,) or not np.isfinite(fixed).all() or (fixed < 0).any() or np.any(fixed[8:] != 0):
        raise ValueError('nonnegative train_amplitudes16 must have an exactly zero validation half')
    if np.any(fixed > problem.amplitude_upper+1.e-12):
        raise ValueError('fixed training amplitude exceeds an original amplitude/price cap')
    started = time.monotonic()
    result = dict(status='not_solved', feasible_incumbent=False, has_incumbent=False,
        prices=None, amplitudes=None, primary=None, secondary=None,
        probability=probability, scenario_count=len(tables), required_scenarios=int(math.ceil(probability*len(tables)-1.e-12)),
        objective_units='micro-pu2 validation exclusion quantile',
        margin_formula='min(no-target validation losses)-1.03*min(with-target losses)-1e-12',
        positive_guard_micro=POSITIVE_GUARD_MICRO, fixed_train_amplitudes=fixed[:8].tolist(),
        scope='finite supplied scenarios and frozen training fits; not topology recovery or refit-domain guarantee')
    training_acceptance = [t['acceptance_margin'] > BAND_EPSILON for t in tables]
    available = [t['comparison_available'] for t in tables]
    acceptance = [bool(a and b) for a,b in zip(training_acceptance,available)]
    if sum(acceptance) < result['required_scenarios']:
        result.update(status='acceptance_infeasible', acceptance_passes=acceptance,
            training_acceptance_passes=training_acceptance,comparison_available=available,
            sample_requirement_met=False, solver_calls=0, elapsed_seconds=time.monotonic()-started)
        return result
    lam = np.asarray(problem.elasticities, float)
    cap_price = np.asarray(problem.amplitude_upper, float)[8:]/lam[8:]
    groups = _price_groups(problem, fixed, cap_price)
    expected = np.asarray(problem.expected_cost_coefficients, float)
    hard = np.asarray(problem.hard_cost_coefficients, float)
    fixed_cost = float(hard@(fixed*fixed))
    remaining = float(problem.budget-fixed_cost)
    if remaining < -1.e-10:
        result.update(status='fixed_training_budget_infeasible', sample_requirement_met=False,
            solver_calls=0, elapsed_seconds=time.monotonic()-started)
        return result
    remaining = max(0.,remaining)
    beta = hard[8:]*lam[8:]**2
    if remaining == 0:
        cap_price[:] = 0.
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True); env.setParam('OutputFlag',0); env.start()
    model = gp.Model('continuous_validation_exclusion',env=env)
    compilation = dict(constant_residuals=0, sign_stable_residuals=0, exact_abs_residuals=0,
        family_min_constraints=0, skipped_device_rows=0, duplicate_device_rows=0,
        exact_device_factor_abs=0,unavailable_comparison_scenarios=0)
    zero_starts=[]
    try:
        for name,value in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=0,NumericFocus=2,
                FeasibilityTol=1.e-9,OptimalityTol=1.e-9,BarQCPConvTol=1.e-8,
                MIPGap=1.e-8,MIPGapAbs=1.e-7,DualReductions=0).items():
            model.setParam(name,value)
        price = model.addVars(8,lb=0.,ub={j:float(cap_price[j]) for j in range(8)},name='validation_price')
        active = model.addVars(8,vtype=GRB.BINARY,name='validation_type')
        for j in range(8):
            model.addConstr(price[j] <= float(cap_price[j])*active[j])
            if cap_price[j] == 0:
                active[j].UB=0
        for gi,group in enumerate(groups):
            model.addConstr(gp.quicksum(active[j] for j in group) <= 1,name=f'type_group_{gi}')
        amplitude = [float(fixed[j]) for j in range(8)]+[float(lam[8+j])*price[j] for j in range(8)]
        hard_expression = fixed_cost+gp.quicksum(float(beta[j])*price[j]*price[j] for j in range(8))
        fee_expression = float(expected@(fixed*fixed))+gp.quicksum(float(expected[8+j]*lam[8+j]**2)*price[j]*price[j] for j in range(8))
        model.addQConstr(hard_expression <= float(problem.budget),name='original_hard_budget')

        # Robust device rows: A*x + ||A*diag(x)*D||_1 <= h, exactly as before.
        unique_rows=set()
        for i,(row,rhs) in enumerate(zip(problem.device_matrix,problem.device_rhs)):
            signature=(np.asarray(row).tobytes(),float(rhs))
            if signature in unique_rows:
                compilation['duplicate_device_rows']+=1; continue
            unique_rows.add(signature)
            mean = float(np.asarray(row[:8])@fixed[:8])+gp.quicksum(float(row[8+j]*lam[8+j])*price[j] for j in range(8) if row[8+j] != 0)
            has_mean = bool(np.any(row[8:]*lam[8:]*cap_price != 0))
            constant_norm=0.; pieces=[]
            for k in range(problem.n_factors):
                const=float((row[:8]*fixed[:8])@problem.response_factors[:8,k])
                coeff=np.asarray(row[8:])*lam[8:]*problem.response_factors[8:,k]
                if not np.any(coeff*cap_price != 0):
                    constant_norm+=abs(const); continue
                expression=const+gp.quicksum(float(coeff[j])*price[j] for j in range(8) if coeff[j] != 0)
                absolute=model.addVar(lb=0.,name=f'device_abs_{i}_{k}')
                model.addConstr(absolute>=expression);model.addConstr(absolute>=-expression)
                zero_starts.append((absolute,abs(const)))
                pieces.append(absolute); compilation['exact_device_factor_abs']+=1
            if not has_mean and not pieces:
                if float(row[:8]@fixed[:8])+constant_norm <= float(rhs)+1.e-12:
                    compilation['skipped_device_rows']+=1; continue
            model.addConstr(mean+constant_norm+gp.quicksum(pieces) <= float(rhs),name=f'original_device_{i}')

        scenario_models=[]
        for s,table in enumerate(tables):
            if not table['comparison_available']:
                scenario_models.append(dict(var=None,lower=0.,upper=0.,zero_margin=None))
                compilation['unavailable_comparison_scenarios']+=1
                continue
            records=[]
            for fi,family in enumerate(table['families']):
                constant=family['constant_loss']*MICRO
                expression=gp.LinExpr(constant); lower=constant; upper=constant
                for i,(base,coeff) in enumerate(zip(family['residual_base']*MICRO,family['residual_price']*MICRO)):
                    if not np.any(coeff*cap_price != 0):
                        contribution=abs(float(base))/table['validation_size']
                        expression+=contribution;lower+=contribution;upper+=contribution
                        compilation['constant_residuals']+=1; continue
                    lo,hi=_row_bounds(float(base),coeff,cap_price,groups,beta,remaining)
                    affine=float(base)+gp.quicksum(float(coeff[j])*price[j] for j in range(8) if coeff[j] != 0)
                    minimum=0. if lo <= 0 <= hi else min(abs(lo),abs(hi))
                    maximum=max(abs(lo),abs(hi))
                    lower+=minimum/table['validation_size'];upper+=maximum/table['validation_size']
                    if lo >= 0:
                        term=affine;compilation['sign_stable_residuals']+=1
                    elif hi <= 0:
                        term=-affine;compilation['sign_stable_residuals']+=1
                    else:
                        residual=model.addVar(lb=lo,ub=hi,name=f'residual_{s}_{fi}_{i}')
                        absolute=model.addVar(lb=0.,ub=maximum,name=f'absolute_{s}_{fi}_{i}')
                        model.addConstr(residual==affine)
                        model.addGenConstrAbs(absolute,residual,name=f'exact_abs_{s}_{fi}_{i}')
                        zero_starts.extend([(residual,float(base)),(absolute,abs(float(base)))])
                        term=absolute;compilation['exact_abs_residuals']+=1
                    expression+=term/table['validation_size']
                loss=model.addVar(lb=lower,ub=upper,name=f'validation_loss_{s}_{fi}')
                model.addConstr(loss==expression)
                zero_loss=(family['constant_loss']+float(np.abs(family['residual_base']).sum())/table['validation_size'])*MICRO
                zero_starts.append((loss,zero_loss))
                records.append(dict(var=loss,lower=lower,upper=upper,target=family['has_target'],zero_loss=zero_loss))
            minima=[];bounds=[]
            for target in (False,True):
                part=[r for r in records if r['target']==target]
                lo=min(r['lower'] for r in part);hi=min(r['upper'] for r in part)
                if len(part)==1:
                    var=part[0]['var']
                else:
                    var=model.addVar(lb=lo,ub=hi,name=f'exact_min_{s}_{int(target)}')
                    model.addGenConstrMin(var,[r['var'] for r in part])
                    zero_starts.append((var,min(r['zero_loss'] for r in part)))
                    compilation['family_min_constraints']+=1
                minima.append(var);bounds.append((lo,hi))
            lo=bounds[0][0]-1.03*bounds[1][1]-BAND_EPSILON*MICRO
            hi=bounds[0][1]-1.03*bounds[1][0]-BAND_EPSILON*MICRO
            margin=model.addVar(lb=lo,ub=hi,name=f'scenario_margin_{s}')
            model.addConstr(margin==minima[0]-1.03*minima[1]-BAND_EPSILON*MICRO)
            zero_no=min(r['zero_loss'] for r in records if not r['target'])
            zero_yes=min(r['zero_loss'] for r in records if r['target'])
            zero_margin=zero_no-1.03*zero_yes-BAND_EPSILON*MICRO
            zero_starts.append((margin,zero_margin))
            scenario_models.append(dict(var=margin,lower=lo,upper=hi,zero_margin=zero_margin))
        k=result['required_scenarios']
        upper=sorted((r['upper'] for r,a in zip(scenario_models,acceptance) if a),reverse=True)[k-1]
        lower=sorted((r['lower'] for r,a in zip(scenario_models,acceptance) if a),reverse=True)[k-1]
        zero_eval=_compact_evaluation(problem,fixed)
        baseline=evaluate_affine(tables,np.r_[fixed[:8]/lam[:8],np.zeros(8)],probability)
        if zero_eval['numerical_feasible']:
            lower=max(lower,float(baseline['quantile_micro'])-1.e-6)
        if lower > upper:
            if lower-upper > 1.e-6:
                raise AssertionError('quantile bounds contradict a feasible baseline')
            lower=upper
        gamma=model.addVar(lb=float(lower),ub=float(upper),name='quantile_micro')
        flags=model.addVars(len(tables),vtype=GRB.BINARY,name='scenario_selected')
        for s,r in enumerate(scenario_models):
            if not acceptance[s]:
                flags[s].UB=0;continue
            big_m=max(0.,upper-r['lower'])+1.e-8
            model.addConstr(r['var'] >= gamma-big_m*(1-flags[s]),name=f'quantile_selection_{s}')
        model.addConstr(gp.quicksum(flags[s] for s in range(len(tables))) >= k)
        if zero_eval['numerical_feasible']:
            for j in range(8):price[j].Start=0.;active[j].Start=0.
            for var,value in zero_starts:var.Start=float(value)
            start_gamma=min(upper,max(lower,float(baseline['quantile_micro'])-5.e-7))
            gamma.Start=float(start_gamma)
            selected=sorted((s for s in range(len(tables)) if acceptance[s]),
                key=lambda s:(-scenario_models[s]['zero_margin'],s))[:k]
            for s in range(len(tables)):flags[s].Start=1. if s in selected else 0.
            result['zero_start_supplied']=True
            result['zero_start_variable_count']=16+len(zero_starts)+1+len(tables)
            result['zero_start_quantile_micro']=float(start_gamma)
        else:
            result['zero_start_supplied']=False
        model.setObjective(gamma,GRB.MAXIMIZE)
        model.Params.TimeLimit=max(.001,seconds-(time.monotonic()-started))
        model.optimize();result['primary']=_receipt(model,'maximize_validation_quantile_micro')
        result['compilation']=compilation
        result['model_size']=dict(variables=int(model.NumVars),binary_variables=int(model.NumBinVars),
            linear_constraints=int(model.NumConstrs),quadratic_constraints=int(model.NumQConstrs),general_constraints=int(model.NumGenConstrs))
        result['solver_calls']=1
        if not model.SolCount:
            result.update(status='no_primary_incumbent',elapsed_seconds=time.monotonic()-started)
            return result

        def extract():
            validation_prices=np.array([price[j].X for j in range(8)],float)
            snapped=validation_prices.copy()
            snapped[np.abs(snapped)<=1.e-10]=0.
            close=np.abs(snapped-cap_price)<=1.e-10;snapped[close]=cap_price[close]
            amplitudes=fixed.copy();amplitudes[8:]=lam[8:]*snapped
            prices=np.r_[fixed[:8]/lam[:8],snapped]
            values=evaluate_affine(tables,prices,probability)
            original=_compact_evaluation(problem,amplitudes)
            quantile=values['quantile_micro'];solver_gamma=float(gamma.X)
            feasible=bool(original['numerical_feasible'] and quantile is not None and solver_gamma <= quantile+1.e-5)
            return dict(prices=prices.tolist(),amplitudes=amplitudes.tolist(),**values,
                original_evaluation=original,feasible_incumbent=feasible,
                optimizer_gamma_micro=solver_gamma,numerical_price_snap_max=float(np.max(np.abs(snapped-validation_prices),initial=0.)))

        chosen=extract();chosen_stage='primary'
        result['primary_recomputed_quantile_micro']=chosen['quantile_micro']
        remaining_time=seconds-(time.monotonic()-started)
        if model.Status==GRB.OPTIMAL and chosen['feasible_incumbent'] and remaining_time>1.:
            optimum=min(float(gamma.X),chosen['quantile_micro'])
            floor=optimum-LEX_TOLERANCE_MICRO
            if optimum>POSITIVE_GUARD_MICRO:
                floor=min(optimum,max(floor,POSITIVE_GUARD_MICRO+1.e-9))
            model.addConstr(gamma>=float(floor),name='preserve_primary_quantile')
            model.setObjective(fee_expression,GRB.MINIMIZE)
            model.Params.TimeLimit=max(.001,seconds-(time.monotonic()-started))
            model.optimize();result['secondary']=_receipt(model,'minimize_original_expected_payment')
            result['solver_calls']+=1;result['secondary_quantile_floor_micro']=floor
            if model.SolCount:
                candidate=extract()
                if candidate['feasible_incumbent'] and candidate['quantile_micro']>=floor-1.e-5:
                    chosen=candidate;chosen_stage='secondary'
                else:
                    result['secondary_rejected']=dict(reason='independent physical/quantile check failed',
                        quantile_micro=candidate['quantile_micro'],original_evaluation=candidate['original_evaluation'])
        else:
            result['secondary_skipped_reason']='primary not optimal/feasible or shared time exhausted'
        result.update(chosen,has_incumbent=True,chosen_stage=chosen_stage,
            status='feasible' if chosen['feasible_incumbent'] else 'incumbent_failed_independent_check',
            elapsed_seconds=time.monotonic()-started)
        result['global_finite_scenario_optimality']=bool(result['primary']['solver_reported_optimal'] and chosen['feasible_incumbent'])
        return result
    finally:
        model.dispose();env.dispose()
