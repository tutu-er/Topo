"""Independent scalar audit of the affine validation-price inner problem.

No solver, actual future, true network, or observed trajectory is read here.
The training-fit provenance belongs to the table producer; this audit replays
the exact ABS table, full-family minima, chance denominator and physical domain.

Schema:
  tables[s] = {validation_size: Nv, acceptance_margin: float,
      families: [{family_id: id, has_target: bool, constant_loss: float,
                  residual_base: (M,), residual_price: (M,8)}]}
  residual_price multiplies the LAST EIGHT entries of full prices16.
  constant_loss is already divided by Nv; the residual ABS sum is not.

The optimizer may have a feasible negative-quantile incumbent.  That is a
feasible PRICE, and this audit expressly does not call it a 90% excitation pass.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping

import numpy as np


def _finite_array(value, name, shape=None):
    result = np.asarray(value, dtype=float)
    if shape is not None and result.shape != shape:
        raise ValueError(f'{name}: shape {result.shape} differs from {shape}')
    if not np.isfinite(result).all():
        raise ValueError(f'{name}: nonfinite value')
    return result


def _finite_scalar(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'{name}: nonfinite value')
    return result


def audit_inner(problem, tables, train_amplitudes, result, *, probability=.9,
                native_tolerance=1e-12, physical_tolerance=1e-7,
                excitation_guard=1e-12):
    """Return scalar checks without solving or modifying any supplied objects.

    gamma/quantile_micro are in 10^-6 p.u.^2; family losses and margins are
    native p.u.^2.  Exactly ceil(probability*S) successes are required from the
    FULL supplied scenario denominator, including failed acceptance states.

    This function does not validate an epsilon-optimal parameter face, the
    completeness of a historical tree pool, or out-of-sample probability.
    """
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import (
        PriceProblem, evaluate_plan,
    )

    checks = []
    replay = []

    def check(name, condition, **details):
        checks.append(dict(name=name, passed=bool(condition), **details))
        return bool(condition)

    def close(a, b, atol=native_tolerance):
        return abs(float(a)-float(b)) <= atol

    def phase_receipt(name, phase, maximizing):
        if not isinstance(phase, Mapping):
            return dict(scope='phase receipt not supplied')
        # Keep absence as None, never replace unavailable bounds/gaps by zero.
        receipt = {key: copy.deepcopy(phase.get(key)) for key in
            ('status', 'status_code', 'solution_count', 'ObjVal', 'objective_value',
             'objective_sense', 'best_bound', 'absolute_gap', 'gap',
             'solver_reported_optimal', 'chosen_stage', 'runtime', 'seconds',
             'stopping_reason')}
        incumbent = phase.get('objective_value',phase.get('ObjVal',phase.get('objective')))
        bound = phase.get('best_bound')
        if incumbent is not None and bound is not None:
            incumbent = _finite_scalar(incumbent, name+' incumbent')
            bound = _finite_scalar(bound, name+' bound')
            tol = 1e-6*max(1., abs(incumbent), abs(bound))
            good = bound+tol >= incumbent if maximizing else bound-tol <= incumbent
            check(name+'_bound_direction', good, incumbent=incumbent, bound=bound)
            if phase.get('absolute_gap') is not None:
                check(name+'_absolute_gap',close(phase['absolute_gap'],abs(incumbent-bound),tol))
        return receipt

    audit = dict(scope='exact affine table and feasible-price replay; not topology probability or statistical coverage',
        solver_calls=0, new_actual_future=0, new_AC=0, new_RNJ_W=0,
        scenario_denominator=len(tables), probability_target=float(probability),
        acceptance_guard=float(excitation_guard), validation_guard=float(excitation_guard),
        training_refit_provenance_audited=False,
        original_3pct_rule_changed=False, old_linear_benefit_used_for_excitation=False,
        phases={}, scenarios=replay)
    try:
        if isinstance(problem, Mapping):
            problem = PriceProblem.from_dict(problem)
        if not 0 < probability <= 1 or not tables:
            raise ValueError('a nonempty scenario table and probability in (0,1] are required')
        if native_tolerance <= 0 or physical_tolerance <= 0 or excitation_guard < 0:
            raise ValueError('invalid numeric tolerances')
        count = len(tables)
        required = int(math.ceil(probability*count-1e-12))
        audit['required_scenarios'] = required
        availability = []
        training_acceptance = []
        for sid,table in enumerate(tables):
            has_yes = any(bool(f['has_target']) for f in table['families'])
            has_no = any(not bool(f['has_target']) for f in table['families'])
            explicit = table.get('comparison_available',table.get('candidate_available'))
            if explicit is not None:
                check(f's{sid}_explicit_available_has_both_families',not bool(explicit) or (has_yes and has_no))
            available = bool(has_yes and has_no and (explicit is None or explicit))
            gain = _finite_scalar(table['acceptance_margin'],'acceptance_margin')
            training_pass = gain > excitation_guard
            availability.append(available)
            training_acceptance.append(training_pass)
            for key,expected in [('comparison_available',available),
                ('training_acceptance_passes',training_pass),
                ('acceptance_passes',bool(available and training_pass))]:
                if key in result:
                    check(f's{sid}_preflight_'+key,bool(result[key][sid])==expected)
        audit['comparison_available_count'] = sum(availability)
        audit['training_acceptance_pass_count'] = sum(training_acceptance)
        audit['chosen_stage'] = copy.deepcopy(result.get('chosen_stage'))
        # Phase bounds remain evidence for their own objective and not for a
        # new topology statement.  Primary max gamma; secondary min payment.
        audit['phases']['primary'] = phase_receipt('primary', result.get('primary'), True)
        audit['phases']['secondary'] = phase_receipt('secondary', result.get('secondary'), False)
        if not result.get('feasible_incumbent', False):
            audit.update(status='no_price_incumbent', excitation_requirement_satisfied=False,
                price_hard_constraints_feasible=None, gamma_replay_micro=None,
                note='No incumbent is not a mathematical infeasibility proof.')
            for key in ('excitation_requirement_met','sample_90pct_requirement_met','sample_requirement_met'):
                if result.get(key,False):
                    check('no_false_'+key+'_without_incumbent', False)
            return _finish(audit, checks)

        if problem.n_contracts != 16:
            raise ValueError('this locked audit expects the original16-contract case')
        fixed = _finite_array(train_amplitudes, 'train_amplitudes', (16,))
        x = _finite_array(result['amplitudes'], 'amplitudes', (16,))
        prices = _finite_array(result['prices'], 'prices', (16,))
        check('fixed_training_vector_has_zero_validation_tail', np.max(np.abs(fixed[8:]), initial=0.) <= native_tolerance)
        check('training_amplitudes_fixed', np.max(np.abs(x[:8]-fixed[:8]), initial=0.) <= physical_tolerance)
        check('prices_equal_amplitudes_over_elasticities',
            np.max(np.abs(prices-x/problem.elasticities), initial=0.) <= physical_tolerance)

        # Recompute the domain formulas here and compare with the established
        # model evaluator.  Do not retain its time-series moment output.
        q = np.abs(problem.response_factors).sum(axis=1)
        upper = np.minimum(problem.elasticities*problem.price_upper,
                           problem.actual_amplitude_upper/(1+q))
        coefficients = problem.reduction_hours/problem.elasticities
        expected_cost = float(coefficients@(x*x))
        hard_cost = float((coefficients*(1+q))@(x*x))
        device = problem.device_matrix@x+np.abs((problem.device_matrix*x)@problem.response_factors).sum(axis=1)
        check('nonnegative_amplitudes', np.min(x, initial=0.) >= -physical_tolerance)
        check('physical_amplitude_cap', np.max(x-upper, initial=0.) <= physical_tolerance)
        check('quoted_price_cap', np.max(prices-problem.price_upper, initial=0.) <= physical_tolerance)
        check('hard_budget', hard_cost <= problem.budget+physical_tolerance,
              hard_cost=hard_cost, budget=float(problem.budget))
        check('whole_support_device_rows', np.max(device-problem.device_rhs, initial=0.) <= physical_tolerance)
        for group_id, group in enumerate(problem.exclusive_groups):
            check('exclusive_group_'+str(group_id), int(np.count_nonzero(x[list(group)]>physical_tolerance)) <= 1)
        domain = evaluate_plan(problem, x, tolerance=physical_tolerance)
        check('original_evaluator_domain', domain['numerical_feasible'])
        check('original_evaluator_expected_payment', close(expected_cost, domain['expected_cost'], physical_tolerance))
        check('original_evaluator_hard_budget', close(hard_cost, domain['hard_budget_bound'], physical_tolerance))
        if 'original_evaluation' in result:
            supplied = result['original_evaluation']
            for key, value in [('expected_cost',expected_cost), ('hard_budget_bound',hard_cost)]:
                if key in supplied:
                    check('saved_'+key, close(value,supplied[key],physical_tolerance))
        audit['price_hard_constraints_feasible'] = bool(domain['numerical_feasible'])
        audit['expected_payment'] = expected_cost
        audit['hard_budget_bound'] = hard_cost

        eligible_margins = []
        successful = 0
        for sid, table in enumerate(tables):
            nv = int(table['validation_size'])
            if nv <= 0 or nv != table['validation_size']:
                raise ValueError('validation_size must be a positive integer')
            family_losses = []
            no, yes = [], []
            for family in table['families']:
                base = _finite_array(family['residual_base'], 'residual_base')
                if base.ndim != 1:
                    raise ValueError('residual_base must be a vector')
                matrix = _finite_array(family['residual_price'], 'residual_price', (len(base),8))
                constant = _finite_scalar(family['constant_loss'], 'constant_loss')
                check(f's{sid}_f{family["family_id"]}_constant_nonnegative', constant >= -native_tolerance)
                loss = constant+float(np.abs(base+matrix@prices[8:]).sum()/nv)
                family_losses.append(loss)
                (yes if family['has_target'] else no).append(loss)
            available = availability[sid]
            margin = min(no)-1.03*min(yes)-1e-12 if available else None
            acceptance = _finite_scalar(table['acceptance_margin'], 'acceptance_margin')
            training_pass = training_acceptance[sid]
            acceptance_pass = bool(training_pass and available)
            excitation_pass = bool(acceptance_pass and margin > excitation_guard)
            successful += int(excitation_pass)
            if acceptance_pass:
                eligible_margins.append(margin)
            replay.append(dict(scenario=sid, family_ids=[f['family_id'] for f in table['families']],
                family_losses=family_losses, best_without_target=min(no) if no else None,
                best_with_target=min(yes) if yes else None, comparison_available=available,
                validation_exit_margin=margin, acceptance_margin=acceptance,
                training_acceptance_pass=training_pass,
                acceptance_pass=acceptance_pass, excitation_pass=excitation_pass))
            if 'scenario_margins' in result:
                saved=result['scenario_margins'][sid]
                check(f's{sid}_saved_margin',saved is None if margin is None else saved is not None and close(margin,saved))
            if 'acceptance_passes' in result:
                check(f's{sid}_saved_acceptance',acceptance_pass==bool(result['acceptance_passes'][sid]))
            if 'excitation_passes' in result:
                check(f's{sid}_saved_excitation',excitation_pass==bool(result['excitation_passes'][sid]))
            if 'scenario_family_losses' in result:
                supplied = result['scenario_family_losses'][sid]
                if isinstance(supplied,Mapping):
                    supplied=[supplied.get(f['family_id'], supplied.get(str(f['family_id']))) for f in table['families']]
                supplied=_finite_array(supplied,'scenario_family_losses',(len(family_losses),))
                check(f's{sid}_saved_family_losses',np.max(np.abs(supplied-family_losses),initial=0.)<=native_tolerance)

        # Full denominator is never replaced by the eligible count.  Taking
        # the kth largest eligible value is equivalent to padding failed
        # acceptance states with -infinity, then using sorted index S-k.
        gamma = None
        if len(eligible_margins) >= required:
            gamma = float(sorted(eligible_margins)[len(eligible_margins)-required]*1e6)
        audit.update(acceptance_pass_count=len(eligible_margins), excitation_pass_count=successful,
            empirical_excitation_rate=successful/count,
            excitation_requirement_satisfied=bool(successful>=required),
            gamma_replay_micro=gamma,
            quantile_definition='kth-largest acceptance-eligible margin; k=ceil(probability*FULL_S); failed acceptance padded by -infinity',
            probability_scope='empirical planning scenario excitation only; no independent actual-probability claim')
        if 'required_scenarios' in result:
            check('saved_required_count_uses_full_denominator',int(result['required_scenarios'])==required)
        for key in ('gamma','quantile_micro'):
            if key in result:
                saved=result[key]
                check('saved_'+key,saved is None if gamma is None else saved is not None and close(saved,gamma,max(1e-6,native_tolerance*1e6)))
        if result.get('optimizer_gamma_micro') is not None and gamma is not None:
            check('optimizer_gamma_below_replayed_quantile',
                float(result['optimizer_gamma_micro']) <= gamma+max(1e-6,native_tolerance*1e6))
        for key in ('excitation_requirement_met','sample_90pct_requirement_met','sample_requirement_met'):
            if key in result:
                check('no_false_'+key,bool(result[key])==bool(successful>=required))
        if gamma is not None:
            check('quantile_and_count_agree',(gamma>excitation_guard*1e6)==(successful>=required))
        else:
            check('insufficient_eligible_keeps_failure_mass',successful<required)
        if successful < required:
            audit['note']='Feasible prices can have a negative gamma; no90% excitation success is claimed.'
    except Exception as error:
        check('audit_input_and_replay',False,error_type=type(error).__name__,message=str(error))
        audit.setdefault('excitation_requirement_satisfied',False)
    return _finish(audit,checks)


def _finish(audit, checks):
    audit['checks'] = checks
    audit['check_count'] = len(checks)
    audit['passed'] = sum(c['passed'] for c in checks)
    audit['failed'] = [c for c in checks if not c['passed']]
    audit['audit_passed'] = not audit['failed']
    audit.setdefault('status','complete' if audit['audit_passed'] else 'failed')
    return audit
