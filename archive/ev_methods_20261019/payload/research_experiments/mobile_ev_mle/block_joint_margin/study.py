"""Independent small pilot for block-local joint-fit-margin EV planning.

The selector accepts only cases.planning_inputs(case), public fit options and
an independent seed. Truth, future trajectories and actual meter noise enter
only after plans are saved. Existing experiments and records are unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

NUMERIC_THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                   'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for _numeric_name in NUMERIC_THREADS:
    os.environ[_numeric_name] = '1'

import numpy as np

from .. import ev_ac_repair_refit as refit
from .. import ev_bayes_response_cases as cases
from .. import ev_bayes_response_study as old
from .. import ev_bayes_small_study as small
from .. import ev_mobile_v3_cases as public
from .. import ev_mobile_v4_domain as nested
from .. import ev_systematic_refit as legacy_refit
from .. import gradient_price_study as io
from .. import likelihood as lk
from ..information_context import meter_to_record
from .block_adapter import build_block_adapter
from .config import CONFIG

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parents[2] / 'outputs' / 'ev_block_joint_margin_v1'


def _seed(seed, tag):
    return int(np.random.SeedSequence([int(seed), int(tag), 20310117])
               .generate_state(1)[0])


def _rng(seed, tag):
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(tag), 20310117]))


def _save(path, value):
    """Atomic lossless gzip JSON, including NumPy and dataclass records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old.save_record(path, value)


def _plain(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(old.serial(value), ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    result = old.source_hashes()
    for name in ('ev_ac_repair_refit.py', 'ev_fixed_train_fit.py', 'ev_rx_fit.py',
                 'ev_systematic_refit.py', 'ev_rx_precision.py', 'ev_mobile_v3_cases.py',
                 'ev_mobile_v4_cases.py', 'ev_mobile_v4_domain.py',
                 'ev_mobile_service_domain.py', 'ev_bayes_response_cases.py',
                 'ev_bayes_response_study.py', 'ev_bayes_small_study.py',
                 'ev_operations_milp.py', 'ev_original_mip_backend.py',
                 'gradient_price_study.py', 'likelihood.py'):
        dependency = HERE.parent / name
        if dependency.exists():
            result[name] = sha256(dependency.read_bytes()).hexdigest()
    for path in sorted(HERE.iterdir()):
        if path.is_file() and (path.suffix == '.py' or path.name in ('math.md', 'protocol.md')):
            result['block_joint_margin/' + path.name] = sha256(path.read_bytes()).hexdigest()
    return result


def freeze(output, config=None):
    """Bind source and settings before new development/evaluation is read."""
    config = deepcopy(CONFIG if config is None else config)
    hashes = source_hashes()
    payload = dict(config=config, source_sha=hashes,
        scope='new independent development and evaluation; old science unchanged')
    identity = sha256(json.dumps(old.serial(payload), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    record = dict(payload, identity=identity)
    target = Path(output) / 'freeze.json'
    if target.exists():
        previous = io.read_json(target)
        if previous.get('identity') != identity:
            raise RuntimeError('source/config differs from freeze; use a new output directory')
        return previous
    _plain(target, record)
    return record


def specifications(phase, config=None):
    cfg = CONFIG if config is None else config
    if phase in ('dev', 'development'):
        phase = 'development'
        combinations = [(32, 0, 'ordinary'), (32, 1, 'ordinary')]
        seeds = cfg['development_seeds']
    elif phase in ('eval', 'evaluation'):
        phase = 'evaluation'
        # Eight fixed cells balance H32/H96, four truths and two load regimes.
        # Each excitation stratum has four independent histories, not repeats.
        combinations = [(H, family, ('ordinary', 'correlated')[(hi + family) % 2])
                        for hi, H in enumerate((32, 96)) for family in range(4)]
        seeds = cfg['evaluation_seeds']
    else:
        raise ValueError('development or evaluation required')
    if len(seeds) != len(combinations):
        raise ValueError('predeclared seed count differs from pilot strata')
    return [dict(phase=phase, index=i, stratum_id=i, seed=int(seeds[i]),
        bank='n3', H=H, family_index=family, excitation=excitation,
        innovation='nominal', innovation_multiplier=1., replicate=0,
        legacy_audit=False) for i, (H, family, excitation) in enumerate(combinations)]


def _fit_options(public_case_config, config):
    values = dict(public_case_config['training'], **public_case_config['source'],
                  ratio_bounds=public_case_config['ratio_bounds'])
    values.update(multistart=int(config['fit_multistart']),
                  max_nfev=int(config['fit_max_nfev']), tol=float(config['fit_tol']))
    return refit._options(values)


def _plan_record(domain_name, domain, x):
    x = np.asarray(x, float)
    decoded = domain.decode(x)
    check = domain.check(x)
    if not check['feasible']:
        raise ValueError('original EV operation domain violated: ' + str(check['issues']))
    power = np.asarray(decoded['u_kw'], float)
    return dict(domain=domain_name, operation_x=x.copy(), decoded=decoded,
        physicalcheck=check, power=power, action_key=nested.action_key(power))


def _perturb_operation(domain, x, direction, step):
    """Same discrete route/modes; adjust two same-site powers and energy."""
    result = np.asarray(x, float).copy()
    ix, cfg = domain.indices, domain.config
    before = np.asarray(x, float)[ix.u]
    after = before + float(step) * direction
    # Directions never cross a charge/discharge mode or perturb zero slots.
    if np.any((before * after <= 0.) & (direction != 0.)):
        raise ValueError('perturbation crosses a discrete charging mode')
    old_charge, old_discharge = result[ix.charge].copy(), result[ix.discharge].copy()
    result[ix.u] = after
    result[ix.charge] = np.maximum(after, 0.)
    result[ix.discharge] = np.maximum(-after, 0.)
    delta = cfg.dt_hours * (
        cfg.eta_charge * (result[ix.charge] - old_charge).sum(axis=1)
        - (result[ix.discharge] - old_discharge).sum(axis=1) / cfg.eta_discharge)
    result[ix.energy] += np.r_[0., np.cumsum(delta)]
    return result


def feasible_exchange_direction(domain, x, step=.001):
    """Find a feasible energy-preserving direction; vertices use one-sided FD."""
    power = domain.decode(x)['u_kw']
    cfg = domain.config
    one_sided = None
    for site in range(power.shape[1]):
        slots = np.flatnonzero(abs(power[:, site]) > 4 * step)
        for pos, first in enumerate(slots):
            for second in slots[pos + 1:]:
                a = cfg.eta_charge if power[first, site] > 0 else 1 / cfg.eta_discharge
                b = cfg.eta_charge if power[second, site] > 0 else 1 / cfg.eta_discharge
                direction = np.zeros_like(power)
                scale = max(abs(a), abs(b))
                direction[first, site], direction[second, site] = b / scale, -a / scale
                plus = _perturb_operation(domain, x, direction, step)
                minus = _perturb_operation(domain, x, direction, -step)
                pcheck, mcheck = domain.check(plus), domain.check(minus)
                if pcheck['feasible'] and mcheck['feasible']:
                    return dict(available=True, direction=direction, plus=plus, minus=minus,
                        site=site, slots=[int(first), int(second)], step_kw=float(step),
                        scheme='central',
                        plus_check=pcheck, minus_check=mcheck,
                        unit='direction max norm is one; perturbation in kW',
                        exact_terminal_energy_preserved=True)
                if one_sided is None:
                    for sign in (1., -1.):
                        orient = sign * direction
                        first_point = _perturb_operation(domain, x, orient, step)
                        second_point = _perturb_operation(domain, x, orient, 2 * step)
                        first_check, second_check = domain.check(first_point), domain.check(second_point)
                        if first_check['feasible'] and second_check['feasible']:
                            one_sided = dict(available=True, direction=orient,
                                one_step=first_point, two_steps=second_point,
                                site=site, slots=[int(first), int(second)], step_kw=float(step),
                                scheme='one_sided_second_order', one_step_check=first_check,
                                two_steps_check=second_check,
                                unit='direction max norm is one; perturbation in kW',
                                exact_terminal_energy_preserved=True)
                            break
    return one_sided if one_sided is not None else dict(available=False,
        reason='no same-site two-slot feasible exchange at SQ, including one-sided directions')


def _plane_map(record):
    planes = {}
    for plane in record.get('planes', []):
        key = (tuple(plane['pair']), str(plane.get('scenario', plane.get('key', ''))))
        if key in planes:
            raise ValueError('nonunique pair/scenario plane key')
        planes[key] = plane
    return planes


def gradient_gate(domain, square_x, objective_get, config):
    """All uncapped scenario margins must pass a feasible finite difference."""
    began = perf_counter()
    direction = feasible_exchange_direction(domain, square_x, config['gradient_step_kw'])
    if not direction['available']:
        return dict(pass_gate=False, reason=direction['reason'], seconds=perf_counter()-began)
    anchor = objective_get(domain.decode(square_x)['u_kw'], 'gradient_anchor')
    if not anchor.get('complete'):
        return dict(pass_gate=False, reason='inner fit failure at SQ anchor',
            direction=direction, failures=anchor.get('failures'), seconds=perf_counter()-began)
    central = direction['scheme'] == 'central'
    point1 = direction['plus'] if central else direction['one_step']
    point2 = direction['minus'] if central else direction['two_steps']
    plus = objective_get(domain.decode(point1)['u_kw'],
                         'gradient_plus' if central else 'gradient_one_step')
    minus = objective_get(domain.decode(point2)['u_kw'],
                          'gradient_minus' if central else 'gradient_two_steps')
    if not all(record.get('complete') for record in (anchor, plus, minus)):
        return dict(pass_gate=False, reason='inner fit failure at finite-difference points',
            direction=direction, complete=[r.get('complete') for r in (anchor, plus, minus)],
            failures=[r.get('failures') for r in (anchor, plus, minus)], seconds=perf_counter()-began)
    am, pm, mm = map(_plane_map, (anchor, plus, minus))
    if not am or set(am) != set(pm) or set(am) != set(mm):
        return dict(pass_gate=False, reason='scenario plane keys changed', direction=direction,
                    seconds=perf_counter()-began)
    checks = []
    h = float(config['gradient_step_kw'])
    for key, plane in am.items():
        numerical = ((float(pm[key]['value']) - float(mm[key]['value'])) / (2 * h)
            if central else (-3 * float(plane['value']) + 4 * float(pm[key]['value'])
                             - float(mm[key]['value'])) / (2 * h))
        analytical = float(np.sum(np.asarray(plane['gradient']) * direction['direction']))
        absolute = abs(numerical - analytical)
        tolerance = float(config['gradient_absolute_tolerance']) + float(
            config['gradient_relative_tolerance']) * max(abs(numerical), abs(analytical))
        checks.append(dict(pair=list(key[0]), scenario=key[1], value=plane['value'],
            scheme=direction['scheme'],
            analytical=analytical, finite_difference=numerical, absolute_error=absolute,
            relative_error=absolute / max(abs(numerical), abs(analytical), 1e-15),
            tolerance=tolerance, passed=bool(absolute <= tolerance)))
    return dict(pass_gate=all(c['passed'] for c in checks),
        reason='all scenario planes checked' if all(c['passed'] for c in checks)
               else 'envelope derivative and feasible finite difference disagree',
        checks=checks, direction=direction, anchor_score=anchor['score'],
        plus_score=plus['score'], minus_score=minus['score'], seconds=perf_counter()-began,
        scope='directional local check, no global optimality or action-ranking guarantee')


def _objective_counters(objective):
    return dict(calls=int(getattr(objective, 'calls', 0)),
        seconds=float(getattr(objective, 'seconds', 0.)),
        setup_seconds=float(getattr(objective, 'setup_seconds', 0.)),
        counters=deepcopy(getattr(objective, 'counters', {})))


def _planning(observed, fit_options, config, seed, report):
    """Observed-only selector; no full case/true label/future noise argument."""
    from .ac_objective import MarginObjective
    from .price_milp import solve_local_margin_milp

    began = perf_counter()
    D = lk.concat_observations(observed['training'], observed['evidence'])
    dm = legacy_refit.concat_models((observed['training'], observed['evidence']),
        (observed['train_meter'], observed['evidence_meter']))
    history = refit.fit_history(observed['candidates'], D, dm, fit_options, _seed(seed, 1000))
    result = dict(history=history.record(), plans={}, failures={}, costs={}, planning_logs={})
    report('history_fitted', result['history'])
    template = replace(public.service_domain(observed['horizon'], D.n, mobile=True).config,
                       n_slots=observed['horizon'])
    domains = nested.build_nested_domains(template)
    constant = np.zeros((observed['horizon'], D.n))
    constant[:, template.initial_site] = (template.terminal_energy_kwh
        - template.initial_soc * template.battery_capacity_kwh) / (
        template.eta_charge * observed['horizon'] * template.dt_hours)
    highlow = np.zeros_like(constant)
    highlow[:len(highlow)//2, template.initial_site] = constant[0, template.initial_site] * 2
    xconstant = small.complete_curve(domains['fixed_charge'], constant)
    result['plans']['constant'] = _plan_record('fixed_charge', domains['fixed_charge'], xconstant)
    squares, logs, seconds = old.square_family(domains, observed['future_meter'],
        constant, highlow, iterations=int(config['square_iterations']))
    square_x = squares['mobile_bidir']
    result['plans']['SQMD'] = _plan_record('mobile_bidir', domains['mobile_bidir'], square_x)
    result['planning_logs']['square'] = logs
    result['costs']['square_seconds'] = seconds
    result['operation_config'] = asdict(template)
    result['block_adapter'] = build_block_adapter(observed['candidates'])
    report('square_fixed', dict(plans=result['plans'], square_logs=logs,
                                operation_config=result['operation_config']))
    if not history.complete:
        result['failures']['history'] = history.failures
        result['gradient_gate'] = dict(pass_gate=False, reason='history fit failed')
        result['plans']['joint_margin'] = deepcopy(result['plans']['SQMD'])
        result['costs']['planning_seconds'] = perf_counter()-began
        return result, D, dm, history
    try:
        objective = MarginObjective(observed['candidates'], D, dm, observed['future_meter'],
            history, config=config, seed=_seed(seed, 2000))
    except Exception as exc:
        result['failures']['objective_setup'] = type(exc).__name__ + ': ' + str(exc)
        result['gradient_gate'] = dict(pass_gate=False, reason='objective setup failed',
            failure=result['failures']['objective_setup'])
        result['plans']['joint_margin'] = deepcopy(result['plans']['SQMD'])
        result['costs']['planning_seconds'] = perf_counter()-began
        report('objective_setup_failed', dict(failure=result['failures']['objective_setup'],
            fallback='SQMD', plans=result['plans']))
        return result, D, dm, history
    result['scenario_support'] = objective.scenario_record()
    report('scenario_support', result['scenario_support'])
    recorded = set()

    def get(power, label):
        key = sha256(np.asarray(power, dtype='<f8').tobytes()).hexdigest()
        value = objective.get(power)
        if key not in recorded:
            report('objective_' + label, dict(key=key, result=value,
                                              counters=_objective_counters(objective)))
            recorded.add(key)
        return value

    domain = domains['mobile_bidir']
    try:
        gate = gradient_gate(domain, square_x, get, config)
    except Exception as exc:
        gate = dict(pass_gate=False, reason='gradient check exception',
                    failure=type(exc).__name__ + ': ' + str(exc))
    result['gradient_gate'] = gate
    report('gradient_gate', gate)
    winner = np.asarray(square_x).copy()
    try:
        current = get(domain.decode(winner)['u_kw'], 'square')
    except Exception as exc:
        current = dict(complete=False, score=None, failures=dict(
            square_objective=type(exc).__name__ + ': ' + str(exc)))
        result['failures']['square_objective'] = current['failures']
    events = []
    if gate['pass_gate'] and current.get('complete'):
        for iteration in range(min(2, int(config['max_proposals']))):
            event = dict(iteration=iteration, accepted=False, parent_score=current['score'])
            try:
                proposal = solve_local_margin_milp(domain, domain.decode(winner)['u_kw'],
                    planes=current['planes'], tau=config['margin_cap'],
                    radius=config['power_radius_kw'], time_limit=config['solver_seconds'])
                event['milp'] = proposal
                proposed_x = proposal.get('operation_x')
                if proposed_x is None:
                    proposed_x = proposal.get('x')
                if proposed_x is None and isinstance(proposal.get('operation'), dict):
                    proposed_x = proposal['operation'].get('x')
                if proposed_x is not None:
                    proposed_x = np.asarray(proposed_x, float)
                    check = domain.check(proposed_x)
                    event['physicalcheck'] = check
                    if not check['feasible']:
                        raise ValueError('MILP incumbent violates original operation domain')
                    candidate = get(domain.decode(proposed_x)['u_kw'], 'proposal_' + str(iteration))
                    event['complete_AC_objective'] = candidate.get('complete', False)
                    event['proposed_score'] = candidate.get('score')
                    event['objective_failures'] = candidate.get('failures')
                    if candidate.get('complete') and candidate['score'] > current['score'] + 1e-9:
                        winner, current = proposed_x.copy(), candidate
                        event['accepted'] = True
                else:
                    event['failure'] = 'MILP returned no feasible incumbent'
            except Exception as exc:
                event['failure'] = type(exc).__name__ + ': ' + str(exc)
            event['retained_score'] = current['score']
            events.append(event)
            report('proposal_' + str(iteration), event)
    else:
        result['failures']['gradient_gate'] = gate['reason']
    result['plans']['joint_margin'] = _plan_record('mobile_bidir', domain, winner)
    try:
        initial = get(domain.decode(square_x)['u_kw'], 'square_final')
    except Exception:
        initial = dict(complete=False, score=None)
    result['planning_logs']['joint_margin'] = dict(events=events,
        initial_score=initial.get('score'), final_score=current.get('score'),
        inherited_square_not_worse=bool(current.get('complete') and initial.get('complete')
            and current['score'] >= initial['score'] - 1e-9),
        accepted_proposals=sum(e['accepted'] for e in events),
        fallback_to_square=bool(nested.action_key(domain.decode(winner)['u_kw'])
                                == result['plans']['SQMD']['action_key']),
        score_scope='full AC attained local joint-fit margin, not classification probability')
    result['costs']['objective'] = _objective_counters(objective)
    result['costs']['planning_seconds'] = perf_counter()-began
    report('plans_fixed', dict(plans=result['plans'], logs=result['planning_logs'],
                               failures=result['failures'], costs=result['costs']))
    return result, D, dm, history


def _decision(fitted, route, true_name, truth_r, truth_x):
    losses = fitted.get('losses', {}).get(route, {})
    complete = bool(fitted.get('complete')) and len(losses) == len(fitted['candidate_names'])
    result = dict(available=complete, keep_all_on_failure=not complete,
                  global_optimality_certified=False)
    if not complete:
        result.update(retained=list(fitted.get('candidate_names', [])), error=None,
            unresolved=True, failures=fitted.get('failures', {}))
        return result
    best = min(losses.values())
    tie = 1e-8 * (1 + abs(best))
    retained = [name for name, value in losses.items() if value <= best + tie]
    predicted = min(losses, key=losses.get)
    state = fitted['comparisons'][true_name][route]['state']
    line_error = {}
    for name, truth in (('r', truth_r), ('x', truth_x)):
        value = np.asarray(state[name], float)
        truth = np.asarray(truth, float)
        line_error[name] = dict(mean_absolute_relative=float(np.mean(abs(value-truth)/abs(truth))),
            relative_l2=float(np.linalg.norm(value-truth)/np.linalg.norm(truth)),
            estimated=value, truth_evaluation_only=truth)
    result.update(predicted=predicted, error=bool(predicted != true_name),
        retained=retained, unresolved=len(retained) != 1,
        tie_tolerance= tie, losses=losses, line_error=line_error,
        decision_scope='argmin SSE; unresolved is numerical tie, no new statistical rejection calibration')
    return result


def _evaluate(case, spec, observed, plans, D, dm, history, fit_options, config, report):
    """Evaluator alone receives simulator truth after the selector is done."""
    actions = {}
    dimension = observed['horizon'] * (3 * D.n + 2)
    white = _rng(spec['seed'], 4000).standard_normal(dimension)
    true_name = next(c.name for c in observed['candidates']
        if c.supports == case['truth']['candidate'].supports)
    for method, plan in plans.items():
        key = plan['action_key']
        if key in actions:
            actions[key]['methods'].append(method)
            continue
        action = dict(methods=[method], power=plan['power'], regimes={}, failures={})
        for ri, regime in enumerate(('regular', 'shift')):
            try:
                future_case = cases.make_case(spec, regime=regime)
                physical, truth_inputs = old._actual_physical(case, np.asarray(plan['power']), future_case)
                E = old.first_observation(physical['physical'], observed['future_meter'], white)
            except Exception as exc:
                failure = type(exc).__name__ + ': ' + str(exc)
                action['failures'][regime + '_physical'] = failure
                action['regimes'][regime] = dict(physical_generation_failed=True,
                    failure=failure, keep_all_on_failure=True,
                    retained_on_failure=[c.name for c in observed['candidates']])
                report('physical_failure_' + method + '_' + regime,
                    dict(failure=failure, action_key=key, retained_all=True))
                continue
            actual = dict(observations=io.observations_record(E), actual_white_noise=white,
                truth_inputs_evaluation_only=truth_inputs,
                physical=dict(ac_equation_residual=float(physical['physical']['ac_equation_residual']),
                    power_balance_residual=float(physical['physical']['power_balance_residual']),
                    min_voltage=float(np.min(physical['physical']['v'])),
                    max_voltage=float(np.max(physical['physical']['v']))))
            for label, observation in [('noisy', E)] + ([('clean', old.first_observation(
                physical['physical'], observed['future_meter'], np.zeros(dimension)))]
                if config.get('clean_diagnostic', True) else []):
                try:
                    fitted = refit.evaluate_refits(observed['candidates'], D, dm,
                        observation, observed['future_meter'], history, fit_options,
                        _seed(spec['seed'], 6000 + 100 * ri + (label == 'clean')),
                        bayes_centers=None)
                    actual[label + '_refit'] = fitted
                    actual[label + '_decisions'] = {route: _decision(fitted, route, true_name,
                        case['truth']['r'], case['truth']['x']) for route in ('fixed', 'joint')}
                    if not fitted['complete']:
                        action['failures'][regime + '_' + label] = fitted['failures']
                except Exception as exc:
                    failure = type(exc).__name__ + ': ' + str(exc)
                    action['failures'][regime + '_' + label] = failure
                    actual[label + '_refit'] = dict(complete=False, failure=failure,
                        keep_all_on_failure=True, retained_on_failure=[c.name for c in observed['candidates']])
                report('evaluated_' + method + '_' + regime + '_' + label,
                    dict(action_key=key, result=actual.get(label + '_refit'),
                         decisions=actual.get(label + '_decisions')))
            action['regimes'][regime] = actual
        actions[key] = action
    return dict(actions=actions, true_name_evaluation_only=true_name,
        true_lines_evaluation_only=dict(r=case['truth']['r'], x=case['truth']['x']),
        actual_noise_paired_across_plans_and_regimes=True,
        n_independent_histories=1, future_regimes_per_history=2,
        clean_used_for_selection=False, bayes_center_used=False)


def run(spec, *, config=None, output=DEFAULT_OUTPUT, progress=None):
    config = deepcopy(CONFIG if config is None else config)
    output = Path(output)
    frozen = freeze(output, config)
    began = perf_counter()
    spec = deepcopy(spec)
    phase = spec['phase']
    folder = output / phase
    path = folder / ('history_%03d.json.gz' % spec['index'])
    if path.exists():
        raise FileExistsError('existing independent history is immutable: ' + str(path))
    steps = folder / ('history_%03d.steps' % spec['index'])
    sequence = [0]

    def report(stage, details=None):
        item = dict(stage=stage, index=spec['index'], phase=phase, seed=spec['seed'],
            elapsed_seconds=perf_counter()-began, details=details or {}, freeze_identity=frozen['identity'])
        _save(steps / ('%03d_%s.json.gz' % (sequence[0], stage)), item)
        sequence[0] += 1
        _plain(output / 'checkpoint.json', dict(phase=phase, index=spec['index'],
            seed=spec['seed'], stage=stage, history_path=str(path),
            elapsed_seconds=perf_counter()-began, freeze_identity=frozen['identity'],
            not_scientifically_complete=True))
        if progress is not None:
            progress(stage, dict(index=spec['index'], elapsed_seconds=perf_counter()-began))

    record = dict(spec=spec, config=config, freeze_identity=frozen['identity'],
        source_sha=frozen['source_sha'], numeric_threads={k: os.environ.get(k) for k in NUMERIC_THREADS},
        failures={}, status='started', n_independent_histories=1)
    report('started')
    try:
        case = cases.make_case(spec)
        observed = cases.planning_inputs(case)
        fit_options = _fit_options(case['config'], config)
        record.update(observed_training=io.observations_record(observed['training']),
            observed_evidence=io.observations_record(observed['evidence']),
            train_meter=meter_to_record(observed['train_meter']),
            evidence_meter=meter_to_record(observed['evidence_meter']),
            future_meter=meter_to_record(observed['future_meter']),
            candidate_structures=[dict(name=c.name, n=c.n, supports=c.supports)
                                  for c in observed['candidates']],
            fit_options=fit_options, provenance=case['provenance'])
        planning, D, dm, history = _planning(observed, fit_options, config, spec['seed'], report)
        record['planning'] = planning
        record['failures'].update(planning['failures'])
        gate_passed = bool(planning['gradient_gate']['pass_gate'])
        record['new_metric_derivative_accepted'] = gate_passed
        record['new_metric_design_failure'] = not gate_passed
        record['new_metric_fallback'] = 'SQMD' if not gate_passed else None
        # Baselines remain evaluated even if the new indicator cannot deploy.
        # Evaluation histories are never dropped because of gradient failure.
        # The sole transition where full simulation case becomes visible.
        report('selection_sealed', dict(plans=planning['plans'],
            truth_or_future_seen_by_selector=False, new_metric_deployed=gate_passed))
        record['evaluation'] = _evaluate(case, spec, observed, planning['plans'],
            D, dm, history, fit_options, config, report)
        failures = {k: v['failures'] for k, v in record['evaluation']['actions'].items()
                    if v['failures']}
        if failures:
            record['failures']['evaluation'] = failures
        record['status'] = 'complete' if not record['failures'] else 'partial_failure'
        if phase == 'development' and not gate_passed:
            record['status'] = 'gradient_gate_failed'
    except Exception as exc:
        record['status'] = 'failed'
        record['failures']['exception'] = dict(message=type(exc).__name__ + ': ' + str(exc),
                                              traceback=traceback.format_exc())
    record.update(seconds=perf_counter()-began,
                  source_unchanged_during_run=source_hashes() == frozen['source_sha'])
    _save(path, record)
    report('history_finished', dict(status=record['status'], history_path=str(path)))
    update_gate(output, config, frozen)
    return record


def update_gate(output, config=None, frozen=None):
    config = CONFIG if config is None else config
    frozen = freeze(output, config) if frozen is None else frozen
    rows = []
    for spec in specifications('development', config):
        path = Path(output) / 'development' / ('history_%03d.json.gz' % spec['index'])
        value = io.read_json(path) if path.exists() else {}
        rows.append(dict(index=spec['index'], seed=spec['seed'], available=bool(value),
            pass_gate=bool(value.get('planning', {}).get('gradient_gate', {}).get('pass_gate')),
            frozen_source_matches=value.get('freeze_identity') == frozen['identity'],
            source_unchanged=bool(value.get('source_unchanged_during_run')),
            status=value.get('status', 'not_run'), path=str(path)))
    gate = dict(pass_gate=all(r['available'] and r['pass_gate'] and r['frozen_source_matches']
                             and r['source_unchanged']
                             for r in rows), development=rows,
                freeze_identity=frozen['identity'], evaluation_used=False,
                scope='local gradient acceptance only; no prior claim of planning improvement')
    _plain(Path(output) / 'development_gate.json', gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'development', 'eval', 'evaluation'), default='development')
    parser.add_argument('--index', type=int)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = deepcopy(CONFIG)
    if args.config is not None:
        config.update(io.read_json(args.config))
    frozen = freeze(args.output, config)
    if args.freeze_only:
        print(json.dumps(dict(freeze_identity=frozen['identity'], output=str(args.output)), ensure_ascii=False))
        return
    phase = 'development' if args.phase in ('dev', 'development') else 'evaluation'
    if phase == 'evaluation' and not update_gate(args.output, config, frozen)['pass_gate']:
        raise SystemExit('evaluation blocked: both independent development gradient gates must pass')
    specs = specifications(phase, config)
    if args.index is not None:
        specs = [s for s in specs if s['index'] == args.index]
        if not specs:
            raise SystemExit('index is outside predeclared phase')
    for spec in specs:
        path = args.output / phase / ('history_%03d.json.gz' % spec['index'])
        if path.exists():
            previous = io.read_json(path)
            if previous.get('freeze_identity') != frozen['identity']:
                raise SystemExit('existing history differs from frozen source/config')
            if args.resume:
                print(json.dumps(dict(stage='resumed_existing', index=spec['index'], status=previous['status'])))
                continue
            raise SystemExit('history already exists; use --resume or a new output directory')
        result = run(spec, config=config, output=args.output,
            progress=lambda stage, details: print(json.dumps(dict(stage=stage, **details)), flush=True))
        print(json.dumps(dict(index=spec['index'], status=result['status'], seconds=result['seconds'])), flush=True)
        if result['status'] in ('gradient_gate_failed', 'failed'):
            break


if __name__ == '__main__':
    main()
