"""Physical-witness exchange MILP for the frozen bounded QP design problem.

Each pair uses one shared quadratic .5*u.T*(B.T*B)*u.  Every additional legal
competitor explanation adds an affine row, not another set of square pieces.
The MILP maximizes an upper model.  Its best bound is an upper bound; its
incumbent objective is not an attained design lower bound.  QP dual bounds at
physically feasible operation vectors supply attained lower bounds instead.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from . import ev_mobile_v3_projection as operation_tools
from . import ev_mobile_v4_bounded as bounded
from .ev_original_mip_backend import solve_linear_mip


def _names(values):
    if isinstance(values, Mapping): return list(values.items())
    return [('operation_'+str(i), value) for i, value in enumerate(values)]


def _shared_structure(pairs, op, raw_domain, enabled):
    """Prove diagonal-on-operation Gram structure or explicitly fall back.

    Exact row-disjoint controls have zero inner product.  Other cross terms
    require the complete operation MILP's zero-product certificate.  Merely
    small cancellation dots are enclosed by a box-uniform constant, including
    their floating point dot error; a real unsupported cross term forces the
    generic factor master.  No tolerance changes the feasible EV domain.
    """
    began = perf_counter()
    if enabled not in ('auto', True, False):
        raise ValueError("shared_control_squares must be 'auto', True or False")
    if enabled is False:
        return dict(used=False, reason='explicitly_disabled', seconds=perf_counter()-began)
    raw = raw_domain.as_dict() if hasattr(raw_domain, 'as_dict') else raw_domain
    n = pairs[0].control_dimension
    claimed = set()
    try:
        for item in raw.get('zero_product_pairs', ()):
            if len(item) != 2: raise ValueError('two indices per claimed zero product required')
            i, j = item
            if int(i) != i or int(j) != j or not 0 <= i < j < n:
                raise ValueError('ordered valid control indices required for zero products')
            claimed.add((int(i), int(j)))
    except (TypeError, ValueError) as exc:
        return dict(used=False, reason='invalid_zero_product_claim: '+str(exc), seconds=perf_counter()-began)
    ubar = np.maximum(abs(op['control_lower']), abs(op['control_upper']))
    records = []; needed = set(); structural = set(); enclosed = []
    for pair in pairs:
        B = pair.B.astype(np.longdouble)
        gram = np.asarray(B.T@B, float)
        absolute_gram = np.asarray(abs(B).T@abs(B), float)
        steps = (2*pair.B.shape[0]+8)*np.finfo(float).eps
        if steps >= 1.:
            return dict(used=False, reason='undefined_floating_dot_enclosure', seconds=perf_counter()-began)
        gamma = steps/(1.-steps)
        diag = .5*np.diag(gram).copy()
        guard = .5*gamma*float(np.diag(absolute_gram)@(ubar**2))
        local_enclosed = []; local_structural = []
        for i in range(n):
            for j in range(i+1, n):
                if ubar[i] == 0. or ubar[j] == 0.:
                    continue
                if absolute_gram[i, j] == 0.:
                    structural.add((i, j)); local_structural.append((i, j)); continue
                if (i, j) in claimed:
                    needed.add((i, j)); continue
                dot_error = gamma*absolute_gram[i, j]
                if abs(gram[i, j]) > dot_error:
                    return dict(used=False, reason='unsupported_nonzero_Gram_cross_term',
                        pair=[pair.a_label, pair.b_label], controls=[i, j],
                        dot=float(gram[i, j]), floating_dot_error=float(dot_error),
                        seconds=perf_counter()-began)
                allowance = float((abs(gram[i, j])+dot_error)*ubar[i]*ubar[j])
                guard += allowance
                local_enclosed.append(dict(controls=[i, j], dot=float(gram[i, j]),
                    floating_dot_error=float(dot_error), upper_allowance=allowance))
        guard += 16*np.finfo(float).eps*(1.+float(abs(diag)@(ubar**2)))
        records.append(dict(a=pair.a_label, b=pair.b_label, diagonal=diag,
                            upward_quadratic_guard=guard, enclosed_cross_terms=local_enclosed,
                            row_disjoint_exact_zero_pairs=local_structural))
        enclosed.extend(local_enclosed)
    try:
        from .ev_shared_square_milp import certify_operation_zero_products
        proof_op = dict(op, U=op['U'].toarray())
        proof = certify_operation_zero_products(proof_op, sorted(needed))
        if not proof.get('valid'): raise ValueError('incomplete zero-product certificate')
    except Exception as exc:
        return dict(used=False, reason='zero_product_certificate_failed: '+type(exc).__name__+': '+str(exc),
                    needed_zero_products=sorted(needed), seconds=perf_counter()-began)
    return dict(used=True, records=records, zero_product_certificate=proof,
                needed_zero_products=sorted(needed), row_disjoint_pairs_in_at_least_one_model=sorted(structural),
                enclosed_numeric_cross_terms=len(enclosed),
                proof='row-disjoint exact zeros or complete-operation binary exclusion; numeric remainder enclosed',
                seconds=perf_counter()-began)


def _global_control_factors(pair, op, segments, provider):
    factors = []
    for i in range(pair.control_dimension):
        coefficient = np.zeros(pair.control_dimension); coefficient[i] = 1.
        low, high = float(op['control_lower'][i]), float(op['control_upper'][i])
        box_low, box_high = low, high; supplied = None
        if provider is not None:
            supplied = provider(pair, i, dict(intercept=0., coefficient=coefficient,
                lower=low, upper=high, control_shape=pair.control_shape,
                reference_u=pair.reference_u.copy()))
            if supplied is not None:
                if not isinstance(supplied, Mapping) or not {'lower', 'upper'} <= set(supplied):
                    raise ValueError('factor provider must return finite lower/upper mapping')
                sl, sh = float(supplied['lower']), float(supplied['upper'])
                if not np.isfinite([sl, sh]).all() or sl > sh:
                    raise ValueError('finite ordered additional factor bounds required')
                low, high = max(low, sl), min(high, sh)
                if low > high: raise ValueError('factor bounds exclude the whole coordinate box')
        grid = np.unique(np.linspace(low, high, segments+1)) if high > low else np.array([low])
        factors.append(dict(coefficient=coefficient, lower=low, upper=high,
            box_lower=box_low, box_upper=box_high, grid=grid, supplied_bounds=supplied,
            square_error=float(np.max(np.diff(grid)**2, initial=0.))/4.))
    return factors


def _factor_data(pair, op, segments, provider):
    # No numerical singular direction is silently dropped.  A rounding
    # remainder on B.T B is bounded independently of each witness's affine row.
    _, singular, Vh = np.linalg.svd(pair.B, full_matrices=False)
    beta = singular[:, None]*Vh
    Qerror = pair.B.T@pair.B-beta.T@beta
    radius2 = float(np.sum(np.maximum(abs(op['control_lower']), abs(op['control_upper']))**2))
    remainder = .5*float(np.linalg.norm(Qerror, 2))*radius2 if Qerror.size else 0.
    factors = []
    for i, coefficient in enumerate(beta):
        low = float(np.where(coefficient >= 0, coefficient*op['control_lower'],
                             coefficient*op['control_upper']).sum())
        high = float(np.where(coefficient >= 0, coefficient*op['control_upper'],
                              coefficient*op['control_lower']).sum())
        box_low, box_high = low, high
        supplied = None
        if provider is not None:
            primitive = dict(intercept=0., coefficient=coefficient.copy(),
                             lower=low, upper=high, control_shape=pair.control_shape,
                             reference_u=pair.reference_u.copy())
            supplied = provider(pair, i, primitive)
            if supplied is not None:
                if not isinstance(supplied, Mapping) or not {'lower', 'upper'} <= set(supplied):
                    raise ValueError('factor provider must return finite lower/upper mapping')
                sl, sh = float(supplied['lower']), float(supplied['upper'])
                if not np.isfinite([sl, sh]).all() or sl > sh:
                    raise ValueError('finite ordered additional factor bounds required')
                low, high = max(low, sl), min(high, sh)
                if low > high: raise ValueError('factor bounds exclude the whole coordinate box')
        grid = np.unique(np.linspace(low, high, segments+1)) if high > low else np.array([low])
        factors.append(dict(coefficient=coefficient.copy(), lower=low, upper=high,
                            box_lower=box_low, box_upper=box_high, grid=grid,
                            supplied_bounds=supplied,
                            chord_error=float(np.max(np.diff(grid)**2, initial=0.))/8.))
    return dict(a=pair.a_label, b=pair.b_label, factors=factors,
                singular_values=singular, quadratic_rounding_upper=remainder,
                chord_error=sum(f['chord_error'] for f in factors),
                n_shared_squares=len(factors))


def _witness_data(pair, witness):
    if not pair.domain.audit(witness, 1e-8)['feasible']:
        raise ValueError('only legal original-domain explanations may enter an upper master')
    cr = pair.residual-pair.B@pair.reference_u.ravel()+pair.J@witness
    return dict(z=np.asarray(witness, float).copy(), residual_intercept=cr,
                control_linear=pair.B.T@cr+pair.affine_control,
                constant=float(.5*(cr@cr)-pair.self_interval[0]+pair.affine_offset
                               -pair.affine_control@pair.affine_reference))


def _box_upper(pair, witnesses, op):
    answers = []
    for z in witnesses:
        cr = pair.residual-pair.B@pair.reference_u.ravel()+pair.J@z
        lo = cr+np.where(pair.B >= 0, pair.B*op['control_lower'], pair.B*op['control_upper']).sum(axis=1)
        hi = cr+np.where(pair.B >= 0, pair.B*op['control_upper'], pair.B*op['control_lower']).sum(axis=1)
        affine_upper = (pair.affine_offset-pair.affine_control@pair.affine_reference
                        +np.where(pair.affine_control >= 0, pair.affine_control*op['control_upper'],
                                  pair.affine_control*op['control_lower']).sum())
        answers.append(float(.5*np.maximum(lo**2, hi**2).sum()-pair.self_interval[0]+affine_upper))
    return min(answers)


def build_master(pairs, witnesses, operation_domain, *, references=None, segments=4,
                 factor_bounds_provider=None, shared_control_squares='auto'):
    """Build one shared-square upper MILP; supplied physical x rows unchanged."""
    pairs, labels = bounded.validate_pairs(pairs, references)
    if isinstance(segments, bool) or int(segments) != segments or segments < 1:
        raise ValueError('segments must be a positive integer')
    if factor_bounds_provider is not None and not callable(factor_bounds_provider):
        raise TypeError('factor_bounds_provider must be callable')
    if isinstance(witnesses, Mapping):
        groups = [witnesses[p.a_label, p.b_label] for p in pairs]
    else: groups = list(witnesses)
    if len(groups) != len(pairs) or any(not len(group) for group in groups):
        raise ValueError('at least one legal physical witness required for every pair')
    op = operation_tools._operation(operation_domain, pairs[0].control_dimension)
    shared = _shared_structure(pairs, op, operation_domain, shared_control_squares)
    n = len(op['lower']); lo = list(op['lower']); hi = list(op['upper']); integer = list(op['integrality'])
    rows, lows, highs, kinds = [], [], [], []
    modes, audits, witness_rows = [], [], []

    def var(low, high, it=0):
        index = len(lo); lo.append(float(low)); hi.append(float(high)); integer.append(it)
        return index

    def row(coefficients, low=-np.inf, high=np.inf, kind='auxiliary'):
        rows.append(coefficients); lows.append(float(low)); highs.append(float(high)); kinds.append(kind)

    for con in op['constraints']:
        A = con.A.tocsr()
        for i in range(A.shape[0]):
            cut = slice(A.indptr[i], A.indptr[i+1])
            row(dict(zip(A.indices[cut], A.data[cut])), con.lb[i], con.ub[i], kind='operation')
    eta = {a: var(-np.inf, np.inf) for a in labels}
    segment_binaries = 0

    def factor_variables(factor, square_scale):
        nonlocal segment_binaries
        grid = factor['grid']
        if len(grid) == 1:
            return {}, square_scale*float(grid[0]**2)
        lam = [var(0., 1.) for _ in grid]
        select = []
        row({j: 1. for j in lam}, 1., 1.)
        equation = {int(i): float(v) for i, v in enumerate(np.asarray(factor['coefficient']@op['U']).ravel()) if v}
        equation.update({i: -float(g) for i, g in zip(lam, grid)})
        row(equation, 0., 0., kind='shared_square_factor_equation')
        if len(grid) > 2:
            select = [var(0., 1., 1) for _ in range(len(grid)-1)]
            segment_binaries += len(select)
            row({j: 1. for j in select}, 1., 1.)
            for i, j in enumerate(lam):
                adjacency = {j: 1.}
                if i: adjacency[select[i-1]] = -1.
                if i < len(select): adjacency[select[i]] = -1.
                row(adjacency, high=0.)
        modes.append(dict(factor=factor, lambda_indices=lam, segment_indices=select))
        return {j: square_scale*float(g*g) for j, g in zip(lam, grid)}, 0.

    global_factors, global_squares = None, None
    if shared['used']:
        global_factors = _global_control_factors(pairs[0], op, int(segments), factor_bounds_provider)
        needed_square = np.any(np.vstack([record['diagonal'] > 0. for record in shared['records']]), axis=0)
        global_squares = [factor_variables(factor, 1.) if needed else ({}, 0.)
                          for factor, needed in zip(global_factors, needed_square)]
        shared['shared_nonzero_control_squares'] = int(np.count_nonzero(needed_square))
    pair_boxes = []
    for pair_index, (pair, group) in enumerate(zip(pairs, groups)):
        if shared['used']:
            proof = shared['records'][pair_index]
            diagonal = proof['diagonal']
            audit = dict(a=pair.a_label, b=pair.b_label, factors=global_factors,
                form='shared_control_square', diagonal=diagonal,
                quadratic_rounding_upper=proof['upward_quadratic_guard'],
                chord_error=float(sum(d*f['square_error'] for d, f in zip(diagonal, global_factors))),
                n_shared_squares=len(global_factors), enclosed_cross_terms=proof['enclosed_cross_terms'])
            square_coeff = {}; square_constant = audit['quadratic_rounding_upper']
            for d, (coefficients, constant) in zip(diagonal, global_squares):
                for j, value in coefficients.items(): square_coeff[j] = square_coeff.get(j, 0.)+float(d*value)
                square_constant += float(d*constant)
        else:
            audit = _factor_data(pair, op, int(segments), factor_bounds_provider)
            audit['form'] = 'generic_affine_factor_square'
            square_coeff = {}; square_constant = audit['quadratic_rounding_upper']
            for factor in audit['factors']:
                coefficients, constant = factor_variables(factor, .5)
                square_coeff.update(coefficients); square_constant += constant
        physical_rows = []
        for z in group:
            witness = _witness_data(pair, z)
            # eta_a <= common quadratic chord + witness affine term - L_aa.
            coefficients = {eta[pair.a_label]: 1.}
            for j, value in square_coeff.items(): coefficients[j] = -value
            linear = np.asarray(witness['control_linear']@op['U']).ravel()
            for j, value in enumerate(linear):
                if value: coefficients[j] = coefficients.get(j, 0.)-float(value)
            row(coefficients, high=witness['constant']+square_constant,
                kind='physical_witness_upper')
            physical_rows.append(witness)
        witness_rows.append(physical_rows); audits.append(audit)
        pair_boxes.append(dict(a=pair.a_label, b=pair.b_label, upper=_box_upper(pair, group, op)))
    ri, ci, values = [], [], []
    for i, coefficients in enumerate(rows):
        for j, value in coefficients.items():
            if value: ri.append(i); ci.append(j); values.append(value)
    A = coo_matrix((values, (ri, ci)), shape=(len(rows), len(lo))).tocsr()
    objective = np.zeros(len(lo))
    for j in eta.values(): objective[j] = -1./len(labels)
    box_upper = float(np.mean([min(r['upper'] for r in pair_boxes if r['a'] == a) for a in labels]))
    generic_nonconstant_factors = sum(np.count_nonzero(np.linalg.svd(p.B, compute_uv=False)) for p in pairs)
    generic_binary_estimate = int(generic_nonconstant_factors*int(segments)) if int(segments) > 1 else 0
    return dict(objective=objective, bounds=Bounds(lo, hi), integrality=np.asarray(integer),
                constraints=LinearConstraint(A, lows, highs), operation=op,
                operation_dimension=n, eta=eta, modes=modes, pair_audits=audits,
                witness_rows=witness_rows, pairs=pairs, references=labels,
                n_segment_binary=segment_binaries, row_kinds=kinds, box_upper=box_upper,
                shared_control_square=shared,
                generic_segment_binary_estimate=generic_binary_estimate,
                segment_binary_saving=max(0, generic_binary_estimate-segment_binaries),
                witness_counts=[len(group) for group in groups],
                scope='fixed-Z frozen residual upper master; not nonlinear AC')


def reconstruct_start(master, operation_x):
    """Complete an unchanged parent operation with contiguous chord variables."""
    x = np.asarray(operation_x, float)
    if not operation_tools._audit(x, master['operation'], 1e-7)['feasible']:
        raise ValueError('legal complete operation required for master start')
    point = np.zeros(len(master['objective'])); point[:len(x)] = x
    u = np.asarray(master['operation']['U']@x).ravel()
    for mode in master['modes']:
        factor, lam, select = mode['factor'], mode['lambda_indices'], mode['segment_indices']
        grid = factor['grid']; value = float(factor['coefficient']@u)
        k = min(max(int(np.searchsorted(grid, value, side='right'))-1, 0), len(grid)-2)
        weight = np.clip((value-grid[k])/(grid[k+1]-grid[k]), 0., 1.)
        point[lam[k]], point[lam[k+1]] = 1.-weight, weight
        if select: point[select[k]] = 1.
    upper_rows = {a: [] for a in master['references']}
    for pair, audit, witnesses in zip(master['pairs'], master['pair_audits'], master['witness_rows']):
        common = audit['quadratic_rounding_upper']
        if audit['form'] == 'shared_control_square':
            common += sum(float(d*np.interp(f['coefficient']@u, f['grid'], f['grid']**2))
                          for d, f in zip(audit['diagonal'], audit['factors']))
        else:
            common += sum(float(np.interp(f['coefficient']@u, f['grid'], .5*f['grid']**2))
                          for f in audit['factors'])
        for witness in witnesses:
            upper_rows[pair.a_label].append(common+witness['control_linear']@u+witness['constant'])
    for a, j in master['eta'].items(): point[j] = min(upper_rows[a])
    return point


def _master_audit(master, x, tolerance):
    return operation_tools._audit(x, dict(lower=master['bounds'].lb, upper=master['bounds'].ub,
        integrality=master['integrality'], constraints=[master['constraints']]), tolerance)


def solve_bounded_milp(pairs, domain, *, initial_operations, references=None,
                       segments=4, max_iterations=2, solver_seconds=3., total_seconds=30.,
                       qp_backend='gurobi', qp_seconds=3., qp_options=None, backend='gurobi',
                       mip_rel_gap=1e-4, absolute_gap=1e-6, relative_gap=1e-4,
                       feasibility_tolerance=1e-7, factor_bounds_provider=None,
                       shared_control_squares='auto'):
    """Exchange legal explanations, retaining parents through interval checks.

    All supplied operations are independently scored.  ``power_kw`` is the
    interval-protected selection: it changes only if a candidate's lower bound
    exceeds the selected parent's upper bound.  ``proposals`` additionally
    exposes every physically valid master proposal for a separate complete-AC
    acceptance rule.  No evaluated action library or random proposal is used.
    """
    began = perf_counter(); pairs, labels = bounded.validate_pairs(pairs, references)
    if (max_iterations < 0 or int(max_iterations) != max_iterations or
            min(solver_seconds, total_seconds, qp_seconds) <= 0):
        raise ValueError('nonnegative integer iterations and positive budgets required')
    op = operation_tools._operation(domain, pairs[0].control_dimension)
    options = dict(backend=qp_backend, seconds=qp_seconds)
    options.update(qp_options or {})
    witnesses = [[] for pair in pairs]; evaluations = []; failures = []; iterations = []
    proposed_records = []; qp_calls = 0; best_upper = None

    def evaluate(x, origin):
        nonlocal qp_calls
        audit = operation_tools._audit(x, op, feasibility_tolerance)
        if not audit['feasible']:
            failures.append(dict(kind='infeasible_operation', origin=origin, audit=audit)); return None
        x = np.asarray(x, float).copy(); u = np.asarray(op['U']@x).reshape(pairs[0].control_shape)
        solutions = []
        for i, pair in enumerate(pairs):
            solution = bounded.solve_pair(pair, u, **options); qp_calls += 1
            solutions.append(solution)
            z = solution['z']
            if not any(np.allclose(z, old, rtol=1e-10, atol=1e-11) for old in witnesses[i]):
                witnesses[i].append(z.copy())
            if not solution['completed'] or solution['failures']:
                failures.append(dict(kind='inner_QP_unfinished_or_failed', origin=origin,
                    a=pair.a_label, b=pair.b_label, inner_gap=solution['inner_gap'],
                    solver=solution['solver'], failures=solution['failures']))
        aggregation = bounded.aggregate_solutions(solutions, labels)
        record = dict(origin=origin, operation_x=x, power_kw=u, audit=audit,
                      aggregation=aggregation, solutions=solutions,
                      score_lower=aggregation['lower'], score_upper=aggregation['upper'])
        evaluations.append(record)
        return record

    for name, x in _names(initial_operations):
        evaluate(x, 'initial_operation:'+str(name))
    if not evaluations:
        return dict(success=False, reason='no_legal_parent_operation', failures=failures,
                    seconds=perf_counter()-began, qp_calls=qp_calls)
    selected = evaluations[0]
    for candidate in evaluations[1:]:
        if candidate['score_lower'] > selected['score_upper']:
            selected = candidate
    best_initial_lower = max(e['score_lower'] for e in evaluations)
    first_parent_upper = evaluations[0]['score_upper']
    for number in range(int(max_iterations)):
        remaining = total_seconds-(perf_counter()-began)
        if remaining <= 0:
            failures.append(dict(kind='total_budget_before_master', iteration=number)); break
        master = build_master(pairs, witnesses, domain, references=labels, segments=segments,
                              factor_bounds_provider=factor_bounds_provider,
                              shared_control_squares=shared_control_squares)
        start_record = max(evaluations, key=lambda e: e['score_lower'])
        start = reconstruct_start(master, start_record['operation_x'])
        start_audit = _master_audit(master, start, feasibility_tolerance)
        if not start_audit['feasible']:
            raise RuntimeError('reconstructed unchanged parent violates upper master')
        solve_began = perf_counter()
        result = solve_linear_mip(master['objective'], master['integrality'], master['bounds'],
            master['constraints'], time_limit=min(solver_seconds, remaining), mip_rel_gap=mip_rel_gap,
            start=start, backend=backend)
        rawdual = getattr(result, 'mip_dual_bound', None)
        dual = (None if int(result.status) not in (0, 1) or rawdual is None
                or not np.isfinite(rawdual) else -float(rawdual))
        candidate_upper = master['box_upper'] if dual is None else min(master['box_upper'], dual)
        best_upper = candidate_upper if best_upper is None else min(best_upper, candidate_upper)
        entry = dict(iteration=number, solver_status=int(result.status), message=str(result.message),
            mip_best_bound=dual, master_box_upper=master['box_upper'], upper=best_upper,
            mip_incumbent_upper_model=None if result.fun is None else -float(result.fun),
            seconds=perf_counter()-solve_began, witness_counts_before=master['witness_counts'],
            n_operation_variables=len(op['lower']), n_master_variables=len(master['objective']),
            n_segment_binary=master['n_segment_binary'], factor_ranges=master['pair_audits'],
            shared_control_square=master['shared_control_square'],
            generic_segment_binary_estimate=master['generic_segment_binary_estimate'],
            segment_binary_saving=master['segment_binary_saving'],
            start_audit=start_audit, backend_used=getattr(result, 'backend_used', None),
            mip_gap=getattr(result, 'mip_gap', None))
        iterations.append(entry)
        if int(result.status) >= 2:
            failures.append(dict(kind='master_solver_failure', iteration=number,
                                  status=int(result.status), message=str(result.message)))
        if result.x is None:
            failures.append(dict(kind='master_no_incumbent', iteration=number, status=int(result.status)))
            break
        vector = np.asarray(result.x, float)
        entry['master_incumbent_audit'] = _master_audit(master, vector, feasibility_tolerance)
        x = vector[:len(op['lower'])]
        entry['operation_incumbent_audit'] = operation_tools._audit(x, op, feasibility_tolerance)
        if not entry['operation_incumbent_audit']['feasible']:
            failures.append(dict(kind='master_physical_violation', iteration=number,
                                  audit=entry['operation_incumbent_audit'])); break
        if not entry['master_incumbent_audit']['feasible']:
            failures.append(dict(kind='master_auxiliary_violation', iteration=number,
                                  audit=entry['master_incumbent_audit']))
        duplicate = next((e for e in evaluations if np.max(abs(e['operation_x']-x), initial=0.) <= 1e-10), None)
        record = duplicate if duplicate is not None else evaluate(x, 'bounded_MILP_'+str(number))
        if record is not None:
            proposed_records.append(record)
            if record['score_lower'] > selected['score_upper']:
                selected = record
        entry['witness_counts_after'] = [len(group) for group in witnesses]
        lower = max(e['score_lower'] for e in evaluations)
        entry['feasible_lower'] = lower
        if best_upper < lower-1e-7*(1.+abs(lower)):
            failures.append(dict(kind='master_upper_below_feasible_lower', lower=lower, upper=best_upper))
            best_upper = None; break
        if best_upper-lower <= absolute_gap+relative_gap*max(1., abs(best_upper), abs(lower)):
            break
        if duplicate is not None and entry['witness_counts_after'] == entry['witness_counts_before']:
            entry['stopped_duplicate_action_no_new_explanation'] = True
            break
    lower = max(e['score_lower'] for e in evaluations)
    if best_upper is None:
        box_rows = [dict(a=p.a_label, upper=_box_upper(p, group, op)) for p, group in zip(pairs, witnesses)]
        best_upper = float(np.mean([min(r['upper'] for r in box_rows if r['a'] == a) for a in labels]))
    guard = 128*np.finfo(float).eps*(1.+abs(best_upper)+abs(lower))
    best_upper += guard; lower -= guard
    gap = best_upper-lower
    completed = gap >= 0 and gap <= absolute_gap+relative_gap*max(1., abs(best_upper), abs(lower))
    parent_checks = [dict(origin=e['origin'], same_plan=e is selected,
        interval_dominance_certified=(e is selected or selected['score_lower'] >= e['score_upper']))
        for e in evaluations if e['origin'].startswith('initial_operation:')]
    return dict(success=True, power_kw=selected['power_kw'], operation_x=selected['operation_x'],
                selected_origin=selected['origin'], selected=selected,
                objective_lower=lower, objective_upper=best_upper, gap=gap,
                completed=bool(completed), unfinished=not completed,
                inheritance_protected=True,
                inheritance_scope='first supplied parent; all replacements require interval dominance',
                all_parent_dominance_certified=all(check['interval_dominance_certified'] for check in parent_checks),
                parent_comparisons=parent_checks,
                interval_selection_rule='replace only when new lower > selected upper',
                first_parent_upper=first_parent_upper, best_initial_lower=best_initial_lower,
                evaluations=evaluations, proposals=proposed_records, witnesses=witnesses,
                failures=failures, iterations=iterations, qp_calls=qp_calls,
                MILP_calls=len(iterations), n_pairs=len(pairs), reference_labels=list(labels),
                seconds=perf_counter()-began, full_route_enumeration=False,
                uses_truth=False, uses_future_noise=False,
                scope='global bounds only for the declared frozen bounded residual design model')
