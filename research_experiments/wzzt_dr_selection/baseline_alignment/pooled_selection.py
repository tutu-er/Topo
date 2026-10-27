"""Compare a complete RNJ model and the WZZT path by the original one-SE rule.

This module does not consume topology truth. RNJ is an additional model, not
an initialization that replaces or constrains the already computed WZZT path.
The production selector counts stored supports, including zero-weight fixed
supports. We retain that convention for every candidate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from types import SimpleNamespace

import numpy as np

from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed

core = observed.core


def _record(point):
    # Keep NaN/Inf until validation rejects them; the JSON helper maps them to
    # null and would otherwise confuse invalid losses with missing validation.
    return asdict(point) if is_dataclass(point) else deepcopy(point)


def _checked_point(point):
    """The core selector needs only these fields; explicitly exclude metadata."""
    value = _record(point)
    supports = value['support_indices']
    if not supports:
        raise ValueError('candidate has no stored support')
    mean, se = value['validation_mae'], value['validation_se']
    if mean is None or se is None:
        raise ValueError('every candidate requires validation MAE and SE')
    if not np.isfinite(mean) or not np.isfinite(se) or mean < 0 or se < 0:
        raise ValueError('validation MAE and SE must be finite and nonnegative')
    return SimpleNamespace(support_indices=supports, validation_mae=float(mean),
                           validation_se=float(se))


def select_with_complete_rnj(path, rnj_candidate=None, *, rnj_failure=None):
    """Apply the production rule once to all original path points plus RNJ.

    Path order is retained and RNJ is appended. Equal complexity is resolved by
    the original path order, so a complete tie cannot silently favor RNJ.
    Without validation we keep the original path selection and mark comparison
    unavailable. A missing or failed RNJ candidate never overwrites WZZT.
    """
    if not path:
        raise ValueError('a nonempty WZZT path is required')
    records = [_record(point) for point in path]
    missing_validation = any(p.get('validation_mae') is None or p.get('validation_se') is None
                             for p in records)
    if missing_validation:
        original = (len(records)-1 if records[0].get('validation_mae') is None else None)
        return {'status':'unavailable', 'reason':'missing_validation',
                'original_wzzt_selected_path_index':original,
                'selected_origin':'wzzt_path' if original is not None else None,
                'selected_path_index':original, 'selected':records[original] if original is not None else None,
                'rnj_candidate_included':False, 'rnj_failure':rnj_failure,
                'candidate_count':len(records)}
    path_points = [_checked_point(p) for p in records]
    original = core._choose_path_point(path_points)
    candidates = records.copy()
    points = path_points.copy()
    failure = rnj_failure
    if rnj_candidate is not None:
        try:
            candidate = _record(rnj_candidate)
            checked = _checked_point(candidate)
            diag = candidate.get('solver')
            if diag is not None:
                valid_keys = {field.name for field in fields(core.SolverDiagnostics)}
                diagnostic = core.SolverDiagnostics(**{k:v for k,v in diag.items() if k in valid_keys})
                if not core.solver_diagnostics_prove_optimality(diagnostic):
                    raise ValueError('complete RNJ fixed-support fit is not numerically optimal')
            candidates.append(candidate)
            points.append(checked)
        except (ValueError, TypeError, KeyError) as exc:
            failure = {'type':type(exc).__name__, 'message':str(exc)}
    selected = core._choose_path_point(points)
    best = int(np.argmin([p.validation_mae for p in points]))
    threshold = points[best].validation_mae + points[best].validation_se
    eligible = [i for i,p in enumerate(points) if p.validation_mae <= threshold]
    included = len(candidates) > len(records)
    return {'status':'complete' if included else 'rnj_candidate_unavailable',
        'rule':'production _choose_path_point; one common minimum-MAE-plus-own-SE threshold',
        'complexity':'stored support count, including zero-weight fixed supports',
        'tie_rule':'original WZZT path order first; complete RNJ candidate appended',
        'candidate_count':len(candidates), 'rnj_candidate_included':included,
        'rnj_failure':failure, 'original_wzzt_selected_path_index':original,
        'minimum_validation_candidate_index':best, 'threshold_pu2':threshold,
        'eligible_candidate_indices':eligible,
        'selected_candidate_index':selected,
        'selected_origin':'complete_rnj' if selected == len(records) else 'wzzt_path',
        'selected_path_index':None if selected == len(records) else selected,
        'selected':candidates[selected],
        'candidates':[{'index':i, 'origin':'complete_rnj' if i == len(records) else 'wzzt_path',
            'path_index':None if i == len(records) else i,
            'stored_support_count':len(p.support_indices),
            'validation_mae':p.validation_mae, 'validation_se':p.validation_se,
            'eligible':i in eligible} for i,p in enumerate(points)]}


def fit_complete_rnj_candidate(training, validation, full_clades, *, r_upper_bound,
                               x_upper_bound, time_limit=None, solver='highs',
                               validation_blocks=4):
    """Fit only weights of the algorithm-generated complete RNJ family.

    Validation data are evaluated after training-only fitting. All fixed RNJ
    supports remain stored for the same complexity convention as a fixed WZZT
    family. No lower bound or artificial positive weight is introduced.
    """
    if validation is None:
        raise ValueError('complete RNJ model comparison requires validation data')
    prepared = core._prepare_scenarios(training)
    positions = {label:i for i,label in enumerate(prepared.labels)}
    family = {(i,) for i in range(prepared.n)}
    for clade in full_clades:
        family.add(tuple(sorted(positions[label] for label in clade)))
    supports = core.normalize_supports(family, prepared.n)
    solution = core.solve_fixed_support_l1(training, supports,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
        time_limit=time_limit, solver=solver)
    if not core.solver_diagnostics_prove_optimality(solution.diagnostics):
        raise RuntimeError('complete RNJ LP did not reach the numerical optimality criterion')
    point = core._make_path_point(iteration=-1, prepared=prepared, solution=solution,
        validation_scenarios=validation, validation_blocks=validation_blocks,
        accepted_gain=None, solver=solution.diagnostics)
    recomputed, _, _ = core.evaluate_l1_matrices(training, point.r_matrix, point.x_matrix)
    if not np.isclose(recomputed, point.train_mae, atol=1e-10, rtol=1e-7):
        raise RuntimeError('complete RNJ training objective does not reproduce')
    return point
