"""Extract finite numerical references from an already completed observed fit.

No fitting, power/voltage arrays, truth or future files are accessed. A recorded
native/proposed family is distinguished from saved coefficients: coefficients
discarded by PoolLP's <=1e-9 rule are NOT reconstructed as exact zeros.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from hashlib import sha256
import json
from math import isfinite


def canonical(supports, *, n=None):
    rows = []
    for support in supports:
        row = tuple(support)
        if not row or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in row):
            raise ValueError('supports need nonempty nonnegative integer indices')
        if len(set(row)) != len(row) or (n is not None and max(row) >= n):
            raise ValueError('invalid support indices')
        rows.append(tuple(sorted(row)))
    if len(set(rows)) != len(rows):
        raise ValueError('duplicate supports in one recorded family')
    return tuple(sorted(rows, key=lambda s: (len(s), s)))


def compatible(left, right):
    a, b = set(left), set(right)
    return not (a & b) or a <= b or b <= a


def _laminar(family):
    return all(compatible(a, b) for i, a in enumerate(family) for b in family[i+1:])


def _saved_model(point, n):
    supports = point['supports']
    canonical(supports, n=n)
    if len(supports) != len(point['r']) or len(supports) != len(point['x']):
        raise ValueError('saved supports and coefficients are not aligned')
    rows = []
    for support, r, x in zip(supports, point['r'], point['x'], strict=True):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0 for v in (r, x)):
            raise ValueError('saved response weights must be finite nonnegative numbers')
        rows.append((tuple(sorted(support)), float(r), float(x)))
    return tuple(sorted(rows, key=lambda row: (len(row[0]), row[0])))


def _signature(model):
    # Include coefficients, not just supports: different parameter explanations
    # must not be collapsed merely because they use the same structural family.
    return tuple((support, r.hex(), x.hex()) for support, r, x in model)


def _strict_best(records):
    best = None
    for record in records:
        value = record.get('objective')
        if not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError('candidate decision has no finite objective')
        if best is None or value < best['objective']-1e-12:
            best = record
    if best is None:
        raise ValueError('accepted path transition has no candidate record')
    return best


def recover_path_containers(result):
    """Reconstruct what each saved path point's LP was allowed to fit.

    The previous *saved/pruned* supports form the next current family, matching
    engine.search. Using the previous unpruned container here would be wrong.
    Ordered candidate comparisons reproduce objective < best - 1e-12 exactly.
    """
    n = len(result['terminal_labels'])
    primary = [p for p in result['bank'] if p.get('config') == 'raw_RX75_0.16']
    recovered = {}
    for name, search in result['searches'].items():
        path = search['path']
        if not path:
            raise ValueError('saved search path is empty')
        if name == 'primary':
            if len(primary) != 1:
                raise ValueError('primary raw_RX75_0.16 native family is ambiguous')
            initial = primary[0]['native_family']
        else:
            initial = search['source_family']
        family = canonical(initial, n=n)
        if not _laminar(family):
            raise ValueError('nonlaminar initial path container')
        if not set(canonical(path[0]['supports'], n=n)) <= set(family):
            raise ValueError('initial saved supports lie outside their full container')
        recovered[(name, 0)] = dict(family=family, source='recorded_initial_family', objective_difference=None)
        for step, point in enumerate(path[1:], 1):
            if step-1 >= len(search['decisions']):
                raise ValueError('missing path transition decision')
            best = _strict_best(search['decisions'][step-1]['candidates'])
            current = canonical(path[step-1]['supports'], n=n)
            candidate = canonical([best['candidate']], n=n)[0]
            removed = canonical(best['removed'], n=n)
            expected_removed = tuple(s for s in current if not compatible(s, candidate))
            if set(removed) != set(expected_removed):
                raise ValueError('cross-deletion record differs from current pruned family')
            retained = [s for s in current if s not in removed]
            # A missing candidate cannot already be part of the current family.
            family = canonical([*retained, candidate], n=n)
            if not _laminar(family) or not set(canonical(point['supports'], n=n)) <= set(family):
                raise ValueError('accepted point lies outside its legal proposed family')
            difference = float(point['objective'])-float(best['objective'])
            if not isfinite(difference) or abs(difference) > 1e-7*(1+abs(float(best['objective']))):
                raise ValueError('accepted point objective does not match ordered best candidate')
            recovered[(name, step)] = dict(family=family, source='ordered_best_candidate_and_cross_deletion',
                                          objective_difference=difference)
    return recovered


def fixed_prediction_loss_bounds(predicted, known_positive, unknown_supports):
    """Exact truth-label bounds for a FIXED predicted family.

    True positive supports are K union T, where T is any subset of U. This does
    not account for how unknown parameters change future observations or the
    identifier's output; it is not by itself a bound on a future price risk.
    """
    s, k, u = set(canonical(predicted)), set(canonical(known_positive)), set(canonical(unknown_supports))
    if k & u:
        raise ValueError('known-positive and unresolved supports must be disjoint')
    if not _laminar(canonical([*k, *u])):
        raise ValueError('known and unknown supports must share a legal laminar container')
    base = len(s ^ k)
    return dict(lower=base-len(s & u), upper=base+len(u-s),
                scope='fixed_prediction_and_truth_support_uncertainty_only',
                bounds_future_input_or_algorithm_changes=False)


def extract_reference(result, *, source_json_sha256=None, scientific_reference_locked=False,
                      pruning_threshold=1e-9):
    """Preserve all selection rows and annotate every unavailable coefficient.

    The scientific reference is a separate modeling decision; extraction does
    not establish it. Neither 3% validation nor output contraction filters rows.
    """
    if result.get('status') != 'complete' or result.get('truth_input_allowed') is not False:
        raise ValueError('require a completed observed-only result')
    if result.get('final_strategy') != 'multistart_original_rule':
        raise ValueError('require the declared original joint strategy')
    if not isfinite(pruning_threshold) or pruning_threshold <= 0:
        raise ValueError('a positive source pruning threshold must be declared')
    labels = result['terminal_labels']
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('terminal labels must be nonempty and unique')
    n = len(labels)
    inputs = result['input_sha256']
    if not all(isinstance(inputs.get(key), str) and inputs[key] for key in ('training', 'validation')):
        raise ValueError('both historical split hashes are required')
    bank_by_config = {}
    for index, point in enumerate(result['bank']):
        config = point['config']
        if config in bank_by_config:
            raise ValueError('duplicate native bank configuration')
        bank_by_config[config] = (index, point)
    recovered = recover_path_containers(result)
    rows, models, ids, full_models, full_ids = [], [], {}, [], {}
    actual_sources = Counter()
    for index, point in enumerate(result['selection_candidates']):
        origin = point.get('origin')
        if origin == 'tree_bank':
            bank_index, bank_point = bank_by_config[point['config']]
            original = bank_point
            family = canonical(bank_point['native_family'], n=n)
            if _saved_model(point, n) != _saved_model(bank_point, n):
                raise ValueError('selection bank point differs from its historical bank source')
            provenance = dict(kind='complete_RNJ_bank', bank_index=bank_index, config=point['config'])
            actual_sources[('bank', point['config'])] += 1
            path_objective = None
        else:
            search_name = 'primary' if origin == 'tree_initialized_correction' else origin
            path_index = point['path_index']
            if isinstance(path_index, bool) or not isinstance(path_index, int):
                raise ValueError('path index must be an integer')
            record = recovered[(search_name, path_index)]
            original = result['searches'][search_name]['path'][path_index]
            if _saved_model(point, n) != _saved_model(original, n):
                raise ValueError('selection path point differs from its historical path source')
            family = record['family']
            provenance = dict(kind='WZZT_path', search=search_name, path_index=path_index,
                              reconstruction=record['source'], objective_difference=record['objective_difference'])
            actual_sources[('path', search_name, path_index)] += 1
            path_objective = float(original['objective'])
        if not _laminar(family):
            raise ValueError('nonlaminar declared reference family')
        model = _saved_model(point, n)
        saved = tuple(row[0] for row in model)
        if not set(saved) <= set(family):
            raise ValueError('saved coefficient support outside complete container')
        discarded = tuple(s for s in family if s not in set(saved))
        if any(len(s) == 1 for s in discarded):
            raise ValueError('source rule always retained singleton coefficients')
        zero = tuple(s for s, r, x in model if r == 0 and x == 0)
        full_fit = point.get('full_fit', original.get('full_fit'))
        full_model = None
        if full_fit is not None:
            full_model = _saved_model(full_fit, n)
            if tuple(row[0] for row in full_model) != family:
                raise ValueError('full_fit coefficients do not cover the reconstructed container')
            full_map = {s: (r, x) for s, r, x in full_model}
            if any(full_map[s] != (r, x) for s, r, x in model):
                raise ValueError('full_fit differs from its retained saved coefficients')
            upper = full_fit.get('weight_upper_bound')
            threshold = full_fit.get('support_prune_tolerance')
            if not isinstance(upper, (int, float)) or not isfinite(upper) or upper <= 0:
                raise ValueError('full_fit needs its numerical solver weight bound')
            if threshold != pruning_threshold or any(max(r, x) > upper for _, r, x in full_model):
                raise ValueError('full_fit numerical settings differ from declared source semantics')
            if any(max(full_map[s]) > pruning_threshold for s in discarded):
                raise ValueError('source dropped coefficient above its stated pruning threshold')
            if not isinstance(full_fit.get('semantics'), str) or not full_fit['semantics']:
                raise ValueError('full_fit numerical provenance must be explicit')
        elif not discarded:
            full_model = model
        unknown = discarded if full_model is None else ()
        known_model = model if full_model is None else full_model
        positive = tuple(s for s, r, x in known_model if r > 0 or x > 0)
        signature = _signature(model)
        if signature not in ids:
            ids[signature] = f'numerical_{len(ids):03d}'
            models.append(dict(id=ids[signature], saved_coefficients=[dict(support=s, r=r, x=x) for s, r, x in model]))
        full_id = None
        if full_model is not None:
            full_signature = _signature(full_model)
            if full_signature not in full_ids:
                full_ids[full_signature] = f'full_numerical_{len(full_ids):03d}'
                full_models.append(dict(id=full_ids[full_signature], coefficients=[dict(support=s, r=r, x=x) for s, r, x in full_model]))
            full_id = full_ids[full_signature]
        row = dict(id=f'selection_{index:03d}', selection_index=index, provenance=provenance,
                   numerical_point_id=ids[signature], complete_container=family,
                   full_parameter_point_id=full_id,
                   saved_supports=saved, known_positive_numerical_supports=positive,
                   retained_exact_zero_numerical_supports=zero,
                   coefficient_knowledge_source='full_fit_before_numerical_pruning' if full_fit is not None else 'legacy_saved_coefficients',
                   unavailable_coefficients=[dict(support=s, r_interval=[0., pruning_threshold],
                                                  x_interval=[0., pruning_threshold],
                                                  status='unresolved_source_discarded_not_proven_zero') for s in unknown],
                   exact_full_parameter_point_available=not unknown,
                   reduced_saved_model_does_not_determine_unpruned_solution=bool(unknown),
                   source_applied_numerical_support_pruning=bool(discarded),
                   original_reported_objective=float(point['objective']),
                   selection_saved_point_objective=float(point['objective']),
                   solver_objective=point.get('solver_objective', original.get('solver_objective')),
                   source_path_objective=path_objective,
                   full_fit_objective=None if full_fit is None else full_fit.get('objective'),
                   objective_semantics='selection top objective may be remeasured after pruning; path/solver/full_fit objectives are not assumed equal',
                   historical_validation_mae=float(point['validation_mae']),
                   row_kept_without_validation_or_contraction_filter=True)
        rows.append(row)
    expected_rows = len(result['bank'])+sum(len(s['path']) for s in result['searches'].values())
    if len(rows) != expected_rows:
        raise ValueError('selection rows do not cover the complete bank and all recorded paths')
    expected_sources = Counter(('bank', point['config']) for point in result['bank'])
    expected_sources.update(('path', name, index)
                            for name, search in result['searches'].items()
                            for index in range(len(search['path'])))
    if actual_sources != expected_sources:
        raise ValueError('selection provenance coverage differs from complete bank/path sources')
    if not rows:
        raise ValueError('no historical reference points')
    missing_rows = sum(bool(row['unavailable_coefficients']) for row in rows)
    return dict(schema='wzzt_saved_history_reference_v1',
                status='finite_numerical_reference_with_explicit_unknowns' if missing_rows else 'finite_saved_numerical_reference',
                source_json_sha256=source_json_sha256, historical_fingerprint=result.get('fingerprint'),
                input_sha256=deepcopy(inputs), source_sha256=deepcopy(result['source_sha256']),
                settings=deepcopy(result['settings']), root_label=result['root_label'], terminal_labels=deepcopy(labels),
                training_rows=result['training_rows'], validation_rows=result['validation_rows'],
                scientific_reference_locked=bool(scientific_reference_locked),
                reference_generation_ready=bool(scientific_reference_locked and not missing_rows),
                risk_ready=False, risk_ready_reason='extraction supplies neither conditional future-loss table nor a calibrated future law',
                topology_posterior=False, continuous_parameter_coverage=False, full_tree_coverage=False,
                algorithm_bound_is_physical_prior=False, weights_are_normalized_response_weights_not_raw_ohms=True,
                numeric_zero_is_physical_zero_claim=False,
                pruning=dict(threshold=pruning_threshold, discarded_interpretation='unknown in interval, never silently filled with zero'),
                references=rows, numerical_points=models, full_parameter_points=full_models,
                summary=dict(selection_rows=len(rows), bank_rows=len(result['bank']), path_rows=len(rows)-len(result['bank']),
                             unique_saved_numerical_points=len(models),
                             unique_complete_full_parameter_points=len(full_models),
                             unique_complete_containers=len({tuple(map(tuple, row['complete_container'])) for row in rows}),
                             unique_saved_families=len({tuple(map(tuple, row['saved_supports'])) for row in rows}),
                             rows_with_unavailable_coefficients=missing_rows,
                             unavailable_support_occurrences=sum(len(row['unavailable_coefficients']) for row in rows),
                             retained_zero_support_occurrences=sum(len(row['retained_exact_zero_numerical_supports']) for row in rows)))


def extract_json_file(source_path, output_path, *, scientific_reference_locked=False):
    """File adapter restricted to the explicitly supplied completed JSON result."""
    from pathlib import Path
    source, output = Path(source_path), Path(output_path)
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing reference packet')
    payload = source.read_bytes()
    result = extract_reference(json.loads(payload), source_json_sha256=sha256(payload).hexdigest(),
                               scientific_reference_locked=scientific_reference_locked)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return result
