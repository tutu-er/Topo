"""Fixed and joint source domains checked against independently enumerated LPs."""

import numpy as np
import pandas as pd
import pytest

scipy_optimize = pytest.importorskip("scipy.optimize")

from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation import laminar_l1_source as source
from rnj_wzzt.estimation.unmetered_inputs import PreparedSourceInputs


def _source_inputs(p, q):
    p_blocks = tuple(np.asarray(block, dtype=float) for block in p)
    return PreparedSourceInputs(
        p=p_blocks,
        q=tuple(np.asarray(block, dtype=float) for block in q),
        time_indices=tuple(pd.RangeIndex(len(block)) for block in p_blocks),
        provenance=tuple("supplied_test_amplitude" for _ in p_blocks),
    )


def _small_prepared():
    return core._PreparedScenarios(
        labels=(0, 1),
        p=(np.array([[0.2, 0.1], [0.4, 0.3]]), np.array([[0.1, 0.5]])),
        q=(np.array([[0.03, -0.02], [0.04, 0.01]]), np.array([[-0.01, 0.02]])),
        target=(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.2, 0.1]])),
    )


def _solve_model_lp(model, objective=None, *, require_success=True):
    """Use LP only after every binary source-path variable has been pinned."""
    rows = model.rows.build(model.variables.size)
    matrix = rows.A.toarray()
    equal = np.isfinite(rows.lb) & (rows.lb == rows.ub)
    upper = np.isfinite(rows.ub) & ~equal
    lower = np.isfinite(rows.lb) & ~equal
    a_ub = np.vstack((matrix[upper], -matrix[lower]))
    b_ub = np.concatenate((rows.ub[upper], -rows.lb[lower]))
    result = scipy_optimize.linprog(
        model.variables.objective if objective is None else objective,
        A_ub=a_ub if len(a_ub) else None,
        b_ub=b_ub if len(b_ub) else None,
        A_eq=matrix[equal] if equal.any() else None,
        b_eq=rows.lb[equal] if equal.any() else None,
        bounds=list(zip(model.variables.lower, model.variables.upper)),
        method="highs",
    )
    if require_success:
        assert result.success, result.message
    return result


def _direct_design(p_blocks, q_blocks, supports, h, amplitudes):
    """Scalar physical formula, independent of the production feature builders."""
    design = []
    n = p_blocks[0].shape[1]
    for scenario_index, (p, q) in enumerate(zip(p_blocks, q_blocks, strict=True)):
        for time_index in range(len(p)):
            for output_index in range(n):
                r_row, x_row = [], []
                for k, support in enumerate(supports):
                    if output_index in support:
                        r_row.append(sum(p[time_index, j] for j in support)
                                     + amplitudes.p[scenario_index][time_index] * h[k])
                        x_row.append(sum(q[time_index, j] for j in support)
                                     + amplitudes.q[scenario_index][time_index] * h[k])
                    else:
                        r_row.append(0.0)
                        x_row.append(0.0)
                design.append(r_row + x_row)
    return np.asarray(design)


def _direct_l1_oracle(design, observed, atom_count):
    observations = len(observed)
    result = scipy_optimize.linprog(
        np.r_[np.zeros(2 * atom_count), np.full(observations, 1.0 / observations)],
        A_ub=np.vstack((np.c_[design, -np.eye(observations)],
                        np.c_[-design, -np.eye(observations)])),
        b_ub=np.r_[observed, -observed],
        bounds=[(0.0, 2.0)] * atom_count + [(0.0, 1.7)] * atom_count
               + [(0.0, None)] * observations,
        method="highs",
    )
    assert result.success, result.message
    return result


def _solve_model_milp(model, *, require_success=True):
    result = scipy_optimize.milp(
        model.variables.objective,
        integrality=model.variables.integrality,
        bounds=scipy_optimize.Bounds(model.variables.lower, model.variables.upper),
        constraints=model.rows.build(model.variables.size),
        options={"time_limit": 15.0, "mip_rel_gap": 0.0},
    )
    if require_success:
        assert result.success, result.message
    return result


def _enumerate_new_clades(n, supports):
    """Enumerate the finite domain by set definitions, without production predicates."""
    existing = [set(support) for support in supports]
    for bits in range(1, 1 << n):
        candidate = {i for i in range(n) if bits & (1 << i)}
        if candidate in existing:
            continue
        if all(not candidate.intersection(other) or candidate <= other or other <= candidate
               for other in existing):
            yield tuple(sorted(candidate))


@pytest.mark.parametrize(
    ("h", "use_raw_scenarios"),
    [([1, 1, 1, 0, 0], False), ([1, 0, 0, 0, 1], True)],
    ids=["nested_terminal_path", "disjoint_branch_path"],
)
def test_supplied_path_matches_independent_linprog_and_direct_mae(h, use_raw_scenarios):
    supports = ((0, 1, 2), (0, 1), (0,), (1,), (2,))
    h = np.asarray(h)
    rng = np.random.default_rng(812)
    p = tuple(rng.uniform(0.05, 0.4, size=(count, 3)) for count in (5, 4))
    q = tuple(rng.uniform(-0.08, 0.12, size=block.shape) for block in p)
    amplitudes = _source_inputs(
        ([0.07, 0.13, 0.02, 0.19, 0.09], [0.15, 0.04, 0.11, 0.08]),
        ([-0.06, 0.04, -0.02, 0.07, -0.05], [0.03, -0.07, 0.05, -0.04]),
    )
    design = _direct_design(p, q, supports, h, amplitudes)
    weights = np.array([0.3, 0.2, 0.15, 0.25, 0.1, 0.12, 0.08, 0.2, 0.06, 0.14])
    observed = design @ weights + 0.02 * np.sin(1.31 * np.arange(len(design)) + 0.5)
    targets = (observed[:15].reshape(5, 3), observed[15:].reshape(4, 3))
    prepared = core._PreparedScenarios(labels=(0, 1, 2), p=p, q=q, target=targets)
    inputs = prepared
    if use_raw_scenarios:
        inputs = [
            {"P_terminal": p_s, "Q_terminal": q_s, "drop_target": y_s}
            for p_s, q_s, y_s in zip(p, q, targets, strict=True)
        ]
    selection_calls = []

    def pin_known_path(model, actual_supports, blocks):
        # This supplies one known location; it does not define the admissible search domain.
        assert actual_supports == supports
        selection_calls.append(blocks)
        for k, membership in enumerate(h):
            model.rows.add({blocks.h.start + k: 1.0}, lower=membership, upper=membership)

    model, blocks = source.build_fixed_source_model(
        inputs, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
        selection_constraints=pin_known_path,
    )
    assert blocks is not None
    assert len(selection_calls) == 1
    assert selection_calls[0] is blocks
    result = _solve_model_lp(model)

    # Independent oracle eliminates h and its products altogether.
    observations = len(observed)
    oracle = scipy_optimize.linprog(
        np.r_[np.zeros(10), np.full(observations, 1.0 / observations)],
        A_ub=np.vstack((np.c_[design, -np.eye(observations)],
                        np.c_[-design, -np.eye(observations)])),
        b_ub=np.r_[observed, -observed],
        bounds=[(0.0, 2.0)] * 5 + [(0.0, 1.7)] * 5 + [(0.0, None)] * observations,
        method="highs",
    )
    assert oracle.success, oracle.message
    assert oracle.fun > 1e-5
    assert result.fun == pytest.approx(oracle.fun, abs=1e-9)
    r_values, x_values = result.x[model.r], result.x[model.x]
    direct_mae = np.mean(np.abs(design @ np.r_[r_values, x_values] - observed))
    assert result.fun == pytest.approx(direct_mae, abs=1e-9)
    np.testing.assert_allclose(result.x[blocks.h], h, atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.w_r], r_values * h, atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.w_x], x_values * h, atol=1e-9)


@pytest.mark.parametrize("amplitudes", [None, "zero"], ids=["none", "zero"])
def test_disabled_source_is_exact_baseline_and_skips_selection(amplitudes):
    prepared = _small_prepared()
    supports = ((0, 1), (0,), (1,))
    if amplitudes == "zero":
        amplitudes = _source_inputs(([0.0, 0.0], [0.0]), ([0.0, 0.0], [0.0]))

    def must_not_select(*args):
        raise AssertionError("disabled sources must not call source selection")

    options = dict(r_upper_bound=2.0, x_upper_bound=1.7)
    baseline = core.build_fixed_model(prepared, supports, **options)
    actual, blocks = source.build_fixed_source_model(
        prepared, supports, amplitudes, **options, selection_constraints=must_not_select
    )
    assert blocks is None
    assert actual.variables.slices == baseline.variables.slices
    for name in ("objective", "lower", "upper", "integrality"):
        np.testing.assert_array_equal(getattr(actual.variables, name), getattr(baseline.variables, name))
    before = baseline.rows.build(baseline.variables.size)
    after = actual.rows.build(actual.variables.size)
    assert after.A.shape == before.A.shape
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(after.A, name), getattr(before.A, name))
    np.testing.assert_array_equal(after.lb, before.lb)
    np.testing.assert_array_equal(after.ub, before.ub)


def test_default_fixed_source_optimum_matches_all_reduced_nodes_enumerated_by_lp():
    supports = ((0, 1, 2), (0, 1), (0,), (1,), (2,))
    # A candidate is an existing clade node. Its ancestors include itself.
    # This enumeration uses elementary set inclusion, not a production path predicate.
    masks = [np.array([int(set(node) <= set(edge)) for edge in supports]) for node in supports]
    rng = np.random.default_rng(20261001)
    p = tuple(rng.uniform(0.05, 0.4, size=(count, 3)) for count in (4, 3))
    q = tuple(rng.uniform(-0.08, 0.12, size=block.shape) for block in p)
    amplitudes = _source_inputs(
        ([0.07, 0.13, 0.02, 0.19], [0.15, 0.04, 0.11]),
        ([-0.06, 0.04, -0.02, 0.07], [0.03, -0.07, 0.05]),
    )
    design = _direct_design(p, q, supports, masks[1], amplitudes)
    weights = np.array([0.3, 0.2, 0.15, 0.25, 0.1, 0.12, 0.08, 0.2, 0.06, 0.14])
    observed = design @ weights + 0.007 * np.cos(1.31 * np.arange(len(design)))
    prepared = core._PreparedScenarios(
        labels=(0, 1, 2), p=p, q=q,
        target=(observed[:12].reshape(4, 3), observed[12:].reshape(3, 3)),
    )
    oracle_values = [
        _direct_l1_oracle(_direct_design(p, q, supports, h, amplitudes), observed, 5).fun
        for h in masks
    ]
    model, blocks = source.build_fixed_source_model(
        prepared, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7
    )
    assert blocks is not None
    result = _solve_model_milp(model)
    assert result.fun == pytest.approx(min(oracle_values), abs=1e-8)
    chosen_h = result.x[blocks.h]
    assert any(np.allclose(chosen_h, candidate, atol=1e-7) for candidate in masks)
    chosen_design = _direct_design(p, q, supports, chosen_h, amplitudes)
    direct_mae = np.mean(np.abs(chosen_design @ np.r_[result.x[model.r], result.x[model.x]] - observed))
    assert result.fun == pytest.approx(direct_mae, abs=1e-8)


@pytest.mark.parametrize(
    ("h", "require_nonempty", "feasible"),
    [([1, 0, 0], True, True), ([0, 1, 0], True, False),
     ([1, 1, 1], True, False), ([0, 0, 0], True, False), ([0, 0, 0], False, True)],
    ids=["internal_clade", "missing_ancestor", "two_disjoint_children", "root_excluded", "root_optional"],
)
def test_fixed_selection_domain_has_ancestor_closure_disjointness_and_optional_root(h, require_nonempty, feasible):
    supports = ((0, 1), (0,), (1,))
    model = core._base_model(_small_prepared(), 3, r_upper_bound=2.0, x_upper_bound=1.7)
    blocks = source._add_source_variables(model, 3, r_upper_bound=2.0, x_upper_bound=1.7)
    source._add_source_selection_constraints(model, supports, blocks, require_nonempty=require_nonempty)
    for k, value in enumerate(h):
        model.rows.add({blocks.h.start + k: 1.0}, lower=value, upper=value)
    result = _solve_model_lp(model, require_success=False)
    assert result.status == (0 if feasible else 2), result.message


def test_nonzero_source_rejects_empty_supports_before_selection():
    def must_not_select(*args):
        raise AssertionError("empty source supports must be rejected first")

    with pytest.raises(ValueError):
        source.build_fixed_source_model(
            _small_prepared(), (),
            _source_inputs(([0.1, 0.2], [0.3]), ([-0.04, 0.02], [-0.05])),
            r_upper_bound=2.0, x_upper_bound=1.7, selection_constraints=must_not_select,
        )


def test_binary_product_constraints_fix_both_extremes_for_zero_and_one_membership():
    model = core._base_model(_small_prepared(), 2, r_upper_bound=2.0, x_upper_bound=1.7)
    blocks = source._add_source_variables(model, 2, r_upper_bound=2.0, x_upper_bound=1.7)
    source._add_source_product_constraints(model, blocks, r_upper_bound=2.0, x_upper_bound=1.7)
    np.testing.assert_array_equal(model.variables.integrality[blocks.h], [1, 1])
    np.testing.assert_array_equal(model.variables.lower[blocks.h], [0.0, 0.0])
    np.testing.assert_array_equal(model.variables.upper[blocks.h], [1.0, 1.0])
    for block, values in ((model.r, [1.1, 0.9]), (model.x, [0.4, 0.6]), (blocks.h, [0, 1])):
        for k, value in enumerate(values):
            model.rows.add({block.start + k: 1.0}, lower=value, upper=value)
    for direction in (-1.0, 1.0):
        objective = np.zeros(model.variables.size)
        objective[blocks.w_r] = direction
        objective[blocks.w_x] = direction
        result = _solve_model_lp(model, objective)
        np.testing.assert_allclose(result.x[blocks.w_r], [0.0, 0.9], atol=1e-9)
        np.testing.assert_allclose(result.x[blocks.w_x], [0.0, 0.6], atol=1e-9)


@pytest.mark.parametrize(
    ("p", "q"),
    [(([0.1, 0.2],), ([0.0, 0.0],)),
     (([0.0], [0.0]), ([0.0], [0.0])),
     (([0.1, 0.2], [0.3]), ([0.0], [0.0])),
     (([-0.1, 0.2], [0.3]), ([0.0, 0.0], [0.0])),
     (([0.1, 0.2], [0.3]), ([np.nan, 0.0], [0.0])),
     (([[0.1, 0.2]], [0.3]), ([0.0, 0.0], [0.0]))],
    ids=["scenario_count", "zero_wrong_time_count", "q_time_count", "negative_p", "nonfinite_q", "p_not_vector"],
)
def test_builder_validates_amplitudes_before_selection(p, q):
    def must_not_select(*args):
        raise AssertionError("invalid amplitudes must be rejected before source selection")

    with pytest.raises(ValueError):
        source.build_fixed_source_model(
            _small_prepared(), ((0, 1),), _source_inputs(p, q),
            r_upper_bound=2.0, x_upper_bound=1.7, selection_constraints=must_not_select,
        )


def test_raw_scenarios_reject_same_length_but_different_time_labels():
    index = pd.Index([10, 20], name="time")
    scenarios = [{
        "P_terminal": pd.DataFrame([[0.1], [0.2]], index=index, columns=[0]),
        "Q_terminal": pd.DataFrame([[0.01], [0.02]], index=index, columns=[0]),
        "drop_target": pd.DataFrame([[0.1], [0.2]], index=index, columns=[0]),
    }]
    amplitudes = PreparedSourceInputs(
        p=(np.array([0.05, 0.06]),), q=(np.array([-0.01, 0.02]),),
        time_indices=(pd.Index([10, 30], name="time"),), provenance=("external_test",),
    )

    def must_not_select(*args):
        raise AssertionError("different time labels must be rejected before source selection")

    with pytest.raises(ValueError):
        source.build_fixed_source_model(
            scenarios, ((0,),), amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
            selection_constraints=must_not_select,
        )


def test_response_helper_copies_amplitudes_and_keeps_signed_q_only_source():
    amplitudes = _source_inputs(([0.0, 0.0], [0.0]), ([-0.4, 0.2], [-0.3]))
    model = core._base_model(_small_prepared(), 1, r_upper_bound=2.0, x_upper_bound=1.7)
    blocks = source._add_source_variables(model, 1, r_upper_bound=2.0, x_upper_bound=1.7)

    def response_terms(current_model, output_index):
        assert current_model is model
        return {blocks.w_r.start: output_index + 1.0}, {blocks.w_x.start: 2.0}

    provider = source.make_source_response_terms(amplitudes, response_terms)
    assert provider is not None
    for block in (*amplitudes.p, *amplitudes.q):
        block[:] = 99.0
    for scenario_index, time_index, expected_q in ((0, 0, -0.8), (0, 1, 0.4), (1, 0, -0.6)):
        coefficients, constant = provider(model, scenario_index, time_index, 1)
        assert coefficients.get(blocks.w_r.start, 0.0) == 0.0
        assert coefficients[blocks.w_x.start] == pytest.approx(expected_q)
        assert constant == 0.0
    assert source.make_source_response_terms(None, response_terms) is None
    assert source.make_source_response_terms(
        _source_inputs(([0.0],), ([0.0],)), response_terms
    ) is None


def test_core_affine_source_adds_shared_weight_and_moves_signed_constants():
    prepared = core._PreparedScenarios(
        labels=(0,), p=(np.array([[1.0], [2.0], [3.0]]),),
        q=(np.zeros((3, 1)),), target=(np.array([[3.0], [5.0], [10.0]]),),
    )
    model = core._base_model(prepared, 1, r_upper_bound=10.0, x_upper_bound=10.0)

    def affine_terms(current_model, scenario_index, time_index, output_index):
        return {current_model.r.start: [1.0, 2.0, 3.0][time_index]}, [1.0, -1.0, 2.0][time_index]

    core._add_absolute_residual_constraints(
        prepared, ((0,),), model, source_terms=affine_terms,
    )
    result = _solve_model_lp(model)
    r_value = result.x[model.r.start]
    # Direct weighted-median solution: coefficients [2,4,6], shifted targets [2,6,8].
    assert r_value == pytest.approx(4.0 / 3.0)
    direct_prediction = np.array([2 * r_value + 1, 4 * r_value - 1, 6 * r_value + 2])
    direct_mae = np.mean(np.abs(direct_prediction - [3.0, 5.0, 10.0]))
    assert direct_mae == pytest.approx(4.0 / 9.0)
    assert result.fun == pytest.approx(direct_mae)
    assert model.rows.size == 6


def test_raw_scenario_reorders_source_by_time_labels_before_prediction():
    index = pd.Index([10, 20], name="time")
    scenarios = [{
        "P_terminal": pd.DataFrame([[0.0], [0.0]], index=index, columns=[0]),
        "Q_terminal": pd.DataFrame([[0.0], [0.0]], index=index, columns=[0]),
        "drop_target": pd.DataFrame([[6.0], [2.0]], index=index, columns=[0]),
    }]
    amplitudes = PreparedSourceInputs(
        p=(np.array([2.0, 6.0]),), q=(np.zeros(2),),
        time_indices=(pd.Index([20, 10], name="time"),), provenance=("external_test",),
    )

    def pin_known_path_and_weights(model, supports, blocks):
        for column, value in ((blocks.h.start, 1.0), (model.r.start, 1.0), (model.x.start, 0.0)):
            model.rows.add({column: 1.0}, lower=value, upper=value)

    model, blocks = source.build_fixed_source_model(
        scenarios, ((0,),), amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
        selection_constraints=pin_known_path_and_weights,
    )
    result = _solve_model_lp(model)
    assert blocks is not None
    assert result.fun == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("h_new", [0, 1], ids=["source_outside_new_clade", "source_inside_new_clade"])
def test_supplied_extension_matches_eliminated_product_lp_and_direct_prediction(h_new):
    supports = ((0, 1, 2), (0,), (1,), (2,))
    new_support = (0, 1)
    z = np.array([1, 1, 0])
    complete_supports = (*supports, new_support)
    complete_h = np.array([1, 1, 0, 0, 1] if h_new else [1, 0, 0, 1, 0])
    rng = np.random.default_rng(404)
    p = tuple(rng.uniform(0.05, 0.4, size=(count, 3)) for count in (5, 4))
    q = tuple(rng.uniform(-0.08, 0.12, size=block.shape) for block in p)
    amplitudes = _source_inputs(
        ([0.07, 0.13, 0.02, 0.19, 0.09], [0.15, 0.04, 0.11, 0.08]),
        ([-0.06, 0.04, -0.02, 0.07, -0.05], [0.03, -0.07, 0.05, -0.04]),
    )
    design = _direct_design(p, q, complete_supports, complete_h, amplitudes)
    weights = np.array([0.3, 0.2, 0.15, 0.25, 0.1, 0.12, 0.08, 0.2, 0.06, 0.14])
    observed = design @ weights + 0.008 * np.sin(1.7 * np.arange(len(design)) + 0.2)
    prepared = core._PreparedScenarios(
        labels=(0, 1, 2), p=p, q=q,
        target=(observed[:15].reshape(5, 3), observed[15:].reshape(4, 3)),
    )
    callback_calls = []

    def pin_known_candidate(model, actual_supports, blocks, relations):
        assert actual_supports == supports
        assert isinstance(relations, slice)
        callback_calls.append(blocks)
        assert isinstance(blocks, source.SourceBlocks)
        for block in (blocks.h, blocks.w_r, blocks.w_x):
            assert block.stop - block.start == len(complete_supports)
        for block, values in ((model.z, z), (blocks.h, complete_h)):
            for k, value in enumerate(values):
                model.rows.add({block.start + k: 1.0}, lower=value, upper=value)

    model, blocks = source.build_extension_source_model(
        prepared, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
        selection_constraints=pin_known_candidate,
    )
    assert blocks is not None
    assert len(callback_calls) == 1 and callback_calls[0] is blocks
    result = _solve_model_lp(model)
    oracle = _direct_l1_oracle(design, observed, len(complete_supports))
    assert result.fun == pytest.approx(oracle.fun, abs=1e-9)
    r_new, x_new = result.x[model.new_r][0], result.x[model.new_x][0]
    fitted_weights = np.r_[result.x[model.r], r_new, result.x[model.x], x_new]
    assert result.fun == pytest.approx(np.mean(np.abs(design @ fitted_weights - observed)), abs=1e-9)
    np.testing.assert_allclose(result.x[blocks.h], complete_h, atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.w_r], np.r_[result.x[model.r], r_new] * complete_h, atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.w_x], np.r_[result.x[model.x], x_new] * complete_h, atol=1e-9)
    # Each run has z_i=0 and z_i=1; the two runs cover all h_new/z_i combinations.
    np.testing.assert_allclose(result.x[blocks.v_r], r_new * h_new * z, atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.v_x], x_new * h_new * z, atol=1e-9)


@pytest.mark.parametrize("h_new", [0, 1], ids=["empty_path_rejected", "first_atom_source"])
def test_extension_with_no_old_atoms_allows_only_nonempty_new_source_path(h_new):
    prepared = _small_prepared()
    amplitudes = _source_inputs(([0.1, 0.2], [0.3]), ([-0.04, 0.02], [-0.05]))

    def pin_first_atom(model, supports, blocks, relations):
        assert supports == ()
        assert isinstance(blocks, source.SourceBlocks)
        for block in (blocks.h, blocks.w_r, blocks.w_x):
            assert block.stop - block.start == 1
        for column, value in ((model.z.start, 1), (model.z.start + 1, 0),
                              (blocks.h.start, h_new),
                              (model.new_r.start, 0.7), (model.new_x.start, 0.3)):
            model.rows.add({column: 1.0}, lower=value, upper=value)

    model, blocks = source.build_extension_source_model(
        prepared, (), amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
        selection_constraints=pin_first_atom,
    )
    assert blocks is not None
    result = _solve_model_lp(model, require_success=False)
    if h_new == 0:
        assert result.status == 2, result.message
        return
    assert result.success, result.message
    design = _direct_design(prepared.p, prepared.q, ((0,),), [1], amplitudes)
    observed = np.concatenate([block.ravel() for block in prepared.target])
    assert result.fun == pytest.approx(np.mean(np.abs(design @ [0.7, 0.3] - observed)), abs=1e-9)
    np.testing.assert_allclose(result.x[blocks.h], [1.0], atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.w_r], [0.7], atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.w_x], [0.3], atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.v_r], [0.7, 0.0], atol=1e-9)
    np.testing.assert_allclose(result.x[blocks.v_x], [0.3, 0.0], atol=1e-9)


@pytest.mark.parametrize("amplitudes", [None, "zero"], ids=["none", "zero"])
def test_disabled_extension_source_is_exact_core_model(amplitudes):
    prepared = _small_prepared()
    supports = ((0,), (1,))
    if amplitudes == "zero":
        amplitudes = _source_inputs(([0.0, 0.0], [0.0]), ([0.0, 0.0], [0.0]))

    def must_not_select(*args):
        raise AssertionError("disabled source must skip joint source selection")

    options = dict(r_upper_bound=2.0, x_upper_bound=1.7)
    baseline = core.build_extension_model(prepared, supports, **options)
    actual, blocks = source.build_extension_source_model(
        prepared, supports, amplitudes, **options, selection_constraints=must_not_select
    )
    assert blocks is None
    assert actual.variables.slices == baseline.variables.slices
    for name in ("objective", "lower", "upper", "integrality"):
        np.testing.assert_array_equal(getattr(actual.variables, name), getattr(baseline.variables, name))
    before = baseline.rows.build(baseline.variables.size)
    after = actual.rows.build(actual.variables.size)
    assert after.A.shape == before.A.shape
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(after.A, name), getattr(before.A, name))
    np.testing.assert_array_equal(after.lb, before.lb)
    np.testing.assert_array_equal(after.ub, before.ub)


@pytest.mark.parametrize(
    ("supports", "generating_support"),
    [(((0, 1), (2,)), (0,)), (((0, 1), (2,)), (0, 1, 2)),
     (((0, 1),), (2,)), ((), (0, 2))],
    ids=["new_subset", "new_superset", "new_disjoint", "no_existing_atoms"],
)
def test_default_joint_extension_matches_all_clades_and_source_nodes_enumerated_by_lp(supports, generating_support):
    n = 3
    rng = np.random.default_rng(183)
    p = tuple(rng.uniform(0.05, 0.4, size=(count, n)) for count in (5, 4))
    q = tuple(rng.uniform(-0.08, 0.12, size=block.shape) for block in p)
    amplitudes = _source_inputs(
        ([0.07, 0.13, 0.02, 0.19, 0.09], [0.15, 0.04, 0.11, 0.08]),
        ([-0.06, 0.04, -0.02, 0.07, -0.05], [0.03, -0.07, 0.05, -0.04]),
    )
    new_clades = tuple(_enumerate_new_clades(n, supports))
    assert generating_support in new_clades
    generating_family = (*supports, generating_support)
    generating_h = [int(set(generating_support) <= set(edge)) for edge in generating_family]
    atom_count = len(generating_family)
    generating_weights = np.r_[np.linspace(0.15, 0.35, atom_count), np.linspace(0.07, 0.17, atom_count)]
    design = _direct_design(p, q, generating_family, generating_h, amplitudes)
    observed = design @ generating_weights + 0.003 * np.cos(1.17 * np.arange(len(design)))
    prepared = core._PreparedScenarios(
        labels=tuple(range(n)), p=p, q=q,
        target=(observed[:15].reshape(5, n), observed[15:].reshape(4, n)),
    )
    # Each LP eliminates z, h, and every product using one complete candidate.
    oracle_candidates = []
    for new_clade in new_clades:
        family = (*supports, new_clade)
        for node in family:
            h = np.array([int(set(node) <= set(edge)) for edge in family])
            candidate_design = _direct_design(p, q, family, h, amplitudes)
            oracle = _direct_l1_oracle(candidate_design, observed, len(family))
            oracle_candidates.append((new_clade, h, oracle.fun))
    model, blocks = source.build_extension_source_model(
        prepared, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
    )
    assert blocks is not None
    baseline = core.build_extension_model(prepared, supports, r_upper_bound=2.0, x_upper_bound=1.7)
    assert sum(model.variables.integrality) == sum(baseline.variables.integrality) + atom_count
    result = _solve_model_milp(model)
    assert result.fun == pytest.approx(min(item[2] for item in oracle_candidates), abs=1e-8)
    chosen_z = np.rint(result.x[model.z]).astype(int)
    chosen_h = np.rint(result.x[blocks.h]).astype(int)
    np.testing.assert_allclose(result.x[model.z], chosen_z, atol=1e-7)
    np.testing.assert_allclose(result.x[blocks.h], chosen_h, atol=1e-7)
    chosen_clade = tuple(np.flatnonzero(chosen_z))
    assert any(clade == chosen_clade and np.array_equal(h, chosen_h)
               for clade, h, _ in oracle_candidates)
    family = (*supports, chosen_clade)
    r_values = np.r_[result.x[model.r], result.x[model.new_r]]
    x_values = np.r_[result.x[model.x], result.x[model.new_x]]
    prediction = _direct_design(p, q, family, chosen_h, amplitudes) @ np.r_[r_values, x_values]
    assert result.fun == pytest.approx(np.mean(np.abs(prediction - observed)), abs=1e-8)
    np.testing.assert_allclose(result.x[blocks.w_r], r_values * chosen_h, atol=1e-7)
    np.testing.assert_allclose(result.x[blocks.w_x], x_values * chosen_h, atol=1e-7)
    np.testing.assert_allclose(result.x[blocks.v_r], r_values[-1] * chosen_h[-1] * chosen_z, atol=1e-7)
    np.testing.assert_allclose(result.x[blocks.v_x], x_values[-1] * chosen_h[-1] * chosen_z, atol=1e-7)


@pytest.mark.parametrize(
    ("supports", "new_support", "h", "feasible"),
    [(((0, 1), (2,)), (0,), [0, 0, 1], False),
     (((0, 1), (2,)), (0,), [1, 0, 0], True),
     (((0, 1), (2,)), (0, 1, 2), [1, 0, 0], False),
     (((0, 1), (2,)), (0, 1, 2), [0, 0, 1], True),
     (((0, 1),), (2,), [1, 1], False),
     (((0, 1),), (2,), [1, 0], True)],
    ids=["subset_missing_parent", "subset_source_at_parent", "superset_missing_parent",
         "superset_source_at_new_parent", "disjoint_two_sources", "disjoint_single_branch"],
)
def test_default_joint_relation_constraints_accept_exactly_consistent_pinned_paths(supports, new_support, h, feasible):
    prepared = core._PreparedScenarios(
        labels=(0, 1, 2), p=(np.full((1, 3), 0.1),),
        q=(np.full((1, 3), 0.02),), target=(np.zeros((1, 3)),),
    )
    model, blocks = source.build_extension_source_model(
        prepared, supports, _source_inputs(([0.1],), ([-0.02],)),
        r_upper_bound=2.0, x_upper_bound=1.7,
    )
    assert blocks is not None
    for i in range(3):
        value = float(i in new_support)
        model.rows.add({model.z.start + i: 1.0}, lower=value, upper=value)
    for k, value in enumerate(h):
        model.rows.add({blocks.h.start + k: 1.0}, lower=value, upper=value)
    result = _solve_model_lp(model, require_success=False)
    assert result.status == (0 if feasible else 2), result.message


def test_default_joint_extension_is_infeasible_when_no_distinct_laminar_clade_remains():
    supports = ((0,), (1,), (0, 1))
    assert list(_enumerate_new_clades(2, supports)) == []
    model, blocks = source.build_extension_source_model(
        _small_prepared(), supports,
        _source_inputs(([0.1, 0.2], [0.3]), ([-0.04, 0.02], [-0.05])),
        r_upper_bound=2.0, x_upper_bound=1.7,
    )
    assert blocks is not None
    result = _solve_model_milp(model, require_success=False)
    assert result.status == 2, result.message


@pytest.mark.parametrize(
    ("supports", "new_support", "prefix_h"),
    [(((0, 1),), (1, 2), [1]), (((0, 1),), (0, 1), [1]),
     (((0, 1, 2), (0,)), (1,), [0, 1])],
    ids=["crossing_support", "duplicate_support", "old_path_missing_ancestor"],
)
def test_extension_keeps_ordinary_laminar_constraints_and_known_support_path_consistency(supports, new_support, prefix_h):
    prepared = core._PreparedScenarios(
        labels=(0, 1, 2), p=(np.full((1, 3), 0.1),),
        q=(np.full((1, 3), 0.02),), target=(np.zeros((1, 3)),),
    )

    def pin_invalid_candidate(model, actual_supports, blocks, relations):
        for i in range(3):
            value = float(i in new_support)
            model.rows.add({model.z.start + i: 1.0}, lower=value, upper=value)
        for k, value in enumerate([*prefix_h, 0]):
            model.rows.add({blocks.h.start + k: 1.0}, lower=value, upper=value)

    model, blocks = source.build_extension_source_model(
        prepared, supports, _source_inputs(([0.1],), ([-0.02],)),
        r_upper_bound=2.0, x_upper_bound=1.7, selection_constraints=pin_invalid_candidate,
    )
    assert blocks is not None
    result = _solve_model_lp(model, require_success=False)
    assert result.status == 2, result.message


def test_same_supports_refit_all_weights_and_choose_different_paths_from_new_observations():
    supports = ((0, 1), (0,), (1,))
    masks = [np.array([int(set(node) <= set(edge)) for edge in supports]) for node in supports]
    rng = np.random.default_rng(701)
    p = tuple(rng.uniform(0.05, 0.4, size=(count, 2)) for count in (5, 4))
    q = tuple(rng.uniform(-0.08, 0.12, size=block.shape) for block in p)
    amplitudes = _source_inputs(
        ([0.11, 0.27, 0.03, 0.19, 0.14], [0.21, 0.04, 0.16, 0.08]),
        ([-0.09, 0.04, -0.02, 0.07, -0.05], [0.03, -0.07, 0.05, -0.04]),
    )
    fitted_paths = []
    fitted_weights = []
    for node_index, weights in (
        (1, np.array([0.25, 0.2, 0.15, 0.12, 0.09, 0.17])),
        (2, np.array([0.32, 0.11, 0.26, 0.07, 0.15, 0.10])),
    ):
        design = _direct_design(p, q, supports, masks[node_index], amplitudes)
        assert np.linalg.matrix_rank(design) == len(weights)
        observed = design @ weights
        oracle_values = [
            _direct_l1_oracle(_direct_design(p, q, supports, h, amplitudes), observed, 3).fun
            for h in masks
        ]
        assert oracle_values[node_index] == pytest.approx(0.0, abs=1e-10)
        assert min(value for i, value in enumerate(oracle_values) if i != node_index) > 1e-4
        prepared = core._PreparedScenarios(
            labels=(0, 1), p=p, q=q,
            target=(observed[:10].reshape(5, 2), observed[10:].reshape(4, 2)),
        )
        model, blocks = source.build_fixed_source_model(
            prepared, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
        )
        assert blocks is not None
        # No observation-specific weight or previous-path values are supplied to the builder.
        for block, bound in ((model.r, 2.0), (model.x, 1.7)):
            np.testing.assert_array_equal(model.variables.lower[block], np.zeros(3))
            np.testing.assert_array_equal(model.variables.upper[block], np.full(3, bound))
        result = _solve_model_milp(model)
        assert result.fun == pytest.approx(min(oracle_values), abs=1e-8)
        np.testing.assert_allclose(result.x[blocks.h], masks[node_index], atol=1e-7)
        actual_weights = np.r_[result.x[model.r], result.x[model.x]]
        np.testing.assert_allclose(actual_weights, weights, atol=1e-7)
        fitted_paths.append(result.x[blocks.h])
        fitted_weights.append(actual_weights)
    assert not np.allclose(fitted_paths[0], fitted_paths[1])
    assert not np.allclose(fitted_weights[0], fitted_weights[1])


def test_pinned_topology_extension_reoptimizes_complete_path_like_accepted_fixed_refit():
    supports = ((0,), (1,))
    new_support = (0, 1)
    complete_supports = (*supports, new_support)
    masks = [np.array([int(set(node) <= set(edge)) for edge in complete_supports])
             for node in complete_supports]
    expected_h = np.array([0, 0, 1])
    rng = np.random.default_rng(913)
    p = tuple(rng.uniform(0.04, 0.4, size=(count, 2)) for count in (6, 5))
    q = tuple(rng.uniform(-0.08, 0.12, size=block.shape) for block in p)
    amplitudes = _source_inputs(
        ([0.11, 0.27, 0.03, 0.19, 0.14, 0.08], [0.21, 0.04, 0.16, 0.08, 0.23]),
        ([-0.09, 0.04, -0.02, 0.07, -0.05, 0.06], [0.03, -0.07, 0.05, -0.04, 0.08]),
    )
    design = _direct_design(p, q, complete_supports, expected_h, amplitudes)
    weights = np.array([0.2, 0.3, 0.25, 0.08, 0.12, 0.1])
    assert np.linalg.matrix_rank(design) == len(weights)
    observed = design @ weights
    prepared = core._PreparedScenarios(
        labels=(0, 1), p=p, q=q,
        target=(observed[:12].reshape(6, 2), observed[12:].reshape(5, 2)),
    )
    initial_model, initial_blocks = source.build_fixed_source_model(
        prepared, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
    )
    initial = _solve_model_milp(initial_model)
    assert initial_blocks is not None
    initial_h = initial.x[initial_blocks.h]
    assert any(np.allclose(initial_h, candidate, atol=1e-7) for candidate in ([1, 0], [0, 1]))
    assert initial.fun > 1e-4

    def pin_topology_and_enumerate_complete_node_domain(model, actual_supports, blocks, relations):
        assert actual_supports == supports
        for i in range(2):
            model.rows.add({model.z.start + i: 1.0}, lower=1.0, upper=1.0)
        # Test-only domain oracle for this FIXED z: choose any complete-clade node.
        # h is optimized jointly with every weight; no historical h or known answer is fixed.
        choice = model.variables.add("test_complete_node_choice", len(masks), upper=1.0, integral=True)
        model.rows.add({choice.start + j: 1.0 for j in range(len(masks))}, lower=1.0, upper=1.0)
        for k in range(len(complete_supports)):
            entries = {blocks.h.start + k: 1.0}
            entries.update({choice.start + j: -1.0 for j, mask in enumerate(masks) if mask[k]})
            model.rows.add(entries, lower=0.0, upper=0.0)

    extension_model, extension_blocks = source.build_extension_source_model(
        prepared, supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
        selection_constraints=pin_topology_and_enumerate_complete_node_domain,
    )
    accepted_model, accepted_blocks = source.build_fixed_source_model(
        prepared, complete_supports, amplitudes, r_upper_bound=2.0, x_upper_bound=1.7,
    )
    assert extension_blocks is not None and accepted_blocks is not None
    extension = _solve_model_milp(extension_model)
    accepted = _solve_model_milp(accepted_model)
    oracle_values = [
        _direct_l1_oracle(_direct_design(p, q, complete_supports, h, amplitudes), observed, 3).fun
        for h in masks
    ]
    assert oracle_values[2] == pytest.approx(0.0, abs=1e-10)
    assert min(oracle_values[:2]) > 1e-4
    for model, blocks, result in (
        (extension_model, extension_blocks, extension),
        (accepted_model, accepted_blocks, accepted),
    ):
        assert result.fun == pytest.approx(min(oracle_values), abs=1e-8)
        np.testing.assert_allclose(result.x[blocks.h], expected_h, atol=1e-7)
        np.testing.assert_allclose(result.x[blocks.w_r], weights[:3] * expected_h, atol=1e-7)
        np.testing.assert_allclose(result.x[blocks.w_x], weights[3:] * expected_h, atol=1e-7)
    extension_weights = np.r_[extension.x[extension_model.r], extension.x[extension_model.new_r],
                              extension.x[extension_model.x], extension.x[extension_model.new_x]]
    accepted_weights = np.r_[accepted.x[accepted_model.r], accepted.x[accepted_model.x]]
    np.testing.assert_allclose(extension_weights, weights, atol=1e-7)
    np.testing.assert_allclose(accepted_weights, weights, atol=1e-7)
    assert not np.allclose(initial_h, extension.x[extension_blocks.h][:len(supports)])
    # This compatibility test proves reoptimization for its supplied z only;
    # the exhaustive default-extension test above covers free-z joint optimization.
