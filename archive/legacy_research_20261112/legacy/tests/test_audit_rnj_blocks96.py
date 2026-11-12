"""Hand-built saved-evidence fixtures; no model fitting or solver calls."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import audit_rnj_blocks96 as subject


def failures(audit):
    return [row["check"] for row in audit.checks if not row["passed"]]


def selector_fixture():
    labels = list(range(11, 19))
    a, b, c = [11, 12], [13, 14, 15], [16, 17, 18]
    tree = {"root": 1, "edges": [[1, -i, 1] for i in range(1, 4)]
            + [[-i, value, 1] for i, group in enumerate([a, b, c], 1) for value in group]}
    trace = []
    for index in range(100):
        trace.append({"identifiable": True, "peers": ([a] if index < 75 else [])
                      + ([b, c] if index < 90 else []), "clades": [],
                      "fine_clades": ([[13, 14]] if index < 15 else [])
                      + ([[16, 17]] if index < 16 else [])})
    screen = {"parameters": dict(subject.SCREEN_POLICY), "full_clades": [a, b, c],
              "confidence_records": [{"clade": a, "frequency": .75}, {"clade": b, "frequency": .9},
                                     {"clade": c, "frequency": .9}],
              "selected_support_labels": [b, a], "selected_support_indices": [[2, 3, 4], [0, 1]],
              "diagnostics": {"full_peers": [a, b, c], "full_tree": tree, "full_fine_clades": [],
                  "full_fine_subclades": [], "source_adjusted_identifiable": True,
                  "valid_bootstrap_replicates": 100, "unidentifiable_bootstrap_replicates": 0,
                  "subclade_frequency_records": [{"clade": a, "frequency": 0}, {"clade": b, "frequency": .15},
                                                 {"clade": c, "frequency": .16}],
                  "eligible": [a, b], "bootstrap_records": trace}}
    return labels, screen


def test_selector_recounts_trace_accepts_size2_and_subfrequency_equality():
    labels, screen = selector_fixture()
    audit = subject.Audit()
    result = subject.audit_selector(screen, labels, audit)
    assert not failures(audit), failures(audit)
    assert result["selected_labels"] == [[13, 14, 15], [11, 12]]
    assert not result["unavailable_checks"] and result["bootstrap_trace_recounted"]
    # A summary with a wrong repeated frequency is not a raw recount.
    screen["confidence_records"][0]["frequency"] = .76
    broken = subject.Audit()
    subject.audit_selector(screen, labels, broken)
    assert any("independently recounted" in name for name in failures(broken))


def test_selector_fine_veto_and_nonidentifiable_denominator_not_diluted():
    labels, screen = selector_fixture()
    diag = screen["diagnostics"]
    diag["full_fine_clades"] = [[13, 14]]
    diag["full_fine_subclades"] = [[13, 14, 15]]
    diag["eligible"] = [[11, 12]]
    screen["selected_support_labels"], screen["selected_support_indices"] = [[11, 12]], [[0, 1]]
    audit = subject.Audit()
    subject.audit_selector(screen, labels, audit)
    assert not failures(audit), failures(audit)
    # Peer numerator counts only valid trees, denominator remains all100.
    for record in diag["bootstrap_records"][80:]:
        record["identifiable"] = False
    diag.update(valid_bootstrap_replicates=80, unidentifiable_bootstrap_replicates=20)
    screen["confidence_records"][1]["frequency"] = .8
    screen["confidence_records"][2]["frequency"] = .8
    diag["subclade_frequency_records"][1]["frequency"] = 15/80
    diag["subclade_frequency_records"][2]["frequency"] = 16/80
    audit = subject.Audit()
    subject.audit_selector(screen, labels, audit)
    assert not failures(audit), failures(audit)
    diag["subclade_frequency_records"][1]["frequency"] = .15
    broken = subject.Audit()
    subject.audit_selector(screen, labels, broken)
    assert any("refinement frequencies independently recounted" in name for name in failures(broken))


def test_peer_definition_detects_unary_internal_child_hidden_by_clades():
    tree = {"root": 1, "edges": [[1, -1, 1], [-1, 11, 1], [-1, -2, 1], [-2, 12, 1], [1, 13, 1]]}
    assert subject.terminal_peer_groups(tree, [11, 12, 13]) == set()
    labels, screen = selector_fixture()
    del screen["diagnostics"]["full_tree"]
    del screen["diagnostics"]["full_fine_clades"]
    del screen["diagnostics"]["bootstrap_records"]
    audit = subject.Audit()
    result = subject.audit_selector(screen, labels, audit)
    assert not failures(audit)
    assert len(result["unavailable_checks"]) == 3 and not result["bootstrap_trace_recounted"]


@pytest.mark.parametrize("compressed,expected_size", [(False, 5), (True, 4)])
def test_initial_family_is_unchanged_when_membership_compression_toggled(compressed, expected_size):
    labels, selected = [101, 103, 102, 104, 105], [[101, 103]]
    model = {"initial_supports": [[i] for i in range(5)]+[[0, 1], [0, 1, 2]],
             "ordinary_initial_supports": [[0, 1, 2]],
             "requested_terminal_groups": [[0, 1]] if compressed else [],
             "terminal_groups": [[0, 1], [2], [3], [4]] if compressed else [[i] for i in range(5)]}
    audit = subject.Audit()
    result = subject.audit_initialization(model, selected, labels, audit, compressed=compressed)
    assert not failures(audit), failures(audit)
    assert result["partition_size"] == expected_size and result["ordinary_nontrivial_count"] == 1
    model["terminal_groups"] = [[0, 1], [2], [3]]
    with pytest.raises(ValueError): subject.audit_initialization(model, selected, labels, subject.Audit(), compressed=compressed)


def raw_fixture(*, source_active=True, grouped=True, extension=True):
    n, supports = 4, [(0,), (1,), (2,), (3,), (0, 1)]
    groups = [(0, 1), (2,), (3,)] if grouped else [(i,) for i in range(n)]
    k, m, slices, cursor = len(supports), len(groups), {}, 0
    def allocate(name, size):
        nonlocal cursor
        slices[name] = [cursor, cursor+size]
        cursor += size
    allocate("old_r" if extension else "r", k)
    allocate("old_x" if extension else "x", k)
    allocate("absolute_residual", 96*n)
    rows = []
    if extension:
        allocate("new_r", 1); allocate("new_x", 1); allocate("z", m)
        for name in ("y", "u_r", "u_x"): allocate(name, m*(m+1)//2)
        allocate("relations", 3*k)
        start = slices["z"][0]
        for support in supports:
            rows.append(({start+i: len(group)-2*len(set(group) & set(support)) for i, group in enumerate(groups)}, 1-len(support), np.inf))
        rows.append(({start+i: 1 for i in range(m)}, 2, np.inf))
        rows.append(({start+i: len(group) for i, group in enumerate(groups)}, 2, np.inf))
    if source_active:
        for name in ("source_h", "source_w_r", "source_w_x"): allocate(name, k+int(extension))
        if extension:
            for name in ("source_new_v_r", "source_new_v_x"): allocate(name, m)
        h = slices["source_h"][0]
        rows.append(({h+i: 1 for i in range(k+int(extension))}, 1, np.inf))
        for i, a in enumerate(map(set, supports)):
            for j in range(i+1, k):
                b = set(supports[j])
                if a < b: rows.append(({h+i: 1, h+j: -1}, -np.inf, 0))
                elif b < a: rows.append(({h+j: 1, h+i: -1}, -np.inf, 0))
                else: rows.append(({h+i: 1, h+j: 1}, -np.inf, 1))
        def product(weight, indicator, result):
            rows.extend([({result: 1, indicator: -2}, -np.inf, 0),
                         ({result: 1, weight: -1}, -np.inf, 0),
                         ({result: 1, weight: -1, indicator: -2}, -2, np.inf)])
        for name in ("r", "x"):
            start = slices[("old_" if extension else "")+name][0]
            weights = list(range(start, start+k))+([slices["new_"+name][0]] if extension else [])
            for atom, weight in enumerate(weights): product(weight, h+atom, slices["source_w_"+name][0]+atom)
            if extension:
                for group in range(m): product(slices["source_w_"+name][1]-1, slices["z"][0]+group, slices["source_new_v_"+name][0]+group)
        if extension:
            for atom in range(k):
                relation = slices["relations"][0]+3*atom
                rows.extend([({h+k: 1, h+atom: -1, relation: 1}, -np.inf, 1),
                             ({h+atom: 1, h+k: -1, relation+1: 1}, -np.inf, 1),
                             ({h+k: 1, h+atom: 1, relation+2: 1}, -np.inf, 2)])
    # Feasible incumbent: full terminal support is different from every old atom.
    primal, integer, upper = np.zeros(cursor), np.zeros(cursor, dtype=int), np.ones(cursor)*2
    for name in ("z", "source_h"):
        if name in slices:
            lo, hi = slices[name]
            integer[lo:hi], upper[lo:hi] = 1, 1
    if extension:
        lo, hi = slices["z"]; primal[lo:hi] = 1
        relation = slices["relations"][0]
        primal[relation+1:relation+3*k:3] = 1
    if source_active:
        h = slices["source_h"][0]
        if extension: primal[h+k] = 1
        else: primal[h], primal[h+4] = 1, 1
    ri, ci, values = [], [], []
    for i, (entries, _, _) in enumerate(rows):
        for j, value in entries.items():
            if value: ri.append(i); ci.append(j); values.append(value)
    saved = {"primal": primal, "objective_coefficients": np.zeros(cursor), "lower_bounds": np.zeros(cursor),
             "upper_bounds": upper, "integrality": integer, "constraint_row_index": np.array(ri, dtype=int),
             "constraint_col_index": np.array(ci, dtype=int), "constraint_data": np.array(values, dtype=float),
             "constraint_lower": np.array([r[1] for r in rows], dtype=float),
             "constraint_upper": np.array([r[2] for r in rows], dtype=float)}
    event = {"variable_slices": slices, "variable_count": cursor, "objective": 0., "dual_bound": 0.,
             "status": 0, "success": True, "binary_variable_count": int(integer.sum()), "mip_gap": 0.}
    return n, groups, supports, event, saved


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("source_active", [False, True])
def test_raw_original_terminal_hamming_cardinality_and_full_source_dimensions(grouped, source_active):
    n, groups, supports, event, saved = raw_fixture(grouped=grouped, source_active=source_active)
    audit = subject.Audit()
    subject.audit_membership_domain(event, saved, n, groups, supports, audit, source_active=source_active)
    assert not failures(audit), failures(audit)
    if grouped:
        # Corrupt only the original-terminal cardinality row; the valid core
        # group-count row stays present, so conflating the two misses this bug.
        saved["constraint_lower"][len(supports)+1] = 1
        broken = subject.Audit()
        subject.audit_membership_domain(event, saved, n, groups, supports, broken, source_active=source_active)
        assert any("cardinality uses ORIGINAL" in name for name in failures(broken))
    if source_active:
        event["variable_slices"]["source_h"][1] -= 1
        broken = subject.Audit()
        subject.audit_membership_domain(event, saved, n, groups, supports, broken, source_active=True)
        assert any("source_h covers full" in name for name in failures(broken))


def test_fixed_support_pair_audits_domain_not_arbitrary_nonunique_coefficients():
    _, _, _, left_event, left = raw_fixture(extension=False)
    right_event, right = deepcopy(left_event), deepcopy(left)
    right["primal"][1] = 1  # h[1]=0: a different zero-cost optimum is valid
    audit = subject.Audit()
    subject.audit_fixed_support_pair(left_event, left, right_event, right, audit)
    assert not failures(audit), failures(audit)
    right["upper_bounds"][0] = 3
    broken = subject.Audit()
    subject.audit_fixed_support_pair(left_event, left, right_event, right, broken)
    assert any("upper_bounds identical" in name for name in failures(broken))


def test_source_nonempty_and_product_rows_are_audited_even_for_feasible_incumbent():
    n, groups, supports, event, saved = raw_fixture()
    h = event["variable_slices"]["source_h"][0]
    row = subject._coo_rows(saved)
    nonempty = next(i for i, entries in enumerate(row) if entries == {h+j: 1. for j in range(len(supports)+1)})
    saved["constraint_lower"][nonempty] = 0
    broken = subject.Audit()
    subject.audit_membership_domain(event, saved, n, groups, supports, broken, source_active=True)
    assert any("nonempty constraint uses the complete" in name for name in failures(broken))
    # Its saved incumbent remains feasible; a primal feasibility check alone
    # would not detect the missing non-root-source domain restriction.
    assert not any("raw primal satisfies original linear constraints" in name for name in failures(broken))
    product = event["variable_slices"]["source_w_r"][0]
    target = next(i for i, entries in enumerate(row) if entries == {product: 1., h: -2.})
    saved["constraint_upper"][target] = 1
    broken = subject.Audit()
    subject.audit_source_linear_domain(event, saved, supports, broken)
    assert any("weight-times-h[0] exact original rows" in name for name in failures(broken))


def test_raw_residual_source_amplitudes_are_once_per_output_not_group_size():
    # Hand-written two-output fixed-family equations. The common ancestor
    # gets source .7/.8 for EACH output, never 2*.7 or 2*.8.
    slices = {"r": [0, 3], "x": [3, 6], "absolute_residual": [6, 8],
              "source_w_r": [8, 11], "source_w_x": [11, 14], "source_h": [14, 17]}
    predictions = [{0: 1., 2: 3., 3: .1, 5: .3, 8: .7, 10: .7, 11: .8, 13: .8},
                   {1: 2., 2: 3., 4: .2, 5: .3, 9: .7, 10: .7, 12: .8, 13: .8}]
    ri, ci, data, lower, upper = [], [], [], [], []
    for output, prediction in enumerate(predictions):
        for sign in (1., -1.):
            row = len(lower)
            for column, coefficient in (prediction | {6+output: sign}).items():
                ri.append(row); ci.append(column); data.append(coefficient)
            lower.append(.3+.1*output if sign == 1. else -np.inf)
            upper.append(np.inf if sign == 1. else .3+.1*output)
    saved = {"constraint_row_index": np.array(ri), "constraint_col_index": np.array(ci), "constraint_data": np.array(data),
             "constraint_lower": np.array(lower), "constraint_upper": np.array(upper),
             "objective_coefficients": np.array([0.]*6+[.5, .5]+[0.]*9)}
    arrays = {"P": np.array([[1., 2.]]), "Q": np.array([[.1, .2]]), "Y": np.array([[.3, .4]]),
              "P_source": np.array([.7]), "Q_source": np.array([.8])}
    audit = subject.Audit()
    subject.audit_residual_rows({"variable_slices": slices}, saved, arrays, [[0], [1], [0, 1]], [[0], [1]], audit)
    assert not failures(audit), failures(audit)
    saved["constraint_data"][np.array(ci) == 10] *= 2
    broken = subject.Audit()
    subject.audit_residual_rows({"variable_slices": slices}, saved, arrays, [[0], [1], [0, 1]], [[0], [1]], broken)
    assert any("exactly once" in name for name in failures(broken))


def test_coo_support_index_aggregates_duplicates_and_clears_exact_zeros_first():
    saved = {"constraint_row_index": np.array([0, 0, 0, 0, 1, 2, 2]),
             "constraint_col_index": np.array([2, 2, 2, 4, 4, 3, 3]),
             "constraint_data": np.array([1., 2., -3., 2., 2., 1., -1.]),
             "constraint_lower": np.array([0., 1., 0.]), "constraint_upper": np.array([np.inf, np.inf, 0.])}
    rows = subject._coo_rows(saved)
    assert rows == [{4: 2.}, {4: 2.}, {}]
    assert rows.by_columns == {(4,): [0, 1], (): [2]}
    assert subject._has_row(rows, saved, {2: 0., 4: 2.}, lower=0.)
    assert subject._has_row(rows, saved, {}, lower=0., upper=0.)
    assert not subject._has_row(rows, saved, {4: 2.}, lower=2.)
    # Keep exactly the original numerical comparison tolerances.
    assert subject._has_row(rows, saved, {4: 2.+1e-10}, lower=1.)
    assert not subject._has_row(rows, saved, {4: 2.+1e-7}, lower=1.)


def test_indexed_original_floor_rows_and_extra_source_activation_tie():
    entries = [({2: 1., 6: -2.}, -np.inf, 0.), ({5: 1., 6: -3.}, -np.inf, 0.),
               ({2: 1., 5: 1., 6: -.2}, 0., np.inf), ({7: 1., 8: 1.}, .2, np.inf),
               ({9: 1., 10: 1.}, 2., np.inf)]
    row, col, data = [], [], []
    for i, (terms, _, _) in enumerate(entries):
        for j, value in terms.items(): row.append(i); col.append(j); data.append(value)
    saved = {"constraint_row_index": np.array(row), "constraint_col_index": np.array(col), "constraint_data": np.array(data),
             "constraint_lower": np.array([v[1] for v in entries]), "constraint_upper": np.array([v[2] for v in entries]),
             "lower_bounds": np.zeros(11), "upper_bounds": np.array([2.]*3+[3.]*3+[1., 2., 3., 1., 1.]),
             "integrality": np.array([0]*6+[1, 0, 0, 1, 1]), "primal": np.array([0.]*7+[.2, 0., 1., 1.])}
    event = {"variable_slices": {"old_r": [0, 3], "old_x": [3, 6], "edge_active": [6, 7],
                                 "new_r": [7, 8], "new_x": [8, 9], "z": [9, 11]}}
    audit = subject.Audit()
    subject.audit_floor_domain(event, saved, 2, [[0], [1]], {"a": 1., "b": 1., "tau": .2}, audit, "floor")
    assert not failures(audit), failures(audit)
    i = len(entries)
    saved["constraint_row_index"] = np.r_[saved["constraint_row_index"], i, i]
    saved["constraint_col_index"] = np.r_[saved["constraint_col_index"], 6, 0]
    saved["constraint_data"] = np.r_[saved["constraint_data"], 1., -1.]
    saved["constraint_lower"] = np.r_[saved["constraint_lower"], 0.]
    saved["constraint_upper"] = np.r_[saved["constraint_upper"], np.inf]
    broken = subject.Audit()
    subject.audit_floor_domain(event, saved, 2, [[0], [1]], {"a": 1., "b": 1., "tau": .2}, broken, "floor")
    assert any("activation does not constrain Source path" in name for name in failures(broken))


def write_inputs(directory):
    observed, source = directory/"observed", directory/"source"
    observed.mkdir(); source.mkdir()
    index, columns = pd.Index(range(96), name="t"), [11, 12]
    p, q = np.arange(192).reshape(96, 2)/1000, np.full((96, 2), .02)
    root, v = np.ones(96), np.ones((96, 2))*.99
    y = root[:, None]**2-v**2
    for name, data in (("P_terminal", p), ("Q_terminal", q), ("V_terminal", v), ("drop_target", y)):
        pd.DataFrame(data, index=index, columns=columns).to_csv(observed/(name+".csv"), float_format="%.17g")
    pd.DataFrame({"V_root_measured": root}, index=index).to_csv(observed/"root_voltage.csv", float_format="%.17g")
    u = np.sin(np.arange(96)/11)
    pd.DataFrame({"P_pu": u, "Q_pu": u/2, "P_kw": u*1000, "Q_kvar": u*500}, index=index).to_csv(source/"amplitudes.csv", float_format="%.17g")
    return {"P": p, "Q": q, "Y": y, "P_source": u, "Q_source": u/2,
            "row_scenario": np.zeros(96, dtype=int), "row_position": np.arange(96)}


def test_exact_96row_source_alignment_and_no_validation(tmp_path):
    arrays = write_inputs(tmp_path)
    audit = subject.Audit()
    subject.audit_training96(arrays, tmp_path, [11, 12], audit, data_use={"training_samples": 96, "validation_samples": 0})
    assert not failures(audit), failures(audit)
    frame = pd.read_csv(tmp_path/"source/amplitudes.csv", index_col=0)
    frame.iloc[::-1].to_csv(tmp_path/"source/amplitudes.csv")
    arrays["P_validation"] = np.ones((1, 2))
    broken = subject.Audit()
    subject.audit_training96(arrays, tmp_path, [11, 12], broken)
    assert any("Source time order" in name for name in failures(broken))
    assert any("no validation arrays" in name for name in failures(broken))


def test_fixed_denominator_keeps_failed_incomplete_pending_and_does_not_zero_metrics():
    jobs = [{"job_id": str(i), "case": "different_network"} for i in range(4)]
    results = {"0": {"status": "complete", "metrics": {"clade_f1": 1}},
               "1": {"status": "incomplete", "metrics": {"clade_f1": .5}},
               "2": {"status": "failed", "stop_reason": "no_model"}}
    report = subject.planned_outcomes(jobs, results)
    assert report["planned_denominator"] == 4 and report["status_counts"] == {"complete": 1, "incomplete": 1, "failed": 1, "pending": 1}
    assert report["jobs"][1]["metrics"]["clade_f1"] == .5
    assert report["jobs"][2]["metrics"] is None and report["jobs"][3]["metrics"] is None
    with pytest.raises(ValueError): subject.planned_outcomes(jobs, results | {"unplanned": {}})


def test_calibration_is_observed_only_and_dimension_independent():
    for n in (2, 7, 18):
        value = subject.independent_calibration(np.eye(n)*2, np.eye(n)*4)
        assert value["a"] == .1875 and value["b"] == .03125 and value["tau"] == .08


@pytest.mark.parametrize("source_rank", [0, 1, 2])
def test_bootstrap_circular_rows_and_synchronized_source_rank_reconstructed(source_rank):
    rng = np.random.default_rng(197)
    arrays = {"P": rng.normal(size=(96, 3)), "Q": rng.normal(size=(96, 3)), "Y": rng.normal(size=(96, 3))}
    p, q = np.sin(np.arange(96)/5), np.cos(np.arange(96)/7)
    if source_rank == 0: p, q = np.zeros(96), np.zeros(96)
    if source_rank == 1: q = p*2
    arrays.update(P_source=p, Q_source=q)
    trace, rng = [], np.random.default_rng(20260903)
    for number in range(4):
        starts = rng.integers(0, 96, size=24)
        indices = np.concatenate([(start+np.arange(4)) % 96 for start in starts])
        ranks = subject.source_adjusted_ranks(arrays["P"][indices], arrays["Q"][indices], p[indices], q[indices])
        trace.append({"replicate": number, "resample_indices": [indices.tolist()],
                      "identifiable": ranks["source_adjusted_identifiable"],
                      "source_design_rank": ranks["source_design_rank"], "projected_design_rank": ranks["projected_design_rank"]})
    screen = {"parameters": {"seed": 20260903, "block_length": 4},
              "diagnostics": {"bootstrap_records": trace, "source_adjusted_identifiable": True}}
    audit = subject.Audit()
    result = subject.audit_resampling_trace(screen, arrays, audit)
    assert not failures(audit), failures(audit)
    assert result["replicates_rebuilt"] == 4 and result["full_data_ranks"]["source_design_rank"] == source_rank
    trace[0]["resample_indices"][0][0] = (trace[0]["resample_indices"][0][0]+1) % 96
    broken = subject.Audit()
    subject.audit_resampling_trace(screen, arrays, broken)
    assert any("original circular moving-block indices" in name for name in failures(broken))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_physical_metrics_use_full_truth_and_accept_null_baseline_supports(tmp_path):
    network = tmp_path/"network"
    write_json(network/"metadata.json", {"root_bus": 1, "base_kv": .4, "base_mva": .2})
    (network/"buses.csv").write_text("bus_id\n1\n2\n11\n12\n13\n", encoding="utf-8")
    (network/"branches.csv").write_text("from_bus,to_bus,r_ohm,x_ohm,is_true_closed\n1,2,.04,.02,true\n2,11,.008,.004,true\n2,12,.008,.004,true\n1,13,.008,.004,true\n", encoding="utf-8")
    labels = [11, 12, 13]
    truth = subject.physical_truth(network, labels)
    baseline = {"terminal_labels": labels, "support_indices": None,
                "tree": {"root": 1, "edges": [[1, -1, .15], [-1, 11, .03], [-1, 12, .03], [1, 13, .03]]}}
    arrays = {"R": truth["R"], "X": truth["X"]}
    metrics = subject.physical_metrics(baseline, arrays, network, calibration={"a": 1., "b": 1., "tau": .2})
    assert metrics["family_clade_f1"] == 1 and metrics["resolution_clade_f1"] == 0
    assert metrics["r_relative_error"] == 0 and metrics["x_relative_error"] == 0
    assert metrics["resolution_true_clades"] == 1  # no contraction of truth


def test_experiment_reader_retains_dynamic_catalog_and_never_reads_pending_model(tmp_path):
    runtime = tmp_path/"runtime"
    for directory in (runtime/"rnj_wzzt", runtime/"source_snapshot/rnj_wzzt"):
        directory.mkdir(parents=True)
        (directory/"__init__.py").write_text("# inert saved runtime\n", encoding="utf-8")
    checksum = subject.evidence.file_hash(runtime/"rnj_wzzt/__init__.py")
    write_json(runtime/"runtime_manifest.json", {"runtime_sha256": {"__init__.py": checksum}, "source_sha256": {"__init__.py": checksum}})
    inputs = tmp_path/"inputs/toy/no_source"
    inputs.mkdir(parents=True)
    write_inputs(inputs)
    for key in ("P0_measured", "Q0_measured"):
        pd.DataFrame({"value": np.zeros(96)}).to_csv(inputs/f"observed/{key}.csv")
    write_json(inputs/"observed/metadata.json", {})
    write_json(inputs/"source/metadata.json", {"location_unknown": True})
    inventory = {relative: subject.evidence.file_hash(inputs/relative) for relative in subject.evidence.INPUT_NAMES}
    write_json(tmp_path/"dataset_manifest.json", {"file_sha256": {"toy/no_source/"+r: h for r, h in inventory.items()}})
    jobs = [{"job_id": f"toy__no_source__{method}", "case": "toy", "variant": "no_source", "method": method,
             "per_solve_seconds": 6000., "method_wall_seconds": 30000.} for method in ("WZZT", "RNJ+WZZT")]
    protocol = {"jobs": jobs, "expected_jobs": 2, "training_samples": 96, "validation_samples": 0,
                "validation_supplied": False, "scenario_count": 1, "samples_per_scenario": 96,
                "truth_used_for_selection": False, "resolution_absolute_tolerance": 1e-10,
                "screen_parameters": subject.SCREEN_POLICY | {"seed": 20260903},
                "measurement_settings": {"v_noise_rel": .0002, "root_meter_noise_rel": .0002},
                "source_sha256": {}, "runtime_directory": str(runtime),
                "runtime_manifest_sha256": subject.evidence.file_hash(runtime/"runtime_manifest.json"),
                "dataset_manifest_sha256": subject.evidence.file_hash(tmp_path/"dataset_manifest.json")}
    write_json(tmp_path/"protocol.json", protocol)
    signature = subject.evidence.file_hash(tmp_path/"protocol.json")
    pending = tmp_path/"jobs"/jobs[0]["job_id"]
    write_json(pending/"execution.json", {"pid": 123})
    (pending/"model.json").write_text("INVALID RUNNING MODEL", encoding="utf-8")
    failed = tmp_path/"jobs"/jobs[1]["job_id"]
    write_json(failed/"inputs_hash.json", {"file_sha256": inventory})
    write_json(failed/"result.json", {"status": "failed", "has_model": False, "metrics": {"clade_f1": None},
               "elapsed_seconds": .2, "protocol_sha256": signature, "stop_reason": "pre_fit_failure"})
    report = subject.audit_experiment(tmp_path)
    assert report["passed"], [row for row in report["checks"] if not row["passed"]]
    assert report["expected_new_jobs"] == 2 and report["audited_final_jobs"] == 1 and not report["run_complete"]
    assert report["status_counts"] == {"running": 1, "failed": 1}
    assert report["rows"][0]["metrics_recomputed"] is None
