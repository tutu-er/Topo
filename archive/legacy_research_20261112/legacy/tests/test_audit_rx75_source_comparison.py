"""Synthetic audits use hand-written trees, atoms and original COO rows only."""
import json

import numpy as np
import pytest

from research_experiments.rnj import audit_rx75_source_comparison as subject


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def failures(audit):
    return [row["check"] for row in audit.checks if not row["passed"]]


def test_calibration_analytic_and_no_truth_input():
    # Distances are all 4 in R, all 8 in X; each normalized depth is .5.
    r, x = np.eye(3)*2, np.eye(3)*4
    actual = subject.independent_calibration(r, x)
    assert actual == {"mu_r": 4., "mu_x": 8., "a": .1875, "b": .03125,
                      "median_root_depth": .5, "factor": .16, "tau": .08}
    # Matrix input has no file or topology-truth argument. An unrelated truth
    # change therefore cannot affect its fixed, observed-only calibration.
    truth = {"clades": [[1, 2]], "F1": 0}
    before = subject.independent_calibration(r, x)
    truth.update(clades=[[1, 3]], F1=1)
    assert subject.independent_calibration(r, x) == before


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.array([[np.nan]]), np.empty((0, 0))])
def test_invalid_calibration_rejected(bad):
    with pytest.raises(ValueError): subject.independent_calibration(bad, bad)


def test_resolution_keeps_singletons_rootstem_equality_and_absolute_tolerance():
    supports = [[0], [1], [2], [0, 1], [0, 1, 2]]
    r = [0, 0, 0, .1-5e-11, .1-2e-10]
    strengths, kept = subject.resolution_atoms(supports, r, [0]*5, {"a": 1, "b": 1, "tau": .1})
    assert kept == [True, True, True, True, False]
    assert strengths[-1] < .1
    with pytest.raises(ValueError): subject.resolution_atoms(supports, [1], [1], {"a": 1, "b": 1, "tau": .1})


def test_independent_tree_traversal_contracts_hidden_edge_and_rootstem_only():
    tree = {"root": 1, "edges": [[1, -1, .01], [-1, -2, .1], [-2, 11, .001],
                                [-2, 12, .001], [-1, -3, .05], [-3, 13, .001], [-3, 14, .001]]}
    result = subject.independent_tree_resolution(tree, [11, 12, 13, 14], .1)
    assert result["family"] == {frozenset([11, 12]), frozenset([13, 14])}
    assert result["resolution"] == {frozenset([11, 12])}
    assert {frozenset([a, b]) for a, b, _ in result["contracted_edges"]} == {frozenset([1, -1]), frozenset([-1, -3])}
    assert sum(11 in [a, b] or 12 in [a, b] or 13 in [a, b] or 14 in [a, b] for a, b, _ in result["edges"]) == 4
    with pytest.raises(ValueError): subject.independent_tree_resolution({"root": 1, "edges": [[1, 2, 1], [2, 1, 1]]}, [2], .1)


def physical_network(path):
    write_json(path/"metadata.json", {"root_bus": 1, "base_kv": .4, "base_mva": .2})
    (path/"buses.csv").write_text("bus_id\n1\n2\n11\n12\n13\n", encoding="utf-8")
    (path/"branches.csv").write_text("from_bus,to_bus,r_ohm,x_ohm,is_true_closed\n1,2,.04,.02,true\n2,11,.008,.004,true\n2,12,.008,.004,true\n1,13,.008,.004,true\n", encoding="utf-8")


def test_true_scale_diagnostic_does_not_contract_scoring_truth(tmp_path):
    physical_network(tmp_path)
    report = subject.true_edge_scales(tmp_path, [11, 12, 13], {"a": 1, "b": 1, "tau": .2})
    assert report["min_true_strength"] == pytest.approx(.15)
    assert report["below_tau_count"] == 1 and not report["assumption_satisfied"]
    truth = subject.physical_truth(tmp_path, [11, 12, 13])
    assert truth["clades"] == {frozenset([11, 12])}
    assert subject.clade_metrics(set(), truth["clades"], [11, 12, 13])["clade_f1"] == 0
    assert report["edges"][0]["r_coefficient"] == pytest.approx(.1)


def coo_fixture(*, grouped=False):
    # This is a manually constructed original domain, not an adapter model.
    n, k = (4, 4) if grouped else (2, 3)
    partition = [[0, 1, 2], [3]] if grouped else [[0], [1]]
    r0, x0, cursor = 0, k, 2*k
    slices = {"old_r": [0, k], "old_x": [k, 2*k]}
    rows = []
    if not grouped:
        e = cursor; cursor += 1; slices["edge_active"] = [e, e+1]
        rows.extend([({2: 1., e: -2.}, -np.inf, 0.), ({k+2: 1., e: -3.}, -np.inf, 0.),
                     ({2: 1., k+2: 1., e: -.2}, 0., np.inf)])
    nr, nx = cursor, cursor+1; cursor += 2
    slices.update(new_r=[nr, nr+1], new_x=[nx, nx+1], z=[cursor, cursor+2])
    rows.extend([({nr: 1., nx: 1.}, .2, np.inf),
                 ({cursor: float(len(partition[0])), cursor+1: 1.}, 2., np.inf)])
    size = cursor+2
    lower, upper = np.zeros(size), np.full(size, 3.)
    upper[:k] = 2.
    primal, integer = np.zeros(size), np.zeros(size, dtype=int)
    primal[nr] = .2
    primal[cursor:] = [1, 0] if grouped else [1, 1]
    upper[cursor:] = 1; integer[cursor:] = 1
    if not grouped: upper[e] = 1; integer[e] = 1
    ri, ci, data = [], [], []
    for index, (entries, _, _) in enumerate(rows):
        for column, coefficient in entries.items(): ri.append(index); ci.append(column); data.append(coefficient)
    saved = {"primal": primal, "objective_coefficients": np.zeros(size), "lower_bounds": lower, "upper_bounds": upper,
             "integrality": integer, "constraint_row_index": np.array(ri), "constraint_col_index": np.array(ci),
             "constraint_data": np.array(data), "constraint_lower": np.array([a for _, a, _ in rows]),
             "constraint_upper": np.array([b for _, _, b in rows])}
    event = {"variable_slices": slices, "variable_count": size, "objective": 0., "dual_bound": 0.}
    return n, partition, event, saved


@pytest.mark.parametrize("grouped", [False, True])
def test_original_floor_coo_and_original_terminal_group_cardinality(grouped):
    n, groups, event, saved = coo_fixture(grouped=grouped)
    audit = subject.Audit()
    subject.audit_floor_domain(event, saved, n, groups, {"a": 1., "b": 1., "tau": .2}, audit, "fixture")
    subject.evidence.audit_raw_primal(event, saved, audit, "fixture")
    assert not failures(audit), failures(audit)
    if grouped:
        # A single selected z group holds three original terminals. A row in
        # group-count units would wrongly require selecting the other group.
        index = np.flatnonzero((saved["constraint_row_index"] == 1) & (saved["constraint_data"] == 3))[0]
        saved["constraint_data"][index] = 1.
        broken = subject.Audit()
        subject.audit_floor_domain(event, saved, n, groups, {"a": 1., "b": 1., "tau": .2}, broken, "bad")
        assert any("original-terminal cardinality row" in row for row in failures(broken))


def test_original_floor_missing_row_and_source_h_activation_tie_are_detected():
    n, groups, event, saved = coo_fixture()
    # Keep a feasible primal but erase the positive extension's floor row.
    saved["constraint_lower"][-2] = 0
    audit = subject.Audit()
    subject.audit_floor_domain(event, saved, n, groups, {"a": 1., "b": 1., "tau": .2}, audit, "missing")
    assert any("positive resolved extensions" in row for row in failures(audit))
    # An extra row e-h>=0 would silently restrict the original Source paths.
    n, groups, event, saved = coo_fixture()
    row = len(saved["constraint_lower"])
    saved["constraint_row_index"] = np.r_[saved["constraint_row_index"], row, row]
    saved["constraint_col_index"] = np.r_[saved["constraint_col_index"], event["variable_slices"]["edge_active"][0], 0]
    saved["constraint_data"] = np.r_[saved["constraint_data"], 1., -1.]
    saved["constraint_lower"] = np.r_[saved["constraint_lower"], 0.]
    saved["constraint_upper"] = np.r_[saved["constraint_upper"], np.inf]
    audit = subject.Audit()
    subject.audit_floor_domain(event, saved, n, groups, {"a": 1., "b": 1., "tau": .2}, audit, "tied")
    assert any("does not alter Source h" in row for row in failures(audit))


def test_primal_objective_feasibility_and_gap_are_independent_of_success_label():
    _, _, event, saved = coo_fixture()
    saved["primal"][event["variable_slices"]["new_r"][0]] = .01
    audit = subject.Audit()
    subject.evidence.audit_raw_primal(event, saved, audit, "bad_incumbent")
    assert any("original linear constraints" in row for row in failures(audit))
    certificate = {"status": 0, "success": True, "binary_variable_count": 1,
                   "objective": 1., "dual_bound": .8, "mip_gap": .2}
    assert not subject.optimality_certificate(certificate)
    certificate.update(dual_bound=1., mip_gap=0.)
    assert subject.optimality_certificate(certificate)


def test_source_h_reconstruction_checks_all_atoms_ancestors_and_disjointness():
    labels = [11, 12, 13]
    point = {"support_indices": [[0], [1], [2], [0, 1], [0, 1, 2]],
             "r_values": [.1, .2, .3, .4, .5], "x_values": [.01, .02, .03, .04, .05],
             "source_h": [1, 0, 0, 1, 1], "source_response_r": [1., .9, .5],
             "source_response_x": [.1, .09, .05]}
    atom = subject.evidence.atom_reconstruction(point, labels)
    audit = subject.Audit()
    gr, gx = subject.evidence.audit_source_path(point, atom, True, audit, "source")
    assert not failures(audit), failures(audit)
    rng = np.random.default_rng(23)
    arrays = {"P": rng.normal(size=(288, 3)), "Q": rng.normal(size=(288, 3)),
              "P_source": np.sin(np.arange(288)/11), "Q_source": np.cos(np.arange(288)/13)}
    arrays["Y"] = arrays["P"]@atom["R"].T + arrays["Q"]@atom["X"].T + arrays["P_source"][:, None]*gr + arrays["Q_source"][:, None]*gx
    assert subject.evidence.prediction_and_loss(arrays, atom["R"], atom["X"], gr, gx)[1] == 0
    point["source_h"] = [1, 1, 0, 0, 1]
    broken = subject.Audit()
    subject.evidence.audit_source_path(point, atom, True, broken, "bad_h")
    assert any("ancestor complete and excludes disjoint" in row for row in failures(broken))
    assert any("independently rebuilt" in row for row in failures(broken))


def test_path_posthoc_retains_last_selection_even_if_fixed_tree_has_better_f1():
    labels = [11, 12, 13, 14]
    first = {"support_indices": [[0], [1], [2], [3], [0, 1, 2]],
             "r_values": [.1]*4+[.4], "x_values": [.05]*4+[.2], "iteration": 0}
    last = {"support_indices": first["support_indices"]+[[0, 1]],
            "r_values": first["r_values"]+[.1], "x_values": first["x_values"]+[.05], "iteration": 1}
    reconstructed = [subject.evidence.atom_reconstruction(point, labels) for point in [first, last]]
    reconstructed[0]["loss"], reconstructed[1]["loss"] = .2, .1
    truth = {"clades": {frozenset([11, 12, 13])}, "R": reconstructed[0]["R"], "X": reconstructed[0]["X"]}
    rows = subject.posthoc_path_metrics([first, last], reconstructed, labels, {"a": 1, "b": 1, "tau": .01}, truth)
    assert rows[0]["family_clade_f1"] == 1 and rows[1]["family_clade_f1"] == pytest.approx(2/3)
    assert [row["selected_last_point"] for row in rows] == [False, True]
    assert rows[0]["stage"].startswith("fixed_initial_family") and rows[1]["stage"].startswith("greedy_extension")
    assert [row["train_mae"] for row in rows] == [.2, .1]
    assert all(row["truth_used_for_selection"] is False for row in rows)
    assert rows[0]["r_relative_error"] == 0 < rows[1]["r_relative_error"]
    with pytest.raises(ValueError): subject.posthoc_path_metrics([first], reconstructed, labels, {}, truth)


def test_tree_initialization_is_not_empty_when_atomic_group_selection_is_empty():
    supplement = {"checks": [{"check": "initial no unreported non-singleton initial supports are frozen", "passed": False},
                              {"check": "terminal partition covers all original terminals", "passed": True}],
                  "empty_initialization": True, "selected_count": 0}
    supports = [[0], [1], [2], [3], [0, 1, 2], [0, 1], [0, 1, 2, 3]]
    result = subject.initialization_diagnostic(supplement, supports, 4, tree_init=True)
    assert result["empty_strict_terminal_groups"] is True
    assert result["ordinary_nontrivial_initial_support_count"] == 2
    assert result["ordinary_full_root_stem_initial_support_count"] == 1
    assert result["initial_support_count"] == 7 and "empty_initialization" not in result
    assert result["passed"] and len(result["excluded_checks"]) == 1
    # The strict variant does not permit the tree-init-only exception.
    assert not subject.initialization_diagnostic(supplement, supports, 4, tree_init=False)["passed"]


def prepared_pending_fixture(path):
    """A frozen thirty-job plan with one failure and an invalid RUNNING model."""
    runtime = path/"runtime"
    for root in (runtime/"rnj_wzzt", runtime/"source_snapshot/rnj_wzzt"):
        root.mkdir(parents=True)
        (root/"__init__.py").write_text("# inert hash fixture\n", encoding="utf-8")
    checksum = subject.evidence.file_hash(runtime/"rnj_wzzt/__init__.py")
    write_json(runtime/"runtime_manifest.json", {"runtime_sha256": {"__init__.py": checksum}, "source_sha256": {"__init__.py": checksum}})
    physical_network(path/"evaluation_network")
    manifest = {"file_sha256": {"soumalas11/network/"+p.name: subject.evidence.file_hash(p) for p in (path/"evaluation_network").iterdir()}}
    # This failure occurs before input parsing. All thirty allowed files are
    # nevertheless present and hashed, so the missing model stays in scope.
    failed = "soumalas11__replicate_0__no_source__NJ"
    copied = path/"inputs/soumalas11/replicate_0/no_source"
    inventory = {}
    for scene in range(3):
        for name in subject.evidence.INPUT_NAMES:
            relative = f"scenario_{scene}/{name}"
            target = copied/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}" if name.endswith(".json") else "t,value\n0,0\n", encoding="utf-8")
            inventory[relative] = subject.evidence.file_hash(target)
            manifest["file_sha256"]["soumalas11/replicate_0/no_source/"+relative] = inventory[relative]
    write_json(path/"dataset_manifest.json", manifest)
    jobs = []
    for r in range(3):
        for variant in ("no_source", "rank2_source"):
            for method in ("NJ", "RNJ", "WZZT", "RNJ+WZZT", "RNJ+WZZT_tree_init"):
                jobs.append({"job_id": f"soumalas11__replicate_{r}__{variant}__{method}", "case": "soumalas11",
                    "replicate": r, "variant": variant, "method": method, "per_solve_seconds": 6000. if "WZZT" in method else None,
                    "method_wall_seconds": 30000. if "WZZT" in method else 300., "screen_parameters": {"seed": 20260903+r}})
    protocol = {"jobs": jobs, "expected_jobs": 30, "main_jobs": 24, "initializer_ablation_jobs": 6,
        "training_samples": 288, "validation_samples": 0, "validation_supplied": False,
        "independent_test_samples": 0, "resolution_absolute_tolerance": 1e-10, "solver_threads": 1, "blas_threads": 1,
        "truth_used_for_selection": False, "core_parameters": dict(subject.EXPECTED_CORE),
        "screen_parameters": dict(subject.EXPECTED_SCREEN), "source_sha256": {},
        "dataset_manifest_sha256": subject.evidence.file_hash(path/"dataset_manifest.json"),
        "runtime_directory": str(runtime), "runtime_manifest_sha256": subject.evidence.file_hash(runtime/"runtime_manifest.json")}
    write_json(path/"protocol.json", protocol)
    signature = subject.evidence.file_hash(path/"protocol.json")
    folder = path/"jobs"/failed
    write_json(folder/"inputs_hash.json", {"file_sha256": inventory})
    write_json(folder/"payload.json", {"input_dir": str(copied), "input_sha256": inventory, "protocol_sha256": signature})
    write_json(folder/"result.json", {"status": "failed", "has_model": False, "search_complete": False,
        "protocol_sha256": signature, "elapsed_seconds": .1, "metrics": {"clade_f1": None}})
    running = path/"jobs/soumalas11__replicate_0__rank2_source__WZZT"
    write_json(running/"execution.json", {"pid": 123})
    (running/"model.json").write_text("deliberately invalid pending checkpoint", encoding="utf-8")
    return failed


def test_partial_failure_and_running_artifacts_keep_thirty_denominator(tmp_path):
    failed = prepared_pending_fixture(tmp_path)
    report = subject.audit_comparison(tmp_path, only_jobs=[failed])
    assert report["passed"], [r for r in report["checks"] if not r["passed"]]
    assert len(report["rows"]) == report["expected_jobs"] == 30
    assert report["status_counts"] == {"failed": 1, "not_run": 28, "running": 1}
    assert report["audited_final_jobs"] == 1 and not report["run_complete"]
    assert all(row["metrics_recomputed"] is None for row in report["rows"])
    with pytest.raises(ValueError, match="unknown"): subject.audit_comparison(tmp_path, only_jobs=["outside_protocol"])
    # A changed frozen worker input is detected despite no estimator result.
    copied = tmp_path/"inputs/soumalas11/replicate_0/no_source/scenario_0/source/amplitudes.csv"
    copied.write_text("modified", encoding="utf-8")
    broken = subject.audit_comparison(tmp_path, only_jobs=[failed])
    assert any("worker input" in r["check"] for r in broken["checks"] if not r["passed"])
