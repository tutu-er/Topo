"""Independent, small synthetic checks of the simplified R/T/E engine.

The structural optimum is checked by enumerating every compatible subfamily
of a tiny fixed candidate pool and fitting it with SciPy bounded least squares.
No real observations, validation/test data, benchmark workers or topology
truth files are opened. Native Gurobi calls use one thread and small deadlines.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(family):
    return tuple(sorted({tuple(sorted(s)) for s in family}, key=lambda s: (len(s), s)))


def compatible(left, right):
    a, b = set(left), set(right)
    return not a & b or a <= b or b <= a


def basis(n):
    return canonical([*((i,) for i in range(n)), tuple(range(n))])


def all_pool(n):
    return canonical(itertools.chain.from_iterable(itertools.combinations(range(n), size)
                                                  for size in range(1, n+1)))


def make_data(p, q, y):
    import numpy as np
    p, q, y = (np.asarray(a, float) for a in (p, q, y))
    if p.shape != q.shape or p.shape != y.shape:
        raise ValueError("Synthetic P/Q/Y shape mismatch")
    return SimpleNamespace(n=p.shape[1], labels=tuple(range(p.shape[1])),
                           p=(p,), q=(q,), target=(y,), observation_count=y.size)


def response(point, n):
    """Independent scalar accumulation; no production response builder."""
    import numpy as np
    R, X = np.zeros((n, n)), np.zeros((n, n))
    for support, r, x in zip(point["supports"], point["r"], point["x"]):
        for i in support:
            for j in support:
                R[i, j] += float(r)
                X[i, j] += float(x)
    return R, X


def measured_loss(data, point):
    import numpy as np
    R, X = response(point, data.n)
    residual = data.target[0] - data.p[0] @ R.T - data.q[0] @ X.T
    return float(np.mean(residual ** 2))


def independent_fit(data, family):
    """BVLS fits the raw physical-MSE objective under the same [0,2] bounds."""
    import numpy as np
    from scipy.optimize import lsq_linear
    family = canonical(family)
    fr, fx = [], []
    for support in family:
        z = np.zeros(data.n); z[list(support)] = 1.
        fr.append(((data.p[0] @ z)[:, None] * z).ravel())
        fx.append(((data.q[0] @ z)[:, None] * z).ravel())
    features = np.column_stack([*fr, *fx])
    fit = lsq_linear(features, data.target[0].ravel(), bounds=(0., 2.),
                     method="bvls", tol=1e-12, max_iter=2000)
    if not fit.success:
        raise AssertionError(f"Independent BVLS failed: {fit.message}")
    k = len(family)
    point = dict(supports=family, r=fit.x[:k].tolist(), x=fit.x[k:].tolist())
    point["objective"] = measured_loss(data, point)
    return point, dict(status=int(fit.status), optimality=float(fit.optimality),
                       objective=point["objective"], seconds_unavailable=True,
                       solver="scipy_lsq_linear_bvls", bounds=[0., 2.], success=True)


def independent_score(data, point, penalty):
    return measured_loss(data, point) + penalty * sum(1 < len(s) < data.n for s in point["supports"])


def enumerate_families(data, pool, *, keep=(), hard=(), penalty):
    mandatory = canonical([*basis(data.n), *keep, *hard])
    optional = canonical(s for s in pool if tuple(s) not in mandatory and 1 < len(s) < data.n)
    if len(optional) > 12:
        raise ValueError("This verification deliberately refuses large exhaustive searches")
    checked, best, rows = 0, None, []
    for mask in range(1 << len(optional)):
        chosen = [s for i, s in enumerate(optional) if mask & (1 << i)]
        family = canonical([*mandatory, *chosen])
        if not all(compatible(left, right) for left, right in itertools.combinations(family, 2)):
            continue
        point, audit = independent_fit(data, family)
        J = independent_score(data, point, penalty)
        checked += 1
        row = dict(family=family, objective=point["objective"], score=J, BVLS_status=audit["status"])
        rows.append(row)
        if best is None or J < best["score"] - 1e-12:
            best = dict(point=point, score=J, audit=audit)
    if best is None:
        raise AssertionError("The fixed candidate domain has no compatible family")
    return dict(best=best, candidate_count=len(optional), masks_examined=1 << len(optional),
                feasible_families=checked, rows=rows)


def independent_frontiers(family, n, limit):
    """Enumerate all complete-subtree cuts, independently of tail_domain."""
    nodes = canonical([*basis(n), *family])
    children = {}
    for parent in nodes:
        proper = [s for s in nodes if set(s) < set(parent)]
        children[parent] = canonical(s for s in proper if not any(set(s) < set(t) for t in proper))
    cache = {}

    def cuts(parent):
        if parent in cache:
            return cache[parent]
        choices = {(parent,)}
        if children[parent]:
            for product in itertools.product(*(cuts(child) for child in children[parent])):
                blocks = canonical(block for cut in product for block in cut)
                if len(blocks) <= limit:
                    choices.add(blocks)
        cache[parent] = tuple(sorted(choices, key=lambda cut: (len(cut), cut)))
        return cache[parent]

    frontiers, candidates = set(), set()
    for parent in nodes:
        if len(parent) <= 1:
            continue
        for blocks in cuts(parent):
            if not 2 <= len(blocks) <= limit:
                continue
            frontiers.add((parent, blocks))
            for size in range(2, len(blocks)):
                for chosen in itertools.combinations(blocks, size):
                    candidates.add(tuple(sorted(i for block in chosen for i in block)))
    return frontiers, canonical(candidates)


def assert_close(left, right, tolerance=1e-7, message="Numerical mismatch"):
    if not math.isfinite(float(left)) or not math.isfinite(float(right)) or abs(float(left)-float(right)) > tolerance:
        raise AssertionError(f"{message}: {left} vs {right}")


def point_checks(data, point, engine, penalty):
    import numpy as np
    for values in (point["r"], point["x"]):
        a = np.asarray(values, float)
        assert np.isfinite(a).all() and np.min(a, initial=0.) >= -1e-8 and np.max(a, initial=0.) <= 2.+1e-8
    family = canonical(point["supports"])
    assert all(compatible(a, b) for a, b in itertools.combinations(family, 2))
    mse = measured_loss(data, point)
    assert_close(point["objective"], mse, message="Native loss must equal independent MSE at K=1")
    J = independent_score(data, point, penalty)
    assert_close(engine.score(point, data.n, penalty), J, message="Native J must equal independently recomputed J")
    return dict(mse=mse, score=J, family=family, weights_in_bounds=True, laminar=True)


def clean_point(engine, bank, point, *, hard=()):
    result = engine.clean(bank, point, hard=hard)
    return result if isinstance(result, dict) else result[0]


def solver_record(event):
    if isinstance(event.get("solver"), dict):
        return event["solver"]
    return event


def verify(source, engine_file):
    import numpy as np
    source, engine_file = Path(source).resolve(), Path(engine_file).resolve()
    if not source.is_dir() or not engine_file.is_file():
        raise FileNotFoundError("Both frozen native source and the new engine must exist")
    started = perf_counter()
    engine_initial_sha256 = sha(engine_file)
    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
    import research_experiments
    research_experiments.__path__ = [str(source / "research_experiments")]
    backend = importlib.import_module("research_experiments.miqp_loss_benchmark_20261111.squared_backend")
    backend_file = Path(backend.__file__).resolve()
    assert backend_file.is_relative_to(source)
    manifest = backend.install(1.)
    spec = importlib.util.spec_from_file_location("verified_simplified_rte_engine", engine_file)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    calibration = dict(a=1., b=1., median_root_depth=1.)
    cases, qp_audits = {}, []
    alpha, reference = .003, 1.
    penalty = alpha * reference

    def make_bank(data, pool):
        return backend.SquaredBank(data, pool, calibration, seconds=15.)

    # A wrong provisional 12 atom blocks the true 23 atom even at zero weight.
    n = 3
    z = np.array([0., 1., 1.])
    data = make_data(np.eye(n), np.zeros((n, n)), np.eye(n)+np.outer(z, z))
    pool = all_pool(n)
    wrong_family = canonical([*basis(n), (0, 1)])
    exhaustive = enumerate_families(data, pool, penalty=penalty)
    bank = make_bank(data, pool)
    try:
        wrong = bank.fit(wrong_family, preserve=True)
        assert_close(measured_loss(data, wrong), 4/27, 1e-8, "Three-terminal hand MSE")
        full_index = list(map(tuple, wrong["supports"])).index(tuple(range(n)))
        assert_close(wrong["r"][full_index], 1/3, 1e-7, "Wrong-family full coefficient")
        fixed, event = engine.reconstruct(bank, wrong, pool=pool, keep=basis(n), reference=reference,
                                           deadline=perf_counter()+15., alpha=alpha, stage="E")
        checked = point_checks(data, fixed, engine, penalty)
        assert event["accepted"] and (1, 2) in canonical(fixed["supports"])
        assert_close(checked["mse"], 0., 1e-8, "Corrected three-terminal prediction")
        assert_close(checked["score"], exhaustive["best"]["score"], 1e-7, "E vs all same-pool families")
        assert solver_record(event).get("status") == 2
        cases["three_terminal_wrong12_to_true23"] = dict(passed=True, before_MSE=measured_loss(data, wrong),
            after=checked, expected_before_MSE=4/27, expected_after_J=penalty,
            exhaustive=exhaustive, event=event)
        # A verified hard 12 set must continue to exclude crossing 23.
        hard_fixed, hard_event = engine.reconstruct(bank, wrong, pool=pool, keep=basis(n), hard=((0, 1),),
            reference=reference, deadline=perf_counter()+15., alpha=alpha, stage="E_hard")
        hard_exhaustive = enumerate_families(data, pool, hard=((0, 1),), penalty=penalty)
        hard_checked = point_checks(data, hard_fixed, engine, penalty)
        assert (0, 1) in canonical(hard_fixed["supports"]) and (1, 2) not in canonical(hard_fixed["supports"])
        assert_close(hard_checked["score"], hard_exhaustive["best"]["score"], 1e-7, "Hard-prior same-pool optimum")
        cases["hard_crossing_prior_retained"] = dict(passed=True, after=hard_checked, event=hard_event,
                                                     exhaustive=hard_exhaustive)
        qp_audits.extend(bank.lp.solves)
    finally:
        bank.close()

    # A nonzero, noisy R/X case checks that the entire loss and count objective
    # agrees with independent enumeration beyond an exact zero-residual toy.
    rng = np.random.default_rng(20261111)
    p, q = rng.uniform(.2, 1.2, (8, 4)), rng.uniform(.1, .7, (8, 4))
    true = dict(supports=[*basis(4), (0, 1), (0, 1, 2)],
                r=[.15]*4+[.04, .12, .10], x=[.08]*4+[.02, .09, .06])
    R, X = response(true, 4)
    data4 = make_data(p, q, p@R.T+q@X.T + .003*rng.normal(size=(8, 4)))
    pool4 = all_pool(4)
    ref4 = .01
    exhaustive4 = enumerate_families(data4, pool4, penalty=alpha*ref4)
    bank = make_bank(data4, pool4)
    try:
        current = bank.fit([*basis(4), (0, 3)], preserve=True)
        fixed, event = engine.reconstruct(bank, current, pool=pool4, keep=basis(4), reference=ref4,
                                           deadline=perf_counter()+20., alpha=alpha, stage="E_enumeration")
        checked = point_checks(data4, fixed, engine, alpha*ref4)
        assert_close(checked["score"], exhaustive4["best"]["score"], 2e-7,
                     "Four-terminal E vs exhaustive laminar bounded-LS optimum")
        assert solver_record(event).get("status") == 2
        assert checked["mse"] > 1e-10
        cases["four_terminal_full_laminar_enumeration"] = dict(passed=True, after=checked,
            exhaustive=exhaustive4, event=event, seed=20261111, nonzero_residual=True)
        qp_audits.extend(bank.lp.solves)
    finally:
        bank.close()

    # A pure deletion cannot improve the best training MSE, but its smaller
    # line count can improve J. This example uses a genuinely positive atom.
    epsilon = .1
    z12 = np.array([1., 1., 0.])
    deletion_data = make_data(np.eye(3), np.zeros((3, 3)), np.eye(3)+epsilon*np.outer(z12, z12))
    before, _ = independent_fit(deletion_data, [*basis(3), (0, 1)])
    after, _ = independent_fit(deletion_data, basis(3))
    assert measured_loss(deletion_data, after) > measured_loss(deletion_data, before) + 1e-6
    assert independent_score(deletion_data, after, penalty) < independent_score(deletion_data, before, penalty) - 1e-6
    assert_close(measured_loss(deletion_data, after), 4*epsilon**2/27, 1e-9, "Pure deletion hand MSE")
    bank = make_bank(deletion_data, [*basis(3), (0, 1)])
    try:
        current = bank.fit([*basis(3), (0, 1)], preserve=True)
        fixed, event = engine.reconstruct(bank, current, pool=[*basis(3), (0, 1)], keep=basis(3),
            reference=reference, deadline=perf_counter()+15., alpha=alpha, stage="E_deletion")
        checked = point_checks(deletion_data, fixed, engine, penalty)
        assert event["accepted"] and (0, 1) not in canonical(fixed["supports"])
        assert_close(checked["score"], independent_score(deletion_data, after, penalty), 1e-7)
        cases["pure_deletion_MSE_increases_J_decreases"] = dict(passed=True, epsilon=epsilon,
            before_MSE=measured_loss(deletion_data, before), after_MSE=checked["mse"],
            before_J=independent_score(deletion_data, before, penalty), after_J=checked["score"], event=event)
        # The wrong warm family has a zero 13 atom. Before cleanup its J is
        # worse than current; after cleanup it is better. A controlled absence
        # of a native incumbent isolates the required QP->clean->compare order.
        original_solve = backend.solve
        try:
            backend.solve = lambda *args, **kwargs: (None, dict(
                status=9, objective=None, dual_bound=None, relative_gap=None,
                solver="verification_stub_no_native_incumbent", synthetic_stub=True))
            warm_family = canonical([*basis(3), (0, 2)])
            warm_raw = bank.fit(warm_family, preserve=True)
            assert independent_score(deletion_data, warm_raw, penalty) > independent_score(deletion_data, current, penalty)
            warm_fixed, warm_event = engine.reconstruct(bank, current,
                pool=[*basis(3), (0, 1), (0, 2)], keep=basis(3), warm_families=(warm_family,),
                reference=reference, deadline=perf_counter()+15., alpha=alpha, stage="E_warm_closure")
            warm_checked = point_checks(deletion_data, warm_fixed, engine, penalty)
            assert warm_event["accepted"] and canonical(warm_fixed["supports"]) == basis(3)
            assert_close(warm_checked["score"], independent_score(deletion_data, after, penalty), 1e-7)
            assert any(row["accepted"] for row in warm_event["zero_events"])
            cases["QP_zero_cleanup_precedes_warm_score_comparison"] = dict(
                passed=True, after=warm_checked, event=warm_event,
                warm_J_before_cleanup=independent_score(deletion_data, warm_raw, penalty),
                current_J=independent_score(deletion_data, current, penalty),
                native_solver_calls_in_this_check=0, native_incumbent_absence_is_controlled_stub=True)
        finally:
            backend.solve = original_solve
        qp_audits.extend(bank.lp.solves)
    finally:
        bank.close()

    zero_data = make_data(np.eye(4), np.zeros((4, 4)), np.eye(4))
    hard, temporary = (1, 2), (0, 3)
    bank = make_bank(zero_data, [*basis(4), hard, temporary])
    try:
        current = bank.fit([*basis(4), hard, temporary], preserve=True)
        cleaned = clean_point(engine, bank, current, hard=(hard,))
        checked = point_checks(zero_data, cleaned, engine, penalty)
        retained = set(canonical(cleaned["supports"]))
        assert set(basis(4)) | {hard} <= retained and temporary not in retained
        assert_close(measured_loss(zero_data, current), checked["mse"], 1e-10)
        cases["zero_release_protects_basis_and_hard"] = dict(passed=True, after=checked,
            hard=hard, removed_temporary=temporary, basis_including_zero_full_retained=True)
        missing_hard_rejected = False
        try:
            engine.clean(bank, bank.fit(basis(4), preserve=True), hard=(hard,))
        except ValueError:
            missing_hard_rejected = True
        assert missing_hard_rejected
        missing_reconstruction_hard_rejected = False
        try:
            engine.reconstruct(bank, bank.fit(basis(4), preserve=True), pool=[*basis(4), hard],
                keep=basis(4), hard=(hard,), reference=reference, deadline=perf_counter()+5.)
        except ValueError:
            missing_reconstruction_hard_rejected = True
        assert missing_reconstruction_hard_rejected
        cases["zero_release_protects_basis_and_hard"].update(
            missing_hard_clean_rejected=True, missing_hard_reconstruction_rejected=True)
        qp_audits.extend(bank.lp.solves)
    finally:
        bank.close()

    # An eight-terminal parent can open the old six-terminal child through a
    # four-block frontier. Both a crossing correction and a seven-terminal
    # candidate are legal: block_limit is never a terminal-count restriction.
    n = 8
    six, left, right = tuple(range(6)), (0, 1, 2), (3, 4, 5)
    old_family = canonical([*basis(n), six, left, right])
    expected_frontiers, expected_candidates = independent_frontiers(old_family, n, 5)
    domain = engine.tail_domain(old_family, n, block_limit=5)
    pool, keep = set(canonical(domain["pool"])), set(canonical(domain["keep"]))
    released = set(canonical(domain["released"]))
    native_frontiers = {(tuple(row["parent"]), canonical(row["blocks"])) for row in domain["frontiers"]}
    assert native_frontiers == {(parent, blocks) for parent, blocks in expected_frontiers if len(blocks) >= 3}
    assert pool == set(old_family) | set(expected_candidates)
    assert released == (set(old_family) & set(expected_candidates)) - set(basis(n))
    assert keep == set(old_family) - released
    assert set(old_family) <= pool | keep
    assert six in released and six not in keep
    crossing, large = (0, 1, 2, 6), tuple(range(7))
    assert crossing in pool and large in pool and len(large) > 5
    assert set(basis(n)) <= keep and not keep & released
    hard_domain = engine.tail_domain(old_family, n, block_limit=5, hard=(six,))
    assert six in set(canonical(hard_domain["keep"])) and six not in set(canonical(hard_domain["released"]))
    # Coefficient copying is an explicit feasible restoration witness.
    old_point = dict(supports=old_family, r=[.1]*len(old_family), x=[.05]*len(old_family))
    witness = dict(supports=canonical(pool | keep), r=[], x=[])
    coefficients = {s: (r, x) for s, r, x in zip(old_family, old_point["r"], old_point["x"])}
    activations = {s: int(s in coefficients) for s in pool if 1 < len(s) < n}
    for support in witness["supports"]:
        r, x = coefficients.get(support, (0., 0.))
        witness["r"].append(r); witness["x"].append(x)
    old_R, old_X = response(old_point, n); restored_R, restored_X = response(witness, n)
    assert np.array_equal(old_R, restored_R) and np.array_equal(old_X, restored_X)
    cases["all_frontiers_open_large_subtree_and_restore_old"] = dict(
        passed=True, terminals=n, old_six_terminal_block=six, block_limit=5,
        independent_frontier_count=len(expected_frontiers), independent_candidate_count=len(expected_candidates),
        native_frontiers=domain["frontiers"], native_pool=domain["pool"], native_keep=domain["keep"],
        native_released=domain["released"], crossing_witness=crossing, seven_terminal_candidate=large,
        old_response_exactly_recovered=True, hard_large_block_retained=True,
        restoration_selected_supports=old_family,
        restoration_activations=[dict(support=s, selected=a) for s, a in sorted(activations.items())])

    # The full path starts from the basis: RNJ is guidance, not a locked seed.
    # With six singleton blocks, W must run before the <=5-block T stage, and
    # its unrestricted domain must find a triple absent from RNJ guidance.
    p6 = np.eye(6); z6 = np.array([1., 1., 1., 0., 0., 0.])
    run_data = make_data(p6, np.zeros((6, 6)), p6+np.outer(z6, z6))
    guidance = canonical([*basis(6), (4, 5)])
    bank = make_bank(run_data, all_pool(6))
    try:
        trajectory = engine.run(bank, rnj_supports=guidance, rnj_warm_families=(guidance,),
            reference=reference, deadline=perf_counter()+20., per_mip=5., alpha=alpha, block_limit=5)
        assert canonical(trajectory["path"][0]["supports"]) == basis(6)
        assert trajectory["initial_seeds"] == []
        assert trajectory["relative_gain_gate"] is None and trajectory["output_contraction"] is None
        assert not trajectory["tree_global_optimal"] and not trajectory["complete_tree_output_fallback"]
        assert any(event["stage"] == "W" and event["accepted"] for event in trajectory["events"])
        assert [event["stage"] for event in trajectory["events"]].count("T") == 1
        assert [event["stage"] for event in trajectory["events"]].count("E") == 1
        assert (0, 1, 2) not in guidance and (0, 1, 2) in canonical(trajectory["point"]["supports"])
        checked = point_checks(run_data, trajectory["point"], engine, penalty)
        assert_close(checked["mse"], 0., 1e-8)
        for before_point, after_point in zip(trajectory["path"], trajectory["path"][1:]):
            assert independent_score(run_data, before_point, penalty) - independent_score(run_data, after_point, penalty) > 1e-10
        cases["full_W_T_E_path_no_RNJ_seed_or_output_contraction"] = dict(
            passed=True, after=checked, guidance_triple_absent=True,
            stage_order=[event["stage"] for event in trajectory["events"]],
            events=trajectory["events"], path_scores=[independent_score(run_data, point, penalty) for point in trajectory["path"]],
            tree_global_optimal=False)
        qp_audits.extend(bank.lp.solves)
    finally:
        bank.close()

    native_sources = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "research_experiments" or name.startswith("research_experiments.") or
                name == "rnj_wzzt" or name.startswith("rnj_wzzt.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        file = Path(filename).resolve()
        assert file.is_relative_to(source), (name, file)
        native_sources[name] = dict(path=str(file), sha256=sha(file))
    unchanged = all(sha(row["path"]) == row["sha256"] for row in manifest["native_module_sources"].values())
    assert unchanged
    assert sha(engine_file) == engine_initial_sha256, "The engine changed while verification was running"
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                seconds=perf_counter()-started, script_sha256=sha(__file__), engine_file=str(engine_file),
                engine_sha256=sha(engine_file), native_source=str(source), backend_sha256=sha(backend_file),
                native_sources=native_sources, install_manifest=manifest, native_sources_unchanged=True,
                cases=cases, case_count=len(cases), QP_audits=qp_audits, solver_threads=1,
                normalization_K=1., real_benchmark_jobs_started=0, validation_or_test_opened=False,
                physical_truth_files_opened=False, positive_impedance_floor=0., ridge=0.,
                global_tree_optimality_claim=False,
                guarantee_scope="Only the finite enumerated candidate domain in each synthetic test; no global network/search guarantee",
                Q_zero_boundary="The three-terminal/deletion/zero toys do not identify x; no unique-x assertion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/miqp_loss_benchmark_20261111/source")
    parser.add_argument("--engine", type=Path, default=Path(__file__).with_name("engine.py"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = verify(args.source, args.engine)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                               encoding="utf-8", newline="\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
