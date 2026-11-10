"""Independent post-training audit of one-day 96-point T comparisons.

Truth and test data are opened only here.  This code never imports the native
topology optimizer or uses truth to create/select candidates.  NumPy rebuilds
responses; SciPy/HiGHS independently profiles fixed selected/true families.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import warnings

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csr_matrix, eye, hstack, vstack

from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


ROOT = Path(__file__).resolve().parents[2]
METADATA = ("panel", "case", "bundle", "N", "voltage_sigma", "condition")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def load_split(path):
    """Use every saved 15-minute slot; N96 is one day, not 32/day x3."""
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(int).tolist()
        n = len(labels)
        raw = [np.asarray(archive[key], float) for key in ("P", "Q", "V")]
        assert all(a.ndim == 3 and a.shape[1:] == (96, n) for a in raw)
        p, q, voltage = [a.reshape(-1, n) for a in raw]
        root = np.asarray(archive["root"], float).reshape(-1)
    y = root[:, None] ** 2 - voltage ** 2
    assert p.shape == q.shape == y.shape and all(np.isfinite(a).all() for a in (p, q, y))
    return dict(p=p, q=q, y=y, labels=labels, n=n, rows=len(p))


def family(point, n):
    sets = tuple(tuple(int(i) for i in C) for C in point["supports"])
    assert len(sets) == len(set(sets))
    assert all(C and C == tuple(sorted(set(C))) and all(0 <= i < n for i in C) for C in sets)
    for C, D in combinations(map(set, sets), 2):
        assert not C & D or C <= D or D <= C, "Nonlaminar support family"
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    assert r.shape == x.shape == (len(sets),) and np.isfinite(r).all() and np.isfinite(x).all()
    assert np.min(np.r_[r, x], initial=0.) >= -1e-9
    assert np.max(np.r_[r, x], initial=0.) <= 2. + 1e-9
    return sets


def matrices(point, n):
    z = np.asarray([[i in C for i in range(n)] for C in point["supports"]], float)
    return (z.T * np.asarray(point["r"])) @ z, (z.T * np.asarray(point["x"])) @ z


def loss(split, point):
    r, x = matrices(point, split["n"])
    return float(np.mean(abs(split["y"] - split["p"] @ r.T - split["q"] @ x.T)))


def topology(point, n):
    return frozenset(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def independent_profile(split, supports):
    p, q, y, n = (split[k] for k in ("p", "q", "y", "n"))
    z = np.asarray([[i in C for i in range(n)] for C in supports], float)
    features = csr_matrix(np.column_stack([((a @ row)[:, None] * row).ravel()
                                          for a in (p, q) for row in z]))
    m, k = features.shape
    I = eye(m, format="csr")
    A = vstack((hstack((features, -I)), hstack((-features, -I))), format="csr")
    b = np.r_[y.ravel(), -y.ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        result = linprog(np.r_[np.zeros(k), np.ones(m) / m], A_ub=A, b_ub=b,
                         bounds=[(0, 2)] * k + [(0, None)] * m,
                         method="highs", options={"threads": 1})
    if not result.success:
        raise RuntimeError("Independent HiGHS family LP: " + result.message)
    violation = float(np.max(A @ result.x - b, initial=0.))
    assert violation <= 1e-7
    return dict(objective=float(result.fun), violation=violation, status=int(result.status))


def metrics(point, splits, truth):
    n = splits["train"]["n"]
    predicted = topology(point, n)
    target = truth["indexed_clades"]
    tp, fp, fn = len(predicted & target), len(predicted - target), len(target - predicted)
    r, x = matrices(point, n)
    return dict(clade_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                exact=predicted == target, precision=tp / len(predicted) if predicted else float(not target),
                recall=tp / len(target) if target else 1., true_positive=tp, false_positive=fp, false_negative=fn,
                predicted_clades=len(predicted), true_clades=len(target),
                R_relative=float(np.linalg.norm(r - truth["R"]) / np.linalg.norm(truth["R"])),
                X_relative=float(np.linalg.norm(x - truth["X"]) / np.linalg.norm(truth["X"])),
                **{name + "_mae": loss(split, point) for name, split in splits.items()})


def solver_records(result):
    for event in result.get("trajectory", {}).get("attempts", []):
        for record in event.get("trials", [event]):
            if record.get("status") is not None:
                yield dict(stage=event.get("kind", "unknown"), **record)
    event = result.get("exit_event", {})
    if event.get("status") is not None:
        yield dict(stage="final_exit", **event)


def coverage(raw_points, output_points, truth, n, exit_pool):
    raw_sets = [topology(p, n) for p in raw_points]
    output_sets = [topology(p, n) for p in output_points]
    raw_union = frozenset().union(*raw_sets)
    output_union = frozenset().union(*output_sets)
    pool = frozenset(tuple(C) for C in exit_pool)
    denominator = len(truth)
    return dict(raw_model_count=len(raw_sets), output_model_count=len(output_sets),
                raw_pool_has_true_tree=truth in raw_sets,
                output_pool_has_true_tree=truth in output_sets,
                raw_union_has_all_true_connections=truth <= raw_union,
                output_union_has_all_true_connections=truth <= output_union,
                exit_pool_has_all_true_connections=truth <= pool,
                raw_union_true_recall=len(truth & raw_union) / denominator if denominator else 1.,
                output_union_true_recall=len(truth & output_union) / denominator if denominator else 1.,
                exit_pool_true_recall=len(truth & pool) / denominator if denominator else 1.)


def analyze_job(args):
    output, job = args
    output = Path(output)
    protocol, catalog = read(output / "protocol.json"), read(output / "catalog.json")
    meta = catalog[job]
    header = dict(job=job, **{k: meta.get(k) for k in METADATA})
    base_rows = [dict(**header, method=c["id"], status="pending", available=False) for c in protocol["methods"]]
    record = dict(job=job, rows=base_rows, solver=[], errors=[], checks=0, independent_LPs=0,
                  raw_hashes={}, protocol_sha256=sha(output / "protocol.json"), analyzer_sha256=sha(__file__))
    if meta.get("data_error"):
        for row in base_rows:
            row.update(status="failed", error=str(meta["data_error"]))
        record["errors"].append(dict(job=job, kind="data_generation", detail=meta["data_error"]))
        return record
    try:
        hashes = {name: sha(meta[name]) for name in ("train", "validation", "test")}
        expected = protocol["input_hashes"][job]
        assert hashes == {key: expected[key] for key in hashes}
        splits = {name: load_split(meta[name]) for name in ("train", "validation", "test")}
        assert {name: data["rows"] for name, data in splits.items()} == {"train": 96, "validation": 288, "test": 96}
        labels = splits["train"]["labels"]
        n = len(labels)
        assert all(split["labels"] == labels for split in splits.values())
        # All training is already complete or bounded before this post-analysis.
        truth = physical_truth(Path(meta["truth_network"]), labels)
        positions = {label: i for i, label in enumerate(labels)}
        truth["indexed_clades"] = frozenset(tuple(sorted(positions[v] for v in C)) for C in truth["clades"])
        true_family = tuple(sorted([*truth["indexed_clades"], *((i,) for i in range(n)), tuple(range(n))]))
        profiles = {true_family: independent_profile(splits["train"], true_family)}
        record["independent_LPs"] += 1
        record["true_family_LP"] = profiles[true_family]
        common_file = output / "common" / job / "common.json"
        common = read(common_file) if common_file.exists() else None
        if common:
            assert common["labels"] == labels
            assert common.get("truth_used", False) is False and common.get("test_opened", False) is False
            assert common["source_input_hashes"] == {k: hashes[k] for k in ("train", "validation")}
            assert not common.get("LP_failures", [])
            family(common["reference_point"], n)
            assert abs(loss(splits["train"], common["reference_point"]) - common["reference"]) <= 1e-10
            record["checks"] += 8
        network_meta = read(Path(meta["truth_network"]) / "metadata.json")
        record["truth_source_hashes"] = {name: sha(Path(meta["truth_network"]) / name)
                                          for name in ("metadata.json", "buses.csv", "branches.csv")}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for row in base_rows:
            row.update(status="invalid", error=message)
        record["errors"].append(dict(job=job, kind="input_audit", detail=message))
        return record

    for row, config in zip(base_rows, protocol["methods"]):
        method = config["id"]
        file = output / "jobs" / job / method / "result.json"
        error_file = file.with_name("error.json")
        if not file.exists():
            if error_file.exists():
                row.update(status="failed", error=read(error_file))
                record["errors"].append(dict(job=job, method=method, kind="worker_error", detail=row["error"]))
            continue
        record["raw_hashes"][method] = sha(file)
        try:
            result = read(file)
            assert result["job"] == job and result["method"] == method
            assert result["input_hashes"] == {key: hashes[key] for key in ("train", "validation")}
            assert result.get("truth_used", False) is False and result.get("test_opened", False) is False
            assert result.get("final_RNJ_fallback", False) is False
            assert result.get("protocol_sha256") == record["protocol_sha256"]
            assert common is not None and result["common_sha256"] == sha(common_file)
            assert not result.get("LP_failures", [])
            selected = result["selected"]
            assert selected.get("source") in ("own_path", "singleton")
            point = selected["point"]
            key = tuple(sorted(family(point, n)))
            measured = metrics(point, splits, truth)
            assert abs(measured["train_mae"] - point["objective"]) <= 1e-8
            assert abs(measured["validation_mae"] - selected["validation"]) <= 1e-10
            if key not in profiles:
                profiles[key] = independent_profile(splits["train"], key)
                record["independent_LPs"] += 1
            assert abs(profiles[key]["objective"] - measured["train_mae"]) <= 2e-8
            trajectory = result["trajectory"]
            raw_path = trajectory["path"]
            output_rows = result["output_candidates"]
            for saved in [*raw_path, *[v["point"] for v in output_rows]]:
                family(saved, n)
                assert abs(loss(splits["train"], saved) - saved["objective"]) <= 1e-8
                record["checks"] += 3
            # Reconstruct the saved strict .24/refit output selector, without refitting.
            best_validation = min(float(v["validation"]) for v in output_rows)
            eligible = [v for v in output_rows if v["validation"] <= best_validation + 1e-12]
            expected_selected = min(eligible, key=lambda v: (v["internal"], v["validation"], v["index"]))
            assert int(expected_selected["index"]) == int(selected["index"])
            assert tuple(expected_selected["point"]["r"]) == tuple(point["r"])
            assert tuple(expected_selected["point"]["x"]) == tuple(point["x"])
            for candidate in output_rows:
                assert abs(loss(splits["validation"], candidate["point"]) - candidate["validation"]) <= 1e-10
            calls = list(solver_records(result))
            record["solver"].extend(dict(job=job, method=method, **call) for call in calls)
            for call in calls:
                incumbent = call.get("objective", call.get("final_objective"))
                bound = call.get("dual_bound", call.get("bound", call.get("final_bound")))
                if incumbent is not None and bound is not None:
                    assert float(bound) <= float(incumbent) + 1e-8
                for field in ("audit", "warm_start"):
                    if field in call and "valid" in call[field]:
                        assert call[field]["valid"]
            tails = [v for v in trajectory.get("attempts", []) if v.get("kind") == "joint_tail"]
            assert method != "noT" or not tails
            reference = float(result["reference"])
            alpha = float(config.get("tail_coefficient") or .003)
            for event in trajectory.get("attempts", []):
                if "penalty" in event:
                    expected_alpha = alpha if event.get("kind") == "joint_tail" else .003
                    assert abs(float(event["penalty"]) - expected_alpha * reference) <= 1e-12
            zeros = [v for v in trajectory.get("zero_events", []) if v.get("accepted")]
            for event in zeros:
                assert event["after"] <= event["before"] + event.get("loss_tolerance", 1e-10) + 1e-12
            timing = result.get("timing", result)
            fresh, shared, total = (float(timing[k]) for k in ("fresh_seconds", "common_seconds", "total_seconds"))
            assert min(fresh, shared, total) >= 0 and abs(fresh + shared - total) < 1e-7
            cap = float(result.get("limits", protocol.get("limits", {})).get("total", 14400.))
            cap_exceeded = bool(result.get("total_cap_exceeded", False)) or total > cap + 1.
            covered = coverage(raw_path, [v["point"] for v in output_rows], truth["indexed_clades"], n,
                               result.get("exit_event", {}).get("pool", []))
            row.update(status="verified", available=True, **measured, **covered,
                       selected_source=selected["source"], selected_index=int(selected["index"]),
                       path_length=len(raw_path), tail_calls=len(tails), tail_accepted=sum(bool(v.get("accepted")) for v in tails),
                       exit_accepted=bool(result.get("exit_record", {}).get("accepted")),
                       zero_release_events=len(zeros), zero_released_sets=sum(len(v.get("removed", [])) for v in zeros),
                       stop_reason=trajectory.get("stop_reason"), fresh_seconds=fresh, common_seconds=shared,
                       total_seconds=total, cap_exceeded=cap_exceeded, MIPs=len(calls),
                       gap_met=sum(v.get("status") == 2 for v in calls), MIP_timeouts=sum(v.get("status") == 9 for v in calls),
                       solver_gaps_all_met=all(v.get("status") == 2 for v in calls),
                       true_family_training_MAE=profiles[true_family]["objective"],
                       selected_minus_true_training_MAE=measured["train_mae"] - profiles[true_family]["objective"])
            # Independently replay the export frozen by the training-only worker.
            from .candidates import verify_export, coverage_diagnostics
            star_rows = [v for v in output_rows if v.get("source") == "singleton"]
            candidate_file = file.with_name("candidates.json")
            assert sha(candidate_file) == result["candidate_export_sha256"]
            exported = read(candidate_file)
            export_audit = verify_export(exported, train=splits["train"], validation=splits["validation"])
            assert export_audit["passed"] and exported["labels"] == labels
            assert exported["sample_counts"] == {"train": 96, "validation": 288}
            assert exported["root_label"] == int(meta.get("root_label", network_meta["root_bus"]))
            assert exported["base_kw"] == float(meta.get("base_kw", float(network_meta["base_mva"]) * 1000.))
            provenance = exported["provenance"]
            assert provenance["input_hashes"] == {k: hashes[k] for k in ("train", "validation")}
            assert provenance.get("truth_used_in_fitting") is False
            assert provenance.get("test_used_in_generation") is False and provenance.get("future_DR_used") is False
            def fingerprint(p):
                return tuple(sorted((tuple(C), float(r), float(x)) for C, r, x in zip(p["supports"], p["r"], p["x"])))
            expected_points = [*raw_path, *[v["point"] for v in star_rows]]
            assert {fingerprint(p) for p in expected_points} == {fingerprint(v["point"]) for v in exported["candidates"]}
            export_coverage = coverage_diagnostics(exported, truth["indexed_clades"])
            row.update(exported_models=len(exported["candidates"]),
                       exported_raw_pool_has_true_tree=export_coverage["any_raw_model_exact"],
                       exported16_output_pool_has_true_tree=export_coverage["any_output_model_exact"],
                       exported_connection_union_has_truth=export_coverage["all_true_connections_in_union"],
                       exported_connection_union_true_recall=export_coverage["true_connection_union_recall"])
            row["candidate_export"] = str(candidate_file.relative_to(output)).replace("\\", "/")
            record["checks"] += 24
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            row.update(status="invalid", available=False, error=message)
            record["errors"].append(dict(job=job, method=method, kind="verification", detail=message))
    record["fixed_family_LPs"] = [dict(supports=[list(C) for C in key], **value) for key, value in profiles.items()]
    return record


def aggregate(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ok = group[group.status.eq("verified")]
        exact = int(ok.exact.sum()) if len(ok) else 0
        rows.append(dict(zip(keys, key)) | dict(planned=len(group), returned=int(group.status.isin(("verified", "invalid")).sum()),
                    verified=len(ok), pending=int(group.status.eq("pending").sum()), failed=int(group.status.eq("failed").sum()),
                    invalid=int(group.status.eq("invalid").sum()), F1=float(ok.clade_f1.mean()) if len(ok) else np.nan,
                    exact=exact, exact_fraction_planned=exact / len(group),
                    recovery_rate_completed=exact / len(ok) if len(ok) else np.nan,
                    total_seconds=float(ok.total_seconds.sum()) if len(ok) else np.nan,
                    median_seconds=float(ok.total_seconds.median()) if len(ok) else np.nan,
                    complete=len(ok) == len(group), MIPs=int(ok.MIPs.sum()) if len(ok) else 0,
                    MIP_timeouts=int(ok.MIP_timeouts.sum()) if len(ok) else 0,
                    cap_exceeded=int(ok.cap_exceeded.sum()) if len(ok) else 0,
                    raw_pool_truth=int(ok.raw_pool_has_true_tree.sum()) if len(ok) else 0,
                    output_pool_truth=int(ok.output_pool_has_true_tree.sum()) if len(ok) else 0,
                    exit_connection_pool_truth_covered=int(ok.exit_pool_has_all_true_connections.sum()) if len(ok) else 0))
    return pd.DataFrame(rows)


def paired_tables(frame, jobs, methods, *, draws=10000):
    good = frame[frame.status.eq("verified")].set_index(["job", "method"])
    metadata = frame.drop_duplicates("job").set_index("job")
    inputs, summaries, bootstrap = [], [], []
    rng = np.random.default_rng(2026111017)
    for before, after in combinations(methods, 2):
        paired = []
        for job in jobs:
            header = dict(job=job, before=before, after=after,
                          **{k: metadata.loc[job, k] for k in ("case", "bundle", "voltage_sigma")})
            if (job, before) not in good.index or (job, after) not in good.index:
                inputs.append(dict(**header, status="incomplete"))
                continue
            a, b = good.loc[(job, before)], good.loc[(job, after)]
            item = dict(**header, status="paired", delta_F1=float(b.clade_f1 - a.clade_f1),
                        delta_exact=int(b.exact) - int(a.exact), delta_seconds=float(b.total_seconds - a.total_seconds),
                        new_exact=bool(b.exact and not a.exact), lost_exact=bool(a.exact and not b.exact))
            inputs.append(item)
            paired.append(item)
        if not paired:
            summaries.append(dict(before=before, after=after, planned=len(jobs), paired=0))
            continue
        data = pd.DataFrame(paired)
        summaries.append(dict(before=before, after=after, planned=len(jobs), paired=len(data),
                              delta_F1=float(data.delta_F1.mean()), delta_exact=int(data.delta_exact.sum()),
                              delta_recovery_rate=float(data.delta_exact.mean()), delta_seconds=float(data.delta_seconds.sum()),
                              F1_wins=int((data.delta_F1 > 1e-10).sum()), F1_ties=int((abs(data.delta_F1) <= 1e-10).sum()),
                              F1_losses=int((data.delta_F1 < -1e-10).sum()), new_exact=int(data.new_exact.sum()),
                              lost_exact=int(data.lost_exact.sum())))
        # The source generator reuses the same physical profile seed across
        # all four cases.  Cases and noise levels are not independent draws.
        blocks = data.groupby("bundle", sort=True).agg(n=("job", "size"),
                    dF1=("delta_F1", "mean"), dExact=("delta_exact", "mean"), dSeconds=("delta_seconds", "sum")).reset_index()
        histories = data[["case", "bundle"]].drop_duplicates()
        complete = len(data) == len(jobs) and len(blocks) == 4 and bool((blocks.n == 12).all())
        boot = dict(before=before, after=after, paired_inputs=len(data), shared_seed_blocks=len(blocks),
                    case_seed_histories=len(histories),
                    complete=complete, resamples=draws, seed=2026111017,
                    scheme="resample four globally shared profile-seed blocks; each block carries all four fixed cases and all three noise levels")
        if complete:
            values = blocks[["dF1", "dExact", "dSeconds"]].to_numpy(float)
            indices = rng.integers(0, 4, size=(draws, 4))
            sampled = values[indices].mean(axis=1)
            for i, name in enumerate(("delta_F1", "delta_recovery_rate", "delta_total_seconds")):
                values = sampled[:, i] * (4 if i == 2 else 1)
                boot[name + "_lower95"], boot[name + "_upper95"] = map(float, np.quantile(values, [.025, .975]))
            boot["interpretation"] = "descriptive only: four shared seed groups, four fixed networks; few resampling units; not a topology confidence probability"
        bootstrap.append(boot)
    return dict(pairwise_inputs=pd.DataFrame(inputs), pairwise_summary=pd.DataFrame(summaries),
                block_bootstrap=pd.DataFrame(bootstrap))


def markdown(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}" if np.isfinite(value) else "unavailable"
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns),
                      *["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def analyze(output, *, workers=2, export=True):
    started = perf_counter()
    output = Path(output).resolve()
    protocol = read(output / "protocol.json")
    methods = tuple(v["id"] for v in protocol["methods"])
    assert len(methods) == len(set(methods)) == 3
    jobs = protocol["jobs"]
    planned = len(jobs) * len(methods)
    assert planned == 144 and len(jobs) == 48
    for name, digest in protocol.get("source_hashes", {}).items():
        assert sha(output / "source" / name) == digest, name
    dest = output / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    analyzer_hash = sha(__file__)
    exporter_file = Path(__file__).with_name("candidates.py")
    exporter_hash = sha(exporter_file)
    for job in jobs:
        path = dest / "jobs" / job / "verified.json"
        raw_hashes = {method: sha(output / "jobs" / job / method / "result.json") for method in methods
                      if (output / "jobs" / job / method / "result.json").exists()}
        cached = read(path) if path.exists() else None
        if (cached and cached.get("analyzer_sha256") == analyzer_hash
                and cached.get("exporter_sha256") == exporter_hash
                and cached.get("protocol_sha256") == sha(output / "protocol.json")
                and cached.get("raw_hashes") == raw_hashes
                and all(v["status"] == "verified" for v in cached["rows"])):
            results[job] = cached
        else:
            todo.append((str(output), job))
    def save(record):
        record["exporter_sha256"] = exporter_hash
        results[record["job"]] = record
        write(dest / "jobs" / record["job"] / "verified.json", record)
        if len(results) % 8 == 0:
            print(f"Audited {len(results)}/{len(jobs)} inputs", flush=True)
    if workers == 1:
        for record in map(analyze_job, todo):
            save(record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(analyze_job, todo):
                save(record)
    records = [results[job] for job in jobs]
    frame = pd.DataFrame([row for r in records for row in r["rows"]])
    assert len(frame) == planned
    for name in ("clade_f1", "exact", "total_seconds", "MIPs", "MIP_timeouts", "cap_exceeded",
                 "raw_pool_has_true_tree", "output_pool_has_true_tree", "exit_pool_has_all_true_connections"):
        if name not in frame:
            frame[name] = np.nan
    coverage_fields = [k for k in frame.columns if "pool" in k or "union" in k]
    tables = dict(per_model=frame, summary=aggregate(frame, ["method"]), cases=aggregate(frame, ["case", "method"]),
                  noise=aggregate(frame, ["voltage_sigma", "method"]),
                  case_noise=aggregate(frame, ["case", "voltage_sigma", "method"]),
                  candidate_coverage=frame[["job", "case", "bundle", "voltage_sigma", "method", "status", *coverage_fields]])
    tables.update(paired_tables(frame, jobs, methods))
    solver = [dict(job=v["job"], method=v["method"], stage=v["stage"], status=v.get("status"),
                   seconds=v.get("seconds"), relative_gap=v.get("relative_gap", v.get("gap")),
                   incumbent=v.get("objective", v.get("final_objective")),
                   bound=v.get("dual_bound", v.get("bound", v.get("final_bound"))))
              for record in records for v in record["solver"]]
    errors = [error for record in records for error in record["errors"]]
    tables["solver"] = pd.DataFrame(solver)
    tables["failures"] = pd.DataFrame(errors)
    for name, table in tables.items():
        table.to_csv(dest / (name + ".csv"), index=False, lineterminator="\n")
    verified = int(frame.status.eq("verified").sum())
    complete = verified == planned
    cap_exceeded = int(frame.loc[frame.status.eq("verified"), "cap_exceeded"].sum())
    receipt = dict(passed=complete and not errors and not cap_exceeded, complete=complete, planned=planned,
                   returned=int(frame.status.isin(("verified", "invalid")).sum()), verified=verified,
                   failed=int(frame.status.eq("failed").sum()), invalid=int(frame.status.eq("invalid").sum()),
                   pending=int(frame.status.eq("pending").sum()), budget_exceeded=cap_exceeded,
                   independent_LPs=sum(r["independent_LPs"] for r in records), checks=sum(r["checks"] for r in records),
                   independent_LP_count_scope="unique per-input family profiles represented in audit records; cached records reused; not cumulative solver calls across reruns",
                   MIPs=len(solver), gap_met=sum(v["status"] == 2 for v in solver),
                   MIP_timeouts=sum(v["status"] == 9 for v in solver), all_MIP_gaps_met=all(v["status"] == 2 for v in solver),
                   source_hashes_verified=True, all_planned_denominators_preserved=True,
                   no_complete_RNJ_output_fallback=True, train_rows=96, validation_rows=288, test_rows=96,
                   baseline_sample_definition="one complete day96, not previous total96 sparse across three days",
                   analyzer_sha256=analyzer_hash, exporter_sha256=exporter_hash,
                   protocol_sha256=sha(output / "protocol.json"), elapsed_seconds=perf_counter() - started,
                   candidate_coverage_semantics="whole-tree membership distinct from union-of-connections coverage; diagnostic truth only after fitting",
                   bootstrap="four globally shared profile-seed blocks; resample cases and noise levels together; 16 case-seed histories are dependent",
                   independent_resampling_groups=4, case_seed_histories=16)
    write(dest / "verification.json", receipt)
    text = "# 一日96点：末端联合方法独立对照\n\n"
    text += f"已核验 {verified}/{planned}，失败 {receipt['failed']}，无效 {receipt['invalid']}，待完成 {receipt['pending']}。\n\n"
    text += "48输入=四网络×四个新种子×三档噪声；每输入训练为一整天96槽，验证为独立三天288槽，测试为独立一天96槽。三方法同预算、同少量RNJ初值、原始WZZT、清零和最终E；仅T开关/末端惩罚系数不同。最终为自身路径的.24收缩重拟合与严格验证，不使用完整RNJ整树备用。\n\n"
    text += markdown(tables["summary"]) + "\n\n"
    text += "未完成时，均值仅描述已核验子集；exact_fraction_planned保留计划分母，不能当成未完成算法最终恢复率。时间为逐输入算法墙钟之和，含每方法一次公共前端记账，不是并行等待时间。\n\n"
    text += "## 成对差异\n\n" + markdown(tables["pairwise_summary"]) + "\n\n"
    text += "## 同种子噪声成组重采样\n\n" + markdown(tables["block_bootstrap"]) + "\n\n"
    text += "生成器在四个网络之间复用同一物理曲线种子。因此，16个case/seed历史只有4个独立种子组；每次重抽整个共享种子组，同时携带四个网络和三档噪声。区间只描述四个固定网络的种子变化，重采样单位很少，不是拓扑真实性概率。\n\n"
    text += "候选库同时报告完整树是否出现、连接并集是否覆盖真连接，两者不等价。候选导出保留历史已拟合完整响应及来源，未用真拓扑或未来DR数据筛选。对接DR时须另声明strict/.24/refit与3%/.16/no-refit的不同选模合同。\n"
    (dest / "RESULTS.md").write_text(text, encoding="utf-8", newline="\n")
    if export:
        delivery = ROOT / "docs" / output.name
        delivery.mkdir(parents=True, exist_ok=True)
        for file in dest.iterdir():
            if file.is_file():
                shutil.copy2(file, delivery / file.name)
    print(markdown(tables["summary"]), flush=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "outputs/day96_t_benchmark_20261110")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    analyze(args.output, workers=args.workers)
