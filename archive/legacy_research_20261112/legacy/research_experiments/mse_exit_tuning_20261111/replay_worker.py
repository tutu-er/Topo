"""Replay one frozen final-E proposal under predeclared training-MSE gates.

Execute this file by absolute path in a fresh process. The forward path and
MIQP proposal remain unchanged. Only a newly admitted family can require new
fixed-family QPs; validation/test/truth never generate structural proposals.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, sleep
from types import SimpleNamespace
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


def read(path):
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for attempt in range(100):
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def canonical(family):
    return tuple(sorted({tuple(sorted(map(int, support))) for support in family}, key=lambda s: (len(s), s)))


def parameters(point):
    rows = sorted(((tuple(sorted(map(int, s))), float(r), float(x))
                   for s, r, x in zip(point["supports"], point["r"], point["x"])),
                  key=lambda row: (len(row[0]), row[0]))
    if len(rows) != len(point["supports"]) or len({row[0] for row in rows}) != len(rows):
        raise ValueError("Malformed or duplicate support coefficients")
    if len(point["r"]) != len(rows) or len(point["x"]) != len(rows):
        raise ValueError("Support/coefficient length mismatch")
    if any(not math.isfinite(v) or v < -1e-7 or v > 2. + 1e-7 for row in rows for v in row[1:]):
        raise ValueError("Invalid frozen response coefficient")
    return dict(supports=[list(row[0]) for row in rows],
                r=[row[1] for row in rows], x=[row[2] for row in rows])


def close(left, right, *, atol=1e-12, rtol=1e-9, message="Numerical replay mismatch"):
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        raise ValueError(message)
    if abs(float(left) - float(right)) > atol + rtol * max(abs(float(left)), abs(float(right))):
        raise ValueError(f"{message}: {left} != {right}")


def load_observations(path, *, slots):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(int).tolist()
        p, q, v = [np.asarray(archive[key], float).reshape(-1, len(labels)) for key in ("P", "Q", "V")]
        root = np.asarray(archive["root"], float).reshape(-1)
    if p.shape != q.shape or p.shape != v.shape or p.shape[0] != slots or len(root) != slots:
        raise ValueError("Every original continuous-day observation must be retained")
    y = root[:, None] ** 2 - v ** 2
    if not all(np.isfinite(a).all() for a in (p, q, y)):
        raise ValueError("Nonfinite observed channel")
    return SimpleNamespace(labels=tuple(labels), n=len(labels), p=(p,), q=(q,),
                           target=(y,), observation_count=y.size)


def matrices(point, n):
    import numpy as np
    point = parameters(point)
    z = np.asarray([[i in support for i in range(n)] for support in point["supports"]], float)
    if z.size == 0:
        return np.zeros((n, n)), np.zeros((n, n))
    return (z.T * np.asarray(point["r"])) @ z, (z.T * np.asarray(point["x"])) @ z


def score(point, data):
    import numpy as np
    R, X = matrices(point, data.n)
    residual = data.target[0] - data.p[0] @ R.T - data.q[0] @ X.T
    day_mae = [float(np.mean(np.abs(part))) for part in np.split(residual, len(residual) // 96)]
    day_mse = [float(np.mean(part ** 2)) for part in np.split(residual, len(residual) // 96)]
    return dict(mae=float(np.mean(np.abs(residual))), mse=float(np.mean(residual ** 2)),
                day_mae=day_mae, day_mse=day_mse)


def count(point, n):
    return sum(1 < len(support) < n for support in point["supports"])


def run(payload_file):
    started = perf_counter()
    payload_file = Path(payload_file).resolve()
    p = read(payload_file)
    target = Path(p["result"]).resolve()
    if target.exists():
        raise FileExistsError("A completed replay must not be overwritten")
    parent = Path(p["parent_output"]).resolve()
    source = parent / "source"
    protocol_file = Path(p["protocol_file"]).resolve()
    if sha(protocol_file) != p["protocol_sha256"]:
        raise ValueError("The tuning protocol changed")
    protocol = read(protocol_file)
    if sha(__file__) != p["worker_sha256"] or sha(p["gate_file"]) != p["gate_sha256"]:
        raise ValueError("A frozen replay controller changed")
    parent_protocol_file = parent / "protocol.json"
    parent_protocol = read(parent_protocol_file)
    if sha(parent_protocol_file) != p["parent_protocol_sha256"]:
        raise ValueError("The original MIQP protocol changed")
    if p["source_hashes"] != parent_protocol["source_hashes"]:
        raise ValueError("The original source manifest changed")
    for name, expected in {**parent_protocol["source_hashes"], **parent_protocol["source_artifact_hashes"]}.items():
        file = (source / name).resolve()
        if not file.is_relative_to(source) or sha(file) != expected:
            raise ValueError(f"An immutable native source changed: {name}")
    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
    if sha(parent / "catalog.json") != parent_protocol["catalog_sha256"]:
        raise ValueError("The original input catalog changed")
    catalog = read(parent / "catalog.json")
    meta = catalog[p["job"]]
    if p["method"] != "T003" or meta["data_error"]:
        raise ValueError("This replay is restricted to the frozen T003 48-input arm")
    result_file = parent / "jobs" / p["job"] / p["method"] / "result.json"
    if sha(result_file) != protocol["parent_result_sha256"][p["job"]]:
        raise ValueError("A frozen parent result changed after the replay protocol was written")
    if sha(parent / "analysis" / "verification.json") != protocol["parent_verification_sha256"]:
        raise ValueError("The parent independent verification changed")
    original = read(result_file)
    common_file = parent / "common" / p["job"] / "common.json"
    common = read(common_file)
    if original["protocol_sha256"] != p["parent_protocol_sha256"] or sha(common_file) != original["common_sha256"]:
        raise ValueError("The native result/shared preprocessing provenance changed")
    if common["protocol_sha256"] != p["parent_protocol_sha256"]:
        raise ValueError("The native QP common has another protocol")
    input_hashes = {split: sha(meta[split]) for split in ("train", "validation")}
    if input_hashes != original["input_hashes"] or any(input_hashes[s] != parent_protocol["input_hashes"][p["job"]][s] for s in input_hashes):
        raise ValueError("Original observations changed")
    if sha(result_file.with_name("candidates.json")) != original["candidate_export_sha256"]:
        raise ValueError("The original model export changed")
    spec = importlib.util.spec_from_file_location("frozen_mse_exit_gate", Path(p["gate_file"]).resolve())
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    if tuple(gate.TAU_GRID) != (0., .0001, .001, .003, .01):
        raise ValueError("The predeclared tau grid changed")
    train = load_observations(meta["train"], slots=96)
    validation = load_observations(meta["validation"], slots=288)
    if train.labels != validation.labels or list(train.labels) != common["labels"]:
        raise ValueError("Observed terminal labels changed")
    K = float(original["normalization"]["K"])
    f0_mse = float(original["normalization"]["f0_mse"])
    close(K * f0_mse, original["reference"], atol=1e-12, message="Frozen penalty reference")
    n = train.n
    models, model_lookup = [], {}

    def add_model(point, origin):
        point = parameters(point)
        fingerprint = digest(point)
        if fingerprint in model_lookup:
            index = model_lookup[fingerprint]
            if origin not in models[index]["origins"]:
                models[index]["origins"].append(origin)
            return models[index]["id"]
        training, validating = score(point, train), score(point, validation)
        R, X = matrices(point, n)
        index = len(models)
        model = dict(
            id=f"model_{index:03d}", point={**point, "objective": K * training["mse"]},
            parameter_sha256=fingerprint, family_sha256=digest(point["supports"]),
            R=R.tolist(), X=X.tolist(), internal=count(point, n),
            train_mae=training["mae"], train_mse=training["mse"],
            validation_mae=validating["mae"], validation_mse=validating["mse"],
            validation_scaled_mse=K * validating["mse"],
            validation_day_mae=validating["day_mae"], validation_day_mse=validating["day_mse"],
            origins=[origin])
        model_lookup[fingerprint] = index; models.append(model)
        return model["id"]

    def get_model(model_id):
        return models[int(model_id.split("_")[-1])]

    trajectory, exit_record = original["trajectory"], original["exit_record"]
    original_path = trajectory["path"]
    final_zero = any(z.get("stage") == "final_exit" and z.get("accepted") for z in trajectory["zero_events"])
    suffix_count = int(bool(exit_record["accepted"])) + int(final_zero)
    prefix_path = original_path[:-suffix_count] if suffix_count else list(original_path)
    if not prefix_path:
        raise ValueError("An original forward path is required")
    before, raw = prefix_path[-1], exit_record["raw_point"]
    before_score = K * score(before, train)["mse"] + .003 * original["reference"] * count(before, n)
    close(before_score, exit_record["before_score"], atol=1e-10, message="Exact E-before path restoration")
    close(K * score(raw, train)["mse"], raw["objective"], atol=1e-10, message="Frozen raw MSE")
    # A native solve started only after its construction/time guard succeeded.
    time_eligible = bool(original["exit_event"].get("solved"))
    feasibility_eligible = bool(exit_record["audit"].get("valid"))
    if not time_eligible or not feasibility_eligible:
        raise ValueError("This replay needs a stored, feasible final-E solve with a satisfied time guard")
    baseline_rows = original["output_candidates"]
    if len(baseline_rows) != len(original_path) + 1:
        raise ValueError("The native output library must contain every path point and the singleton model")
    raw_to_output, fitted_by_family = {}, {}
    original_path_ids = []
    for index, point in enumerate(original_path):
        model_id = add_model(point, dict(source="parent_path", index=index))
        original_path_ids.append(model_id)
        fitted_by_family.setdefault(canonical(point["supports"]), deepcopy(point))
    baseline_library = []
    for index, row in enumerate(baseline_rows):
        if int(row["index"]) != index:
            raise ValueError("The native library indices changed")
        model_id = add_model(row["point"], dict(source="parent_output", index=index))
        model = get_model(model_id)
        close(model["validation_mae"], row["validation_mae"], message="Native library MAE")
        close(model["validation_mse"], row["validation_mse"], atol=1e-15, message="Native library MSE")
        fitted_by_family.setdefault(canonical(row["point"]["supports"]), deepcopy(row["point"]))
        raw_id = original_path_ids[index] if index < len(original_path) else model_id
        baseline_library.append(dict(index=index, model_id=model_id, source_model_id=raw_id,
                                     internal=model["internal"]))
        if index < len(original_path):
            fingerprint = digest(parameters(original_path[index]))
            previous = raw_to_output.setdefault(fingerprint, model_id)
            if previous != model_id:
                raise ValueError("Identical native parameters produced inconsistent cached output refits")
    star = deepcopy(baseline_rows[-1]["point"])
    star_id = baseline_library[-1]["model_id"]
    before_raw_id = add_model(before, dict(source="before_E"))
    before_endpoint_id = baseline_library[len(prefix_path)-1]["model_id"]
    raw_id = add_model(raw, dict(source="frozen_E_proposal"))
    # The cache contains actual stored QP points, never L1 or raw MIQP weights.
    if exit_record["accepted"]:
        fitted_by_family.setdefault(canonical(exit_record["point"]["supports"]), deepcopy(exit_record["point"]))
    new_fit_audits, qp_cache_reuses = [], []
    native_bank = None
    overlay = None
    source_receipts = {}

    class ReplayBank:
        data = train

        def fit(self, family, *, preserve=False):
            nonlocal native_bank, overlay
            family = canonical(family)
            if family in fitted_by_family:
                point = deepcopy(fitted_by_family[family])
                qp_cache_reuses.append(dict(family=[list(s) for s in family], preserve=bool(preserve)))
            else:
                if len(new_fit_audits) >= 64:
                    raise RuntimeError("The replay exceeded its 64-new-family QP guard")
                if native_bank is None:
                    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
                    backend = importlib.import_module("research_experiments.miqp_loss_benchmark_20261111.squared_backend")
                    overlay = backend.install(K)
                    for row in overlay["native_module_sources"].values():
                        file = Path(row["path"]).resolve()
                        relative = file.relative_to(source).as_posix()
                        if row["sha256"] != parent_protocol["source_hashes"][relative]:
                            raise ValueError("A QP backend alias escaped the immutable source manifest")
                    native_bank = backend.SquaredBank(train, common["pool"], common["calibration"], seconds=300.)
                before_solves = len(native_bank.lp.solves)
                point = native_bank.fit(family, preserve=True)
                fitted_by_family[family] = deepcopy(point)
                model_id = add_model(point, dict(source="new_QP", requested_family=[list(s) for s in family]))
                audits = native_bank.lp.solves[before_solves:]
                if len(audits) != 1:
                    raise ValueError("A new replay family must have exactly one fresh native QP audit")
                new_fit_audits.append(dict(**deepcopy(audits[0]), requested_family=[list(s) for s in family],
                                          preserve=True, returned_model_id=model_id))
            if preserve:
                return deepcopy(point)
            keep = [i for i, (s, r, x) in enumerate(zip(point["supports"], point["r"], point["x"]))
                    if len(s) == 1 or max(r, x) > 1e-9]
            point = {**point, "supports": [point["supports"][i] for i in keep],
                     "r": [point["r"][i] for i in keep], "x": [point["x"][i] for i in keep]}
            point["objective"] = K * score(point, train)["mse"]
            return point

    bank = ReplayBank()

    def output_for(point):
        fingerprint = digest(parameters(point))
        if fingerprint in raw_to_output:
            return raw_to_output[fingerprint]
        calibration = common["calibration"]
        family = [s for s, r, x in zip(point["supports"], point["r"], point["x"])
                  if len(s) == 1 or calibration["a"]*r + calibration["b"]*x + 1e-10 >= .24*calibration["median_root_depth"]]
        fitted = bank.fit(family)
        model_id = add_model(fitted, dict(source="new_output_refit_024", raw_parameter_sha256=fingerprint))
        raw_to_output[fingerprint] = model_id
        return model_id

    def select(library, metric):
        key = "validation_mae" if metric == "mae" else "validation_scaled_mse"
        best = min(library, key=lambda row: (get_model(row["model_id"])[key], row["internal"], row["index"]))
        best_value = get_model(best["model_id"])[key]
        eligible = [row for row in library if get_model(row["model_id"])[key] <= best_value + 1e-12]
        chosen = min(eligible, key=lambda row: (row["internal"], get_model(row["model_id"])[key], row["index"]))
        model = get_model(chosen["model_id"])
        return dict(**chosen, rule="strict_mae" if metric == "mae" else "strict_scaled_mse",
                    validation_mae=model["validation_mae"], validation_mse=model["validation_mse"],
                    validation_scaled_mse=model["validation_scaled_mse"], eligible_count=len(eligible),
                    best_validation=best_value)

    before_mse = get_model(before_raw_id)["train_mse"]
    raw_mse = get_model(raw_id)["train_mse"]
    old_gate = gate.old_regularized_gate(before_mse, raw_mse, count(before, n), count(raw, n),
                                         K=K, f0_mse=f0_mse, alpha=.003)
    if bool(old_gate["accepted"]) != bool(exit_record["accepted"]):
        raise ValueError("The original regularized gate did not replay exactly")
    variants = {}
    try:
        specifications = [("old_J", None)] + [(f"tau_{tau:g}", float(tau)) for tau in gate.TAU_GRID]
        for variant_id, tau in specifications:
            numeric_gate = old_gate if tau is None else gate.pure_mse_gate(before_mse, raw_mse, K=K, tau=tau)
            accepted = bool(numeric_gate["accepted"] and time_eligible and feasibility_eligible)
            cleanup_events, removed = [], []
            if accepted == bool(exit_record["accepted"]):
                path = list(original_path)
            else:
                path = list(prefix_path)
                if accepted:
                    fitted = bank.fit(raw["supports"], preserve=True)
                    # Match exit_repair's first cleanup before its path append.
                    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
                    revocable = importlib.import_module("research_experiments.revocable.engine")
                    fitted, removed = revocable.clean_zero(bank, fitted)
                    path.append(fitted)
                else:
                    fitted = before
                # Match finish's second cleanup: one appended final point if any
                # of its repeated numerical-zero refits was accepted.
                zero = importlib.import_module("research_experiments.iterative_exit.zero")
                final, cleanup_events = zero.release_zero_atoms(bank, fitted, stage="final_exit")
                if any(event["accepted"] for event in cleanup_events):
                    path.append(final)
            path_ids = [add_model(point, dict(source="variant_path", variant=variant_id, index=index))
                        for index, point in enumerate(path)]
            if variant_id == "old_J":
                library = deepcopy(baseline_library)
            else:
                library = [dict(index=index, model_id=output_for(point), source_model_id=path_ids[index],
                                internal=get_model(output_for(point))["internal"])
                           for index, point in enumerate(path)]
                library.append(dict(index=len(path), model_id=star_id, source_model_id=star_id,
                                    internal=get_model(star_id)["internal"]))
            endpoint_id = library[len(path)-1]["model_id"]
            before_val = get_model(before_endpoint_id)["validation_mse"]
            endpoint_val = get_model(endpoint_id)["validation_mse"]
            if before_val <= 0.:
                raise ValueError("Endpoint ratio needs a positive observed validation MSE")
            variants[variant_id] = dict(
                gate_mode="regularized" if tau is None else "pure_mse", tau=.001 if tau is None else tau,
                gate={**numeric_gate, "time_eligible": time_eligible, "feasibility_eligible": feasibility_eligible},
                accepted=accepted, prefix_length=len(prefix_path), path_model_ids=path_ids,
                path_length=len(path), library=library, selected_mae=select(library, "mae"),
                selected_mse=select(library, "mse"), endpoint_raw_model_id=path_ids[-1], endpoint_model_id=endpoint_id,
                endpoint_validation_mse=endpoint_val, endpoint_relative_to_before=endpoint_val/before_val,
                zero_removed_inside_exit=removed, zero_events=cleanup_events)
        old = variants["old_J"]
        for name, key in (("selected", "selected_mae"), ("selected_mse_view", "selected_mse")):
            selected, expected = old[key], original[name]
            if int(selected["index"]) != int(expected["index"]):
                raise ValueError(f"The baseline {key} selected another path index")
            if get_model(selected["model_id"])["parameter_sha256"] != digest(parameters(expected["point"])):
                raise ValueError(f"The baseline {key} selected another fitted model")
            close(selected["validation_mae"], expected["validation_mae"], message="Selected baseline MAE")
            close(selected["validation_mse"], expected["validation_mse"], atol=1e-15, message="Selected baseline MSE")
        if old["path_length"] != len(original_path) or old["path_model_ids"] != original_path_ids:
            raise ValueError("The full baseline raw path did not replay")
        for name, module in sorted(sys.modules.items()):
            if not (name == "research_experiments" or name.startswith("research_experiments.") or
                    name == "rnj_wzzt" or name.startswith("rnj_wzzt.")):
                continue
            filename = getattr(module, "__file__", None)
            if filename is None:
                continue
            file = Path(filename).resolve()
            relative = file.relative_to(source).as_posix()
            measured = sha(file)
            if measured != parent_protocol["source_hashes"].get(relative):
                raise ValueError(f"A loaded native module differs from frozen source: {name}")
            source_receipts[name] = dict(file=str(file), relative_path=relative, sha256=measured)
        result = dict(
            schema="mse_exit_gate_replay_v1", job=p["job"], method=p["method"],
            models=models, variants=variants, before_raw_model_id=before_raw_id,
            before_endpoint_model_id=before_endpoint_id, raw_model_id=raw_id,
            before_endpoint_validation_mse=get_model(before_endpoint_id)["validation_mse"],
            original_exit_accepted=bool(exit_record["accepted"]), normalization=deepcopy(original["normalization"]),
            fixed_parameters=dict(proposal_alpha=.003, output_resolution=.24, QP_seconds=300.,
                                  maximum_new_QP_families=64, zero_weight_tolerance=1e-9,
                                  zero_loss_tolerance_internal=1e-10),
            baseline_replay=dict(passed=True, path_length=old["path_length"], library_rows=len(old["library"]),
                                 selected_mae_exact_parameters=True, selected_mse_exact_parameters=True,
                                 all_library_rows_scores_replayed=True),
            new_fit_audits=new_fit_audits, qp_cache_reuses=qp_cache_reuses,
            new_QP_calls=len(new_fit_audits), new_MIQP_calls=0, runtime_overlay=overlay,
            loaded_native_sources=source_receipts, native_source_modified=False,
            provenance=dict(
                protocol_file=str(protocol_file), protocol_sha256=p["protocol_sha256"],
                parent_output=str(parent), parent_protocol_sha256=sha(parent_protocol_file),
                parent_result_file=str(result_file), parent_result_sha256=sha(result_file),
                parent_common_file=str(common_file), parent_common_sha256=sha(common_file),
                parent_candidates_sha256=sha(result_file.with_name("candidates.json")),
                source_hashes=p["source_hashes"], input_hashes=input_hashes,
                gate_file=str(Path(p["gate_file"]).resolve()), gate_sha256=sha(p["gate_file"]),
                worker_file=str(Path(__file__).resolve()), worker_sha256=sha(__file__),
                payload_file=str(payload_file), payload_sha256=sha(payload_file),
                controller_hashes=p.get("controller_hashes", {})),
            timing=dict(replay_seconds=perf_counter()-started,
                        new_QP_native_seconds=sum(row["seconds"] for row in new_fit_audits),
                        reused_parent_successful_seconds=float(original["total_seconds"]),
                        reused_parent_operational_seconds=float(original.get("operational_total_seconds", original["total_seconds"]))),
            truth_opened=False, test_opened=False, validation_used_for_gate=False,
            completed_utc=datetime.now(timezone.utc).isoformat())
        write(target, result)
        print("REPLAY_COMPLETE", p["job"], "newQP", len(new_fit_audits), flush=True)
    finally:
        if native_bank is not None:
            native_bank.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    try:
        run(args.payload)
    except Exception:
        p = read(args.payload)
        if not Path(p["result"]).exists():
            write(Path(p["result"]).with_name("error.json"), dict(stage="replay", traceback=traceback.format_exc()))
        raise
