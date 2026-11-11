"""Independent post-seal replay: NumPy/stdlib only; no native imports or solves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(first, second, message):
    require(np.isclose(first, second, rtol=1e-8, atol=1e-15), message)


def inside(root, name):
    path = (root/name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root), "Path escapes frozen output")
    return path


def load96(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"P", "Q", "V", "root", "labels"}, "Unexpected sole-input fields")
        labels = tuple(map(int, archive["labels"])); n = len(labels)
        p, q, v = [np.asarray(archive[k], float).reshape(96, n) for k in ("P", "Q", "V")]
        y = np.asarray(archive["root"], float).reshape(96, 1)**2-v**2
    require(all(np.isfinite(a).all() for a in (p, q, y)), "Nonfinite sole observations")
    return labels, (p[:64], q[:64], y[:64]), (p[64:], q[64:], y[64:])


def fingerprint(labels, rows):
    digest = hashlib.sha256(repr(labels).encode())
    for a in rows:
        a = np.ascontiguousarray(a, dtype=np.float64); digest.update(repr(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def replay(rows, point):
    p, q, y = rows; n = p.shape[1]; R, X = np.zeros((n, n)), np.zeros((n, n))
    require(len(point["supports"]) == len(point["r"]) == len(point["x"]), "Coefficient dimensions differ")
    family = [frozenset(C) for C in point["supports"]]
    require(len(set(family)) == len(family) and all(C and C <= set(range(n)) for C in family), "Invalid supports")
    require(all(not C & D or C <= D or D <= C for C in family for D in family), "Crossing model")
    for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True):
        require(np.isfinite(r+x) and min(r, x) >= 0, "Invalid response coefficients")
        z = np.zeros(n); z[C] = 1; outer = np.outer(z, z); R += r*outer; X += x*outer
    return float(np.mean((y-p@R.T-q@X.T)**2))


def identity(point, n, zero):
    return tuple(sorted(tuple(sorted(C)) for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
                        if 1 < len(C) < n and max(r, x) > zero))


def count(point, n):
    return sum(1 < len(C) < n for C in point["supports"])


def model_signature(point):
    return signature({key: point[key] for key in ("supports", "r", "x")})


def choose(rows, tie):
    best = min(row["selection_scaled_mse"] for row in rows)
    return min((row for row in rows if row["selection_scaled_mse"] <= best+tie),
               key=lambda row: (row["internal"], row["selection_scaled_mse"], row["index"]))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def analyze(output):
    out = Path(output).resolve(); runtime = out/"runtime"
    complete, seal, protocol = [read(out/name) for name in ("completion.json", "selection_seal.json", "protocol.json")]
    require(complete["all_planned_terminal"] and seal["sealed"] and seal["all_planned_terminal"], "Batch is not sealed and complete")
    require(complete["selection_seal_sha256"] == sha(out/"selection_seal.json"), "Completion seal SHA mismatch")
    require(seal["protocol_sha256"] == sha(out/"protocol.json") == sha(runtime/"protocol.json"), "Protocol SHA mismatch")
    require(protocol["split"] == dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), "Unexpected data split")
    require(len(protocol["inputs"]) == 4 and protocol["kappas"] == [0., .001, .003], "Unexpected planned cohort")
    tracked = {"protocol.json": sha(out/"protocol.json"), "selection_seal.json": sha(out/"selection_seal.json")}
    tracked["completion.json"] = sha(out/"completion.json")
    require(seal["source_sha256"] == protocol["source_sha256"], "Seal source mismatch")
    for name, digest in protocol["source_sha256"].items():
        require(sha(inside(runtime, name)) == digest, "Frozen source SHA mismatch: "+name); tracked["runtime/"+name] = digest
    planned = {f"runs/{i}/{k}.json" for i in protocol["inputs"] for k in protocol["kappas"]}
    require(set(seal["result_sha256"]) == planned, "Sealed method denominator changed")
    for name, digest in seal["result_sha256"].items():
        require(sha(inside(out, name)) == digest, "Result SHA mismatch: "+name); tracked[name] = digest
    summary, candidates, decisions, local_mapping, exact_models = [], [], [], {}, {}
    for input_id, info in protocol["inputs"].items():
        path = inside(runtime, info["path"]); require(sha(path) == info["sha256"], "Sole input SHA mismatch")
        tracked["runtime/"+info["path"]] = info["sha256"]
        labels, train, selection = load96(path); n = len(labels)
        require(list(labels) == info["labels"], "Input labels changed")
        for rows, key in ((train, "training_sha256"), (selection, "selection_sha256")):
            require(fingerprint(labels, rows) == info[key], "Split fingerprint mismatch")
        local, signatures = {}, {}
        for kappa in protocol["kappas"]:
            record = read(out/f"runs/{input_id}/{kappa}.json"); cfg = dict(protocol["config"], relative_gain=kappa)
            status = record["status"]; require(status in {"fitted", "failed", "budget_exceeded", "interrupted"}, "Nonterminal method")
            row = dict(input=input_id, kappa=kappa, status=status, configured_seconds=cfg["total_seconds"],
                       total_seconds=record.get("total_seconds"), common_seconds=record.get("common_seconds"), fresh_seconds=record.get("fresh_seconds"))
            summary.append(row)
            if status != "fitted":
                continue
            require(record["config"] == cfg and record["source_sha256_before"] == record["source_sha256_after"] == protocol["source_sha256"], "Method source/config changed")
            require(record["input_sha256"] == info["sha256"] and record["protocol_sha256"] == seal["protocol_sha256"], "Method provenance changed")
            result = record["result"]; reference = float(result["reference_mse"]); tie, zero, alpha = [cfg[k] for k in ("validation_tie", "zero_tol", "alpha")]
            require(reference > 0 and np.isfinite(reference) and result["selection_metric"] == "validation_mse", "Invalid selection/reference")
            require(not result["total_cap_exceeded"] and record["total_seconds"] <= cfg["total_seconds"]+.05, "Fitted method exceeds budget")
            bank = []
            for index, candidate in enumerate(result["frozen_candidates"]):
                point = candidate["point"]; train_mse, val_mse = replay(train, point), replay(selection, point)
                require(candidate["index"] == index and candidate["labels"] == list(labels), "Frozen model order changed")
                close(train_mse/reference, point["objective"], "Training fit does not replay"); close(train_mse, point["mse"], "Physical training MSE changed")
                close(val_mse, candidate["validation_mse"], "Selection MSE changed"); close(val_mse/reference, candidate["validation_scaled_mse"], "Scaled selection MSE changed")
                model = dict(input=input_id, kappa=kappa, index=index, internal=count(point, n), train_mse=train_mse,
                    selection_mse=val_mse, selection_scaled_mse=val_mse/reference, model_sha256=model_signature(point), topology_sha256=signature(identity(point, n, zero)))
                bank.append(model)
            primary = choose(bank, tie); alternatives = [c for c in bank if c["topology_sha256"] != primary["topology_sha256"]]
            secondary = choose(alternatives, tie) if alternatives else None; saved = result["competition"]
            require(primary["index"] == result["selected"]["index"] == saved["primary_index"], "Primary does not follow MSE rule")
            require((secondary["index"] if secondary else None) == saved["secondary_index"], "Secondary differs from best distinct active family")
            require(model_signature(result["selected"]["point"]) == primary["model_sha256"], "Selected point differs from indexed model")
            for model in bank:
                model.update(primary=model is primary, secondary=model is secondary); candidates.append(model)
            local[str(kappa)] = primary["selection_scaled_mse"]
            signatures[str(kappa)] = dict(model=primary["model_sha256"], path=signature([model_signature(p) for p in result["path"]]),
                decisions=signature([(e["stage"], e["accepted"], model_signature(e["point"]) if "point" in e else None) for e in result["events"]]))
            ledger = result["MIQP"]; row.update(selected_index=primary["index"], selected_model_sha256=primary["model_sha256"],
                selected_topology_sha256=primary["topology_sha256"], selection_mse=primary["selection_mse"], native_calls=len(ledger),
                native_nodes=sum(r.get("nodes", 0) for r in ledger), native_seconds=sum(r.get("seconds", 0) for r in ledger),
                time_limits=sum(r.get("status") == 9 for r in ledger), bound_false=sum(r.get("bound_consistent_with_fixed_family_qp") is False for r in ledger),
                bound_missing=sum("bound_consistent_with_fixed_family_qp" not in r for r in ledger), qp_calls=len(result["QP"]),
                stop_reason=result["stop_reason"], relative_gate_rejections=0, selected_by_sealed_policy=seal["selected_kappa_by_input"][input_id] == kappa)
            current = result["path"][0]
            for number, event in enumerate(result["events"]):
                decision = dict(input=input_id, kappa=kappa, event=number, stage=event["stage"], accepted=event["accepted"])
                native = event.get("solver", {}); decision.update({"native_"+k: native.get(k) for k in ("status", "objective", "bound", "gap", "nodes", "seconds", "bound_consistent_with_fixed_family_qp")})
                if "point" in event:
                    proposal = event["point"]; old_mse, new_mse = replay(train, current), replay(train, proposal)
                    delta_k = count(proposal, n)-count(current, n); gain = (old_mse-new_mse)/reference-alpha*delta_k
                    before = current["objective"]+alpha*count(current, n); after = proposal["objective"]+alpha*count(proposal, n)
                    old = {tuple(C): max(r, x) for C, r, x in zip(current["supports"], current["r"], current["x"], strict=True)}
                    new = set(map(tuple, proposal["supports"])); removed = set(old)-new
                    zero_only = bool(removed) and new <= set(old) and all(old[C] <= zero for C in removed)
                    zero_only = zero_only and identity(current, n, zero) == identity(proposal, n, zero) and proposal["objective"] <= current["objective"]+cfg["gain_tol"]
                    required = cfg["gain_tol"] if zero_only else max(cfg["gain_tol"], kappa*before)
                    close(required, event["required_gain"], "Relative gain floor changed")
                    require(event["acceptance_rule"] == ("numerical_zero_cleanup" if zero_only else "relative_regularized_training_gain"), "Cleanup bypass changed")
                    close(before, event["before_score"], "Event old score changed"); close(after, event["after_score"], "Event new score changed")
                    close(before-after, event["gain"], "Event gain changed")
                    expected = before-after > event["required_gain"] and not event.get("repeated_family", False)
                    require(bool(event["accepted"]) == expected, "Event gate changed")
                    row["relative_gate_rejections"] += int(not zero_only and cfg["gain_tol"] < before-after <= required)
                    decision.update(fit_gain_mse=old_mse-new_mse, fit_gain_normalized=(old_mse-new_mse)/reference, delta_internal=delta_k,
                        count_gain=-alpha*delta_k, replayed_J_gain=gain, recorded_J_gain=event["gain"], required_gain=event["required_gain"], acceptance_rule=event["acceptance_rule"])
                    if event["accepted"]:
                        current = proposal
                decisions.append(decision)
        require(all(np.isclose(local[k], seal["normalized_selection_mse_by_input"][input_id][k], rtol=1e-8, atol=1e-15) for k in local), "Sealed local MSE changed")
        best = min(local.values()) if local else None
        local_mapping[input_id] = min((float(k) for k, v in local.items() if v <= best+protocol["config"]["validation_tie"]), default=None)
        exact_models[input_id] = {key+"_sha256_by_kappa": {k: v[key] for k, v in signatures.items()} for key in ("model", "path", "decisions")}
        exact_models[input_id].update({"all_"+key+"_identical": len({v[key] for v in signatures.values()}) == 1 for key in ("model", "path", "decisions")})
        for row in summary:
            if row["input"] == input_id and str(row["kappa"]) in signatures:
                row.update({"exact_"+key+"_same_as_kappa0": signatures[str(row["kappa"])][key] == signatures.get("0.0", {}).get(key) for key in ("model", "path", "decisions")})
    require(local_mapping == seal["selected_kappa_by_input"], "Per-input sealed kappa rule changed")
    grade = read(out/"grade.json") if (out/"grade.json").exists() else None
    if grade:
        for row in summary:
            matched = next((g for g in grade["rows"] if g["input"] == row["input"] and g["kappa"] == row["kappa"]), {})
            row.update({k: matched.get(k) for k in ("F1", "exact", "top2_contains_true")})
    for name, digest in tracked.items():
        require(sha(out/name) == digest, "Frozen file changed during replay: "+name)
    folder = out/"analysis"; folder.mkdir(exist_ok=True); source = Path(__file__).resolve(); digest = sha(source)
    archived = folder/"source"/digest/source.name; archived.parent.mkdir(parents=True, exist_ok=True); archived.write_bytes(source.read_bytes())
    verification = dict(verified=True, created_utc=datetime.now(timezone.utc).isoformat(), planned=len(planned),
        fitted=sum(r["status"] == "fitted" for r in summary), candidates=len(candidates), decisions=len(decisions),
        frozen_file_sha256=tracked, analysis_source_sha256=digest, archived_source=archived.relative_to(out).as_posix(),
        selected_kappa_by_input=local_mapping, exact_selected_model_comparison=exact_models,
        calendar_seconds=complete["calendar_seconds"], logical_total_seconds=sum(r["total_seconds"] or 0 for r in summary),
        chosen_fits_logical_seconds=sum(r["total_seconds"] or 0 for r in summary if r.get("selected_by_sealed_policy")),
        physical_common_once_seconds=sum(r["fresh_seconds"] or 0 for r in summary)+sum(next(r["common_seconds"] for r in summary if r["input"] == i) or 0 for i in protocol["inputs"]),
        grade_included=grade is not None, grade_sha256=sha(out/"grade.json") if grade else None,
        scope="finite frozen library replay; four sole 96-point NPZ; no native imports, solves, truth, extra validation or test files")
    for name, rows in (("summary.csv", summary), ("candidates.csv", candidates), ("decisions.csv", decisions)):
        write_csv(folder/name, rows)
    (folder/"verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output", type=Path)
    result = analyze(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ("verified", "planned", "fitted", "candidates", "decisions", "analysis_source_sha256")}))
