"""Independent replay of saved local R/X evidence; no fitting or new samples.

Truth is read only in this evaluation tool, never forwarded to an estimator.
Every configured slot remains in the audit, including scientific failures and
missing files.  Numerical comparisons are floating-point checks, not interval
certificates or a proof of global optimality.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from .. import likelihood as lk
from ..reference_benchmark.representation import physical_tree_ac


def _read(path):
    p = Path(path)
    body = p.read_bytes()
    return json.loads(gzip.decompress(body) if p.suffix == ".gz" else body)


def _write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def clean(item):
        if isinstance(item, dict):
            return {k:clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item
    p.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def _key(spec):
    return f"{spec['network']}_{spec['regime']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def _base(spec):
    return spec["network"], spec["regime"], int(spec["seed"])


def _expected_specs():
    return [dict(network=network, regime=regime, seed=seed, N=n, boundary=b)
            for network in ("paper15", "paper11") for regime in ("ordinary", "correlated")
            for seed in (31001, 31002, 31003) for n in (96, 288) for b in ("K", "U")]


def _issue(issues, code, path, detail, severity="error"):
    issues.append(dict(code=code, path=str(path), detail=str(detail), severity=severity))


def _close(a, b, *, atol=1e-8, rtol=1e-9):
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= atol+rtol*max(abs(a), abs(b)))


def _truth_leaks(value, path="inference"):
    leaks = []
    if isinstance(value, dict):
        for key, child in value.items():
            lower = str(key).lower()
            here = f"{path}.{key}"
            permitted_flag = lower in ("truth_used", "truth_seen", "truth_inputs_to_inference") and child is False
            if (("truth" in lower and not permitted_flag)
                    or lower in ("true_r", "true_x", "true_p", "true_q", "true_v", "true_root_v")):
                leaks.append(here)
            leaks.extend(_truth_leaks(child, here))
    elif isinstance(value, list):
        for j, child in enumerate(value):
            leaks.extend(_truth_leaks(child, f"{path}[{j}]"))
    return leaks


def _candidate(name):
    supports = {"star": (), "pair01": ((0, 1),),
                "pair02": ((0, 2),), "pair12": ((1, 2),)}
    return lk.RadialCandidate.from_supports(name, 3, supports[name])


def _audit_fit(candidate, fitted, obs, sigma, domain, issues, path, *,
               root_known=None, base_kv=.4, base_kw=200.):
    """Check a saved physical state against meters and an independent solver."""
    required = ("r", "x", "p", "q", "root_v", "loss")
    missing = [k for k in required if k not in fitted]
    row = dict(status=fitted.get("status", "unknown"), qualified=bool(fitted.get("qualified", False)))
    if missing:
        failed = fitted.get("status") == "failed" and not fitted.get("qualified")
        row.update(replayed=False, missing=missing, failure_preserved=failed)
        _issue(issues, "fit_missing_states", path, missing, "warning" if failed else "error")
        return row
    try:
        r, x, p, q, root = (np.asarray(fitted[k], float) for k in ("r", "x", "p", "q", "root_v"))
        response = lk.ac_predict(candidate, r, x, p, q, source_p=0., source_q=0., root_v=root)
        sse = sum(float(np.sum(((response[k]-np.asarray(obs[k]))/np.asarray(sigma[k]))**2))
                  for k in ("p", "q", "v"))
        saved_loss = float(fitted["loss"])
        row.update(replayed=True, raw_sse=sse, saved_loss=saved_loss,
                   raw_sse_absolute_error=abs(sse-saved_loss))
        if not _close(sse, saved_loss):
            _issue(issues, "loss_mismatch", path, f"recomputed={sse:.16g}; saved={saved_loss:.16g}")
        if "v" in fitted:
            delta = float(np.max(abs(response["v"]-np.asarray(fitted["v"], float))))
            row["saved_voltage_max_error"] = delta
            if delta > 2e-10:
                _issue(issues, "saved_voltage_mismatch", path, delta)
        if root_known is not None and np.max(abs(root-np.asarray(root_known))) > 1e-12:
            _issue(issues, "known_boundary_changed", path, np.max(abs(root-np.asarray(root_known))))
        z = np.hypot(r, x)
        ratio = x/r
        all_v = np.c_[root, abs(response["voltage"])]
        cap = np.asarray(domain["power_caps"], float)
        checks = dict(
            impedance_magnitude=bool(np.min(z) >= float(domain["z_min"])-1e-10 and
                                     np.max(z) <= float(domain["z_max"])+1e-10),
            ratio=bool(np.min(ratio) >= float(domain["ratio_min"])-1e-10 and
                       np.max(ratio) <= float(domain["ratio_max"])+1e-10),
            root_range=bool(np.min(root) >= float(domain["root_min"])-1e-10 and
                            np.max(root) <= float(domain["root_max"])+1e-10),
            all_node_voltage=bool(np.min(all_v) >= float(domain["v_min"])-1e-10 and
                                  np.max(all_v) <= float(domain["v_max"])+1e-10),
            apparent_power=bool(np.all(np.hypot(p, q) <= cap+1e-10)))
        for channel, values in (("p", p), ("q", q)):
            if channel+"_bounds" in domain:
                lo, hi = (np.asarray(v, float) for v in domain[channel+"_bounds"])
                checks[channel+"_range"] = bool(np.all(values >= lo-1e-10) and np.all(values <= hi+1e-10))
        row["domain_checks"] = checks
        for label, passed in checks.items():
            if not passed:
                _issue(issues, "physical_domain_violation", path, label)
        zbase = base_kv**2/(base_kw/1000.)
        branches = [dict(from_bus=u, to_bus=v, r_ohm=float(r[e]*zbase), x_ohm=float(x[e]*zbase))
                    for e, (u, v) in enumerate(candidate.edges)]
        independent = physical_tree_ac(branches, tuple(e+1 for e in candidate.terminal_edges),
            0, base_kv, base_kw, p*base_kw, q*base_kw, 1, 0., 0., root)
        delta = float(np.max(abs(independent["v"]-response["v"])))
        equation = max(float(response["ac_equation_residual"]), float(independent["ac_equation_residual"]))
        balance = max(float(response["power_balance_residual"]), float(independent["power_balance_residual"]))
        row.update(independent_voltage_max_error=delta, ac_equation_residual=equation,
                   power_balance_residual=balance, all_node_voltage_range=[float(all_v.min()), float(all_v.max())])
        if delta > 2e-10 or equation > 2e-10 or balance > 2e-10:
            _issue(issues, "independent_ac_replay", path, f"voltage={delta}; equation={equation}; balance={balance}")
        if "information" in fitted and root_known is None and candidate.name == "star":
            rank = int(fitted["information"].get("rank", 2*candidate.m))
            row["unknown_boundary_rank_ceiling"] = 2*candidate.m-2
            if rank > 2*candidate.m-2:
                _issue(issues, "unknown_boundary_rank", path, f"rank={rank}; ceiling={2*candidate.m-2}")
    except Exception as exc:
        row.update(replayed=False, replay_error=f"{type(exc).__name__}: {exc}")
        _issue(issues, "fit_replay_failed", path, row["replay_error"])
    return row


def _bootstrap_recompute(observed, records):
    failed = tail = 0
    for record in records:
        try:
            value = float(record.get("sse", np.nan))
            valid = (record.get("success") is True and record.get("stationary") is True
                     and np.isfinite(value) and value >= 0)
        except (TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            failed += 1
            tail += 1
        else:
            tail += int(value >= observed)
    return dict(pvalue=(1+tail)/(1+len(records)), failed=failed,
                valid=len(records)-failed, replicates=len(records), upper_tail=tail)


def audit_directory(directory, *, destination=None):
    """Audit all configured slots and write audit.json, retaining every issue."""
    started = perf_counter()
    root = Path(directory)
    issues, slots = [], []
    try:
        config = _read(root/"configuration.json")
        tasks = config["tasks"]
    except Exception as exc:
        _issue(issues, "configuration_unreadable", root, f"{type(exc).__name__}: {exc}")
        tasks, config = [], {}
    bases = {_base(s) for s in tasks}
    expected_keys = {_key(s) for s in _expected_specs()}
    actual_keys = [_key(s) for s in tasks]
    pilot = len(tasks) == 2
    if len(set(actual_keys)) != len(actual_keys):
        _issue(issues, "duplicate_slots", root, actual_keys)
    if pilot:
        expected_pilot = {_key(s) for s in _expected_specs()[:2]}
        if set(actual_keys) != expected_pilot or len(bases) != 1:
            _issue(issues, "pilot_slot_matrix", root, actual_keys)
    elif set(actual_keys) != expected_keys or len(tasks) != 48 or len(bases) != 12:
        _issue(issues, "main_slot_matrix", root,
               f"slots={len(tasks)}; bases={len(bases)}; missing={sorted(expected_keys-set(actual_keys))}")
    protocol_path = Path(__file__).with_name("protocol.md")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if config.get("protocol_sha256") != protocol_hash:
        _issue(issues, "protocol_hash_mismatch", root, config.get("protocol_sha256"))
    truth_names, certificate_counts = {}, {}
    for spec in tasks:
        key = _key(spec)
        row = dict(key=key, spec=spec, fits={}, diagnostics={})
        slots.append(row)
        try:
            record_path = root/"results"/(key+".json.gz")
            record = _read(record_path)
            row["result_sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
            row["result_status"] = record.get("status", "unknown")
            if record.get("spec") != spec:
                _issue(issues, "result_spec_mismatch", key, record.get("spec"))
            if record.get("status") != "completed":
                row["failure_preserved"] = True
                row["error"] = record.get("error")
                _issue(issues, "scientific_slot_failed", key, row["error"], "warning")
                continue
            inference_path = root/"inference"/(key+".json.gz")
            inference = _read(inference_path)
            digest = hashlib.sha256(inference_path.read_bytes()).hexdigest()
            row["inference_sha256"] = digest
            if digest != record.get("inference_sha256"):
                _issue(issues, "inference_hash_mismatch", key, record.get("inference_sha256"))
            for leak in _truth_leaks(inference):
                _issue(issues, "truth_field_leak", key, leak)
            if inference.get("spec") != spec:
                _issue(issues, "inference_spec_mismatch", key, inference.get("spec"))
            if spec["boundary"] == "U" and inference.get("root_known") is not None:
                _issue(issues, "unknown_boundary_exposed", key, "root_known is present")
            if spec["boundary"] == "K" and inference.get("root_known") is None:
                _issue(issues, "known_boundary_missing", key, "root_known is absent")
            obs, sigma, domain = (inference[k] for k in ("observed", "sigma", "domain"))
            if set(obs) != {"p", "q", "v"}:
                _issue(issues, "observation_channels", key, list(obs))
            if any(np.shape(obs.get(k)) != (spec["N"], 3) for k in ("p", "q", "v")):
                _issue(issues, "observation_dimensions", key, {k:np.shape(v) for k, v in obs.items()})
            truth_path = root/"truth"/(key+".json.gz")
            truth = _read(truth_path)
            row["truth_sha256"] = hashlib.sha256(truth_path.read_bytes()).hexdigest()
            truth_names[key] = truth["candidate_name"]
            for name in ("star", "pair01", "pair02", "pair12"):
                fitted = inference.get("fits", {}).get(name)
                if fitted is None:
                    _issue(issues, "candidate_missing", key, name)
                    row["fits"][name] = dict(status="missing")
                    continue
                candidate = _candidate(name)
                row["fits"][name] = _audit_fit(candidate, fitted, obs, sigma, domain, issues,
                    f"{key}.fits.{name}", root_known=inference.get("root_known"))
                cert = inference.get("certificates", {}).get(name, {})
                for item in cert.get("alpha_results", []):
                    for method, lower_name in (("raw", "sse_lower"), ("residual", "qp_lower")):
                        decision = item.get(method, {})
                        lower, threshold = decision.get(lower_name), decision.get("threshold")
                        if lower is not None and threshold is not None:
                            expected = float(lower) > float(threshold)
                            if expected != bool(decision.get("reject")):
                                _issue(issues, "certificate_decision_mismatch", f"{key}.{name}", method)
                        count_key = f"{method}_alpha_{item.get('alpha')}"
                        count = certificate_counts.setdefault(count_key,
                            dict(candidate_slots=0, true_candidate_slots=0, false_reject=0, wrong_reject=0,
                                 calculation_failed=0))
                        count["candidate_slots"] += 1
                        count["calculation_failed"] += int(item.get("status") != "completed")
                        is_true = name == truth["candidate_name"]
                        count["true_candidate_slots"] += int(is_true)
                        count["false_reject"] += int(is_true and decision.get("reject", False))
                        count["wrong_reject"] += int(not is_true and decision.get("reject", False))
                if not cert:
                    _issue(issues, "certificate_missing", f"{key}.{name}", "no certificate")
            diagnostics = record.get("diagnostics", {})
            for name, entry in diagnostics.get("candidates", {}).items():
                checks = {}
                decomposition = entry.get("decomposition_holdout")
                if decomposition:
                    subtotal = sum(float(decomposition[k]) for k in ("parameter", "fitted_model_difference", "cross"))
                    total = float(decomposition["total"])
                    checks["decomposition_identity_error"] = abs(total-subtotal)
                    if not _close(total, subtotal, atol=1e-7, rtol=1e-10):
                        _issue(issues, "response_decomposition_identity", f"{key}.{name}", checks)
                clean_obs = {k:np.asarray(truth[k], float)[:spec["N"]] for k in ("p", "q", "v")}
                for label in ("clean_fixed_PQ_fit", "clean_EIV_fit"):
                    if label in entry:
                        checks[label] = _audit_fit(_candidate(name), entry[label], clean_obs, sigma, domain, issues,
                            f"{key}.{name}.{label}", root_known=inference.get("root_known"))
                fixed, eiv = entry.get("clean_fixed_PQ_fit", {}), entry.get("clean_EIV_fit", {})
                if "loss" in fixed and "loss" in eiv:
                    checks["clean_EIV_not_above_fixed_PQ"] = float(eiv["loss"]) <= float(fixed["loss"])+1e-6
                    if not checks["clean_EIV_not_above_fixed_PQ"]:
                        _issue(issues, "clean_distance_inclusion", f"{key}.{name}",
                               f"EIV={eiv['loss']}; fixedPQ={fixed['loss']}")
                row["diagnostics"][name] = checks
        except Exception as exc:
            row["audit_exception"] = f"{type(exc).__name__}: {exc}"
            _issue(issues, "slot_audit_exception", key, row["audit_exception"])
    extra_results = sorted(p.name for p in (root/"results").glob("*.json.gz")
                           if p.name[:-8] not in set(actual_keys))
    if extra_results:
        _issue(issues, "unconfigured_result_files", root, extra_results)
    bootstrap = []
    bootstrap_keys = set()
    for path in sorted((root/"bootstrap").glob("*.json")):
        item_row = dict(file=path.name)
        bootstrap.append(item_row)
        try:
            item = _read(path)
            bootstrap_keys.add((_key(item["spec"]), item.get("candidate")))
            item_row.update(status=item.get("status"), candidate=item.get("candidate"))
            if item.get("status") != "completed":
                item_row["failure_preserved"] = True
                continue
            computed = _bootstrap_recompute(float(item["observed_sse"]), item["records"])
            if len(item["records"]) != 39:
                _issue(issues, "bootstrap_repeat_count", path, len(item["records"]))
            item_row.update(computed)
            decision = item["decision"]
            if not _close(computed["pvalue"], float(decision["pvalue"]), atol=1e-14, rtol=0):
                _issue(issues, "bootstrap_pvalue_mismatch", path, computed)
            if computed["failed"] != int(decision["failed_or_nonstationary_replicates"]):
                _issue(issues, "bootstrap_failure_count_mismatch", path, computed)
            expected_reject = computed["pvalue"] <= .05
            if bool(item.get("reject")) != expected_reject:
                _issue(issues, "bootstrap_decision_mismatch", path, computed)
            key = _key(item["spec"])
            item_row["true_candidate"] = item["candidate"] == truth_names.get(key)
            item_row["false_reject"] = bool(expected_reject and item_row["true_candidate"])
        except Exception as exc:
            _issue(issues, "bootstrap_audit_exception", path, f"{type(exc).__name__}: {exc}")
    expected_bootstrap = {(_key(dict(network=network, regime=regime, seed=31001, N=96, boundary="K")), candidate)
                          for network in ("paper15", "paper11") for regime in ("ordinary", "correlated")
                          for candidate in ("star", "pair01", "pair02", "pair12")}
    if not pilot and bootstrap_keys != expected_bootstrap:
        _issue(issues, "bootstrap_slot_matrix", root,
               f"present={len(bootstrap_keys)}; expected=16; missing={sorted(expected_bootstrap-bootstrap_keys)}")
    source_files = [Path(__file__), Path(__file__).with_name("protocol.md"),
                    Path(lk.__file__), Path(__file__).parents[1]/"reference_benchmark"/"representation.py"]
    source_files += [Path(__file__).with_name(name) for name in ("cases.py", "fitting.py", "rejection.py", "experiment.py")]
    source_sha = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source_files if p.exists()}
    errors = sum(i["severity"] == "error" for i in issues)
    report = dict(schema="local_rx_saved_evidence_audit_v1", directory=str(root.resolve()),
        audit_pass=errors == 0, mode="pilot" if pilot else "main", configured_slots=len(tasks),
        base_condition_trajectories=len(bases),
        independent_random_families=len({(s["network"], int(s["seed"])) for s in tasks}),
        condition_pairing="ordinary/correlated share network/seed RNG streams and are paired conditions",
        completed_slots=sum(s.get("result_status") == "completed" for s in slots),
        failed_slots=sum(s.get("result_status") == "failed" for s in slots),
        configured_candidate_slots=4*len(tasks), slots=slots, certificate_counts=certificate_counts,
        bootstrap=bootstrap, expected_bootstrap_slots=0 if pilot else 16,
        bootstrap_false_reject=sum(b.get("false_reject", False) for b in bootstrap),
        issues=issues, error_count=errors, warning_count=len(issues)-errors,
        source_sha256=source_sha,
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         numpy=np.__version__, scipy=scipy.__version__),
        fit_calls=0, newly_generated_samples=0, elapsed_seconds=perf_counter()-started)
    target = Path(destination) if destination else root/"audit.json"
    _write(target, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args()
    result = audit_directory(args.directory, destination=args.destination)
    print(json.dumps({k:result[k] for k in ("audit_pass", "configured_slots", "base_condition_trajectories", "independent_random_families",
                                           "completed_slots", "failed_slots", "error_count", "warning_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
