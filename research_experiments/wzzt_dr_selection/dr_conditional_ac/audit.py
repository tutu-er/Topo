"""Independent saved-artifact audit; no producer, fit, scorer or solver calls.

Recomputes measurements, physical identities, frozen losses, w2/tau3 selection,
contraction and truth from saved arrays / network CSVs. This auditor is written
after the experiment and is deliberately not inserted into its frozen manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import sys
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def arr(x):
    return np.asarray(x, dtype=float)


def family(items):
    return {tuple(sorted(map(int, s))) for s in items}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, tag, condition, detail=None):
        self.checks.append(dict(tag=tag, passed=bool(condition), detail=detail))

    def close(self, tag, actual, expected, atol=2e-12, rtol=2e-12):
        a, b = arr(actual), arr(expected)
        same = a.shape == b.shape
        error = float(np.max(np.abs(a-b), initial=0)) if same else None
        ok = same and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=atol, rtol=rtol)
        self.check(tag, ok, dict(max_abs_error=error, actual_shape=list(a.shape), expected_shape=list(b.shape)))

    def reference(self, tag, reference):
        path = Path(reference["path"])
        self.check(tag, path.is_file() and sha(path) == reference["sha256"])
        return path


def target(scene):
    return arr(scene["V_root_observed"]).reshape(-1, 1)**2-arr(scene["V_terminal"])**2


def prediction(model, scene):
    return arr(scene["P_terminal"]) @ arr(model["R"]).T + arr(scene["Q_terminal"]) @ arr(model["X"]).T


def root_replay(settings, seed, count):
    rng = np.random.default_rng(seed)
    sigma = settings["root_sigma"]
    if settings["root_process"] == "iid":
        fluctuation = rng.normal(0, sigma, count)
    else:
        hours = np.arange(count)*settings["day_hours"]/count
        phase = rng.uniform(0, 2*np.pi)
        fraction = settings["daily_fraction"]
        daily = np.sqrt(2)*fraction*sigma*np.sin(2*np.pi*hours/24+phase)
        ss = sigma*np.sqrt(1-fraction**2)
        rho = np.exp(-(settings["day_hours"]/count)/settings["ou_tau_hours"])
        correlated = np.empty(count)
        correlated[0] = rng.normal(0, ss)
        innovations = rng.normal(0, ss*np.sqrt(1-rho*rho), count-1)
        for i in range(1, count):
            correlated[i] = rho*correlated[i-1]+innovations[i-1]
        pulse = settings["tap_step_amplitude"]*((hours >= settings["tap_start_hour"]) & (hours < settings["tap_end_hour"]))
        fluctuation = daily+correlated+pulse-pulse.mean()
    return settings["root_mean"]+fluctuation


def physical_replay(private, branches, metadata):
    ac = private["AC"]
    voltage = ac["V_bus_mag"]
    buses = list(voltage.columns)
    positions = {int(bus): i for i, bus in enumerate(buses)}
    p = arr(private["physical_P_pu"].reindex(columns=buses, fill_value=0))
    q = arr(private["physical_Q_pu"].reindex(columns=buses, fill_value=0))
    netp, netq = np.zeros_like(p), np.zeros_like(q)
    zbase = metadata["base_kv"]**2/metadata["base_mva"]
    edge_map = {(int(r.from_bus), int(r.to_bus)): (r.r_ohm/zbase, r.x_ohm/zbase) for r in branches.itertuples()}
    max_loss = max_drop = max_current = max_phasor = 0.0
    for edge in ac["metadata"]["branch_edges"]:
        label, parent, child = edge["label"], int(edge["from_bus"]), int(edge["to_bus"])
        r, x = edge_map[(parent, child)]
        pf, qf = arr(ac["branch_p_from_pu"][label]), arr(ac["branch_q_from_pu"][label])
        pt, qt = arr(ac["branch_p_to_pu"][label]), arr(ac["branch_q_to_pu"][label])
        ell = arr(ac["I_branch_mag"][label])**2
        netp[:, positions[child]] += pt
        netq[:, positions[child]] += qt
        netp[:, positions[parent]] -= pf
        netq[:, positions[parent]] -= qf
        vp, vc = arr(voltage[parent]), arr(voltage[child])
        lp, lq = arr(ac["branch_loss_p_pu"][label]), arr(ac["branch_loss_q_pu"][label])
        max_loss = max(max_loss, float(np.max(np.abs(lp-r*ell))), float(np.max(np.abs(lq-x*ell))),
                       float(np.max(np.abs(pf-pt-lp))), float(np.max(np.abs(qf-qt-lq))))
        max_drop = max(max_drop, float(np.max(np.abs(vp*vp-vc*vc-2*(r*pf+x*qf)+(r*r+x*x)*ell))))
        max_current = max(max_current, float(np.max(np.abs(ell*vp*vp-pf*pf-qf*qf))))
        vp_complex = vp*np.exp(1j*np.deg2rad(arr(ac["V_bus_angle_deg"][parent])))
        vc_complex = vc*np.exp(1j*np.deg2rad(arr(ac["V_bus_angle_deg"][child])))
        current = np.conj((pf+1j*qf)/vp_complex)
        max_phasor = max(max_phasor, float(np.max(np.abs(vc_complex-(vp_complex-(r+1j*x)*current)))))
    nonroot = [i for i, b in enumerate(buses) if int(b) != int(metadata["root_bus"])]
    rootpos = positions[int(metadata["root_bus"])]
    metrics = dict(nodal_p=float(np.max(np.abs((netp-p)[:, nonroot]))),
                   nodal_q=float(np.max(np.abs((netq-q)[:, nonroot]))),
                   global_p=float(np.max(np.abs(-netp[:, rootpos]-p.sum(axis=1)-arr(ac["branch_loss_p_pu"]).sum(axis=1)))),
                   global_q=float(np.max(np.abs(-netq[:, rootpos]-q.sum(axis=1)-arr(ac["branch_loss_q_pu"]).sum(axis=1)))),
                   branch_loss=max_loss, squared_drop=max_drop, branch_current=max_current, phasor=max_phasor)
    return metrics


def run(root):
    root = Path(root).resolve()
    a = Audit()
    protocol = read(root/"protocol.json")
    prices = read(root/"price_freeze.json")
    frozen_before = {str(p): sha(p) for p in [root/"protocol.json", root/"price_freeze.json", root/"summary.json", root/"scored_private.json"]}
    for path, digest in protocol["source_sha256"].items():
        a.check("source:"+str(path), sha(path) == digest)
    with zipfile.ZipFile(root/"source_snapshot.zip") as archive:
        members = archive.namelist()
        sources = sorted(protocol["source_sha256"])
        a.check("source_snapshot.count", len(members) == len(sources))
        for member, path in zip(members, sources):
            a.check("snapshot:"+member, hashlib.sha256(archive.read(member)).hexdigest() == protocol["source_sha256"][path])
    for key, reference in protocol["inputs"].items():
        a.reference("input:"+key, reference)
    for key in ["high_prices", "low_prices", "source_metadata"]:
        a.reference("input:"+key, protocol[key])
    pool = read(protocol["inputs"]["pool"]["path"])
    history = load(protocol["inputs"]["history"]["path"])["history"]
    metadata_path = Path(protocol["source_metadata"]["path"])
    metadata = read(metadata_path)
    branches = pd.read_csv(metadata_path.parent/"branches.csv")
    labels = protocol["labels"]
    n = len(labels)
    order = pool["selection_order"]
    models = {x["candidate_id"]: x for x in pool["models"]}
    a.check("case", protocol["case"] == "paper15" and protocol["condition"] == "paper15_b2_v0005")
    a.close("base_kw", metadata["base_mva"]*1000, protocol["base_kw"])
    a.close("fixed_noise", [protocol["voltage_sigma"], protocol["pq_sigma"], protocol["root_physical_sigma"]], [.0005, .005, .003])
    a.close("cv", protocol["cv"], np.sqrt(.025/3))
    a.check("no_refit_scope", "no new fit" in protocol["identification"])
    a.check("explicit_rule", protocol["rule"] == dict(history_gate=True, new_weight=2, new_fraction=.03, history_fraction=.03))

    # Parent prices and Gaussian-response contract are checked, not reoptimized.
    for name, p in prices["policies"].items():
        parent_key = "low_prices" if name.endswith("low") else "high_prices"
        parent = read(protocol[parent_key]["path"])
        parent_name = name.split("_")[0]
        a.close("price_parent:"+name, p["mean_u"], parent["policies"][parent_name]["mean_u"], 0, 0)
        a.close("prepaid_fee:"+name, p["planned_cost"], .25*np.sum(arr(p["mean_u"])**2/.5), 1e-10)
        a.check("price_shape:"+name, arr(p["mean_u"]).shape == (2, 96))
    for tier in ["high", "low"]:
        a.close("matched_fee:"+tier, prices["policies"]["continuous_"+tier]["planned_cost"], prices["policies"]["rotation_"+tier]["planned_cost"], 1e-8)
    parent_root = root.parent.parent/"wzzt_conditional_dr_stage"
    parent_problem = load(parent_root/"run_v1/problem.pkl")
    a.close("parent_problem_cv", parent_problem["cv"], np.full((2, 96), protocol["cv"]))
    a.close("parent_problem_elasticity", parent_problem["elasticities"], np.full((2, 96), .5))
    a.close("parent_problem_kappa", parent_problem["kappa"], 1.03)
    for split in ["training", "validation"]:
        a.check("full_history:"+split, sum(len(s["P_terminal"]) for s in history[split]) == 288)

    historical = {}
    raw_families, output_families = {}, {}
    for identity, model in models.items():
        point = model["point"]
        raw_families[identity] = family([[labels[i] for i in s] for s in point["supports"]])
        for channel, weights in [("R", point["r"]), ("X", point["x"])]:
            matrix = np.zeros((n, n))
            for block, weight in zip(point["supports"], weights):
                matrix[np.ix_(block, block)] += weight
            a.close("frozen_matrix:"+identity+":"+channel, model[channel], matrix)
        for split in ["training", "validation"]:
            total = math.fsum(float(np.abs(target(s)-prediction(model, s)).sum()) for s in history[split])
            a.close("historical_loss:"+identity+":"+split, total, model["historical_"+split+"_abs_sum"], 1e-10)
            if split == "validation":
                historical[identity] = total/(288*n)
        cal = pool["calibration"]
        out = [s for j, s in enumerate(point["supports"]) if len(s) == 1 or cal["a"]*point["r"][j]+cal["b"]*point["x"][j]+1e-10 >= .16*cal["median_root_depth"]]
        output_families[identity] = family([[labels[i] for i in s] for s in out])
        a.check("frozen_contraction:"+identity, family(out) == family(model["frozen_output_supports"]))
        a.check("frozen_output_labels:"+identity, output_families[identity] == family(model["frozen_output_support_labels"]))
    hmin = min(historical.values())
    historical_eligible = [k for k in order if historical[k] <= 1.03*hmin+1e-12]
    # h is the threshold for the unweighted new-day L_C - 1.03 L_F.
    h = (1e-12-.6*(historical["coarse"]-1.03*historical["refined"]))/ .4
    # The planning kernel stores residuals and its threshold in micro p.u.^2.
    a.close("planning_threshold_micro", parent_problem["threshold"], h*1e6, 1e-8)

    # Derive truth independently from nonzero physical CSV edges, with no fit.
    children = defaultdict(list)
    for edge in branches.itertuples():
        children[int(edge.from_bus)].append(int(edge.to_bus))
    def descendants(node):
        result = {node}
        for child in children[node]:
            result |= descendants(child)
        return result
    truth = set()
    for edge in branches.itertuples():
        if edge.r_ohm == 0 and edge.x_ohm == 0:
            continue
        block = tuple(sorted(set(labels) & descendants(int(edge.to_bus))))
        if block:
            truth.add(block)
    a.check("truth_from_physical_edges", truth == family(read(root/"truth_private.json")["missed_blocks"]))
    ledger = read(root/"observations_frozen.json")
    a.reference("ledger_price", ledger["price_freeze"])
    a.reference("ledger_protocol", ledger["protocol"])
    scored = read(root/"scored_private.json")["rows"]
    indexed = {(r["day"], r["policy"], r["noise"]): r for r in scored}
    a.check("denominator_physical", len(ledger["rows"]) == 60)
    a.check("denominator_selection", len(indexed) == len(scored) == 240)
    by_day = {}
    reconstructed = []
    maxima = defaultdict(float)
    resource_days = Counter()
    timestamps = {}
    ids = {(int(d["profile_index"]), int(d["replicate"])) for d in protocol["days"]}
    a.check("unique_new_ids", len(ids) == 12)
    a.check("profile_balance", Counter(d["profile_index"] for d in protocol["days"]) == {0: 4, 1: 4, 2: 4})
    for d in protocol["days"]:
        a.check("declared_identity:"+str(d["day"]), d == dict(day=d["day"], profile_index=d["day"]%3, replicate=930000+97*d["day"], response_seed=2026110210+d["day"]))
    physical_samples = 0
    for record in ledger["rows"]:
        tag = record["record_id"]
        a.check(tag+":observed", record["status"] == "observed")
        path = a.reference(tag+":packet", record["packet"])
        payload = load(path)
        private, summary = payload["private"], payload["summary"]
        day = record["day"]
        u = arr(prices["policies"][record["policy"]]["mean_u"])
        z = np.random.default_rng(record["response_seed"]).standard_normal((2, 96))
        a.close(tag+":u_frozen", private["mean_u"], u, 0, 0)
        a.close(tag+":Z_seed", private["Z"], z, 0, 0)
        delivered = u*(1+protocol["cv"]*z)
        a.close(tag+":delivered_unclipped", private["delivered_kw"], delivered, 0, 0)
        dp = np.zeros((96, n))
        for j, label in enumerate(protocol["resource_labels"]):
            dp[:, labels.index(label)] = delivered[j]
        a.close(tag+":node_P_increment", private["node_increment_kw"], dp)
        a.close(tag+":node_Q_increment", private["node_increment_kvar"], .35*dp)
        p, q = arr(private["physical_P_pu"]), arr(private["physical_Q_pu"])
        a.close(tag+":physical_P", p, arr(private["natural_P_pu"])+dp/200)
        a.close(tag+":physical_Q", q, arr(private["natural_Q_pu"])+.35*dp/200)
        a.close(tag+":component", private["resource_component_kw"], 1+delivered)
        bad = (1+delivered < -1e-10) | (1+delivered > 2+1e-10)
        a.check(tag+":resource_flag", summary["resource_violation"] == bool(bad.any()) and summary["resource_violation_cells"] == int(bad.sum()))
        resource_days[record["policy"]] += int(bad.any())
        fee = .25*float(np.sum(u*u/.5))
        a.close(tag+":fees", [summary["planned_cost"], record["fee"]], [fee, fee], 1e-10)
        a.close(tag+":net_energy", summary["actual_net_energy_kwh"], .25*delivered.sum(axis=1))
        a.check(tag+":not_skipped_or_clipped", summary["no_clipping"] and summary["no_redraw"] and summary["resource_limit_does_not_skip_AC"])
        shifted = [42, 7, 21][record["profile_index"]]+100003*record["replicate"]
        rootseed = 70000019+100003*record["replicate"]+record["profile_index"]
        a.check(tag+":producer_seeds", summary["profile_seed"] == shifted and summary["root_seed"] == shifted+1000 and summary["meter_seed"] == shifted+2000 and summary["root_meter_seed"] == rootseed)
        measured = private["both_measured_channels"]
        a.close(tag+":physical_root_seed", private["root_true"], root_replay(measured["scenario_settings"], shifted+1000, 96))
        rng = np.random.default_rng(shifted+2000)
        normals = [rng.standard_normal((96, n)) for _ in range(3)]
        true_v = arr(private["voltage_true"])
        root_true = arr(private["root_true"])
        predictions_measured = [p+.005*np.maximum(np.abs(p), 1e-12)*normals[0],
                                q+.005*np.maximum(np.abs(q), 1e-12)*normals[1],
                                true_v+.0005*np.maximum(np.abs(true_v), 1e-12)*normals[2]]
        for name, expected in zip(["p_measured", "q_measured", "voltage_measured"], predictions_measured):
            a.close(tag+":meter_seed:"+name, measured[name], expected, 3e-15, 3e-15)
        root_noise = np.random.default_rng(rootseed).standard_normal(96)
        root_observed = root_true+.0005*np.maximum(np.abs(root_true), 1e-12)*root_noise
        a.close(tag+":root_meter_seed", measured["root_observed"], root_observed, 3e-15, 3e-15)
        if day not in by_day:
            by_day[day] = private
        else:
            for field in ["natural_P_pu", "natural_Q_pu", "root_true", "Z"]:
                a.close(tag+":paired:"+field, private[field], by_day[day][field], 0, 0)
        ac = private["AC"]
        a.check(tag+":AC_96_converged", np.asarray(ac["converged"]).shape == (96,) and bool(np.asarray(ac["converged"]).all()))
        physical_samples += len(ac["converged"])
        a.close(tag+":AC_voltage_terminal", private["voltage_true"], ac["V_bus_mag"].loc[:, labels], 0, 0)
        a.close(tag+":AC_root", ac["V_bus_mag"][protocol["root_label"]], root_true, 3e-15)
        metrics = physical_replay(private, branches, metadata)
        for key, value in metrics.items():
            maxima[key] = max(maxima[key], value)
            a.check(tag+":physical:"+key, value <= 1e-8, value)
        a.check(tag+":solver_error", bool((arr(ac["max_voltage_error"]) <= 1e-10).all()))
        e0 = root_observed-root_true
        ei = predictions_measured[2]-true_v
        expected_drop_noise = (2*root_true*e0+e0*e0)[:, None] - (2*true_v*ei+ei*ei)
        for variant in protocol["observation_variants"]:
            scene = payload["scenes"][variant]
            ktag = tag+":"+variant
            pn, vn = variant in ["pq", "both"], variant in ["v", "both"]
            for field, expected in [("P_terminal", predictions_measured[0] if pn else p),
                                    ("Q_terminal", predictions_measured[1] if pn else q),
                                    ("V_terminal", predictions_measured[2] if vn else true_v),
                                    ("V_root_observed", root_observed if vn else root_true)]:
                a.close(ktag+":channel:"+field, scene[field], expected, 3e-15, 3e-15)
            a.close(ktag+":target_from_root", scene["drop_target"], target(scene), 2e-15)
            a.close(ktag+":root_common_error", target(scene)-(root_true[:, None]**2-true_v**2), expected_drop_noise if vn else np.zeros_like(true_v), 8e-16)
            predictions = {k: prediction(models[k], scene) for k in order}
            losses = {k: float(np.mean(np.abs(target(scene)-predictions[k]))) for k in order}
            weighted = {k: (288*historical[k]+2*96*losses[k])/480 for k in order}
            direct = {k: float(np.mean(np.concatenate([np.concatenate([np.abs(target(s)-prediction(models[k], s)).reshape(-1) for s in history["validation"]]),
                                                       np.abs(target(scene)-predictions[k]).reshape(-1), np.abs(target(scene)-predictions[k]).reshape(-1)]))) for k in order}
            reference = min(historical_eligible, key=lambda k: (weighted[k], order.index(k)))
            threshold = 1.03*weighted[reference]+1e-12
            eligible = [k for k in historical_eligible if weighted[k] <= threshold]
            selected = min(eligible, key=lambda k: (len(models[k]["point"]["supports"]), weighted[k], order.index(k)))
            row = indexed[(day, record["policy"], variant)]
            score_path = a.reference(ktag+":scorehash", row["score"])
            saved = read(score_path)
            rule = next(r for r in saved["rules"] if r["new_day_weight"] == 2 and r["new_day_fraction"] == .03)
            for k in order:
                a.close(ktag+":direct_mean:"+k, weighted[k], direct[k], 2e-16)
                a.close(ktag+":weighted_loss:"+k, rule["weighted_validation_mae"][k], weighted[k], 2e-16)
                a.close(ktag+":new_loss:"+k, row[k+"_L_micro"], 1e6*losses[k], 1e-8)
            a.check(ktag+":eligible", rule["eligible_candidate_ids"] == eligible)
            a.check(ktag+":selected", selected == rule["selected_candidate_id"] == row["selected"])
            a.check(ktag+":reference", reference == rule["reference_candidate_id"])
            margin = weighted["coarse"]-threshold
            D = losses["coarse"]-1.03*losses["refined"]
            a.close(ktag+":D", row["D_micro"], D*1e6, 1e-8)
            a.close(ktag+":margin", row["coarse_margin_micro"], margin*1e6, 1e-8)
            a.close(ktag+":pair_distance", row["distance_micro"], np.mean(np.abs(predictions["coarse"]-predictions["refined"]))*1e6, 1e-8)
            a.check(ktag+":exit_exact", row["coarse_excluded"] == ("coarse" not in eligible))
            if reference == "refined":
                a.close(ktag+":D_threshold_equivalence", margin, .4*(D-h), 2e-16)
                a.check(ktag+":strict_D_exit", bool(D > h) == row["coarse_excluded"])
            raw_errors = len(raw_families[selected]^truth)
            final_errors = len(output_families[selected]^truth)
            a.check(ktag+":raw_errors", row["raw_errors"] == raw_errors)
            a.check(ktag+":final_errors", row["errors"] == final_errors and row["missing"] == len(truth-output_families[selected]) and row["extra"] == len(output_families[selected]-truth))
            reconstructed.append(dict(day=day, policy=record["policy"], noise=variant, selected=selected, errors=final_errors,
                                      raw_errors=raw_errors, coarse_excluded=bool("coarse" not in eligible), D_micro=1e6*D,
                                      margin_micro=1e6*margin, fee=fee))

    summary = read(root/"summary.json")
    tables = []
    for noise in protocol["observation_variants"]:
        for policy in prices["policies"]:
            rows = [r for r in reconstructed if r["noise"] == noise and r["policy"] == policy]
            table = dict(noise=noise, policy=policy, complete=len(rows), recoveries=sum(r["errors"] == 0 for r in rows),
                         full_errors=sum(r["errors"] for r in rows), raw_errors=sum(r["raw_errors"] for r in rows),
                         coarse_excluded=sum(r["coarse_excluded"] for r in rows), fee_per_day=prices["policies"][policy]["planned_cost"])
            original = next(r for r in summary["tables"] if r["noise"] == noise and r["policy"] == policy)
            for field in table:
                a.check("summary:"+noise+":"+policy+":"+field, original[field] == table[field])
            tables.append(table)
    paired = []
    lookup = {(r["day"], r["policy"], r["noise"]): r for r in reconstructed}
    for noise in protocol["observation_variants"]:
        for tier in ["high", "low"]:
            delta = [lookup[(d, "continuous_"+tier, noise)]["errors"]-lookup[(d, "rotation_"+tier, noise)]["errors"] for d in range(12)]
            row = dict(noise=noise, tier=tier, better=sum(v < 0 for v in delta), worse=sum(v > 0 for v in delta), same=sum(v == 0 for v in delta), unresolved=0)
            a.check("paired:"+noise+":"+tier, row == next(r for r in summary["pairs"] if r["noise"] == noise and r["tier"] == tier))
            paired.append(row)

    # Bounded historical identity scope: explicit prior request records only.
    prior_ids, prior_files = set(), []
    for file in root.parent.parent.rglob("*request*.json"):
        try:
            data = read(file)
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and data.get("case") == "paper15" and "profile_index" in data and "replicate" in data:
            prior_ids.add((int(data["profile_index"]), int(data["replicate"])))
            prior_files.append(str(file))
    a.check("identity_explicit_prior_requests", not(ids & prior_ids))
    a.check("physical_sample_total", physical_samples == 5760)
    a.check("no_new_fit_search_optimizations", all(summary[k] == 0 for k in ["new_fits", "new_W_search", "new_optimizations"]))
    for path, digest in frozen_before.items():
        a.check("unmodified:"+path, sha(path) == digest)
    failures = [c for c in a.checks if not c["passed"]]
    result = dict(status="passed" if not failures else "failed", assertions=len(a.checks), passed=len(a.checks)-len(failures), failed=len(failures),
                  physical_days=60, physical_timepoints=physical_samples, frozen_selections=len(reconstructed),
                  historical_rows=dict(training=288, validation=288), profile_strata=3, new_physical_replicates=12,
                  sigma_v=.0005, sigma_pq=.005, cv=protocol["cv"], theoretical_new_day_threshold_micro=h*1e6,
                  source_files_verified=len(protocol["source_sha256"]), max_physical_residuals=maxima,
                  resource_violation_days=dict(resource_days), tables=tables, pairs=paired,
                  auditor_source_sha256=sha(__file__), auditor_written_after_experiment=True,
                  audit_environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform()),
                  producer_environment=protocol["environment"],
                  supplemental_file_hashes={str(p): sha(p) for p in [metadata_path.parent/"branches.csv", metadata_path.parent/"buses.csv", parent_root/"run_v1/problem.pkl"]},
                  identity_scope=dict(explicit_prior_requests=len(prior_files), distinct_prior_ids=len(prior_ids), overlap=len(ids & prior_ids),
                                      limitation="Only explicit paper15/profile_index/replicate request JSONs audited; older replicate_base+event schemes not exhaustively expanded."),
                  scope="Saved nonlinear AC equations, metering RNG/root-common law, frozen historical pair w2/3% choice and output mapping; no new W search, no optimality or 95% certification.",
                  failures=failures)
    (root/"audit_checks.json").write_text(json.dumps(dict(checks=a.checks, prior_request_files=prior_files, reconstructed=reconstructed), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (root/"audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "assertions", "passed", "failed", "physical_timepoints", "frozen_selections", "max_physical_residuals", "auditor_source_sha256", "failures"]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    raise SystemExit(run(parser.parse_args().root))
