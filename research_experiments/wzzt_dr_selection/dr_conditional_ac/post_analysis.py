"""Post hoc aggregate analysis of saved AC observations; never solves or reprices."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys

import numpy as np

STAGE = Path(__file__).resolve().parents[1]
RUN = STAGE / "run_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float).ravel()
    if not len(a):
        return None
    return {"count": int(len(a)), "mean": float(a.mean()), "sd": float(a.std()),
            "min": float(a.min()), "p05": float(np.quantile(a, .05)),
            "median": float(np.median(a)), "p95": float(np.quantile(a, .95)),
            "max": float(a.max())}


def main():
    checks = []
    inputs = {}

    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def read_json(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected:
            check("hash " + str(path), actual == expected)
        inputs[str(path)] = actual
        return json.loads(path.read_text(encoding="utf-8"))

    protocol = read_json(RUN / "protocol.json")
    observed = read_json(RUN / "observations_frozen.json")
    predicted = read_json(RUN / "predictions_frozen.json")
    check("observations frozen", observed["status"] == "frozen")
    check("predictions frozen", predicted["status"] == "frozen")
    fit_ref = protocol["inputs"]["historical_fit"]
    fit = read_json(fit_ref["path"], fit_ref["sha256"])
    pool_ref = protocol["inputs"]["pool"]
    pool = read_json(pool_ref["path"], pool_ref["sha256"])
    R0 = np.asarray(fit["fronts"]["raw"]["R"], float)
    X0 = np.asarray(fit["fronts"]["raw"]["X"], float)
    labels = protocol["labels"]
    positions = [labels.index(i) for i in protocol["resource_labels"]]
    check("A0 labels match actual", fit["terminal_labels"] == labels)
    check("pool labels match actual", pool["labels"] == labels)
    check("A0 shape", R0.shape == X0.shape == (15, 15))
    models = {m["candidate_id"]: m for m in pool["models"]}
    matrices = {k: (np.asarray(m["R"], float), np.asarray(m["X"], float)) for k, m in models.items()}
    hc, hf = (models[k]["historical_validation_mae"] for k in ("coarse", "refined"))
    h = 1.5 * (1.03 * hf - hc) + 2.5e-12
    check("original rule retained", protocol["rule"] == {"history_gate": True, "new_weight": 2, "new_fraction": .03, "history_fraction": .03})
    check("all historical validation rows retained", pool["validation_rows"] == 288 and protocol["history_rows"] == [288, 288])
    base_kw = protocol["base_kw"]
    dt = protocol["dt_hours"]
    packets = {}
    for row in observed["rows"]:
        ref = row["packet"]
        actual = sha(ref["path"])
        check("packet hash " + row["record_id"], actual == ref["sha256"])
        inputs[ref["path"]] = actual
        with open(ref["path"], "rb") as f:
            packets[(row["day"], row["policy"])] = pickle.load(f)
    score_rows = {}
    for row in predicted["rows"]:
        ref = row["score"]
        score = read_json(ref["path"], ref["sha256"])
        score_rows[(row["day"], row["policy"], row["noise"])] = (row, score)
    policies = sorted({policy for _, policy in packets})
    days = sorted({day for day, _ in packets})
    check("12 days and five policies", len(days) == 12 and len(policies) == 5 and len(packets) == 60)
    check("all four noise views", len(score_rows) == 240)
    natural_days, physical_rows, transfer_rows, decision_rows = [], [], [], []
    arrays = {policy: {"dv": [], "dy": [], "dy_pred": [], "dy_error": [],
                       "node_dv": [], "node_dy": [], "node_pred": [], "node_error": []} for policy in policies}
    for day in days:
        zero = packets[(day, "zero")]["private"]
        p0 = np.asarray(zero["physical_P_pu"], float)
        q0 = np.asarray(zero["physical_Q_pu"], float)
        v0 = np.asarray(zero["voltage_true"], float)
        root0 = np.asarray(zero["root_true"], float)
        natural_kw = p0 * base_kw
        natural_energy = dt * natural_kw.sum(axis=0)
        denom_system, denom_pair = natural_energy.sum(), natural_energy[positions].sum()
        natural_days.append({"day": day, "system_kwh": float(denom_system), "controlled_kwh": float(denom_pair),
                             "system_mean_kw": float(natural_kw.sum(axis=1).mean()),
                             "controlled_mean_kw": float(natural_kw[:, positions].sum(axis=1).mean()),
                             "positive_terminal_kwh": float(dt * np.maximum(natural_kw, 0).sum()),
                             "negative_terminal_magnitude_kwh": float(dt * np.maximum(-natural_kw, 0).sum())})
        for policy in policies:
            packet = packets[(day, policy)]
            private = packet["private"]
            p = np.asarray(private["physical_P_pu"], float)
            q = np.asarray(private["physical_Q_pu"], float)
            v = np.asarray(private["voltage_true"], float)
            delivered = np.asarray(private["delivered_kw"], float)
            check(f"same natural P day{day}/{policy}", np.array_equal(private["natural_P_pu"], zero["natural_P_pu"]))
            check(f"same natural Q day{day}/{policy}", np.array_equal(private["natural_Q_pu"], zero["natural_Q_pu"]))
            check(f"same physical root day{day}/{policy}", np.array_equal(private["root_true"], root0))
            dp, dq = p - p0, q - q0
            check(f"delivered P reaches physical input day{day}/{policy}", np.allclose(dp[:, positions] * base_kw, delivered.T, rtol=1e-12, atol=1e-12))
            inc, dec = dt * np.maximum(delivered, 0).sum(), dt * np.maximum(-delivered, 0).sum()
            absolute, net = inc + dec, inc - dec
            sumabs_peak = np.abs(delivered).sum(axis=0).max()
            physical_rows.append({"day": day, "policy": policy, "increase_kwh": float(inc),
                "decrease_magnitude_kwh": float(dec), "absolute_kwh": float(absolute), "net_kwh": float(net),
                "sum_absolute_peak_kw": float(sumabs_peak), "absolute_net_peak_kw": float(np.abs(delivered.sum(axis=0)).max()),
                "absolute_over_system_energy": float(absolute / denom_system), "absolute_over_controlled_energy": float(absolute / denom_pair),
                "net_over_system_energy": float(net / denom_system), "net_over_controlled_energy": float(net / denom_pair),
                "max_slot_absolute_over_system_kw": float(np.max(np.abs(delivered).sum(axis=0) / natural_kw.sum(axis=1))),
                "max_slot_absolute_over_controlled_kw": float(np.max(np.abs(delivered).sum(axis=0) / natural_kw[:, positions].sum(axis=1))),
                "fee": float(packet["summary"]["planned_cost"]), "resource_violating": bool(packet["summary"]["resource_violation"])})
            dv = v - v0
            dy = np.square(v0) - np.square(v)
            dy_pred = dp @ R0.T + dq @ X0.T
            error = dy_pred - dy
            for key, val in (("dv", dv), ("dy", dy), ("dy_pred", dy_pred), ("dy_error", error),
                             ("node_dv", dv[:, positions]), ("node_dy", dy[:, positions]),
                             ("node_pred", dy_pred[:, positions]), ("node_error", error[:, positions])):
                arrays[policy][key].append(val)
            transfer_rows.append({"day": day, "policy": policy, "actual_deltaY_rms_micro": float(1e6 * np.sqrt(np.mean(dy**2))),
                "A0_deltaY_rms_micro": float(1e6 * np.sqrt(np.mean(dy_pred**2))),
                "increment_error_rms_micro": float(1e6 * np.sqrt(np.mean(error**2))),
                "increment_error_mae_micro": float(1e6 * np.mean(np.abs(error))),
                "actual_deltaV_mean_abs_millipu": float(1e3 * np.mean(np.abs(dv))),
                "actual_deltaV_max_abs_millipu": float(1e3 * np.max(np.abs(dv)))})
            for noise, scene in packet["scenes"].items():
                measured_p = np.asarray(scene["P_terminal"], float)
                measured_q = np.asarray(scene["Q_terminal"], float)
                y = np.asarray(scene["drop_target"], float)
                # Post hoc substitution only: retain actual background and meter error,
                # replacing the paired true DR increment by the frozen A0 increment.
                y_hybrid = y + error
                losses, hybrid_losses = {}, {}
                for key, (r, x) in matrices.items():
                    prediction = measured_p @ r.T + measured_q @ x.T
                    losses[key] = float(np.mean(np.abs(y - prediction)))
                    hybrid_losses[key] = float(np.mean(np.abs(y_hybrid - prediction)))
                D = losses["coarse"] - 1.03 * losses["refined"]
                D_hybrid = hybrid_losses["coarse"] - 1.03 * hybrid_losses["refined"]
                saved_row, saved_score = score_rows[(day, policy, noise)]
                check(f"saved original D day{day}/{policy}/{noise}", np.isclose(D * 1e6, saved_row["D_micro"], rtol=1e-10, atol=1e-8))
                actual_rule = next(x for x in saved_score["rules"] if x["rule_id"] == "w2_tau3pct")
                check(f"exact two-candidate rejection day{day}/{policy}/{noise}", bool(D > h) == bool(saved_row["coarse_excluded"]))
                check(f"saved original selection day{day}/{policy}/{noise}", actual_rule["selected_candidate_id"] == saved_row["selected"])
                decision_rows.append({"day": day, "policy": policy, "noise": noise,
                    "actual_D_micro": D * 1e6, "A0_increment_hybrid_D_micro": D_hybrid * 1e6,
                    "A0_increment_minus_actual_D_micro": (D_hybrid - D) * 1e6,
                    "actual_coarse_excluded": bool(D > h), "hybrid_coarse_excluded": bool(D_hybrid > h)})

    aggregates = []
    for policy in policies:
        rows = [r for r in physical_rows if r["policy"] == policy]
        agg = {"policy": policy, "days": len(rows), "actual_DR": {k: stats([r[k] for r in rows]) for k in rows[0] if k not in ("day", "policy", "resource_violating")},
               "resource_violating_days": sum(r["resource_violating"] for r in rows), "response": {}, "noise_views": {}}
        cat = {key: np.concatenate(value) for key, value in arrays[policy].items()}
        for scope, dyk, predk, errk, dvk in (("all_15", "dy", "dy_pred", "dy_error", "dv"),
                                          ("controlled_2", "node_dy", "node_pred", "node_error", "node_dv")):
            actual, estimate, error, dv = (cat[k].ravel() for k in (dyk, predk, errk, dvk))
            actual_norm = np.linalg.norm(actual)
            agg["response"][scope] = {"deltaV_signed_millipu": stats(dv * 1e3), "deltaV_absolute_millipu": stats(np.abs(dv) * 1e3),
                "actual_deltaY_micro": stats(actual * 1e6), "A0_deltaY_micro": stats(estimate * 1e6),
                "A0_increment_error_micro": stats(error * 1e6), "A0_increment_error_MAE_micro": float(np.abs(error).mean() * 1e6),
                "A0_increment_error_RMSE_micro": float(np.sqrt(np.mean(error**2)) * 1e6),
                "actual_deltaY_RMS_micro": float(np.sqrt(np.mean(actual**2)) * 1e6),
                "relative_increment_L2_error": float(np.linalg.norm(error) / actual_norm) if actual_norm > 0 else None,
                "increment_correlation": float(np.corrcoef(actual, estimate)[0, 1]) if actual.std() > 0 and estimate.std() > 0 else None}
        for noise in ("none", "v", "pq", "both"):
            dr = [r for r in decision_rows if r["policy"] == policy and r["noise"] == noise]
            agg["noise_views"][noise] = {"days": len(dr), "actual_coarse_excluded": sum(r["actual_coarse_excluded"] for r in dr),
                "hybrid_coarse_excluded": sum(r["hybrid_coarse_excluded"] for r in dr),
                "actual_D_micro": stats([r["actual_D_micro"] for r in dr]),
                "A0_increment_hybrid_D_micro": stats([r["A0_increment_hybrid_D_micro"] for r in dr]),
                "A0_increment_minus_actual_D_micro": stats([r["A0_increment_minus_actual_D_micro"] for r in dr])}
        aggregates.append(agg)
    output = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}, "input_hashes": inputs,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "definitions": {"natural_load": "Saved physical natural P, not measured P; 200 kW base; signed net terminal demand.",
            "DR": "Saved delivered stochastic response without clipping; 0.25-hour energy sums. Absolute energy counts both directions; net energy is signed.",
            "voltage_response": "Same-background paired AC V(strategy)-V(zero), actual physical voltages, no measurement noise.",
            "increment_prediction": "Frozen historical non-tree A0: deltaP@R0.T+deltaQ@X0.T versus paired physical squared-drop increment Vzero**2-Vstrategy**2; common root cancels.",
            "units": "millipu=0.001 pu voltage; micro=1e-6 pu^2 squared voltage drop.",
            "hybrid": "Post hoc mechanism diagnosis: actual observed Y plus (A0 predicted increment minus true AC increment), retaining same actual background, same meter realization, actual measured P/Q and frozen candidate matrices. This uses actual paired physical data and is not deployable prediction or new optimization.",
            "inference_unit": "12 complete independent synthetic paths, three profiles with four replicates each. Node/time cells in descriptive quantiles are not independent trials.",
            "boundaries": "No new AC, no refit, no new search, no repricing, no topology truth read. Existing unsafe component deliveries remain included. Results concern the fixed two-candidate validation stage."},
        "threshold_D_micro": float(h * 1e6), "natural_background": {k: stats([r[k] for r in natural_days]) for k in natural_days[0] if k != "day"},
        "policy_aggregates": aggregates, "daily_scalar_DR": physical_rows,
        "daily_scalar_transfer": transfer_rows, "daily_scalar_decisions": decision_rows,
        "check_count": len(checks), "passed_count": sum(c["passed"] for c in checks), "checks": checks}
    destination = RUN / "ac_post_analysis.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "output": str(destination)}))
    for a in aggregates:
        print(json.dumps({"policy": a["policy"], "absolute_kwh": a["actual_DR"]["absolute_kwh"]["mean"],
            "system_fraction": a["actual_DR"]["absolute_over_system_energy"]["mean"],
            "pair_fraction": a["actual_DR"]["absolute_over_controlled_energy"]["mean"],
            "increment_relative_error": a["response"]["all_15"]["relative_increment_L2_error"],
            "increment_correlation": a["response"]["all_15"]["increment_correlation"],
            "both_actual_pass": a["noise_views"]["both"]["actual_coarse_excluded"],
            "both_hybrid_pass": a["noise_views"]["both"]["hybrid_coarse_excluded"],
            "both_D_bias_micro": a["noise_views"]["both"]["A0_increment_minus_actual_D_micro"]["mean"]}))


if __name__ == "__main__":
    main()
