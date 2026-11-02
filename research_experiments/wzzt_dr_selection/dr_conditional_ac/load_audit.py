"""Recompute aggregate DR/load ratios from frozen historical observations only.

No future observations, physical truth, fitted models, or forecast member of the
history packet are accessed.  The history pickle is a trusted, hash-locked local
artifact.  Output contains scalar aggregates, not P/Q/V or price trajectories.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys

import numpy as np


STAGE = Path(__file__).resolve().parents[1]
INPUT_HASHES = {
    "run_v1/protocol.json": "76c4908d3358442b8ac76f098750135b747c43ba8720db8592b7b5810cafdc31",
    "delivery_v1/protocol.json": "90482257b3ad33f3c7afe09d6553b1165b34f3f8d0028eeee56877ad584bb1b4",
    "run_v1/price_freeze.json": "a8e53fa24a8ea2c7af27f7d44d51b245804df68a864e012d146cfc105b9fe501",
    "delivery_v1/price_freeze.json": "523cc4e435b98795e5618082d83ce9a369a6a5b2bda82a64083e43d7a84e591d",
}
HISTORY_HASH = "7ada36398650968ddb35d195e264e3e52bcc2b0544439731315d7c7131ebf5a2"
CONTROL_LABELS = [103, 101]
DT_HOURS = 0.25
ELASTICITY = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks: list[dict] = []

    def check(name: str, ok: bool) -> None:
        checks.append({"check": name, "passed": bool(ok)})
        if not ok:
            raise AssertionError(name)

    inputs = {}
    json_inputs = {}
    for relative, expected in INPUT_HASHES.items():
        path = STAGE / relative
        actual = sha256(path)
        check(f"input hash: {relative}", actual == expected)
        inputs[relative] = {"path": str(path), "sha256": actual}
        json_inputs[relative] = json.loads(path.read_text(encoding="utf-8"))
    history_ref = json_inputs["run_v1/protocol.json"]["inputs"]["history"]
    history_path = Path(history_ref["path"])
    history_actual_hash = sha256(history_path)
    check("history protocol expected hash", history_ref["sha256"] == HISTORY_HASH)
    check("history file expected hash", history_actual_hash == HISTORY_HASH)
    inputs["history"] = {"path": str(history_path), "sha256": history_actual_hash}
    with history_path.open("rb") as handle:
        history = pickle.load(handle)["history"]
    labels = list(history["labels"])
    positions = [labels.index(label) for label in CONTROL_LABELS]
    check("original labels", labels == list(range(101, 116)))
    check("200 kW power base", history["base_kw"] == 200.0)
    records = []
    for split in ("training", "validation"):
        scenes = history[split]
        check(f"three complete {split} days", len(scenes) == 3)
        for index, scene in enumerate(scenes):
            frame = scene["P_terminal"]
            check(f"{split}/{index} shape", frame.shape == (96, 15))
            check(f"{split}/{index} labels", list(frame.columns) == labels)
            check(f"{split}/{index} units", scene["units"]["P_terminal"] == "pu")
            power_kw = np.asarray(frame, dtype=float) * history["base_kw"]
            check(f"{split}/{index} finite loads and positive ratio denominators", bool(np.isfinite(power_kw).all() and (power_kw.sum(axis=1) > 0).all() and (power_kw[:, positions] > 0).all()))
            records.append({"split": split, "index": index, "power_kw": power_kw,
                            "energy_kwh": DT_HOURS * power_kw.sum(axis=0)})

    denominators = {}
    for split in ("training", "validation", "all_history"):
        selected = [r for r in records if split == "all_history" or r["split"] == split]
        mean_energy = np.mean([r["energy_kwh"] for r in selected], axis=0)
        power = np.concatenate([r["power_kw"] for r in selected])
        denominators[split] = {
            "day_count": len(selected), "row_count": int(len(power)),
            "system_daily_kwh": float(mean_energy.sum()),
            "terminal_positive_power_daily_kwh": float(DT_HOURS * np.maximum(power, 0).sum() / len(selected)),
            "terminal_negative_power_magnitude_daily_kwh": float(DT_HOURS * np.maximum(-power, 0).sum() / len(selected)),
            "negative_terminal_observation_count": int((power < 0).sum()),
            "minimum_terminal_observed_kw": float(power.min()),
            "controlled_pair_daily_kwh": float(mean_energy[positions].sum()),
            "controlled_node_daily_kwh": {str(label): float(mean_energy[pos]) for label, pos in zip(CONTROL_LABELS, positions)},
            "system_mean_kw": float(power.sum(axis=1).mean()),
            "controlled_pair_mean_kw": float(power[:, positions].sum(axis=1).mean()),
            "system_peak_kw": float(power.sum(axis=1).max()),
            "controlled_pair_peak_kw": float(power[:, positions].sum(axis=1).max()),
            "controlled_node_mean_kw": {str(label): float(power[:, pos].mean()) for label, pos in zip(CONTROL_LABELS, positions)},
            "controlled_node_peak_kw": {str(label): float(power[:, pos].max()) for label, pos in zip(CONTROL_LABELS, positions)},
        }

    policies = []
    for batch, names in (("run_v1", ("continuous", "rotation")),
                         ("delivery_v1", ("continuous", "rotation", "single"))):
        frozen = json_inputs[f"{batch}/price_freeze.json"]["policies"]
        for name in names:
            saved = frozen[name]
            u = np.asarray(saved["mean_u"], dtype=float)
            check(f"{batch}/{name} response shape", u.shape == (2, 96))
            check(f"{batch}/{name} response finite", bool(np.isfinite(u).all()))
            fee = float(DT_HOURS * np.square(u).sum() / ELASTICITY)
            check(f"{batch}/{name} fee agrees with frozen quote", bool(np.isclose(fee, saved["planned_cost"], rtol=1e-12, atol=1e-12)))
            positive = float(DT_HOURS * np.maximum(u, 0).sum())
            negative = float(DT_HOURS * np.maximum(-u, 0).sum())
            absolute = float(DT_HOURS * np.abs(u).sum())
            net = float(DT_HOURS * u.sum())
            check(f"{batch}/{name} energy balance", bool(np.isclose(positive + negative, absolute) and np.isclose(positive - negative, net)))
            abs_peak = float(np.abs(u).sum(axis=0).max())
            net_peak = float(np.abs(u.sum(axis=0)).max())
            node_abs = DT_HOURS * np.abs(u).sum(axis=1)
            node_net = DT_HOURS * u.sum(axis=1)
            item = {
                "batch": batch, "policy": name, "fee_contract_units": fee,
                "increase_kwh": positive, "decrease_magnitude_kwh": negative,
                "absolute_kwh": absolute, "net_kwh": net,
                "sum_absolute_peak_kw": abs_peak, "absolute_net_peak_kw": net_peak,
                "max_node_absolute_kw": float(np.abs(u).max()),
                "controlled_node_absolute_kwh": {str(label): float(v) for label, v in zip(CONTROL_LABELS, node_abs)},
                "controlled_node_net_kwh": {str(label): float(v) for label, v in zip(CONTROL_LABELS, node_net)},
                "mean_price_all_resource_slots": float(np.abs(u).mean() / ELASTICITY),
                "peak_price": float(np.abs(u).max() / ELASTICITY),
                "energy_weighted_price": fee / absolute,
                "ratios": {},
            }
            for split, denom in denominators.items():
                selected = [r for r in records if split == "all_history" or r["split"] == split]
                node_denom = np.array([denom["controlled_node_daily_kwh"][str(label)] for label in CONTROL_LABELS])
                item["ratios"][split] = {
                    "absolute_energy_over_system_daily_energy": absolute / denom["system_daily_kwh"],
                    "net_energy_over_system_daily_energy": net / denom["system_daily_kwh"],
                    "absolute_energy_over_pair_daily_energy": absolute / denom["controlled_pair_daily_kwh"],
                    "net_energy_over_pair_daily_energy": net / denom["controlled_pair_daily_kwh"],
                    "node_absolute_energy_over_node_daily_energy": {str(label): float(v) for label, v in zip(CONTROL_LABELS, node_abs / node_denom)},
                    "absolute_peak_over_system_mean_kw": abs_peak / denom["system_mean_kw"],
                    "absolute_peak_over_pair_mean_kw": abs_peak / denom["controlled_pair_mean_kw"],
                    "max_aligned_absolute_over_system_kw": float(max(np.max(np.abs(u).sum(axis=0) / r["power_kw"].sum(axis=1)) for r in selected)),
                    "max_aligned_absolute_over_pair_kw": float(max(np.max(np.abs(u).sum(axis=0) / r["power_kw"][:, positions].sum(axis=1)) for r in selected)),
                    "max_aligned_node_absolute_over_node_kw": {str(label): float(max(np.max(np.abs(u[j]) / r["power_kw"][:, pos]) for r in selected)) for j, (label, pos) in enumerate(zip(CONTROL_LABELS, positions))},
                }
            policies.append(item)
    check("exactly five paid policies", len(policies) == 5)
    output = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": inputs,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "definitions": {
            "load": "Historical observed net-demand P only, original per-unit values multiplied by 200 kW; includes original P measurement noise. Signed terminal power is preserved, including local negative injections. Positive and negative aggregates are reported separately; gross behind-meter demand is unavailable.",
            "time": "96 equal quarter-hour slots per day; historical day labels are synthetic independent replicate ordering.",
            "control_order": CONTROL_LABELS,
            "u": "Frozen planned mean active response in kW; positive increases load; no response draws or clipping used.",
            "absolute_energy": "0.25*sum(abs(u)); includes increases and decreases, not net additional consumption.",
            "fee": "0.25*sum(u**2/0.5), uncalibrated synthetic contract units, not established currency.",
            "elasticity": "0.5 kW per reward-price unit, matching frozen contract.",
            "ratios": "Dimensionless fractions, not percentages. Training is planning-safe historical denominator. Validation/all_history are descriptive existing-history denominators.",
            "aligned": "The same frozen mean day is aligned to each old observed day; the maximum ratio is descriptive, not a future delivery guarantee.",
            "exclusions": "No new future, forecast packet member, physical truth, topology, model fitting, AC solve, or stochastic evaluation is accessed.",
        },
        "historical_case": history["source_case_name"],
        "base_kw": history["base_kw"], "denominators": denominators,
        "policies": policies, "checks": checks,
        "check_count": len(checks), "passed_count": sum(item["passed"] for item in checks),
    }
    destination = STAGE / "history_load_audit.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks": len(checks), "policies": len(policies), "output": str(destination)}))


if __name__ == "__main__":
    main()
