"""Measured-only L1 detection entry; the M7 experiment harness stays unchanged.

Run from this standalone workspace: python detect_simple.py --help
Only the existing H0/H1 pair is solved. No oracle loss, scan, or fixed-RX fit.
Calibration must be supplied explicitly for the same declared protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from theft_wzzt.theft.credibility import calibrated_rank, compare_h0_h1
from theft_wzzt.theft.identified_tree import edge_weight_bounds, load_identified, to_theft_tree
from theft_wzzt.theft.loss_models import amplitude_envelope, estimate_loss
from theft_wzzt.theft.theft_model import TheftData, data_from_scenario

ROOT = Path(__file__).resolve().parent
MATRIX_CHANNELS = ("P_terminal", "Q_terminal", "drop_target")
SERIES_CHANNELS = ("root_voltage", "P0_measured", "Q0_measured")


def prepare(measurements, identified, window=None):
    """Build fixed inputs from six measured channels; never inspect truth keys."""
    tree, weights = to_theft_tree(identified)
    index = measurements["P_terminal"].index
    observed = list(tree.observed)
    if not len(index) or not index.is_unique:
        raise ValueError("Require a nonempty unique time index")
    measured = {}
    for name in MATRIX_CHANNELS:
        frame = measurements[name]
        if (not frame.index.equals(index) or not frame.columns.is_unique
                or set(frame.columns) != set(observed)):
            raise ValueError(f"{name}: inconsistent time index or terminal labels")
        measured[name] = frame.loc[:, observed]
    for name in SERIES_CHANNELS:
        value = measurements[name]
        if not value.index.equals(index):
            raise ValueError(f"{name}: inconsistent time index")
        measured[name] = value
    if any(not np.all(np.isfinite(v.to_numpy(dtype=float))) for v in measured.values()):
        raise ValueError("Measured channels must be finite")
    settings = measurements.get("scenario_settings", {})
    measured["scenario_settings"] = settings
    start, stop = (0, len(index)) if window is None else window
    if (not isinstance(start, int) or not isinstance(stop, int)
            or not 0 <= start < stop <= len(index)):
        raise ValueError("Require 0 <= window start < stop <= sample count")
    loss_p, loss_q = estimate_loss(measured, weights, tree, model="L1")
    full = data_from_scenario(measured, loss_p_estimate=loss_p, loss_q_estimate=loss_q)
    full.validate(len(observed))
    sl = slice(start, stop)
    data = TheftData(*(getattr(full, k)[sl] for k in
                       ("p", "q", "y", "balance", "amplitude", "extra_q")),
                     full.voltage_scale, full.balance_scale)
    # Exclude replicate-specific RNG seeds. This signature checks declared
    # configuration, not exchangeability or absence of distribution shift.
    protocol = {
        "version": 1, "loss_model": "L1", "case": identified.case_key,
        "edges": tree.edges, "observed": tree.observed,
        "weights": [v.tolist() for v in weights], "weight_factors": [.5, 2.],
        "sample_count": len(index), "window": [start, stop],
        "scales": {k: settings.get(k, default) for k, default in
                   (("root_mean", 1.02), ("v_noise_rel", .0002),
                    ("master_noise_rel", .002), ("pq_noise_rel", .005))},
        "conditions": {k: settings.get(k) for k in
                       ("root_observation", "root_meter_noise_rel", "profile_scenario")},
    }
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return tree, data, edge_weight_bounds(weights), amplitude_envelope(measured, loss_p).iloc[sl], signature


def decision(gain, calibration, signature, alpha=.05):
    """One rule only: empirical rank. None means unavailable, not no theft."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Require 0 < alpha < 1")
    if calibration is None:
        return {"alarm": None, "status": "uncalibrated", "rank": None, "null_n": 0}
    if calibration.get("protocol_signature") != signature:
        raise ValueError("Calibration protocol mismatch (tree, L1, window, noise or profile)")
    values = np.asarray(calibration.get("null_gains", []), dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Require nonempty finite one-dimensional null gains")
    rank = calibrated_rank(gain, values)
    enough = 1 / (len(values) + 1) <= alpha
    return {"alarm": bool(rank <= alpha) if enough else None,
            "status": "calibrated" if enough else "insufficient_calibration",
            "rank": rank, "null_n": len(values), "alpha": alpha,
            "min_resolvable_rank": 1 / (len(values) + 1)}


def detect(measurements, identified, *, window=None, calibration=None, alpha=.05, time_limit=240.):
    tree, data, bounds, envelope, signature = prepare(measurements, identified, window)
    decision(0., calibration, signature, alpha)  # reject invalid calibration before fitting
    comparison, _, fit = compare_h0_h1(tree, data, weight_bounds=bounds, time_limit=time_limit)
    verdict = decision(comparison["gain"], calibration, signature, alpha)
    selected = fit.selected_locations(tree)
    return {
        "protocol_signature": signature, "loss_model": "L1", "decision": verdict,
        "gain": comparison["gain"],
        "gain_voltage": comparison["h0"]["voltage_loss"] - comparison["h1"]["voltage_loss"],
        "gain_balance": comparison["h0"]["balance_loss"] - comparison["h1"]["balance_loss"],
        "signed_balance": data.balance.tolist(), "amplitude_input_pu": data.amplitude.tolist(),
        "extra_q_input_pu": data.extra_q.tolist(),
        "amplitude_envelope_pu": envelope[["lower", "point", "upper"]].to_numpy().T.tolist(),
        "envelope_order": ["lower", "point", "upper"],
        "envelope_assumption": "actual loss in [0.75,1.25]*estimate; meter error excluded",
        "candidate_locations_by_time": selected,
        "reported_locations_by_time": selected if verdict["alarm"] else None,
        "candidate_regions": {str(identified.label_of(s)): sorted(s) for s in identified.supports},
        "localization_status": "optimizer_choices_only_not_unique_location_certificates",
        "comparison": comparison,
    }


def load_measurements(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    index = pd.Index(raw["index"])
    result = {k: pd.DataFrame(raw[k], index=index, columns=raw["terminals"]) for k in MATRIX_CHANNELS}
    result.update({k: pd.Series(raw[k], index=index) for k in SERIES_CHANNELS})
    result["scenario_settings"] = raw.get("scenario_settings", {})
    return result


def main():
    parser = argparse.ArgumentParser(description="量测 → L1 线损 → 固定幅值 → H0/H1 → 经验秩判定")
    parser.add_argument("measurements", type=Path, help="仅含量测的 JSON")
    parser.add_argument("--tree", type=Path, default=ROOT / "outputs/theft/identified_tree.json")
    parser.add_argument("--window", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--calibration", type=Path, help="同协议 null_gains；缺省仅输出未校准评分")
    parser.add_argument("--output", type=Path, required=True, help="新结果路径，拒绝覆盖已有文件")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请选择新文件名")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    result = detect(load_measurements(args.measurements), load_identified(args.tree),
                    window=args.window, calibration=calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("gain", "gain_voltage", "gain_balance", "decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
