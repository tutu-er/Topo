"""Locked, resumable prediction -> price -> response -> AC -> WZZT run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import socket
import sys
import traceback
import uuid

import numpy as np
import pandas as pd

from . import physical_case, price_response, wzzt_adapter


BUNDLE_FILE = "privatebundle.pkl"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path, value):
    wzzt_adapter._write_json(Path(path), value)


def _pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)
    temporary.replace(path)


def _unpickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


@contextmanager
def _exclusive_lock(output_dir, config_hash):
    path = output_dir / "runner.lock"
    token = uuid.uuid4().hex
    record = {"pid": os.getpid(), "host": socket.gethostname(), "created_at": _now(),
              "config_hash": config_hash, "owner_token": token}
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
    except FileExistsError as exc:
        raise RuntimeError(f"run lock already exists: {path}; inspect its PID before removing a stale lock") from exc
    try:
        yield
    finally:
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("owner_token") == token:
                path.unlink()


def _source_hashes():
    files = list(Path(__file__).parent.glob("*.py"))
    core_file = Path(wzzt_adapter.core.__file__).resolve()
    # Include physical generation, preprocessing and solver adapters, not only
    # the top-level optimizer: they also determine the recorded observations.
    files.extend(core_file.parents[1].rglob("*.py"))
    return {str(p.resolve()): _sha(p) for p in sorted(set(files))}


def _observed(raw):
    return wzzt_adapter.make_observed_scenario(raw["P_terminal"], raw["Q_terminal"],
        raw["V_terminal"], raw["root_voltage"], raw["name"])


def _check_day(frame, day, terminals):
    index = pd.date_range(day, periods=24, freq="h")
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != terminals or not frame.index.equals(index):
        raise ValueError(f"{day} requires exactly 24 chronological hourly rows and declared terminal order")
    if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
        raise ValueError(f"{day} contains missing or negative household demand")


def _validate_bundle(bundle, terminals, config):
    days = sorted(bundle["history_days"])
    if len(days) != config["source"]["forecast_days"]:
        raise ValueError("history-day count differs from the locked forecast contract")
    cutoff = pd.Timestamp(config["action_date"]) - pd.Timedelta(days=2)
    for day in days:
        if pd.Timestamp(day) > cutoff or pd.Timestamp(day).weekday() >= 5:
            raise ValueError("forecast history includes an ineligible date")
        _check_day(bundle["history_days"][day], day, terminals)
    for phase in ("action", "validation"):
        _check_day(bundle["actual_days"][phase], config[phase + "_date"], terminals)
    if pd.Timestamp(config["validation_date"]) <= pd.Timestamp(config["action_date"]):
        raise ValueError("validation must follow the action day")
    return days


def _physical_options(config):
    physical = dict(config["physical"])
    physical["voltage_limits_pu"] = (physical.pop("voltage_min_pu"), physical.pop("voltage_max_pu"))
    return physical


def prepare(config, data_root, calibration_dir, output_dir):
    """May run in the data reader's environment; no optimizer/AC is called."""
    from .data_inputs import prepare_iflex_bundle
    net = physical_case.build_reference_case(config["case_key"])
    terminals = sorted(int(x) for x in net.terminal_to_original)
    bundle = prepare_iflex_bundle(data_root, calibration_dir, terminals, config, output_dir)
    _validate_bundle(bundle, terminals, config)
    _pickle(output_dir / BUNDLE_FILE, bundle)
    receipt = {"status": "prepared", "created_at": _now(), "config_hash": _fingerprint(config),
               "bundle_sha256": _sha(output_dir / BUNDLE_FILE), "data_manifest": bundle["data_manifest"],
               "bank_metadata": bundle["bank_metadata"], "python": sys.version}
    _json(output_dir / "prepared_receipt.json", receipt)
    return receipt


def _fit_summary(strategy):
    fit = strategy.get("fit", {"status": "not_called", "has_model": False})
    result = {"strategy": strategy["strategy"], "stage_status": strategy["stage_status"],
              "assignment": strategy["assignment"], "planning_feasible": strategy["planning_audit"]["feasible"],
              "fit": wzzt_adapter.summarize_fit(fit), "core_call_count": strategy.get("core_call_count", 0),
              "validation_role": "selection_not_untouched_test"}
    if "periods" in strategy:
        result["periods"] = {phase: {"payment_nok": data["total_payment_nok"],
            "response_constraints_pass": data["response_audit"]["valid"],
            "usable_ac_solution": data["physical_checks"]["usable_ac_solution"],
            "voltage_limits_pass": data["physical_checks"]["voltage_limits_pass"],
            "nonconverged_samples": data["physical_checks"]["nonconverged_samples"]}
            for phase, data in strategy["periods"].items()}
    result["recovery"] = strategy.get("recovery")
    result["exception"] = strategy.get("exception")
    return result


def run(config, data_root, calibration_dir, output_dir, *, resume=False):
    """Run each strategy once; unresolved solves are final records, not retries."""
    bundle_path = output_dir / BUNDLE_FILE
    prepared_path = output_dir / "prepared_receipt.json"
    if not bundle_path.exists() or not prepared_path.exists():
        raise RuntimeError("prepared private bundle missing; run --prepare-only in the data-reader environment first")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["config_hash"] != _fingerprint(config) or prepared["bundle_sha256"] != _sha(bundle_path):
        raise ValueError("prepared bundle/config fingerprint mismatch")
    mean_csv = calibration_dir / "outputs" / "control_training_menu_factors.csv"
    price_csv = data_root / "raw" / "price_signals.csv"
    saved_inputs = prepared.get("data_manifest", {}).get("source_hashes", {})
    for input_path in (mean_csv, price_csv):
        if saved_inputs and saved_inputs.get(str(input_path)) != _sha(input_path):
            raise ValueError("prepared planning-input hash mismatch: " + str(input_path))
    provenance = {"config": config, "source_sha256": _source_hashes(), "bundle_sha256": prepared["bundle_sha256"],
                  "menu_mean_sha256": _sha(mean_csv), "price_signal_sha256": _sha(price_csv)}
    run_fingerprint = _fingerprint(provenance)
    state_path = output_dir / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("run already exists; use --resume to avoid replacing completed records")
        if state["run_fingerprint"] != run_fingerprint:
            raise ValueError("resume refused: configuration, code, bundle or menu inputs changed")
        if state["status"] == "complete":
            return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    else:
        state = {"status": "running", "created_at": _now(), "run_fingerprint": run_fingerprint,
                 "completed_history_dates": [], "completed_strategies": [], "provenance": provenance,
                 "environment": {"python": sys.version, "executable": sys.executable,
                     "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}}
    state["status"] = "running"
    _json(state_path, state)
    bundle = _unpickle(bundle_path)
    net = physical_case.build_reference_case(config["case_key"])
    terminals = sorted(int(x) for x in net.terminal_to_original)
    days = _validate_bundle(bundle, terminals, config)
    physical_options = _physical_options(config)
    history = {}
    for i, day in enumerate(days):
        path = output_dir / "history" / (day + ".pkl")
        if path.exists() and resume:
            simulated = _unpickle(path)
        else:
            p = bundle["history_days"][day]
            q = physical_case.synthetic_q_from_reference_pf(p, case_key=config["case_key"])
            simulated = physical_case.simulate_observations(net, p, q, seed=config["seeds"]["history"] + i,
                name="history_" + day, **physical_options)
            _pickle(path, simulated)
        _json(path.with_suffix(".checks.json"), simulated["checks"])
        if simulated["observed"] is None:
            raise RuntimeError(f"history AC unusable on {day}; all physical evidence retained")
        history[day] = simulated
        if day not in state["completed_history_dates"]:
            state["completed_history_dates"].append(day)
        _json(state_path, state)

    # Only historical noisy meters enter the mean forecast and ordinary D-opt.
    base_kw = net.base_mva * 1000.
    hp = [history[d]["observed"]["P_terminal"].to_numpy() * base_kw for d in days]
    hq = [history[d]["observed"]["Q_terminal"].to_numpy() * base_kw for d in days]
    forecast_p, forecast_q = np.mean(hp, axis=0), np.mean(hq, axis=0)
    model = price_response.MenuModel.load(mean_csv, price_csv)
    planning = config["planning"]
    limits = {"power_limits_kw": planning["power_limits_kw"],
              "max_daily_energy_change_fraction": planning["max_daily_energy_change_fraction"]}
    planned = price_response.plan_prices(forecast_p, forecast_q, model,
        history_p_kw=np.concatenate(hp), history_q_kvar=np.concatenate(hq),
        budget_nok=planning["budget_nok"], objective=planning["objective"], ridge=planning["ridge"],
        budget_mode="conservative", settlement_baseline_kw=forecast_p, **limits)
    _json(output_dir / "plan.json", {**planned, "forecast_p_kw": forecast_p, "forecast_q_kvar": forecast_q,
        "forecast_dates": days, "forecast_source": "mean of ten past noisy observed P/Q weekday profiles",
        "settlement_baseline": "same frozen action forecast for each independently simulated 24h period",
        "validation_action": "repeat frozen menu assignment; no replan"})
    by_name = {a["name"]: a for a in planned["candidates"]}
    strategies = [("no_reward", {"name": "no_reward", "menus": ["no_reward"] * len(terminals)}),
                  ("fixed_P2_first_node", {"name": "node_0_P_2", "menus": ["P_2"] + ["no_reward"] * (len(terminals)-1)}),
                  ("planned_d_opt", planned["selected_assignment"])]
    draws, draw_metadata = {}, {}
    for phase in ("action", "validation"):
        draws[phase], draw_metadata[phase] = price_response.draw_per_menu_banks(
            bundle["response_banks"][phase], config["seeds"]["response_" + phase], len(terminals))
    _json(output_dir / "response_draws.json", {"draws": draw_metadata, "bank_metadata": bundle["bank_metadata"],
        "coupling": "same entire realized menu trajectories across strategies; independent action/validation draws"})
    train_days = days[-config["source"]["history_fit_days"]:]
    training_history = [_observed(history[d]["observed"]) for d in train_days]
    records = []
    for strategy_name, assignment in strategies:
        strategy_dir = output_dir / "strategies" / strategy_name
        record_path = strategy_dir / "result.json"
        if record_path.exists() and resume:
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        record = {"strategy": strategy_name, "assignment": assignment, "planning_audit": by_name[assignment["name"]],
                  "stage_status": "running", "core_call_count": 0, "created_at": _now(), "periods": {}}
        try:
            if not record["planning_audit"]["feasible"]:
                record["stage_status"] = "planning_infeasible"
            else:
                simulations = {}
                for phase in ("action", "validation"):
                    period_path = strategy_dir / (phase + ".pkl")
                    if period_path.exists() and resume:
                        cached = _unpickle(period_path)
                        response, simulation = cached["response"], cached["simulation"]
                    else:
                        actual_p = bundle["actual_days"][phase]
                        actual_q = physical_case.synthetic_q_from_reference_pf(actual_p, case_key=config["case_key"])
                        response = price_response.realize_response(actual_p, actual_q, assignment, model, draws[phase],
                            settlement_baseline_kw=forecast_p, budget_nok=planning["budget_nok"], **limits)
                        realized_p = pd.DataFrame(response["p_kw"], index=actual_p.index, columns=terminals)
                        realized_q = pd.DataFrame(response["q_kvar"], index=actual_p.index, columns=terminals)
                        simulation = physical_case.simulate_observations(net, realized_p, realized_q,
                            seed=config["seeds"][phase], name=strategy_name + "_" + phase, **physical_options)
                        _pickle(period_path, {"response": response, "simulation": simulation})
                    simulations[phase] = simulation
                    record["periods"][phase] = {"total_payment_nok": response["total_payment_nok"],
                        "payment_nok_per_node": response["payment_nok_per_node"],
                        "credits_nok_per_node": response["credits_nok_per_node"],
                        "response_audit": response["audit"], "physical_checks": simulation["checks"]}
                if any(s["observed"] is None for s in simulations.values()):
                    record["stage_status"] = "skipped_invalid_ac"
                    record["fit"] = {"status": "skipped_invalid_ac", "has_model": False}
                else:
                    record["core_call_count"] = 1
                    fit_path = strategy_dir / "wzzt.json"
                    if fit_path.exists() and resume:
                        fit = json.loads(fit_path.read_text(encoding="utf-8"))
                        if fit["status"] == "running":
                            fit.update({"status": "interrupted", "has_model": False,
                                "resume_note": "partial solver checkpoint retained; no automatic duplicate solve"})
                    else:
                        fit = wzzt_adapter.fit_observed_wzzt(
                            training_history + [_observed(simulations["action"]["observed"])],
                            [_observed(simulations["validation"]["observed"])], config["wzzt"], fit_path)
                    record["fit"] = fit
                    record["recovery"] = wzzt_adapter.evaluate_effective_supports(fit, simulations["action"]["truth"]["truth_supports"])
                    record["stage_status"] = "evaluated" if fit["has_model"] else "fit_failed"
        except Exception as exc:
            record["stage_status"] = "failed"
            record["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        record["completed_at"] = _now()
        _json(record_path, record)
        records.append(wzzt_adapter._plain(record))
        if strategy_name not in state["completed_strategies"]:
            state["completed_strategies"].append(strategy_name)
        _json(state_path, state)
    summaries = [_fit_summary(record) for record in records]
    summary = {"status": "complete", "completed_at": _now(), "run_fingerprint": run_fingerprint,
        "strategy_denominator": len(strategies), "recorded_strategies": len(records),
        "technical_flow_success": all(s["stage_status"] == "evaluated" for s in records),
        "fit_complete_count": sum(s.get("fit", {}).get("status") == "complete" for s in records),
        "fit_incomplete_count": sum(s.get("fit", {}).get("status") == "incomplete" for s in records),
        "failed_or_skipped_count": sum(s["stage_status"] != "evaluated" for s in records),
        "operationally_feasible_strategy_count": sum(
            len(s.get("periods", {})) == 2 and all(
                p["response_audit"]["valid"] and p["physical_checks"]["usable_ac_solution"]
                and p["physical_checks"]["voltage_limits_pass"] for p in s["periods"].values())
            for s in records),
        "core_call_count": sum(s["core_call_count"] for s in records),
        "strategies": summaries, "history_dates": days, "wzzt_training_dates": train_days,
        "case": {"name": config["case_key"], "terminals": terminals, "terminal_count": len(terminals),
                 "base_mva": net.base_mva, "impedance_scale": net.metadata["impedance_scale"]},
        "scope": "one paired implementation-check realization; no price superiority or recovery-probability claim",
        "validation_role": "independent observations used for WZZT path selection, not an untouched test set",
        "unresolved_fits_retained": True, "sample_dropping": False, "data_manifest": bundle["data_manifest"]}
    _json(output_dir / "summary.json", summary)
    state.update({"status": "complete", "completed_at": summary["completed_at"], "summary_file": "summary.json"})
    _json(state_path, state)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data_calibration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_acquired = False
    try:
        with _exclusive_lock(output, _fingerprint(config)):
            lock_acquired = True
            if args.prepare_only:
                if (output / "run_state.json").exists():
                    raise RuntimeError("do not replace a prepared bundle after a run has begun; use another output directory")
                result = prepare(config, args.data_root.resolve(), args.calibration_dir.resolve(), output)
            else:
                result = run(config, args.data_root.resolve(), args.calibration_dir.resolve(), output, resume=args.resume)
        print(json.dumps({"status": result["status"], "output_dir": str(output),
            "technical_flow_success": result.get("technical_flow_success"),
            "fit_complete_count": result.get("fit_complete_count"), "fit_incomplete_count": result.get("fit_incomplete_count")}, indent=2))
        return 0 if result.get("technical_flow_success", True) else 2
    except Exception as exc:
        failure = {"status": "failed", "time": _now(), "exception_type": type(exc).__name__,
                   "exception": str(exc), "traceback": traceback.format_exc()}
        if lock_acquired:
            _json(output / "runner_failure.json", failure)
        print(json.dumps({"status": "failed", "exception_type": type(exc).__name__, "exception": str(exc)}, indent=2))
        return 1
