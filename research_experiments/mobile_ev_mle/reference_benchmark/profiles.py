"""Calendar-only reference windows and development-only forecast selection.

The saved future arrays are simulator inputs.  A policy must receive only the
arrived historical prefix through the caller's separate observed-data API.
No AC fit, topology score, future magnitude filter or evaluation tuning occurs.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..full_pipeline.forecast import ForecastConfig, select_on_development
from .data_sources import load_simbench_profiles, load_simbench_pv, profile_diagnostics
from .networks import NETWORK_NAMES, build_network


PROFILE_NAMES = ("lv_rural1", "lv_rural2", "lv_rural3", "lv_semiurb4", "lv_semiurb5", "lv_urban6")
PHASE_BOUNDS = {
    "development": ("2016-01-01T00:00:00Z", "2016-04-01T00:00:00Z"),
    "calibration": ("2016-04-01T00:00:00Z", "2016-10-01T00:00:00Z"),
    "evaluation": ("2016-10-01T00:00:00Z", "2017-01-01T00:00:00Z"),
}
DEFAULT_COUNTS = dict(development=4, calibration=16, evaluation=12)
FORECAST_ORIGINS = (96, 120, 144, 168)
FORECAST_CANDIDATES = (
    ForecastConfig(method="persistence"),
    ForecastConfig(method="seasonal_naive", seasonal_slots=96),
    ForecastConfig(method="ridge", ridge_alpha=10.),
    ForecastConfig(method="ridge", ridge_alpha=100.),
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json_bytes(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def calendar_windows(timestamps, *, history=192, horizon=12, counts=None):
    """Select equally spaced starts inside phase calendars, without values."""
    if history != 192 or horizon != 12:
        raise ValueError("reference v1 freezes H=192 and T=12")
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(PHASE_BOUNDS) or any(int(v) != v or v < 1 for v in counts.values()):
        raise ValueError("positive counts for development/calibration/evaluation required")
    t = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError("ordered unique UTC timestamps required")
    clock = t.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    if not np.all(np.diff(clock) == np.timedelta64(15, "m")):
        raise ValueError("native regular 15-minute clock required")
    length = history + horizon
    out = {}
    for phase, bounds in PHASE_BOUNDS.items():
        lo = int(t.searchsorted(pd.Timestamp(bounds[0]), side="left"))
        end = int(t.searchsorted(pd.Timestamp(bounds[1]), side="left"))
        last = end - length
        if last < lo or end - lo < counts[phase] * length:
            raise ValueError(f"not enough nonoverlapping calendar support for {phase}")
        starts = np.linspace(lo, last, int(counts[phase]), dtype=int)
        if len(starts) > 1 and np.any(np.diff(starts) < length):
            raise ValueError("calendar windows overlap")
        out[phase] = [dict(index=i, phase=phase, start_row=int(s), stop_row=int(s + length),
            history_rows=history, horizon_rows=horizon,
            history_start_utc=t[s].isoformat(), forecast_cutoff_utc=t[s + history - 1].isoformat(),
            future_start_utc=t[s + history].isoformat(), future_stop_utc=t[s + length - 1].isoformat(),
            selection="equally spaced calendar indices only; no load values or AC scores")
            for i, s in enumerate(starts)]
    all_intervals = sorted((w["start_row"], w["stop_row"]) for rows in out.values() for w in rows)
    if any(a[1] > b[0] for a, b in zip(all_intervals, all_intervals[1:])):
        raise AssertionError("cross-phase windows overlap")
    return out


def main():
    """Prepare native 15-minute windows; no AC fitting or EV optimization."""
    import argparse
    from .config import CONFIG
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--output',type=Path,default=root/'outputs/inputs_v2')
    parser.add_argument('--profile-directory',type=Path,
        default=root/'data_external/simbench/scenario0')
    args=parser.parse_args()
    value=prepare(args.output,dict(CONFIG,profile_directory=str(args.profile_directory)))
    print(json.dumps(dict(output=str(args.output),windows={p:len(rows) for p,rows in value['windows'].items()},
                          scientific_fits=0,scientific_planning_calls=0)))


def profile_assignment(networks, seed=20421018):
    """Balanced fixed-seed reuse of the six native LV aggregate profiles."""
    assignment = {}
    for j, (name, net) in enumerate(sorted(networks.items())):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), j]))
        ids = list(net["observation_bus_ids"])
        values = np.resize(np.asarray(PROFILE_NAMES, object), len(ids))
        rng.shuffle(values)
        assignment[name] = dict(zip(ids, values.tolist()))
    return assignment


def select_forecasters(development_windows, *, history=192, horizon=12):
    """Only raw development history, with internal targets before EV planning."""
    if not development_windows or any(w["phase"] != "development" for w in development_windows):
        raise ValueError("forecast model selection accepts development windows only")
    result = {}
    for signal in ("p", "q"):
        records = [dict(phase="development", values=np.asarray(w[signal + "_relative"], float)[:history],
                        timestamps=w["timestamps"][:history]) for w in development_windows]
        selected, receipt = select_on_development(records, candidates=FORECAST_CANDIDATES,
            horizon=horizon, origins=FORECAST_ORIGINS, phase="development", nonnegative=False)
        result[signal] = dict(config=asdict(selected), selection=receipt)
    result.update(history_prefix_only=True, planned_EV_future_used=False,
        calibration_used=False, evaluation_used=False, nonnegative=False,
        independent_P_and_Q_selection=True)
    return result


def _node_history_diagnostics(network, assignment, window, history=192):
    ids = network["observation_bus_ids"]
    node = {x["bus_id"]: x for x in network["buses"]}
    cols = [PROFILE_NAMES.index(assignment[i]) for i in ids]
    p = np.asarray(window["p_relative"], float)[:history, cols] * np.array([node[i]["pd_kw"] for i in ids])
    q = np.asarray(window["q_relative"], float)[:history, cols] * np.array([node[i]["qd_kvar"] for i in ids])
    quality = profile_diagnostics(p, q)
    center = p - p.mean(0)
    norm = np.linalg.norm(center, axis=0)
    z = np.divide(center, norm, out=np.zeros_like(center), where=norm > 1e-15)
    corr = np.abs((z.T @ z)[np.triu_indices(len(ids), 1)])
    quality.update(window_index=window["index"], phase=window["phase"], history_only=True,
        absolute_P_correlation_max=float(corr.max()) if len(corr) else None,
        absolute_P_correlation_mean=float(corr.mean()) if len(corr) else None,
        unique_PQ_profile_count=len(set(assignment.values())),
        assigned_nodes=len(ids), nominal_P_Q_rescaled=False,
        used_to_select_windows_forecasters_or_actions=False)
    return quality


def prepare(output, config=None):
    """Save frozen native windows and development-only forecast configs.

    Optional config keys: profile_directory, history/H, horizon/T,
    phase_counts, network_names, assignment_seed.  Returned arrays are lists
    for direct JSON serialization.  The selector must never receive this full
    prepared object, because it also stores simulator-only future arrays.
    """
    if is_dataclass(config):
        config = asdict(config)
    config = dict(config or {})
    root = Path(__file__).resolve().parents[3]
    directory = Path(config.get("profile_directory", root / "data_external/simbench/scenario0"))
    history = int(config.get("history", config.get("H", 192)))
    horizon = int(config.get("horizon", config.get("T", 12)))
    counts = dict(config.get("phase_counts", DEFAULT_COUNTS))
    names = tuple(config.get("network_names", NETWORK_NAMES))
    seed = int(config.get("assignment_seed", 20421018))
    profiles = load_simbench_profiles(directory, PROFILE_NAMES)
    pv = load_simbench_pv(directory, "PV1")
    if profiles["time"] != pv["time"] or profiles["dt_hours"] != pv["dt_hours"]:
        raise ValueError("P/Q and PV native time axes differ")
    starts = calendar_windows(profiles["time"], history=history, horizon=horizon, counts=counts)
    networks = {name: build_network(name) for name in names}
    assignment = profile_assignment(networks, seed)
    windows = {}
    for phase, rows in starts.items():
        windows[phase] = []
        for row in rows:
            s, e = row["start_row"], row["stop_row"]
            windows[phase].append(dict(**row, timestamps=list(profiles["time"][s:e]),
                p_relative=profiles["p_relative"][s:e].tolist(),
                q_relative=profiles["q_relative"][s:e].tolist(),
                pv_relative=pv["p_relative"][s:e].tolist(),
                profile_names=list(PROFILE_NAMES),
                future_arrays_visibility="simulator only; policy receives arrived prefix through allowlist"))
    forecasts = select_forecasters(windows["development"], history=history, horizon=horizon)
    diagnostics = {name: {phase: [_node_history_diagnostics(net, assignment[name], w, history)
        for w in ws] for phase, ws in windows.items()} for name, net in networks.items()}
    out = dict(schema="reference_prepared_profiles_v1", config=dict(
        supplied=_plain(config), history=history, horizon=horizon, dt_hours=.25,
        phase_counts=counts, phase_bounds=PHASE_BOUNDS, assignment_seed=seed,
        network_names=names, nonnegative=False, total_window_rows=history + horizon,
        no_data_dependent_network_or_time_selection=True), profile_names=list(PROFILE_NAMES),
        networks={name: dict(physical_sha256=net["physical_sha256"],
            observation_bus_ids=net["observation_bus_ids"], base_kw=net["base_kw"],
            total_pd_kw=net["total_pd_kw"], total_qd_kvar=net["total_qd_kvar"],
            provenance=net["provenance"], metadata=net["metadata"]) for name, net in networks.items()},
        windows=windows, assignments=assignment, assignment=assignment,
        forecast_configs={signal: forecasts[signal]["config"] for signal in ("p", "q")},
        forecast_config_p=forecasts["p"]["config"], forecast_config_q=forecasts["q"]["config"],
        forecast_selection=forecasts, historical_design_diagnostics=diagnostics,
        data_provenance=dict(load=profiles["provenance"], pv=pv["provenance"],
            profile_scope="six native LV aggregation profiles; no HS/EHV or independent-household claim",
            mixed_standard_network_and_external_profiles=True,
            raw_synchronous_field_PQV=False, no_future_normalization=True,
            license_file="audit/SimBench_1.6.3_LICENSE.txt"))
    out = _plain(out)
    payload = _json_bytes(out)
    digest = hashlib.sha256(payload).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "prepared.json.gz"
    if path.exists():
        existing = gzip.decompress(path.read_bytes())
        if existing != payload:
            raise FileExistsError("different prepared inputs already exist; preserve and choose a new output directory")
    else:
        path.write_bytes(gzip.compress(payload, mtime=0))
    manifest = dict(schema=out["schema"], prepared_file=path.name,
        payload_bytes=len(payload), payload_sha256=digest,
        compressed_bytes=path.stat().st_size, compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        windows={k: len(v) for k, v in windows.items()},
        source_sha256=profiles["provenance"]["sha256"],
        PV_sha256=pv["provenance"]["sha256"],
        forecast_config_p=out["forecast_config_p"], forecast_config_q=out["forecast_config_q"],
        scientific_fits=0, scientific_planning_calls=0,
        assignment_seed=seed, future_data_visibility="simulator only")
    (output / "manifest.json").write_bytes(_json_bytes(manifest))
    return out


if __name__=='__main__':main()
