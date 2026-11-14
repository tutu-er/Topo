"""Use paired measured P/Q without fixing X/R or synthesizing reactive power."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from ... import io
from .fetch import sha256

SPLIT = dict(train=[0, 192], validation=[192, 288], new_day=[288, 384])
START_DATES = ("2024-01-15", "2024-04-15", "2024-07-15", "2024-10-14")
CASES = ("paper15", "flynn16", "pengwah18", "soumalas11")
LOAD_RMS_FACTOR = .65


def quarter_hour_frame(frame):
    """Paired, finite P/Q means with >=12/15 valid minutes; no gap filling.

    Source one-phase voltage is only a quality flag, never a terminal voltage.
    No P/Q-ratio, power-sign or recovery-based filtering is applied.
    """
    required = {"timestamp_UTC", "active_power_kW", "reactive_power_kVar", "voltage_one_phase_V"}
    if not required <= set(frame.columns):
        raise ValueError("Required measured channels missing")
    time = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_UTC"], utc=True))
    if time.has_duplicates or time.isna().any() or (time != time.floor("min")).any():
        raise ValueError("Duplicate, invalid or non-minute source timestamps")
    d = frame.copy()
    d.index = time
    d = d.sort_index()
    pq = d[["active_power_kW", "reactive_power_kVar"]].astype(float)
    valid = np.isfinite(pq).all(axis=1) & d.voltage_one_phase_V.between(85, 264)
    paired = pq.where(valid, np.nan).rename(columns={"active_power_kW": "p_kw", "reactive_power_kVar": "q_kvar"})
    mean = paired.resample("15min").mean()
    count = paired.resample("15min").count().min(axis=1).astype(np.uint8)
    mean.loc[count < 12, ["p_kw", "q_kvar"]] = np.nan
    mean["valid_minutes"] = count
    return mean


def make_archive(directory):
    """Verify source bytes and derive a synchronized small archive."""
    directory = Path(directory).resolve()
    if (directory / "paired_pq_15min.npz").exists():
        raise FileExistsError("Preserve the completed archive; choose a new directory")
    manifest = io.read(directory / "sources.json")
    metadata_path = directory / "feeder_metadata.csv"
    if sha256(metadata_path) != manifest["metadata"]["sha256"]:
        raise ValueError("Source resource metadata hash differs")
    assets = pd.read_csv(metadata_path)
    assets = assets.loc[assets.feeder.isin(manifest["feeder_ids"])]
    resource_records = json.loads(assets.to_json(orient="records"))
    frames = []
    for number in manifest["feeder_ids"]:
        name = f"feeder_{number:03}.parquet"
        receipt = next(x for x in manifest["files"] if Path(x["name"]).name == name)
        path = directory / "raw" / name
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Source file hash differs: " + name)
        frame = pd.read_parquet(path, columns=["timestamp_UTC", "active_power_kW", "reactive_power_kVar", "voltage_one_phase_V"])
        frames.append(quarter_hour_frame(frame))
    if len(frames) < 18:
        raise ValueError("At least 18 distinct source feeders are required")
    start = min(f.index.min() for f in frames)
    end = max(f.index.max() for f in frames)
    times = pd.date_range(start, end, freq="15min", tz="UTC")
    aligned = [f.reindex(times) for f in frames]
    p = np.column_stack([f.p_kw.to_numpy() for f in aligned])
    q = np.column_stack([f.q_kvar.to_numpy() for f in aligned])
    count = np.column_stack([f.valid_minutes.fillna(0).to_numpy(np.uint8) for f in aligned])
    meta = dict(dataset=manifest["dataset"], version=manifest["version"], doi=manifest["doi"],
                source_manifest=io.ref(directory / "sources.json"), source_files=manifest["files"],
                source_resource_metadata=manifest["metadata"], resource_metadata_records=resource_records,
                feeder_ids=manifest["feeder_ids"], license=manifest["license"],
                aggregation="15-minute paired mean; >=12 valid source minutes; otherwise NaN; no interpolation",
                source_quality="Finite paired P/Q and source busbar voltage in [85,264] V",
                signs="P>0 consumption, P<0 export; Q>0 inductive, Q<0 capacitive",
                source_voltage_used_as_terminal=False, units=dict(p="kW", q="kvar"))
    archive = directory / "paired_pq_15min.npz"
    np.savez_compressed(archive, p_kw=p, q_kvar=q, valid_minutes=count,
                        timestamps_utc=np.asarray(times.strftime("%Y-%m-%dT%H:%M:%SZ"), dtype="U20"),
                        feeder_ids=np.asarray(manifest["feeder_ids"], int), metadata_json=np.asarray(json.dumps(meta)))
    io.write(directory / "archive.json", dict(**meta, archive=io.ref(archive), shape=list(p.shape),
             valid_bins_per_feeder=np.isfinite(p).sum(axis=0), environment=io.environment()))
    return archive


def build_from_arrays(labels, p_kw, q_kvar, timestamps, nominal_p_kw, feeder_ids, source_metadata):
    """Build one 4-day input; only its first 192 rows determine scaling."""
    labels = tuple(labels)
    p, q = np.asarray(p_kw, float), np.asarray(q_kvar, float)
    nominal = np.asarray(nominal_p_kw, float)
    ids = tuple(map(int, feeder_ids))
    if len(labels) != len(set(labels)) or not labels or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Unique labels and distinct source feeder IDs are required")
    if p.shape != (384, len(labels)) or q.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Require 384 complete paired P/Q rows; missing bins are rejected")
    if nominal.shape != (len(labels),) or not np.isfinite(nominal).all() or np.any(nominal <= 0):
        raise ValueError("Positive finite nominal active powers required")
    time = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if len(time) != 384 or time.isna().any() or np.any(time[1:]-time[:-1] != pd.Timedelta(minutes=15)):
        raise ValueError("Four contiguous days at 15 minutes are required")
    rms = np.sqrt(np.mean(p[:192]**2, axis=0))
    if np.any(rms <= 0):
        raise ValueError("Training active-power RMS must be positive")
    scale = LOAD_RMS_FACTOR * nominal / rms
    original = dict(source_metadata)
    metadata = dict(**original)
    metadata.update(node_mapping=[dict(label=label, feeder_id=i, scale=float(a),
                                     source_channel="measured net feeder P/Q", training_p_rms_kw=float(r))
                                  for label, i, a, r in zip(labels, ids, scale, rms, strict=True)],
                    scales=scale.tolist(), scale_fit_split="train", load_rms_factor=LOAD_RMS_FACTOR,
                    scaling="Per-node positive scalar fitted from train P RMS; exactly the same scalar applies to Q and every later row",
                    source_timestamps_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                    missing_policy="reject incomplete selected 15-minute window; upstream paired averages require >=12/15 valid minutes",
                    signs="P>0 consumption, P<0 export; Q>0 inductive, Q<0 capacitive",
                    units=dict(p="kW", q="kvar"), extra_device_baseline_added=False,
                    rx_constraint="none; R and X independently fitted", dr_ready=False,
                    dr_pending="Refit candidates on this new history and define baseline-consistent resource availability and fees before procurement",
                    interpretation="Measured aggregate-user trajectories driving a synthetic test network; not field terminal P/Q/V")
    return dict(labels=labels, p_kw=p*scale, q_kvar=q*scale,
                timestamps=time.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                split={key:list(value) for key,value in SPLIT.items()}, metadata=metadata)


def build_case(case_name, archive_path, start_date, *, expected_sha256=None):
    """Map the first N distinct feeders to the network's original terminals.

    Start at 08:00 Europe/Berlin. DST-changing windows are explicitly rejected.
    Network data supplies labels and nominal P for size only, not P/Q shape.
    """
    from topology_identification_standalone.data import load_case
    if case_name not in CASES:
        raise ValueError("Unknown predeclared test network")
    archive_path = Path(archive_path).resolve()
    digest = sha256(archive_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Archive SHA256 differs from the locked source")
    network = load_case(case_name)
    terminal = [b for b in network["buses"] if b["is_terminal"]]
    n = len(terminal)
    local_start = pd.Timestamp(start_date + "T08:00:00", tz="Europe/Berlin")
    local_stop = local_start + pd.DateOffset(days=4)
    if local_stop.tz_convert("UTC")-local_start.tz_convert("UTC") != pd.Timedelta(days=4):
        raise ValueError("Selected local window crosses a daylight-saving time change")
    wanted = pd.date_range(local_start.tz_convert("UTC"), periods=384, freq="15min")
    with np.load(archive_path, allow_pickle=False) as data:
        ids = data["feeder_ids"]
        if len(ids) < max(18, n) or len(set(map(int, ids))) != len(ids):
            raise ValueError("Source archive needs >=18 distinct feeders")
        times = pd.DatetimeIndex(pd.to_datetime(data["timestamps_utc"], utc=True))
        if times.has_duplicates:
            raise ValueError("Archive timestamps duplicate")
        ix = times.get_indexer(wanted)
        if np.any(ix < 0):
            raise ValueError("Selected dates absent from the source archive")
        p, q = data["p_kw"][ix, :n], data["q_kvar"][ix, :n]
        counts = data["valid_minutes"][ix, :n]
        if np.any(counts < 12):
            raise ValueError(f"Selected window has {int(np.sum(counts<12))} incomplete node-time bins")
        metadata = json.loads(str(data["metadata_json"]))
        records = metadata.get("resource_metadata_records", [])
        current = []
        for feeder in ids[:n]:
            known = [row for row in records if int(row["feeder"]) == int(feeder) and row["date"] <= start_date]
            if known:
                current.append(max(known, key=lambda row: row["date"]))
        changes = [row for row in records if int(row["feeder"]) in ids[:n]
                   and start_date < row["date"] < local_stop.strftime("%Y-%m-%d")]
        metadata.update(archive_path=str(archive_path), archive_sha256=digest,
                        source_archive_sha256=digest, case=case_name, start_date=start_date,
                        local_timezone="Europe/Berlin", local_start=str(local_start),
                        source_valid_minutes_min=int(counts.min()), source_valid_minutes_mean=float(counts.mean()),
                        resource_metadata_at_history_start=current, resource_metadata_changes_in_window=changes,
                        resource_metadata_note="Registration records, not measured real-time availability; later installations are never backfilled",
                        node_assignment="Original terminal order -> first N numbered distinct source feeders")
        return build_from_arrays([b["bus_id"] for b in terminal], p, q,
                                 wanted.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                                 [b["pd_kw"] for b in terminal], ids[:n], metadata)
