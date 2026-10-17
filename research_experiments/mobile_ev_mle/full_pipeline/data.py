"""OPSD cumulative-energy preprocessing and calendar-only simulation windows.

The official 2020-04-15 release reports cumulative kWh, including previously
filled gaps.  A power interval is usable only when both cumulative endpoints
are unmarked, finite, nondecreasing and exactly 15 minutes apart.  No dataset
normalization, interpolation, future fill, or household duplication is used.
Returned future values belong to the simulator, never the planning interface.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd


SOURCE_URL = "https://data.open-power-system-data.org/household_data/2020-04-15/"
CSV_NAME = "household_data_15min_singleindex.csv"
RESIDENTIAL_COLUMNS = tuple(f"DE_KN_residential{i}_grid_import" for i in range(1, 7))
MIXED8_COLUMNS = RESIDENTIAL_COLUMNS + ("DE_KN_industrial2_grid_import", "DE_KN_industrial3_grid_import")
MIXED10_COLUMNS = RESIDENTIAL_COLUMNS + tuple(
    f"DE_KN_{name}_grid_import" for name in ("industrial1", "industrial2", "public1", "public2"))
PHASES = ("training", "development", "calibration", "evaluation")


def file_sha256(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def utc64(value):
    value = pd.Timestamp(value)
    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return value.tz_localize(None).to_datetime64().astype("datetime64[ns]")


def utc_string(value):
    return pd.Timestamp(value).isoformat() + ("Z" if pd.Timestamp(value).tzinfo is None else "")


@dataclass(frozen=True)
class PowerSeries:
    timestamps: np.ndarray
    p_kw: np.ndarray
    valid: np.ndarray
    columns: tuple[str, ...]
    csv_line_numbers: np.ndarray
    provenance: dict

    def __post_init__(self):
        stamps = np.asarray(self.timestamps, dtype="datetime64[ns]").copy()
        power, valid = np.asarray(self.p_kw, float).copy(), np.asarray(self.valid, bool).copy()
        lines = np.asarray(self.csv_line_numbers, int).copy()
        if power.ndim != 2 or valid.shape != power.shape or power.shape != (len(stamps), len(self.columns)):
            raise ValueError("power/valid must have shape (time, distinct physical columns)")
        if len(set(self.columns)) != len(self.columns) or not self.columns:
            raise ValueError("distinct physical meter columns required")
        if lines.shape != (len(stamps),) or np.any(np.diff(stamps) <= np.timedelta64(0, "ns")):
            raise ValueError("strictly increasing timestamps and matching CSV lines required")
        if np.any(valid & (~np.isfinite(power) | (power < 0))):
            raise ValueError("usable grid-import powers must be finite and nonnegative")
        for name, value in (("timestamps", stamps), ("p_kw", power), ("valid", valid), ("csv_line_numbers", lines)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def load_opsd_power(path, columns=RESIDENTIAL_COLUMNS, *, expected_sha256=None):
    """Difference cumulative kWh into backward-interval average kW.

    Row t becomes (E[t]-E[t-1])/0.25, available at timestamp t and describing
    [t-15min,t]. Marking either endpoint discards that interval for that column.
    Marks for unrelated appliance columns do not discard a whole building.
    """
    path = Path(path)
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must be nonempty and distinct; no replicated households")
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("raw CSV SHA256 differs from the registered dataset")
    frame = pd.read_csv(path, usecols=["utc_timestamp", "interpolated", *columns])
    dates = pd.to_datetime(frame["utc_timestamp"], utc=True, errors="raise")
    stamps = dates.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    if len(stamps) < 2 or np.any(np.diff(stamps) <= np.timedelta64(0, "ns")):
        raise ValueError("source rows must have distinct increasing UTC timestamps")
    energies = frame[list(columns)].to_numpy(float)
    marks = frame["interpolated"].fillna("").astype(str).str.lower()
    # The README's example uses ';', while the released CSV uses ' | '.
    # Both separators and metadata capitalization are handled explicitly.
    tokens = [{part.strip().lower() for part in re.split(r"[;|]", value) if part.strip()} for value in marks]
    flagged = np.asarray([[column.lower() in row for column in columns] for row in tokens], bool)
    changes = np.diff(energies, axis=0)
    regular = np.diff(stamps) == np.timedelta64(15, "m")
    endpoint_missing = ~np.isfinite(energies[:-1]) | ~np.isfinite(energies[1:])
    marked = flagged[:-1] | flagged[1:]
    resets = np.isfinite(changes) & (changes < 0)
    valid = (~endpoint_missing) & (~marked) & (~resets) & regular[:, None]
    power = changes * 4.
    power[~valid] = np.nan
    metadata = dict(source_url=SOURCE_URL, raw_file=str(path.resolve()), raw_sha256=digest,
        raw_bytes=path.stat().st_size, raw_data_rows=len(frame), columns=list(columns),
        physical_buildings=[column.removesuffix("_grid_import") for column in columns],
        unit_in="cumulative kWh", unit_out="kW", dt_minutes=15,
        timestamp_meaning="interval end and causal availability time; difference uses previous endpoint",
        first_source_utc=utc_string(stamps[0]), last_source_utc=utc_string(stamps[-1]),
        interpolation_policy="discard an interval if either endpoint marks its selected column; no reinterpolation",
        missing_policy="remain missing; complete simulation windows selected by calendar and availability only",
        normalization="none; no global mean/range scaling", household_time_shifts=False,
        counts=dict(marked_intervals=marked.sum(axis=0).tolist(),
                    missing_endpoint_intervals=endpoint_missing.sum(axis=0).tolist(),
                    reset_intervals=resets.sum(axis=0).tolist(),
                    irregular_intervals=int((~regular).sum()), valid_intervals=valid.sum(axis=0).tolist()),
        electrical_scope="observed building grid import, not gross building demand or measured Q; behind-meter PV/EV may be present")
    # CSV line 1 is the header: the second energy endpoint is line 3.
    return PowerSeries(stamps[1:], power, valid, columns, np.arange(3, len(frame) + 2), metadata)


def causal_forward_fill(values, *, max_gap=4):
    """Fill only from an earlier valid observation; never interpolate/backfill.

    Initial missing values and gaps longer than max_gap remain missing.
    This helper is for observed-history forecast input, not real-data truth.
    """
    values = np.asarray(values, float)
    one_dimensional = values.ndim == 1
    result = values[:, None].copy() if one_dimensional else values.copy()
    if result.ndim != 2 or max_gap < 0 or int(max_gap) != max_gap:
        raise ValueError("time-by-channel values and nonnegative integer max_gap required")
    filled = np.zeros(result.shape, bool)
    for j in range(result.shape[1]):
        last, gap = np.nan, 0
        for t in range(len(result)):
            if np.isfinite(result[t, j]):
                last, gap = result[t, j], 0
            else:
                gap += 1
                if np.isfinite(last) and gap <= max_gap:
                    result[t, j], filled[t, j] = last, True
    return (result[:, 0], filled[:, 0]) if one_dimensional else (result, filled)


@dataclass(frozen=True)
class CalendarSplit:
    name: str
    start_utc: str
    end_utc: str

    def __post_init__(self):
        if self.name not in PHASES or utc64(self.start_utc) >= utc64(self.end_utc):
            raise ValueError("ordered half-open calendar split and registered phase required")


def validate_calendar_splits(splits):
    if [split.name for split in splits] != list(PHASES):
        raise ValueError("training, development, calibration, evaluation must be registered in calendar order")
    for left, right in zip(splits, splits[1:]):
        if utc64(left.end_utc) > utc64(right.start_utc):
            raise ValueError("calendar phases overlap")
    return True


def availability_report(series, *, history_slots=192, future_slots=12, origin_hour_utc=0):
    """Data-quality coverage only; it never ranks window load values."""
    valid = series.valid.all(axis=1)
    starts = pd.DatetimeIndex(series.timestamps)
    run, longest = 0, 0
    eligible = []
    width = history_slots + future_slots
    for i, good in enumerate(valid):
        run = run + 1 if good else 0
        longest = max(longest, run)
        if starts[i].hour == origin_hour_utc and starts[i].minute == 0 and i + width <= len(valid):
            if valid[i:i + width].all() and np.all(np.diff(series.timestamps[i:i + width]) == np.timedelta64(15, "m")):
                eligible.append(utc_string(series.timestamps[i]))
    return dict(columns=list(series.columns), common_valid_intervals=int(valid.sum()),
        longest_common_contiguous_valid_slots=longest,
        history_slots=history_slots, future_slots=future_slots, eligible_calendar_starts_utc=eligible,
        selection_uses="calendar and valid/missing flags only; no load magnitude or AC outcome")


def chronological_windows(series, H=96, T=12, *, split=None, count=None, minimum_gap_slots=0):
    """Greedy earliest complete-window indices, allowing any 15-minute origin.

    The output indexes PowerSeries.p_kw/timestamps, not original CSV rows.
    Complete history and future lie in the same half-open calendar split.
    Nonoverlap of real time windows does not assert iid household samples.
    """
    if any(int(v) != v or v < 1 for v in (H, T)) or minimum_gap_slots < 0:
        raise ValueError("positive integer window lengths and nonnegative gap required")
    width = int(H + T)
    valid = series.valid.all(axis=1)
    if split is not None:
        valid &= (series.timestamps >= utc64(split.start_utc)) & (series.timestamps < utc64(split.end_utc))
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    result, i = [], 0
    while i + width <= len(valid):
        if invalid_prefix[i + width] == invalid_prefix[i] and np.all(
                np.diff(series.timestamps[i:i + width]) == np.timedelta64(15, "m")):
            result.append(i)
            if count is not None and len(result) >= count:
                break
            i += width + int(minimum_gap_slots)
        else:
            i += 1
    return result


load_opsd = load_opsd_power


def select_window(series, *, phase, start_utc, history_slots=192, future_slots=12, split=None):
    """A sealed measured-P window for generation, separated into past/future."""
    if phase not in PHASES or history_slots < 1 or future_slots < 1:
        raise ValueError("registered phase and positive history/future lengths required")
    start = utc64(start_utc)
    i = int(np.searchsorted(series.timestamps, start))
    stop = i + history_slots + future_slots
    if i == len(series.timestamps) or series.timestamps[i] != start or stop > len(series.timestamps):
        raise ValueError("requested fixed window is outside the raw time grid")
    stamps, power = series.timestamps[i:stop], series.p_kw[i:stop]
    if not series.valid[i:stop].all() or not np.all(np.diff(stamps) == np.timedelta64(15, "m")):
        raise ValueError("predeclared window contains missing, marked, reset or irregular intervals")
    if split is not None:
        if phase != split.name or stamps[0] < utc64(split.start_utc) or stamps[-1] >= utc64(split.end_utc):
            raise ValueError("history/future window crosses its registered calendar split")
    def block(a, b):
        return dict(p_kw=power[a:b].copy(), timestamps_utc=[utc_string(t) for t in stamps[a:b]])
    return dict(history=block(0, history_slots), future=block(history_slots, len(stamps)),
        provenance=dict(**series.provenance, phase=phase,
            history_slots=history_slots, future_slots=future_slots,
            start_utc=utc_string(stamps[0]), history_end_utc=utc_string(stamps[history_slots - 1]),
            end_utc=utc_string(stamps[-1]),
            csv_line_numbers=series.csv_line_numbers[i:stop].tolist(),
            cumulative_endpoint_csv_line_numbers=[int(series.csv_line_numbers[i]) - 1, *series.csv_line_numbers[i:stop].tolist()],
            simulator_only_future=True, field_voltage_available=False, field_Q_available=False,
            selector_may_read="history only"))


def register_calendar_windows(series, split, *, history_slots=192, future_slots=12,
                              count=None, origin_hour_utc=0, start_dates=None):
    """First quality-valid nonoverlapping windows in a preregistered phase.

    Explicit start_dates retain a missing-data failure instead of substituting
    another date.  Automatic dates are selected only by calendar/availability.
    """
    if start_dates is None:
        candidates = availability_report(series, history_slots=history_slots, future_slots=future_slots,
                                         origin_hour_utc=origin_hour_utc)["eligible_calendar_starts_utc"]
    else:
        candidates = list(start_dates)
    windows, failures, last_end = [], [], None
    for value in sorted(candidates, key=utc64):
        start = utc64(value)
        if start < utc64(split.start_utc) or start >= utc64(split.end_utc):
            continue
        if last_end is not None and start <= last_end:
            if start_dates is not None:
                raise ValueError("explicit historical/future windows overlap")
            continue
        try:
            window = select_window(series, phase=split.name, start_utc=value, history_slots=history_slots,
                                   future_slots=future_slots, split=split)
        except ValueError as exc:
            if start_dates is not None:
                failures.append(dict(start_utc=str(value), failure=str(exc)))
            continue
        windows.append(window)
        last_end = utc64(window["provenance"]["end_utc"])
        if count is not None and len(windows) >= count:
            break
    return dict(windows=windows, failures=failures, requested_count=count, available_count=len(windows),
                complete=count is None or len(windows) == count,
                split=asdict(split), calendar_rule="chronological nonoverlap; no future-value or topology-outcome ranking")


def assert_windows_disjoint(windows):
    ordered = sorted(windows, key=lambda value: utc64(value["provenance"]["start_utc"]))
    for left, right in zip(ordered, ordered[1:]):
        if utc64(left["provenance"]["end_utc"]) >= utc64(right["provenance"]["start_utc"]):
            raise ValueError("independent-history date windows overlap")
    return True


def synthetic_reactive_power(p_kw, timestamps, *, seed, pf_mean=.96, pf_amplitude=.01,
                             pf_bounds=(.90, .995), innovation_sd=.003, persistence=.85):
    """Explicit synthetic inductive Q for a measured grid-import P path.

    This is a causal stochastic generator, not measured power factor or a
    forecast. The forecast only receives its noisy historical Q afterwards.
    """
    p = np.asarray(p_kw, float)
    stamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if p.ndim != 2 or len(stamps) != len(p) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("nonnegative measured P and matching timestamps required")
    if not (0 < pf_bounds[0] <= pf_mean <= pf_bounds[1] < 1 and abs(persistence) < 1 and innovation_sd >= 0):
        raise ValueError("valid explicit power-factor process required")
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, p.shape[1])
    hours = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60.
    state = np.zeros(p.shape[1])
    pf = np.empty_like(p)
    for t in range(len(p)):
        state = persistence * state + innovation_sd * rng.standard_normal(p.shape[1])
        pf[t] = np.clip(pf_mean + pf_amplitude * np.sin(2 * np.pi * hours[t] / 24 + phase) + state, *pf_bounds)
    q = p * np.tan(np.arccos(pf))
    return dict(q_kvar=q, power_factor=pf,
        metadata=dict(synthetic=True, measured_Q=False, seed=int(seed), pf_mean=pf_mean,
            pf_amplitude=pf_amplitude, pf_bounds=list(pf_bounds), innovation_sd=innovation_sd,
            persistence=persistence, source="declared synthetic inductive PF process; EV Q handled separately"))
