"""Public profile loaders; no AC, fitting, forecasts, or hidden truth access.

Static grid parameters and public profiles are separate sources.  The caller
must declare node/profile assignments.  SINTEF raw power units are deliberately
not guessed: its public description is not explicit about the load CSV units.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


SIMBENCH_VERSION = "1.6.3"
SIMBENCH_SOURCE = "https://pypi.org/project/simbench/1.6.3/"
SINTEF_SOURCE = "https://doi.org/10.5281/zenodo.14528192"
SINTEF_ZIP_SHA256 = "e3babc3c8faeef086518a82b4912a7bd33c10a903ab5192125649391e72adf41"
SINTEF_GRIDS = (
    "39_bus_semi_urban_reference_grid", "50_bus_rural_reference_grid",
    "56_bus_semi_urban_reference_grid", "80_bus_rural_reference_grid",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _time_axis(values, *, date_format=None, expected_hours=None, local_timezone=None):
    t = pd.to_datetime(list(values), format=date_format, errors="raise")
    if local_timezone is not None:
        # Current official CSV labels contain the German DST gap and fold.
        # Retain each distinct physical interval rather than dropping the fold.
        t = t.tz_localize(local_timezone, ambiguous="infer", nonexistent="raise").tz_convert("UTC")
    if len(t) < 2 or t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError("unique increasing timestamps with at least two samples required")
    # pandas 3 may choose microsecond storage; do not assume asi8 is nanoseconds.
    delta = np.diff(t.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "h")
    if not np.allclose(delta, delta[0], rtol=0., atol=1e-12):
        raise ValueError("irregular timestamps; loader does not fill missing intervals")
    dt = float(delta[0])
    if expected_hours is not None and not np.isclose(dt, expected_hours):
        raise ValueError("unexpected native sampling resolution")
    return tuple(x.isoformat() for x in t), dt


def _finite(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"finite two-dimensional {name} required; no interpolation applied")
    return a


def available_simbench_profiles(directory):
    columns = pd.read_csv(Path(directory) / "LoadProfile.csv", sep=";", nrows=0).columns
    return tuple(c[:-6] for c in columns if c.endswith("_pload") and c[:-6] + "_qload" in columns)


def simbench_profile_usage(directory):
    """Trace profile scope from native Load.csv; HS means EHV, not households."""
    path = Path(directory) / "Load.csv"
    frame = pd.read_csv(path, sep=";")
    result = {}
    for name, group in frame.groupby("profile", sort=True):
        result[str(name)] = dict(native_bindings=len(group),
            voltage_levels=sorted(int(v) for v in group.voltLvl.unique()),
            nominal_p_mw_min=float(group.pLoad.min()),
            nominal_p_mw_max=float(group.pLoad.max()),
            example_subnets=list(dict.fromkeys(group.subnet))[:3])
    return result


def load_simbench_profiles(directory, names):
    """Return native relative P/Q; never normalize by the evaluation future."""
    directory = Path(directory)
    names = tuple(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("request nonempty distinct profile names; reuse is declared in node assignments")
    path = directory / "LoadProfile.csv"
    pcols = [n + "_pload" for n in names]
    qcols = [n + "_qload" for n in names]
    frame = pd.read_csv(path, sep=";", usecols=["time", *pcols, *qcols])
    time, dt = _time_axis(frame.time, date_format="%d.%m.%Y %H:%M", expected_hours=.25,
                         local_timezone="Europe/Berlin")
    p = _finite(frame[pcols].to_numpy(), "P relative profiles")
    q = _finite(frame[qcols].to_numpy(), "Q relative profiles")
    return dict(time=time, dt_hours=dt, profile_names=names,
        p_relative=p, q_relative=q, provenance=dict(source=SIMBENCH_SOURCE,
            version=SIMBENCH_VERSION, path=str(path), sha256=sha256_file(path),
            data_kind="published_benchmark_profiles_with_measured_and_constructed_components",
            license="ODbL-1.0/DbCL-1.0", native_sampling_preserved=True,
            normalization="publisher supplied relative scaling; no new whole-year normalization",
            source_clock="observed German CET/CEST labels localized as Europe/Berlin",
            returned_clock="UTC, unique native physical intervals",
            raw_synchronous_feeder_measurements=False))


def load_simbench_pv(directory, name="PV1"):
    path = Path(directory) / "RESProfile.csv"
    frame = pd.read_csv(path, sep=";", usecols=["time", name])
    time, dt = _time_axis(frame.time, date_format="%d.%m.%Y %H:%M", expected_hours=.25,
                         local_timezone="Europe/Berlin")
    p = _finite(frame[[name]].to_numpy(), "PV relative profile")[:, 0]
    return dict(time=time, dt_hours=dt, profile_name=name, p_relative=p,
        provenance=dict(source=SIMBENCH_SOURCE, version=SIMBENCH_VERSION,
            path=str(path), sha256=sha256_file(path), license="ODbL-1.0/DbCL-1.0",
            source_clock="Europe/Berlin CET/CEST", returned_clock="UTC",
            data_kind="weather_driven_simulated_generation_profile", measured_Q=False))


def apply_standard_profiles(network, profiles, assignment):
    """Multiply standard nominal kW/kvar by separately supplied P/Q profiles.

    ``assignment`` maps each observation bus ID to one requested profile name.
    Reuse is retained and counted, never relabelled as independent customers.
    """
    ids = tuple(network["observation_bus_ids"])
    if set(assignment) != set(ids):
        raise ValueError("one explicit profile assignment for every observation bus required")
    source_names = profiles["profile_names"]
    names = tuple(assignment[i] for i in ids)
    if any(n not in source_names for n in names):
        raise ValueError("node references a profile that was not loaded")
    columns = [source_names.index(n) for n in names]
    nominal = {r["bus_id"]: r for r in network["buses"]}
    p0 = np.array([nominal[i]["pd_kw"] for i in ids])
    q0 = np.array([nominal[i]["qd_kvar"] for i in ids])
    p = profiles["p_relative"][:, columns] * p0[None, :]
    q = profiles["q_relative"][:, columns] * q0[None, :]
    return dict(time=profiles["time"], dt_hours=profiles["dt_hours"],
        observation_bus_ids=ids, p_kw=_finite(p, "P kW"),
        q_kvar=_finite(q, "Q kvar"), nominal_p_kw=p0, nominal_q_kvar=q0,
        profile_assignments=dict(zip(ids, names)),
        provenance=dict(profile_source=profiles["provenance"],
            network_source=network.get("provenance", {}),
            data_kind="standard_static_network_with_explicit_external_profile_assignment",
            original_static_power_magnitude_preserved=True,
            no_global_rescaling_to_make_EV_effect_larger=True,
            unique_profile_count=len(set(names)), assignment_count=len(names),
            profile_reuse={k: v for k, v in Counter(names).items() if v > 1},
            raw_synchronous_feeder_measurements=False))


def _csv_table(z, name):
    return pd.read_csv(io.BytesIO(z.read(name)))


def _consumer_matrix(z, name, bus_ids):
    """Read by CSV position: repeated physical bus labels are intentional."""
    rows = csv.reader(io.StringIO(z.read(name).decode("utf-8-sig")))
    header = next(rows)
    if header[0] != "Date" or len(header) != len(bus_ids) + 1:
        raise ValueError("load header and consumer metadata have different widths")
    occurrence = Counter()
    for label, bus in zip(header[1:], bus_ids):
        n = occurrence[bus]
        allowed = {str(bus), f"{bus}.{n}"} if n else {str(bus)}
        if label not in allowed:
            raise ValueError(f"consumer position/header mismatch: {label}, bus {bus}")
        occurrence[bus] += 1
    time, values = [], []
    for row in rows:
        if len(row) != len(header):
            raise ValueError("ragged load CSV")
        time.append(row[0]); values.append([float(v) for v in row[1:]])
    time, dt = _time_axis(time, expected_hours=1.)
    return time, dt, _finite(values, "raw consumer power"), header[1:]


def load_sintef_raw(zip_path, grid_name):
    """Load native hourly data, aggregate co-located customers, retain raw units.

    No assumption is made that all Q columns are independently measured.
    Semi-urban grids contain Q exactly proportional to P (PF 0.98).
    """
    if grid_name not in SINTEF_GRIDS:
        raise ValueError("unknown SINTEF grid")
    zip_path = Path(zip_path)
    digest = sha256_file(zip_path)
    if digest != SINTEF_ZIP_SHA256:
        raise ValueError("SINTEF archive SHA256 mismatch")
    folder = f"dataset/{grid_name}"
    with zipfile.ZipFile(zip_path) as z:
        extra = _csv_table(z, folder + "/load_bus_extra.csv")
        consumer_buses = tuple(int(v) for v in extra.bus_i)
        time, dt, p, p_labels = _consumer_matrix(z, folder + "/p_load.csv", consumer_buses)
        tq, dq, q, q_labels = _consumer_matrix(z, folder + "/q_load.csv", consumer_buses)
        if time != tq or dt != dq or p.shape != q.shape:
            raise ValueError("P/Q time or consumer mismatch")
        bus_table = _csv_table(z, folder + "/mpc_bus.csv")
        branch_table = _csv_table(z, folder + "/mpc_branch.csv")
        base_table = _csv_table(z, folder + "/mpc_base_mva.csv")
        base_mva = float(base_table["Base MVA"].iloc[0])
    ids = tuple(dict.fromkeys(consumer_buses))
    group = {b: [j for j, x in enumerate(consumer_buses) if x == b] for b in ids}
    pg = np.column_stack([p[:, group[b]].sum(axis=1) for b in ids])
    qg = np.column_stack([q[:, group[b]].sum(axis=1) for b in ids])
    nonzero = np.abs(p) > 1e-12
    ratio = q[nonzero] / p[nonzero]
    const_ratio = bool(ratio.size and np.ptp(ratio) < 1e-9)
    return dict(time=time, dt_hours=dt, observation_bus_ids=ids,
        p_raw=pg, q_raw=qg, consumer_p_raw=p, consumer_q_raw=q,
        consumer_bus_ids=consumer_buses, consumer_metadata=extra.to_dict("records"),
        bus_table=bus_table.to_dict("records"), branch_table=branch_table.to_dict("records"),
        base_mva=base_mva,
        provenance=dict(source=SINTEF_SOURCE, version="v1 2024-12-20", license="CC-BY-4.0",
            path=str(zip_path), sha256=digest, grid=grid_name,
            data_kind="anonymized_real_network_based_reference_dataset",
            time_series_unit="not_explicitly_resolved_by_public_description",
            direct_AC_units_ready=False, native_sampling_preserved=True,
            native_dt_hours=dt, source_rows=len(time), consumer_count=p.shape[1],
            unique_load_bus_count=len(ids), co_located_consumer_counts={
                b: len(js) for b, js in group.items() if len(js) > 1},
            constant_Q_over_P=const_ratio,
            constant_Q_over_P_value=float(ratio[0]) if const_ratio else None,
            independent_measured_Q_not_asserted=True,
            raw_P_header=p_labels, raw_Q_header=q_labels))


def convert_sintef_power_units(raw, *, declared_unit, declaration_basis):
    """Explicit conversion, requiring a documented basis; never auto-detect.

    This function does not certify the caller's unit interpretation.
    Use raw profiles only until the original unit statement is resolved.
    """
    if not str(declaration_basis).strip():
        raise ValueError("document the source of the unit declaration")
    factors = {"MW": 1000., "kW": 1., "pu_on_declared_base": raw["base_mva"] * 1000.}
    if declared_unit not in factors:
        raise ValueError("declared_unit must be MW, kW, or pu_on_declared_base")
    scale = factors[declared_unit]
    return dict(time=raw["time"], dt_hours=raw["dt_hours"],
        observation_bus_ids=raw["observation_bus_ids"],
        p_kw=raw["p_raw"] * scale, q_kvar=raw["q_raw"] * scale,
        provenance=dict(**raw["provenance"], declared_raw_power_unit=declared_unit,
            declaration_basis=declaration_basis, conversion_factor_to_kW=scale,
            caller_declaration_is_not_independently_verified=True))


def window(series, start, stop):
    """A half-open time slice.  It does no fitting, scaling, or future filling."""
    times = pd.to_datetime(series["time"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(stop)
    if times.tz is not None:
        lo = lo.tz_localize(times.tz) if lo.tzinfo is None else lo.tz_convert(times.tz)
        hi = hi.tz_localize(times.tz) if hi.tzinfo is None else hi.tz_convert(times.tz)
    elif lo.tzinfo is not None or hi.tzinfo is not None:
        raise ValueError("timezone-aware bounds require a declared timezone on the source data")
    keep = (times >= lo) & (times < hi)
    if np.count_nonzero(keep) < 2:
        raise ValueError("window must contain at least two native observations")
    out = {**series, "time": tuple(t for t, k in zip(series["time"], keep) if k)}
    for k in ("p_kw", "q_kvar", "p_raw", "q_raw", "p_relative", "q_relative",
              "consumer_p_raw", "consumer_q_raw"):
        if k in series:
            out[k] = series[k][keep].copy()
    return out


def profile_diagnostics(p, q):
    """Describe rank/correlation; these offline summaries are not policy input."""
    p, q = _finite(p, "P"), _finite(q, "Q")
    if p.shape != q.shape:
        raise ValueError("P and Q shape mismatch")
    design = np.concatenate([p - p.mean(0), q - q.mean(0)], axis=1)
    scales = np.linalg.norm(design, axis=0)
    scaled = np.divide(design, scales, out=np.zeros_like(design), where=scales > 1e-15)
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.count_nonzero(singular > (singular[0] * 1e-10 if singular.size else 0.)))
    return dict(rows=p.shape[0], loads=p.shape[1], centered_PQ_columns=2 * p.shape[1],
        centered_scaled_rank_1e_10=rank, zero_columns=int(np.count_nonzero(scales <= 1e-15)),
        smallest_scaled_singular=float(singular[-1]) if singular.size else None,
        diagnostics_are_offline_descriptive_not_selector_inputs=True)
