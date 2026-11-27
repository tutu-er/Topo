"""Load the four development feeders and construct causal, paired 2024 observations.

Set PTID_FOUR_DATA to the project's data/ directory when it is outside this checkout.
Archived output paths are relocated under data/ by their manifest-relative suffixes.
The raw archived P/Q are per unit. Original per-bus and transformer power in full_ac
are kW/kvar. Only history() output is supplied to the identifier.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path, PureWindowsPath

import numpy as np

from ptid.cases import physics

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from rnj_wzzt_dr.physical.backend import load_case

CASES = ("soumalas11", "paper15", "flynn16", "pengwah18")
REGISTRY = Path("wzzt_long_grid_data_20261117_v1/worlds.json")
HISTORY_DAYS = 8
METER_SEED = 20261128
V_CHECK_PU, POWER_CHECK_PU = 1e-8, 1e-8


def data_root():
    configured = os.environ.get("PTID_FOUR_DATA")
    if configured:
        return Path(configured).resolve()
    for folder in (REPO, *REPO.parents):
        if (folder/"data"/REGISTRY).is_file():
            return folder/"data"
    raise FileNotFoundError("Specify PTID_FOUR_DATA with the four existing AC archives")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(root, ref):
    parts = PureWindowsPath(ref["path"]).parts
    path = root.joinpath(*parts[parts.index("outputs")+1:])
    if sha256(path) != ref["sha256"]:
        raise ValueError(f"File does not match the physical-world manifest: {path}")
    return path


def groupings(network, labels):
    order, parent, incoming = physics.radial_order(network)
    below = {bus: set() for bus in order}
    for i, bus in enumerate(labels):
        below[bus].add(i)
    for bus in reversed(order[1:]):
        below[parent[bus]].update(below[bus])
    eligible = [b for b in order[1:] if 1 < len(below[b]) < len(labels)]
    truth = {tuple(sorted(below[b])) for b in eligible if incoming[b]["r_ohm"] >= .003}
    exact = {tuple(sorted(below[b])) for b in eligible}
    return tuple(sorted(truth)), tuple(sorted(exact))


def load_feeder(name, start_day="2024-01-31", execution_days=1):
    """Eight complete local days followed by the specified near-term execution period."""
    if name not in CASES or execution_days < 1:
        raise ValueError((name, execution_days))
    root = data_root()
    registry = json.loads((root/REGISTRY).read_text(encoding="utf-8"))
    meta = next(r for r in registry if r["case"] == name)
    clean_file = checked(root, meta["physical"]["clean"])
    ac_file = checked(root, meta["physical"]["full_ac"])
    resource_file = checked(root, meta["resource_context"])
    network = load_case(name)
    labels = tuple(int(x) for x in meta["labels"])
    source_hashes = {k: meta["physical"][k]["sha256"] for k in ("clean", "full_ac")}
    source_hashes["resource_context"] = meta["resource_context"]["sha256"]
    for field in ("metadata.json", "buses.csv", "branches.csv"):
        path = REPO/"rnj_wzzt_dr"/"physical"/"cases"/name/field
        source_hashes[field] = sha256(path)
    with np.load(clean_file, allow_pickle=False) as c, np.load(ac_file, allow_pickle=False) as ac:
        raw_times = c["timestamps"].astype("U")
        ac_times = ac["timestamps_utc"].astype("U")
        if not np.array_equal(raw_times, ac_times):
            raise ValueError("Timestamps differ across archives")
        # Fixed UTC+1 is the local offset during both January experiments.
        local = datetime.combine(date.fromisoformat(start_day), time.min, timezone(timedelta(hours=1)))
        utc = (local - timedelta(days=HISTORY_DAYS)).astimezone(timezone.utc)
        first = np.flatnonzero(raw_times == utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if len(first) != 1:
            raise ValueError(f"Missing history start {utc}")
        index = int(first[0])
        count = (HISTORY_DAYS+execution_days)*96
        s = slice(index, index+count)
        times = raw_times[s]
        expected = [(utc+timedelta(minutes=15*i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(count)]
        if list(times) != expected:
            raise ValueError("Window incomplete, duplicated or out of order")
        if tuple(c["labels"]) != labels or tuple(ac["terminal_labels"]) != labels:
            raise ValueError("Meter labels differ across manifests")
        base = network["base_kw"]
        if base != meta["base_kw"]:
            raise ValueError("Power base mismatch")
        bus_ids = tuple(int(b) for b in ac["bus_ids"])
        indices = [bus_ids.index(b) for b in labels]
        true = dict(P=c["P"][s], Q=c["Q"][s], V=c["V"][s], V0=c["root"][s],
                    S_pcc=(ac["root_supply_kw"][s]+1j*ac["root_supply_kvar"][s])/base,
                    V_all=ac["voltage_magnitude_pu"][s])
        for key, full in (("P", "specified_load_kw"), ("Q", "specified_load_kvar")):
            if not np.allclose(true[key]*base, ac[full][s][:, indices], atol=1e-9, rtol=1e-10):
                raise ValueError(f"{key} units or sign differ")
    truth, exact = groupings(network, labels)
    return dict(name=name, n=len(labels), network=network, labels=labels, source=None, key=(METER_SEED, CASES.index(name)),
                true=true, truth=truth, exact_truth=exact, bus_ids=bus_ids, timestamps=times, offset=index,
                start_day=start_day, execution_days=execution_days, source_hashes=source_hashes,
                resource_context=json.loads(resource_file.read_text(encoding="utf-8")))


def _history_observations(rec, noise_scale, fine):
    """Meter random stream is indexed from the same archived hour in all comparisons."""
    n = rec["n"]
    meters = physics.Meters(n, [*rec["key"], int(fine)], v_noise=.0005*noise_scale, pq_noise=.005*noise_scale)
    # Spend the same random draws for any earlier archived readings outside this window.
    step = 1 if fine else 4
    prior = rec["offset"]//step
    def read_batch(p, q, v, v0, s):
        return [meters.read(pi, qi, vi, v0i, si) for pi, qi, vi, v0i, si in zip(p, q, v, v0, s)]
    empty = np.zeros((prior, n))
    read_batch(empty, empty, empty, np.zeros(prior), np.zeros(prior, complex))
    a = [rec["true"][k][:96*HISTORY_DAYS] for k in ("P", "Q", "V", "V0", "S_pcc")]
    if not fine:
        a = [physics.hourly(x) for x in a]
    rows = read_batch(*a)
    obs = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    return obs, meters


def meter_stream(rec, noise_scale=1.):
    """The same calibrated hourly meters advanced through history, ready for hour zero."""
    return _history_observations(rec, noise_scale, False)[1]


def history(rec, days=HISTORY_DAYS, noise_scale=1., fine=False, per_hour=1):
    if days not in range(1, HISTORY_DAYS+1) or per_hour not in ((1, 4) if fine else (1,)):
        raise ValueError("Unsupported duration or sampling for the four-feeder history")
    obs, _ = _history_observations(rec, noise_scale, fine)
    if fine and per_hour == 1:
        obs = {k: physics.hourly(v) for k, v in obs.items()}
    r = {k: v[-days*24*per_hour:] for k, v in obs.items()}
    return dict(P=r["P"], Q=r["Q"], V0sq=r["V0"]**2, Y=r["V0"][:, None]**2-r["V"]**2,
                Ppcc=r["Ppcc"], Qpcc=r["Qpcc"])


def validate_replay(rec):
    true = rec["true"]
    order, v, supply, _ = physics.power_flow(rec["network"], rec["labels"], true["P"], true["Q"], true["V0"])
    indices = [order.index(b) for b in rec["bus_ids"]]
    verdict = dict(all_bus_voltage_error_pu=float(np.max(abs(v[:, indices]-true["V_all"]))),
                   terminal_voltage_error_pu=float(np.max(abs(v[:, [order.index(b) for b in rec["labels"]]]-true["V"]))),
                   root_p_error_pu=float(np.max(abs(supply.real-true["S_pcc"].real))),
                   root_q_error_pu=float(np.max(abs(supply.imag-true["S_pcc"].imag))))
    if (max(verdict["all_bus_voltage_error_pu"], verdict["terminal_voltage_error_pu"]) > V_CHECK_PU
            or max(verdict["root_p_error_pu"], verdict["root_q_error_pu"]) > POWER_CHECK_PU):
        raise ValueError(f"Zero-action AC replay mismatch: {verdict}")
    return dict(name=rec["name"], points=len(v), tolerance_voltage_pu=V_CHECK_PU, tolerance_power_pu=POWER_CHECK_PU,
                source_hashes=rec["source_hashes"], **verdict)
