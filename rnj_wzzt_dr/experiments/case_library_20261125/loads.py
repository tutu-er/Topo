"""True loads: real customers' hourly readings, split into quarter-hours; and the unmetered load.

Customers: the 8,087 smart-meter records of the Non-Synthetic European LV Test System (20 days, hourly, kW). Even
record numbers serve development, odd numbers the held-out test, so the two sets share no customer.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PROFILES = (REPO/"data"/"nonsynthetic_eu_lv_20261123"/"extracted"/"Sim_files_190128_OK_V0"/"RunDss"/"day_20_profile")
CACHE = REPO/"outputs"/"case_library_20261125"/"customers.npz"
RECORDED_DAYS = 20
QUARTERS = 4
LARGEST_KW = 14.5      # a single-phase supply of 63 A at 230 V; 0.4% of the records are larger (shops, works) and unused

_customers = None


def customers():
    """All recorded profiles as (hours, customers) in kW, with their record numbers; cached outside Git."""
    global _customers
    if _customers is None:
        if not CACHE.exists():
            files = sorted(PROFILES.glob("shape_*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
            kw = np.column_stack([np.loadtxt(f)[:24*RECORDED_DAYS] for f in files]).astype(np.float32)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE, kw=kw, record=[int(re.findall(r"\d+", f.stem)[0]) for f in files])
        with np.load(CACHE) as z:
            _customers = z["kw"].astype(float), z["record"]
    return _customers


def pool(stream):
    """Column numbers of the customers a stream may use."""
    kw, record = customers()
    return np.flatnonzero((record % 2 == {"dev": 0, "test": 1}[stream]) & (kw.max(axis=0) <= LARGEST_KW))


def assign(counts, stream, rng):
    """Draw customers for every connection without repetition; a connection must not be vacant (mean 0.02 kW)."""
    kw, _ = customers()
    free = list(rng.permutation(pool(stream)))
    out = []
    for m in counts:
        while True:
            chosen, free = free[:m], free[m:]
            if kw[:, chosen].mean(axis=0).sum() >= .02:
                break
        out.append(np.array(chosen))
    return out


def quarter_hours(hourly, rng):
    """Split every hourly reading over its four quarter-hours; the hour's energy stays the recorded one."""
    weights = rng.dirichlet(np.full(QUARTERS, 2.), hourly.shape)           # hours x customers x 4
    fine = hourly[..., None]*QUARTERS*weights
    return fine.transpose(0, 2, 1).reshape(-1, hourly.shape[1])


def reactive(p, rng):
    """Reactive power: each customer has a mean power factor (0.90 to 0.99) that drifts from quarter to quarter."""
    T, m = p.shape
    tan = np.tan(np.arccos(rng.uniform(.90, .99, m)))
    drift = np.zeros((T, m)); shock = rng.normal(size=(T, m))
    drift[0] = shock[0]
    for t in range(1, T):
        drift[t] = .9*drift[t-1]+np.sqrt(1-.81)*shock[t]
    return p*tan*np.exp(.25*drift-.5*.25**2)


def metered_loads(counts, stream, seed_key, days, start_day):
    """Quarter-hourly P and Q in kW (kvar) of every connection."""
    rng = np.random.default_rng(np.random.SeedSequence([20261125, 11, *seed_key]))
    kw, _ = customers()
    hours = slice(24*start_day, 24*(start_day+days))
    P, Q = [], []
    for chosen in assign(counts, stream, rng):
        p = quarter_hours(kw[hours][:, chosen], rng)
        P.append(p.sum(axis=1)); Q.append(reactive(p, rng).sum(axis=1))
    return np.column_stack(P), np.column_stack(Q)


def root_voltage(slots, seed_key):
    """Voltage at the transformer's LV side (pu): daily pattern of 0.2% plus slow random movement of 0.2%."""
    rng = np.random.default_rng(np.random.SeedSequence([20261125, 12, *seed_key]))
    phase = 2*np.pi*np.arange(slots)/96
    slow = np.zeros(slots); shock = rng.normal(size=slots)
    slow[0] = shock[0]
    for t in range(1, slots):
        slow[t] = .95*slow[t-1]+np.sqrt(1-.95**2)*shock[t]
    return 1.02+.002*np.sin(phase+.3)+.002*slow


def source_profile(kind, slots, stream, start_day, seed_key):
    """Quarter-hourly P of one unmetered load, scaled to mean 1, and its tan(phi).

    lamp   street lighting, constant from 19:00 to 06:00
    house  customers without a working meter (three real records)
    block  a pump or charger: one or two on-periods of 2 to 4 hours a day
    """
    rng = np.random.default_rng(np.random.SeedSequence([20261125, 13, *seed_key]))
    days = slots//96
    quarter = np.arange(slots) % 96
    if kind == "lamp":
        p, tan = ((quarter >= 76) | (quarter < 24)).astype(float), .30
    elif kind == "house":
        kw, _ = customers()
        chosen = assign([3], stream, rng)[0]
        p, tan = quarter_hours(kw[24*start_day:24*(start_day+days)][:, chosen], rng).sum(axis=1), .25
    elif kind == "block":
        p, tan = np.zeros(slots), .20
        for day in range(days):
            for _ in range(int(rng.integers(1, 3))):
                begin, length = int(rng.integers(0, 80)), int(rng.integers(8, 17))
                p[96*day+begin:96*day+min(begin+length, 96)] = 1.
    else:
        raise ValueError(kind)
    return p/p.mean(), tan
