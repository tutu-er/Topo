"""The development library (PROTOCOL.md, section 2): 288 cases from 72 feeders, four twins each.

A case file holds the network, the true quarter-hourly loads and AC voltages of 9 days, the unmetered load (or none) and
the truth; meter readings are made when a case is opened. Every file is checked against its SHA-256 in
manifest_dev.json (a copy of the case library's committed manifest). The sealed test set is never opened here.

    open_case(name)            the case record (hash-checked)
    readings(rec)              8 days of hourly meter readings: P, Q, Y = V0^2 - V^2 (T x n), V0sq, Ppcc, Qpcc (T)
    truth(rec)                 the identifiable true meter groupings (lines of at least 3 milliohm; 2 to n-1 meters)
    exact_truth(rec)           every true meter grouping, also of lines below 3 milliohm
    kind(name)                 star, street, small, medium, large or huge
    one_twin_per_feeder()      72 names: one twin per feeder, the load's location in rotation within each kind
                               (no load 20, head 20, middle 16, end 16)
"""
import hashlib
import json
import pickle
from collections import defaultdict

from . import physics
from .paths import LIBRARY, MANIFEST_DEV

DAYS = 9                          # 8 days of history and one day for probing
HISTORY = 8*24                    # hourly rows of history
LOCATIONS = ("none", "head", "middle", "end")


def manifest():
    return json.loads(MANIFEST_DEV.read_text(encoding="utf-8"))


def open_case(name):
    """A development case's record, checked against its SHA-256 in the manifest."""
    rows = {r["name"]: r["sha256"] for r in manifest()}
    if name not in rows:
        raise SystemExit(f"{name} is not a development case")
    data = (LIBRARY/"dev"/f"{name}.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != rows[name]:
        raise SystemExit(f"{name}: the case file does not match its manifest hash")
    return pickle.loads(data)


def readings(rec, meter_seed=0):
    """The case library's meter readings (build.to_input with its defaults): hourly averages of the quarter-hours, the
    8 days before day 9, the meters' fixed errors drawn from the case key and the meter seed."""
    r = physics.read(rec["true"], physics.Meters(rec["n"], [*rec["key"], int(meter_seed)]), 1)
    end = (DAYS-1)*24
    rows = slice(end-HISTORY, end)
    r = {k: a[rows] for k, a in r.items()}
    return dict(P=r["P"], Q=r["Q"], Y=r["V0"][:, None]**2-r["V"]**2, V0sq=r["V0"]**2, Ppcc=r["Ppcc"], Qpcc=r["Qpcc"])


def truth(rec):
    """The identifiable true meter groupings."""
    n = rec["n"]
    return {tuple(sorted(int(i) for i in C)) for C in rec["truth"] if 1 < len(C) < n}


def exact_truth(rec):
    """Every true meter grouping, before lines shorter than 3 milliohm are merged."""
    n = rec["n"]
    return {M for M in (tuple(sorted(int(i) for i in C if i < n)) for C in rec["exact_truth"]) if 1 < len(M) < n}


def load_where(rec):
    return rec["source"]["where"] if rec["source"] else None


def kind(name):
    if "_star_" in name:
        return "star"
    if "_street_" in name:
        return "street"
    return name.split("_")[1]                   # dev_<size>_NNN_<location>


def one_twin_per_feeder():
    """One twin per feeder: within each kind, the feeders (sorted by name) take no load, head, middle and end in turn.
    Star and street give 6 per location; each random size (6 feeders) gives 2, 2, 1, 1. In all: no load 20, head 20,
    middle 16, end 16 of 72 (the selection of the 2026-11-27 run)."""
    groups = defaultdict(set)
    for r in manifest():
        groups[kind(r["name"])].add(r["group"])
    pick = {g: LOCATIONS[k % 4] for members in groups.values() for k, g in enumerate(sorted(members))}
    return sorted(r["name"] for r in manifest() if r["source"] == pick[r["group"]])
