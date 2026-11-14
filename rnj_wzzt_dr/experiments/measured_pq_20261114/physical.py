"""Measured P/Q profiles driving the original four-network AC simulator.

This is an offline evaluator. Physical branches never enter identification.
Profiles are used unchanged: no extra DER baseline, power-factor construction,
R/X adjustment, voltage-dependent rescaling, or replacement of failed days.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np

from topology_identification_standalone import data as ac_backend
from ... import io
from ...observations import Observation, fingerprint


CASES = tuple(ac_backend.CASES)
SPLIT = {"train": [0, 192], "validation": [192, 288], "new_day": [288, 384]}


def _validated_profile(case_name, profile):
    if case_name not in CASES:
        raise ValueError(f"Unknown original network: {case_name}")
    network = ac_backend.load_case(case_name)
    labels = tuple(int(row["bus_id"]) for row in network["buses"] if row["is_terminal"])
    if tuple(map(int, profile["labels"])) != labels:
        raise ValueError("Profile terminal labels/order differ from the original physical network")
    p, q = (np.asarray(profile[key], dtype=float) for key in ("p_kw", "q_kvar"))
    if p.shape != (384, len(labels)) or q.shape != p.shape:
        raise ValueError("Exactly four complete 96-point days of paired P/Q are required")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Profile contains missing or nonfinite P/Q; no interpolation here")
    split = {key: list(map(int, value)) for key, value in profile["split"].items()}
    if split != SPLIT:
        raise ValueError("Required chronological split: train 0:192, validation 192:288, new_day 288:384")
    times = [datetime.fromisoformat(str(value).replace("Z", "+00:00"))
             for value in profile["timestamps"]]
    if len(times) != 384 or any(b-a != timedelta(minutes=15) for a, b in zip(times, times[1:])):
        raise ValueError("Timestamps must be 384 consecutive quarter-hour measurements")
    if len({value.utcoffset() for value in times}) != 1:
        raise ValueError("A four-day block must not cross a daylight-saving transition")
    return network, labels, p, q, [value.isoformat() for value in times]


def _network_hash(network):
    text = json.dumps(network, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _slice(observed, start, stop):
    return Observation(observed.p[start:stop].copy(), observed.q[start:stop].copy(),
        observed.y[start:stop].copy(), observed.labels,
        None if observed.voltage is None else observed.voltage[start:stop].copy(),
        None if observed.root is None else observed.root[start:stop].copy())


def generate(case_name, profile, *, seed=20261114, voltage_sigma=.0005,
             pq_sigma=.005, root_sigma=.003):
    """Return observed splits, AC diagnostics, and separate private simulator state.

    P/Q source values are the imposed physical trajectory, in kW/kvar. Meter
    errors below are *additional simulated* errors, not a calibration of the
    source meters. Negative P/Q are preserved. Root variation and meter streams
    remain separate, following the existing simulator's multiplicative law.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(not np.isfinite(value) or value < 0 for value in (voltage_sigma, pq_sigma, root_sigma)):
        raise ValueError("Noise standard deviations must be finite and nonnegative")
    network, labels, p_kw, q_kvar, timestamps = _validated_profile(case_name, profile)
    original_hash = _network_hash(network)
    p, q = p_kw/network["base_kw"], q_kvar/network["base_kw"]
    # The archive uses UTC. Root's diurnal phase follows the declared local clock.
    local_times = [datetime.fromisoformat(t) for t in timestamps]
    timezone_name = profile.get("metadata", {}).get("local_timezone")
    if timezone_name:
        if any(t.tzinfo is None for t in local_times):
            raise ValueError("A declared local timezone requires timezone-aware source timestamps")
        local_times = [t.astimezone(ZoneInfo(timezone_name)) for t in local_times]
        if len({t.utcoffset() for t in local_times}) != 1:
            raise ValueError("A four-day block must not cross a local daylight-saving transition")
    local_hour = np.array([t.hour+t.minute/60 for t in local_times])
    phase = 2*np.pi*local_hour/24
    entropy = [20261114, int(seed), CASES.index(case_name)]
    def rng(component):
        return np.random.default_rng(np.random.SeedSequence([*entropy, component]))
    root = 1.02+root_sigma*(.7*np.sin(phase+.3)+.3*rng(1).normal(size=384))
    if np.any(root <= 0):
        raise ValueError("Root-voltage trajectory is nonpositive")
    voltage, ac = ac_backend._ac_voltages(network, labels, p, q, root,
                                        tolerance=1e-10, max_iterations=200)
    if not np.isfinite(voltage).all() or np.any(voltage <= 0):
        raise ArithmeticError("AC returned nonfinite or nonpositive terminal voltage")
    if _network_hash(network) != original_hash:
        raise ArithmeticError("The original network changed during AC evaluation")
    noises = [rng(component).normal(size=shape) for component, shape in
              ((2, p.shape), (3, q.shape), (4, voltage.shape), (5, root.shape))]
    pm, qm = p*(1+pq_sigma*noises[0]), q*(1+pq_sigma*noises[1])
    vm, rm = voltage*(1+voltage_sigma*noises[2]), root*(1+voltage_sigma*noises[3])
    if np.any(vm <= 0) or np.any(rm <= 0):
        raise ArithmeticError("Additional meter noise produced nonpositive measured voltage")
    observed = Observation(pm, qm, rm[:, None]**2-vm**2, labels, vm, rm)
    splits = {name: _slice(observed, *bounds) for name, bounds in SPLIT.items()}
    split_diagnostics = {}
    for name, (start, stop) in SPLIT.items():
        v = voltage[start:stop]
        split_diagnostics[name] = dict(points=stop-start, observed_sha256=fingerprint(splits[name]),
            minimum_voltage_pu=float(v.min()), maximum_voltage_pu=float(v.max()),
            outside_090_110_cells=int(((v < .9) | (v > 1.1)).sum()),
            outside_095_105_cells=int(((v < .95) | (v > 1.05)).sum()))
    directory = ac_backend.ROOT/"cases"/case_name
    diagnostics = dict(case=case_name, status="converged", physical_model="original balanced radial AC",
        ac=ac, original_network_sha256=original_hash,
        sources={"AC": io.ref(ac_backend.__file__), **{name: io.ref(directory/name)
                 for name in ("metadata.json", "buses.csv", "branches.csv")}},
        base_kw=network["base_kw"], base_kv=network["base_kv"],
        physical_impedances="original case files, unchanged; never given to identification",
        added_device_baseline=False, automatic_power_rescaling=False,
        additional_simulated_meter_relative_sd=dict(P=pq_sigma, Q=pq_sigma, V=voltage_sigma, root=voltage_sigma),
        root_physical_variation_sigma=root_sigma, seed_entropy=entropy,
        source_PQ_interpretation="measured-profile-driven imposed injection, not noise-free field truth",
        power_sign="positive consumption, negative injection; signed P/Q retained",
        thermal_branch_limits_checked=False,
        voltage_screen="counts only; violations retained, no rescaling or day replacement",
        splits=split_diagnostics, profile_metadata=profile.get("metadata", {}))
    return dict(observed=observed, splits=splits, diagnostics=diagnostics,
                private=dict(network=network, p=p, q=q, voltage=voltage, root=root,
                             noises=noises, labels=labels, timestamps=timestamps))


def offline_truth(case_name, labels):
    """Offline structure only; never call this to populate an algorithm bank."""
    return ac_backend.case_truth(case_name, labels)


def self_test():
    """Small deterministic checks of units, independent noise, and AC zero load."""
    checks = []
    for case in CASES:
        network = ac_backend.load_case(case)
        labels = tuple(row["bus_id"] for row in network["buses"] if row["is_terminal"])
        start = datetime(2024, 1, 15, 8)
        profile = dict(labels=labels, p_kw=np.zeros((384, len(labels))), q_kvar=np.zeros((384, len(labels))),
                       timestamps=[(start+timedelta(minutes=15*k)).isoformat() for k in range(384)],
                       split=SPLIT)
        output = generate(case, profile, seed=4, voltage_sigma=0, pq_sigma=0, root_sigma=0)
        assert np.array_equal(output["observed"].y, np.zeros_like(profile["p_kw"]))
        assert np.all(output["observed"].voltage == 1.02)
        assert [v.points for v in output["splits"].values()] == [192, 96, 96]
        profile["p_kw"][:] = -.2
        profile["q_kvar"][:] = -.05
        signed = generate(case, profile, seed=4, voltage_sigma=0, pq_sigma=0, root_sigma=0)
        assert np.all(signed["observed"].p == -.2/network["base_kw"])
        assert np.all(signed["observed"].q == -.05/network["base_kw"])
        assert signed["observed"].voltage.min() > 1.02
        a, b = generate(case, profile, seed=5), generate(case, profile, seed=5)
        assert fingerprint(a["observed"]) == fingerprint(b["observed"])
        assert not np.array_equal(a["private"]["noises"][0], a["private"]["noises"][1])
        bad = dict(profile, timestamps=profile["timestamps"][:-1])
        try:
            generate(case, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete timestamps were accepted")
        checks.append(case)
    return dict(status="passed", cases=checks, tests="zero-load AC, kW/kvar units, signed export, split, paired repeatability, separate P/Q noise, incomplete block rejection")


if __name__ == "__main__":
    print(json.dumps(self_test(), ensure_ascii=False))
