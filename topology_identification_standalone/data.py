"""Local benchmark observations and a small, self-contained radial AC generator.

Historical observations are bundled byte for byte. New synthetic trajectories
use the explicitly documented generator below, rather than the historical
project's simulator. Algorithms receive Observation objects, never a true tree.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")


@dataclass(frozen=True)
class Observation:
    """Time-by-terminal P/Q and observed squared-voltage drop in per unit."""

    p: np.ndarray
    q: np.ndarray
    y: np.ndarray
    labels: tuple[int, ...]
    voltage: np.ndarray | None = None
    root: np.ndarray | None = None

    def __post_init__(self):
        labels = tuple(map(int, self.labels))
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("Observation labels must be nonempty and unique")
        arrays = [np.asarray(value, dtype=float) for value in (self.p, self.q, self.y)]
        if any(value.ndim != 2 for value in arrays) or not all(value.shape == arrays[0].shape for value in arrays):
            raise ValueError("p, q and y must have the same time-by-terminal shape")
        if arrays[0].shape[1] != len(labels) or not arrays[0].shape[0] or not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Observation dimensions or finite-value checks failed")
        for name, value in zip(("p", "q", "y"), arrays):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "labels", labels)
        if self.voltage is not None:
            voltage = np.asarray(self.voltage, dtype=float)
            if voltage.shape != arrays[0].shape or not np.isfinite(voltage).all():
                raise ValueError("Voltage shape or finite-value check failed")
            object.__setattr__(self, "voltage", voltage)
        if self.root is not None:
            root = np.asarray(self.root, dtype=float).reshape(-1)
            if root.shape != (arrays[0].shape[0],) or not np.isfinite(root).all():
                raise ValueError("Root-voltage shape or finite-value check failed")
            object.__setattr__(self, "root", root)

    @property
    def n(self):
        return len(self.labels)

    @property
    def points(self):
        return self.p.shape[0]

    @property
    def rows(self):
        return self.points


Observations = Observation


def _local(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog():
    """Return the 48 bundled jobs; all data and case paths are folder-relative."""
    return json.loads((ROOT/"data/catalog.json").read_text(encoding="utf-8"))


def load_observations(path):
    """Read observed P/Q/V/root only; flatten days without discarding slots."""
    with np.load(_local(path), allow_pickle=False) as archive:
        labels = tuple(map(int, archive["labels"]))
        arrays = [np.asarray(archive[key], float).reshape(-1, len(labels)) for key in ("P", "Q", "V")]
        root = np.asarray(archive["root"], float).reshape(-1)
    p, q, voltage = arrays
    return Observation(p, q, root[:, None]**2-voltage**2, labels, voltage, root)


def load_case(case):
    """Load a bundled physical network; this function is not called by fitting."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}")
    directory = ROOT/"cases"/case
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    with (directory/"buses.csv").open(encoding="utf-8", newline="") as stream:
        buses = list(csv.DictReader(stream))
    with (directory/"branches.csv").open(encoding="utf-8", newline="") as stream:
        branches = list(csv.DictReader(stream))
    for row in buses:
        row["bus_id"] = int(row["bus_id"])
        for field in ("pd_kw", "qd_kvar"):
            row[field] = float(row[field])
        for field in ("is_terminal", "is_observed", "has_load"):
            row[field] = row[field].strip().lower() == "true"
    for row in branches:
        for field in ("from_bus", "to_bus"):
            row[field] = int(row[field])
        for field in ("r_ohm", "x_ohm"):
            row[field] = float(row[field])
        row["status"] = row["status"].strip().lower() == "true"
    return dict(metadata=metadata, buses=buses, branches=branches,
                root_bus=int(metadata["root_bus"]), base_kw=1000.*float(metadata["base_mva"]),
                base_kv=float(metadata["base_kv"]), base_mva=float(metadata["base_mva"]))


def _radial_order(network):
    """Orient a connected radial graph away from its physical root."""
    root = network["root_bus"]
    buses = {row["bus_id"] for row in network["buses"]}
    active = [row for row in network["branches"] if row["status"]]
    if len(active) != len(buses)-1:
        raise ValueError("The bundled generator requires a connected radial case")
    adjacency = {bus: [] for bus in buses}
    for edge in active:
        a, b = edge["from_bus"], edge["to_bus"]
        adjacency[a].append((b, edge)); adjacency[b].append((a, edge))
    order, parent, incoming = [root], {root: None}, {}
    for bus in order:
        for child, edge in sorted(adjacency[bus], key=lambda value: value[0]):
            if child == parent[bus]:
                continue
            if child in parent:
                raise ValueError("The physical case contains a cycle")
            parent[child], incoming[child] = bus, edge
            order.append(child)
    if set(order) != buses:
        raise ValueError("The physical case is disconnected")
    return order, parent, incoming


def case_truth(case, labels):
    """Return physical branch descendant sets, using zero-based terminal indices."""
    network, labels = load_case(case), tuple(map(int, labels))
    order, parent, _ = _radial_order(network)
    if not set(labels) <= set(order):
        raise ValueError("An observed terminal label is absent from the physical case")
    descendants = {node: set() for node in order}
    for index, label in enumerate(labels):
        descendants[label].add(index)
    supports = set()
    for child in reversed(order[1:]):
        if descendants[child]:
            supports.add(tuple(sorted(descendants[child])))
        descendants[parent[child]].update(descendants[child])
    return tuple(sorted(supports, key=lambda support: (len(support), support)))


def load_input(job):
    """Convenience API for benchmark evaluation, including separately held truth."""
    metadata = dict(load_catalog()[job])
    train, validation, test = [load_observations(metadata[split]) for split in ("train", "validation", "test")]
    if not train.labels == validation.labels == test.labels:
        raise ValueError("Terminal ordering differs across splits")
    truth = case_truth(metadata["case"], train.labels)
    return train, validation, test, truth, metadata


def verify_bundled_data():
    """Check local bytes against the bundled manifest; no external path is opened."""
    manifest = json.loads((ROOT/"data/manifest.json").read_text(encoding="utf-8"))
    mismatches = [name for name, expected in manifest["files"].items() if _sha(ROOT/name) != expected]
    return dict(passed=not mismatches, checked_files=len(manifest["files"]), mismatches=mismatches)


def _ac_voltages(network, labels, p, q, root, *, tolerance=1e-10, max_iterations=200):
    """Balanced radial PQ AC flow by backward/forward current sweeps.

    P/Q are positive consumption in per unit, voltage is phase-equivalent per
    unit. Zbase=kV_line_line²/MVA_three_phase, yielding ohms. Hidden buses in
    these cases have zero injection; terminal meters carry all specified load.
    """
    order, parent, incoming = _radial_order(network)
    position = {bus: index for index, bus in enumerate(order)}
    load = np.zeros((len(root), len(order)), dtype=complex)
    for k, label in enumerate(labels):
        load[:, position[label]] = p[:, k]+1j*q[:, k]
    zbase = network["base_kv"]**2/network["base_mva"]
    impedance = {child: complex(incoming[child]["r_ohm"], incoming[child]["x_ohm"])/zbase for child in order[1:]}
    voltage = np.repeat(np.asarray(root, float)[:, None], len(order), axis=1).astype(complex)
    for iteration in range(max_iterations):
        if np.any(np.abs(voltage) < .2):
            raise RuntimeError("AC iteration reached a collapsed-voltage branch")
        current = np.conj(load/voltage)
        for child in reversed(order[1:]):
            current[:, position[parent[child]]] += current[:, position[child]]
        updated = voltage.copy(); updated[:, 0] = root
        for child in order[1:]:
            updated[:, position[child]] = updated[:, position[parent[child]]]-impedance[child]*current[:, position[child]]
        error = float(np.max(np.abs(updated-voltage)))
        voltage = updated
        if error <= tolerance:
            terminal_voltage = np.abs(voltage[:, [position[label] for label in labels]])
            return terminal_voltage, dict(iterations=iteration+1, maximum_voltage_update=error,
                                          minimum_voltage=float(terminal_voltage.min()),
                                          maximum_voltage=float(terminal_voltage.max()))
    raise RuntimeError(f"AC backward/forward sweeps did not converge after {max_iterations} iterations")


def generate_synthetic(case="paper15", seed=0, points=96, voltage_sigma=.0005,
                       pq_sigma=.005, root_sigma=.003):
    """Generate independent train/validation/test AC splits from local case files.

    This is a new reproducible synthetic generator, not a byte-equivalent
    recreation of the historical panel. Validation has three days, other
    splits one. The same physical root sample enters every terminal target.
    """
    if not isinstance(points, int) or points < 4:
        raise ValueError("points must be an integer at least four")
    if any(not np.isfinite(v) or v < 0 for v in (voltage_sigma, pq_sigma, root_sigma)):
        raise ValueError("Noise and root variation scales must be finite and nonnegative")
    network = load_case(case)
    terminal_buses = [row for row in network["buses"] if row["is_terminal"]]
    labels = tuple(row["bus_id"] for row in terminal_buses)
    p_nominal = np.asarray([row["pd_kw"] for row in terminal_buses])/network["base_kw"]
    q_nominal = np.asarray([row["qd_kvar"] for row in terminal_buses])/network["base_kw"]
    generated, diagnostics = [], {}
    for split_index, (name, days) in enumerate((("train", 1), ("validation", 3), ("test", 1))):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), split_index]))
        count = days*points
        phase = 2*np.pi*np.arange(count)/points
        daily = .65+.20*np.sin(phase-.5*np.pi)+.10*np.sin(2*phase-.9*np.pi)
        shifts = rng.uniform(-np.pi, np.pi, len(labels))
        fluctuations = rng.normal(0., .04, (count, len(labels)))
        for t in range(1, count):
            fluctuations[t] += .8*fluctuations[t-1]
        p_factor = np.clip(daily[:, None]+.12*np.sin(phase[:, None]+shifts)+fluctuations, .15, 1.4)
        q_factor = p_factor*np.clip(1.+.15*np.sin(phase[:, None]-shifts)+rng.normal(0., .04, p_factor.shape), .5, 1.5)
        p_true, q_true = p_factor*p_nominal, q_factor*q_nominal
        root_true = 1.02+root_sigma*(.7*np.sin(phase+.3)+.3*rng.normal(size=count))
        v_true, diagnostic = _ac_voltages(network, labels, p_true, q_true, root_true)
        p = p_true*(1.+pq_sigma*rng.normal(size=p_true.shape))
        q = q_true*(1.+pq_sigma*rng.normal(size=q_true.shape))
        voltage = v_true*(1.+voltage_sigma*rng.normal(size=v_true.shape))
        root = root_true*(1.+voltage_sigma*rng.normal(size=count))
        generated.append(Observation(p, q, root[:, None]**2-voltage**2, labels, voltage, root))
        diagnostics[name] = dict(diagnostic, points=count)
    metadata = dict(case=case, seed=int(seed), points=points, voltage_sigma=float(voltage_sigma),
                    pq_sigma=float(pq_sigma), root_sigma=float(root_sigma), root_label=network["root_bus"],
                    labels=list(labels), base_kw=network["base_kw"], base_kv=network["base_kv"],
                    generator="local balanced radial backward/forward AC sweep; new daily load law",
                    historical_byte_equivalence=False, diagnostics=diagnostics)
    return *generated, case_truth(case, labels), metadata
