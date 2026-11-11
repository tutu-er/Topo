"""Historical QP output models, original resources, and a strict-MSE DR gate.

Public interface
----------------
``data = load_history_case(result, catalog_row, allocation, library)`` reads
only the declared training and validation NPZ files. ``result`` is a native
MIQP result dict or its JSON path. Its .24-contracted/QP-refitted output library
is retained, with exact duplicate parameter points merged in original order.
No raw-path model, test observation, network or true support is opened.

``gate = build_gate(data, p=None, full=True, beta=.9)`` uses the current strict
historical MSE winner as model0 and the nearest different-support historical
competitor as model1. It returns the existing ``miqp_mse.MSEGate``. ``full=False``
computes the zero-control score only; its zero factor is not a price model.
``gate.scale`` is in PHYSICAL voltage-response units (pu^2), unlike the older
paper15 micro-unit input. Do not multiply it by 1e-6 in this interface.

``data['gate_inputs']`` contains the declared Gaussian inputs for independent
working-model sampling. Power noise has a fixed per-channel standard deviation
equal to 5% of that channel's maximum absolute forecast value. P/Q at a node
share one factor, different nodes are independent, and time AR is .7.
Remaining voltage error is independent and diagonal, with a node-specific SD
max(2*sqrt(2)*meter_relative_SD, validation RMS of model1). These are working
assumptions, not calibrated probabilities. Random actual DR delivery is not
included in this fixed covariance and remains a separate execution test.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

from research_experiments.wzzt_dr_selection.dr_heterogeneous_pilot.devices import Devices as OriginalDevices
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles


_LEGACY = Path(__file__).resolve().parents[1] / "paper15_exponential"
_GATE_CLASS = None
_SOLVER_DEVICES = None
STRICT_SCALED_EPSILON = 1e-12


def _old_types():
    """Import the unchanged device class and Gaussian gate, without AC imports.

    The old file uses ``import run``. Bind that name only while loading it, so
    the caller's driver named run cannot silently become its dependency.
    """
    global _GATE_CLASS, _SOLVER_DEVICES
    if _GATE_CLASS is not None:
        return _SOLVER_DEVICES, _GATE_CLASS
    root = str(_LEGACY)
    if root not in sys.path:
        sys.path.insert(0, root)
    problem = importlib.import_module("vendor.problem")
    if Path(problem.__file__).resolve() != (_LEGACY / "vendor/problem.py").resolve():
        raise RuntimeError("A different vendor.problem is loaded; use a clean worker")
    saved_run = sys.modules.get("run")
    try:
        key = "_mse_existing_original_run"
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, _LEGACY / "run.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        sys.modules["run"] = sys.modules[key]
        key = "_mse_existing_original_miqp_mse"
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, _LEGACY / "miqp_mse.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        _GATE_CLASS = sys.modules[key].MSEGate
        _SOLVER_DEVICES = problem.Devices
    finally:
        if saved_run is None:
            sys.modules.pop("run", None)
        else:
            sys.modules["run"] = saved_run
    return _SOLVER_DEVICES, _GATE_CLASS


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path)}


def _finite(value, name, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError(f"Invalid finite shape for {name}: {a.shape}, expected {shape}")
    return a


class ResourceOnly:
    """Original device/fee contract without any MAE candidate qualification.

    The resource controller is the existing vendor.problem.Devices; the small
    active-hour mapping below has the same semantics as PricingProblem. This
    adapter never constructs its MAE eligibility or probability model.
    """
    def __init__(self, original, labels):
        SolverDevices, _ = _old_types()
        terminal = np.zeros_like(original.reset, dtype=bool)
        terminal[original.kind != 1, -1] = True
        terminal[original.kind == 2, 31] = True
        arrays = dict(resource_lower_kw=original.delta_lower_kw,
            resource_upper_kw=original.delta_upper_kw,
            resource_cost=original.cost_full_activation,
            resource_efficiency=original.eta,
            resource_energy_lower_kwh=original.energy_low,
            resource_energy_upper_kwh=original.energy_high,
            resource_reset=original.reset, resource_terminal=terminal,
            resource_future_lower_kwh=original.future_lo,
            resource_future_upper_kwh=original.future_hi)
        self.device = SolverDevices(arrays, original.dt, 4)
        self.T, self.H, self.N = self.device.T, self.device.H, len(labels)
        self.shape = (self.device.R, self.device.H)
        self.node_labels, self.resource_ids = list(labels), list(original.ids)
        self.config = self.metadata = dict(node_labels=self.node_labels,
            resource_ids=self.resource_ids, dt_hours=original.dt, decision_slots=4,
            resource_contract="unchanged heterogeneous full inventory",
            selection="external strict unweighted physical MSE; no MAE gate")
        scale = self.device.scale.reshape(self.device.R, self.H, 4)
        self.active = np.flatnonzero(scale.max(axis=2).ravel() > 0)
        if not self.active.size:
            raise ValueError("No available control intervals in the original resource inventory")
        low = self.device.delta_lower_kw.reshape(scale.shape).min(axis=2) < 0
        high = self.device.delta_upper_kw.reshape(scale.shape).max(axis=2) > 0
        self.lower = np.where(low.ravel()[self.active], -1., 0.)
        self.upper = np.where(high.ravel()[self.active], 1., 0.)
        self.bounds = list(zip(self.lower, self.upper))
        self.zero = np.zeros(self.active.size)
        self.warm = self.old = self.zero.copy()
        for name in ("delta_lower_kw", "delta_upper_kw", "cost_full_activation",
                     "eta", "energy_low", "energy_high", "reset", "future_lo", "future_hi"):
            if not np.array_equal(getattr(self.device, name), getattr(original, name)):
                raise ValueError("Solver and original device differ: " + name)
        if not self.audit_delivery(np.zeros((self.device.R, self.T)))["passed"]:
            raise ValueError("Zero delivery is not feasible for the declared device contract")

    def expand(self, values):
        values = _finite(values, "active hourly requests", self.active.shape)
        x = np.zeros(self.device.R * self.H)
        x[self.active] = values
        return x.reshape(self.shape)

    def requests_to_values(self, requests):
        x = _finite(requests, "hourly requests", self.shape)
        values = x.ravel()[self.active]
        if (np.any(values < self.lower) or np.any(values > self.upper)
                or not np.array_equal(self.expand(values), x)):
            raise ValueError("Requests use an unavailable interval or violate original bounds")
        return values.copy()

    def nominal(self, values):
        x = self.expand(values)
        u, cache = self.device.nominal(x, cache=True)
        fee, derivative = self.device.fees(u, gradient=True)
        return dict(x=x, mean_u=u, cache=cache, fee=fee, fee_gradient=derivative)

    def audit_delivery(self, delivery):
        return self.device.audit(delivery)

    def unit_reward(self, delivery):
        u = _finite(delivery, "delivery", (self.device.R, self.T))
        headroom = np.where(u >= 0, self.device.delta_upper_kw, -self.device.delta_lower_kw)
        return np.divide(self.device.cost_full_activation * np.abs(u), headroom,
                         out=np.zeros_like(u), where=headroom > 0)


def _observations(path, expected_slots, labels):
    """Only load named observed arrays, never a test or truth file."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as z:
        declared = np.asarray(z["labels"]).astype(int).tolist()
        if declared != labels:
            raise ValueError("Historical terminal order differs from the declared candidate order")
        n = len(labels)
        pp, qq, vv = [np.asarray(z[k], float).reshape(-1, n) for k in ("P", "Q", "V")]
        root = np.asarray(z["root"], float).reshape(-1)
    for name, value in (("P", pp), ("Q", qq), ("V", vv)):
        _finite(value, name, (expected_slots, n))
    _finite(root, "observed root voltage", (expected_slots,))
    if np.any(vv <= 0) or np.any(root <= 0):
        raise ValueError("Historical measured voltages must be positive")
    return np.column_stack((pp, qq)), root[:, None]**2 - vv**2


def _point(point, n):
    supports = [tuple(sorted(map(int, c))) for c in point["supports"]]
    r = _finite(point["r"], "r", (len(supports),))
    x = _finite(point["x"], "x", (len(supports),))
    if (not supports or len(set(supports)) != len(supports)
            or any(not c or len(set(c)) != len(c) or min(c) < 0 or max(c) >= n for c in supports)
            or np.any(r < 0) or np.any(x < 0)):
        raise ValueError("Invalid nonnegative fixed candidate parameters/supports")
    order = sorted(range(len(supports)), key=lambda i: (len(supports[i]), supports[i]))
    normalized = dict(supports=[list(supports[i]) for i in order],
                      r=[float(r[i]) for i in order], x=[float(x[i]) for i in order])
    z = np.array([[j in c for j in range(n)] for c in supports], float)
    R, X = (z.T * r) @ z, (z.T * x) @ z
    key = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return normalized, np.column_stack((R, X)), key


def load_history_case(result, catalog_row, allocation, library):
    """Return all frozen .24/QP output models and the original resource adapter.

    Required catalog fields: case, labels, train, validation, voltage_sigma,
    pq_sigma. Paths may alternatively be train_path/validation_path. Only the
    96 training and 288 validation observations are read. No test/truth key in
    catalog_row is accessed. ``current`` and ``ref``/``reference`` index models.
    A missing different-support competitor is reported as reference=None.
    """
    references = []
    if isinstance(result, (str, Path)):
        result_path = Path(result).resolve()
        references.append(_reference(result_path))
        native = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        native = result
    if native.get("fixed_parameters", {}).get("output_resolution") != .24:
        raise ValueError("The frozen native output resolution must be .24")
    fitted_hashes = native.get("provenance", {}).get("input_hashes", {})
    labels = [int(x) for x in catalog_row["labels"]]
    if len(set(labels)) != len(labels) or not labels:
        raise ValueError("Unique historical labels required")
    paths = {}
    for role in ("train", "validation"):
        value = catalog_row.get(role, catalog_row.get(role + "_path"))
        if isinstance(value, dict):
            path, expected = value["path"], value.get("sha256")
        else:
            path = value
            expected = catalog_row.get(role + "_sha256")
        if path is None:
            raise ValueError("Explicit historical path required: " + role)
        path = Path(path).resolve()
        actual = _sha(path)
        if role not in fitted_hashes or actual != fitted_hashes[role]:
            raise ValueError("Historical observations differ from the original fitted input: " + role)
        if expected is not None and actual != expected:
            raise ValueError("Historical input hash changed: " + role)
        paths[role] = path
        references.append(_reference(path))
    train_power, train_drop = _observations(paths["train"], 96, labels)
    history_power, history_drop = _observations(paths["validation"], 288, labels)
    K = float(native["normalization"]["K"])
    if not np.isfinite(K) or K <= 0:
        raise ValueError("Positive native frozen K is required")
    variant = native.get("variants", {}).get("tau_0.001")
    if not isinstance(variant, dict) or not variant.get("library"):
        raise ValueError("Native tau_0.001 .24-contracted/QP output library required")
    rows = variant["library"]
    stored = native.get("models")
    if isinstance(stored, list):
        model_by_id = {row["id"]: row for row in stored}
    elif isinstance(stored, dict):
        model_by_id = stored
    else:
        raise ValueError("Native output model table required")
    models, by_key = [], {}
    n = len(labels)
    for ordinal, row in enumerate(rows):
        model_id = str(row["model_id"])
        stored_model = model_by_id[model_id]
        point, H, identity = _point(stored_model["point"], n)
        for name, matrix in (("R", H[:, :n]), ("X", H[:, n:])):
            if not np.allclose(_finite(stored_model[name], name, (n, n)),
                               matrix, atol=1e-12, rtol=1e-12):
                raise ValueError("Stored output matrix and support weights disagree: " + model_id)
        if identity in by_key:
            previous = models[by_key[identity]]
            previous["output_indices"].append(int(row.get("index", ordinal)))
            previous["model_ids"].append(model_id)
            previous["source_model_ids"].append(str(row.get("source_model_id", model_id)))
            continue
        if stored_model["point"].get("fixed_fit") not in (None, "QP"):
            raise ValueError("Output library contains a non-QP fitting declaration")
        residual = history_drop - history_power @ H.T
        training_residual = train_drop - train_power @ H.T
        model = dict(id=model_id, model_ids=[model_id],
            source_model_ids=[str(row.get("source_model_id", model_id))],
            parameter_sha256=stored_model.get("parameter_sha256", identity),
            canonical_parameter_sha256=identity, point=point, H=H,
            supports=point["supports"],
            support_labels=[[labels[i] for i in c] for c in point["supports"]],
            internal=sum(1 < len(c) < n for c in point["supports"]),
            history_sse=float(np.square(residual).sum()),
            history_mse=float(np.square(residual).mean()),
            train_mse=float(np.square(training_residual).mean()),
            output_indices=[int(row.get("index", ordinal))], output_resolution=.24)
        by_key[identity] = len(models)
        models.append(model)
    values = K * np.array([m["history_mse"] for m in models])
    eligible = np.flatnonzero(values <= values.min() + STRICT_SCALED_EPSILON)
    current = min(eligible, key=lambda i: (models[i]["internal"], values[i], i))
    selected = variant.get("selected_mse")
    if not isinstance(selected, dict) or selected.get("model_id") not in models[current]["model_ids"]:
        raise ValueError("Recomputed strict historical MSE winner differs from native selected_mse")
    family = lambda i: tuple(map(tuple, models[i]["point"]["supports"]))
    alternatives = [i for i in range(len(models)) if family(i) != family(current)]
    reference = (min(alternatives, key=lambda i: (abs(values[i] - values[current]),
                 models[i]["internal"], i)) if alternatives else None)
    if allocation["case"] != catalog_row["case"]:
        raise ValueError("Resource allocation belongs to a different physical case")
    original = OriginalDevices(allocation, library, profiles)
    if original.labels != labels:
        raise ValueError("Resource and observation terminal orders differ")
    base_kw = float(allocation["base_kw"])
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError("Positive original power base required")
    forecast = np.concatenate((train_power.reshape(1, 96, 2*n),
                               history_power.reshape(3, 96, 2*n))).mean(axis=0)
    p = ResourceOnly(original, labels)
    voltage_sigma, pq_sigma = float(catalog_row["voltage_sigma"]), float(catalog_row["pq_sigma"])
    if not np.isfinite([voltage_sigma, pq_sigma]).all() or min(voltage_sigma, pq_sigma) < 0:
        raise ValueError("Nonnegative fixed meter noise required")
    return dict(case=catalog_row["case"], labels=labels, models=models,
        H=np.stack([m["H"] for m in models]), current=int(current),
        ref=None if reference is None else int(reference),
        reference=None if reference is None else int(reference),
        history_power=history_power, history_drop=history_drop,
        train_power=train_power, train_drop=train_drop, forecast=forecast, K=K,
        history_sse=np.array([m["history_sse"] for m in models]),
        history_scaled_mse=values, strict_scaled_epsilon=STRICT_SCALED_EPSILON,
        base_kw=base_kw, voltage_sigma=voltage_sigma, pq_sigma=pq_sigma,
        p=p, originaldev=original, original_device=original,
        input_references=references, parameter_rule="native .24/QP outputs fixed",
        current_rule="strict K*MSE; internal count then score then original order",
        reference_rule="nearest different-support historical MSE candidate; no truth")


def build_gate(data, p=None, *, full=True, beta=.9):
    """Return the existing MSEGate for strict MSE exclusion (kappa=1).

    The numerical floor is the final selector's 1e-12 in K*MSE units, not the
    E-training rule's 1e-10. ``data['gate_inputs']`` exposes exact sampling
    parameters. Actual random DR is deliberately outside this fixed Gaussian
    forecast model and is evaluated by the unchanged physical devices.
    """
    _, MSEGate = _old_types()
    p = data["p"] if p is None else p
    if data["reference"] is None:
        raise ValueError("No different-support historical output competitor")
    if not 0 < beta < 1:
        raise ValueError("beta must lie strictly between zero and one")
    H0, H1 = data["H"][[data["current"], data["reference"]]]
    dh = H1-H0
    n, T = len(data["labels"]), p.T
    reference_residual = data["history_drop"]-data["history_power"]@H1.T
    node_sd = np.maximum(2*np.sqrt(2)*data["voltage_sigma"],
                         np.sqrt(np.square(reference_residual).mean(axis=0)))
    s = float(node_sd.max())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Remaining voltage error is degenerate; no artificial variance floor is added")
    E = np.tile((node_sd/s)**2, T)
    lam = 1/(4*float(E.max()))
    t = lam*(1-2*lam*E)  # kappa=1, so b_i=1.
    if np.any(t <= 0):
        raise ValueError("Invalid fixed exponential parameter")
    power_sd = .05*np.max(np.abs(data["forecast"]), axis=0)
    node_factor = dh[:, :n]*power_sd[:n]+dh[:, n:]*power_sd[n:]
    time_ar = .7
    time_corr = time_ar**np.abs(np.arange(T)[:, None]-np.arange(T)[None, :])
    C = np.kron(time_corr, (node_factor@node_factor.T)/(s*s))
    G = np.eye(T*n)+2*np.sqrt(t)[:, None]*C*np.sqrt(t)[None, :]
    chol = np.linalg.cholesky((G+G.T)/2)
    mean = (data["forecast"]@dh.T/s).ravel()
    offset = solve_triangular(chol, np.sqrt(t)*mean, lower=True)
    H0sse, H1sse = data["history_sse"][[data["current"], data["reference"]]]
    total_cells = data["history_drop"].size+T*n
    delta = float((total_cells*STRICT_SCALED_EPSILON/data["K"]-H0sse+H1sse)/(s*s))
    h = float(lam*delta-np.log1p(-beta)-np.log(np.diag(chol)).sum())
    dev = data["original_device"]
    effect = (dh[:, dev.node]+dh[:, n+dev.node]*dev.qratio[None, :])/data["base_kw"]
    active = np.flatnonzero(np.any(effect != 0, axis=0))
    if full:
        design = np.zeros((T*n, len(active)*T))
        for ti in range(T):
            design[ti*n:(ti+1)*n, np.arange(len(active))*T+ti] = effect[:, active]/s
        factor = solve_triangular(chol, np.sqrt(t)[:, None]*design, lower=True)
    else:
        factor = np.zeros((T*n, len(active)*T))
    data["gate_inputs"] = dict(mean=(data["forecast"]@dh.T)/s,
        node_sd=node_sd, voltage_sd=np.broadcast_to(node_sd, (T, n)).copy(),
        normalized_voltage_sd=np.broadcast_to(node_sd/s, (T, n)).copy(),
        power_sd=power_sd, normalized_difference_node_factor=node_factor/s,
        time_ar=time_ar, lambda_=lam, scale=s, scale_unit="physical pu^2",
        history_cells=int(data["history_drop"].size), total_cells=total_cells,
        kappa=1., epsilon_scaled=STRICT_SCALED_EPSILON,
        epsilon_physical_MSE=STRICT_SCALED_EPSILON/data["K"],
        remaining_error_assumption="independent Gaussian, fixed per-node validation RMS or meter reference",
        power_error_assumption="fixed5percent max forecast, same-node P/Q shared, independent nodes, timeAR.7",
        zero_control_screen_only=not full, nonzero_effect_resources=active.copy())
    return MSEGate(offset, factor, active, h, 1., delta, float(beta), s,
                   data["K"], 0., (float(H0sse/(s*s)), float(H1sse/(s*s))),
                   int(data["history_drop"].size))
