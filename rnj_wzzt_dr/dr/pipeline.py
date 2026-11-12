"""Native frozen candidates -> zero-action certificate -> existing pair solver."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import io
from .conditional import build_gate
from .devices import Devices
from .miqp_mse import solve_factorized
from .profiles import profiles
from .resource_adapter import ResourceOnly
from .zero import spectral_pair, joint


def load_resources(case, *, labels, directory=None):
    directory = Path(directory) if directory else Path(__file__).resolve().parents[1]/"resources"
    allocation = next(c for c in io.read(directory/"case_allocations.json")["cases"] if c["case"] == case)
    original = Devices(allocation, io.read(directory/"library.json"), profiles)
    if tuple(original.labels) != tuple(labels):
        raise ValueError("Original resource allocation has different terminal labels/order")
    return original, ResourceOnly(original, labels), float(allocation["base_kw"])


def prepare_dr(history, forecast, *, voltage_sigma, resources=None,
               power_relative_sd=.05, time_ar=.7):
    """Forecast is supplied from history, with explicit slot alignment/units.

    No time-of-day inference, truth, test, physical network or new day is read.
    Defaults retain the old fixed Gaussian working law, without calibration.
    """
    forecast = np.array(forecast, dtype=float, copy=True)
    n = len(history.labels)
    if (forecast.ndim != 2 or not forecast.shape[0] or forecast.shape[1] != 2*n
            or not np.isfinite(forecast).all()):
        raise ValueError("Forecast must be finite [new slots, P channels then Q channels] in pu")
    if (not np.isfinite([voltage_sigma, power_relative_sd, time_ar]).all()
            or min(voltage_sigma, power_relative_sd) < 0 or not 0 <= time_ar < 1):
        raise ValueError("Nonnegative error scales and temporal AR in [0, 1) required")
    data = dict(labels=list(history.labels), models=history.models, H=history.H,
        current=history.current, history_sse=history.sse, K=history.K,
        strict_scaled_epsilon=history.epsilon_scaled,
        history_power=np.column_stack((history.observed.p, history.observed.q)),
        history_drop=history.observed.y.copy(), forecast=forecast,
        voltage_sigma=float(voltage_sigma), power_relative_sd=float(power_relative_sd),
        time_ar=float(time_ar), parameter_sha256=history.parameter_sha256)
    if resources is not None:
        original, adapter, base_kw = resources
        if (original.T != forecast.shape[0] or adapter.T != original.T
                or tuple(original.labels) != history.labels
                or not np.isfinite(base_kw) or base_kw <= 0):
            raise ValueError("Original resource day/order/base disagrees with forecast")
        data.update(original_device=original, p=adapter, base_kw=float(base_kw))
    return data


def _index(data, index):
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(data["models"]):
        raise ValueError("Target/competitor must index the complete frozen bank")
    return int(index)


def zero_screen(data, *, target):
    target = _index(data, target)
    T = len(data["forecast"])
    corr = data["time_ar"]**np.abs(np.arange(T)[:, None]-np.arange(T)[None, :])
    values, vectors = np.linalg.eigh(corr)
    temporal = np.maximum(values, 0.), vectors
    view = dict(data, current=target)
    competitors = [j for j, m in enumerate(data["models"])
                   if m["topology_id"] != data["models"][target]["topology_id"]]
    rows = [spectral_pair(view, j, temporal) for j in competitors]
    certificate = joint(rows)
    fee, device = None, None
    if "p" in data:
        p = data["p"]
        zero = p.nominal(p.zero)
        device = p.audit_delivery(zero["mean_u"])
        fee = float(zero["fee"])
        if not device["passed"] or fee != 0 or np.any(zero["mean_u"] != 0):
            raise ValueError("Zero device replay does not have zero feasible delivery/cost")
    return dict(target=target, pairs=rows, joint=certificate,
        minimum_fee=0. if fee == 0 and certificate["sufficient_90"] else None,
        zero_device=device, parameter_sha256=data["parameter_sha256"],
        coverage="all different active topologies in native frozen bank",
        qualification="truth coverage and actual AC success not evaluated here")


def pair_gate(data, *, target, competitor, alpha=.1, full=True):
    target, competitor = _index(data, target), _index(data, competitor)
    if target == competitor or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Distinct candidates and fixed risk alpha in (0, 1) required")
    if "p" not in data:
        raise ValueError("Original resources required for the dense/control gate")
    # Existing core treats reference as the conditionally correct generator.
    view = dict(data, current=competitor, reference=target, ref=target)
    gate = build_gate(view, full=full, beta=1-alpha)
    gate.zero_control_only = not full
    gate.input_receipt = dict(target=target, competitor=competitor, alpha=alpha,
        parameter_sha256=data["parameter_sha256"],
        forecast_sha256=io.contract_hash(data["forecast"]),
        history_power_sha256=io.contract_hash(data["history_power"]),
        history_drop_sha256=io.contract_hash(data["history_drop"]),
        error_law=dict(voltage_sigma=data["voltage_sigma"],
                       power_relative_sd=data["power_relative_sd"], time_ar=data["time_ar"]),
        K=data["K"], epsilon_scaled=data["strict_scaled_epsilon"], sources=io.source_receipt())
    return gate


def plan(data, *, target, output, seconds=300., mip_gap=.01, warm=None):
    """Finish a zero-fee plan, or use the retained single-competitor optimizer.

    Nonzero joint pricing remains a separate implementation task. It must
    retain the original graph and prelock risk budgets across competitors.
    """
    if "p" not in data:
        raise ValueError("Original resource adapter is required for pricing")
    if not np.isfinite([seconds, mip_gap]).all() or seconds <= 0 or not 0 <= mip_gap < 1:
        raise ValueError("Finite positive time budget and MIP gap in [0, 1) required")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new private output directory; preserve prior attempts")
    screen = zero_screen(data, target=target)
    if screen["minimum_fee"] == 0:
        point = data["p"].nominal(data["p"].zero)
        output.mkdir(parents=True)
        io.write(output/"request.json", dict(target=target, source=io.source_receipt(),
            forecast_sha256=io.contract_hash(data["forecast"]),
            parameter_sha256=data["parameter_sha256"], error_law=dict(
                voltage_sigma=data["voltage_sigma"], power_relative_sd=data["power_relative_sd"],
                time_ar=data["time_ar"]), limits=dict(seconds=seconds, mip_gap=mip_gap)))
        np.savez_compressed(output/"quote.npz", x=point["x"], mean_u=point["mean_u"])
        result = dict(status="complete", qualified=True, objective=0., lower_bound=0.,
                      certified_minimum_fee=True, optimization_calls=0, screen=screen)
        io.write(output/"summary.json", result)
        io.write(output/"complete.json", dict(status="complete", qualified=True))
        return result
    if len(screen["pairs"]) != 1:
        raise NotImplementedError("Nonzero multi-competitor pricing is not connected; retain zero-screen result")
    competitor = screen["pairs"][0]["competitor"]
    gate = pair_gate(data, target=target, competitor=competitor, full=True)
    p = data["p"]
    warm = p.expand(p.zero) if warm is None else warm
    return solve_factorized(p, gate, output, warm, dict(seconds=seconds, mip_gap=mip_gap))
