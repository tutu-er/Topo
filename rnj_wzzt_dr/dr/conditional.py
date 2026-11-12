"""Existing dense Gaussian gate, with explicit scale and forecast-law inputs."""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular
from .miqp_mse import MSEGate

def build_gate(data, p=None, *, full=True, beta=.9):
    """Return the existing MSEGate for strict MSE exclusion (kappa=1).

    The numerical floor is the final selector's declared K*MSE tolerance.
    ``data['gate_inputs']`` exposes exact sampling
    parameters. Actual random DR is deliberately outside this fixed Gaussian
    forecast model and is evaluated by the unchanged physical devices.
    """
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
    power_sd = data["power_relative_sd"]*np.max(np.abs(data["forecast"]), axis=0)
    node_factor = dh[:, :n]*power_sd[:n]+dh[:, n:]*power_sd[n:]
    time_ar = data["time_ar"]
    time_corr = time_ar**np.abs(np.arange(T)[:, None]-np.arange(T)[None, :])
    C = np.kron(time_corr, (node_factor@node_factor.T)/(s*s))
    G = np.eye(T*n)+2*np.sqrt(t)[:, None]*C*np.sqrt(t)[None, :]
    chol = np.linalg.cholesky((G+G.T)/2)
    mean = (data["forecast"]@dh.T/s).ravel()
    offset = solve_triangular(chol, np.sqrt(t)*mean, lower=True)
    H0sse, H1sse = data["history_sse"][[data["current"], data["reference"]]]
    total_cells = data["history_drop"].size+T*n
    delta = float((total_cells*data["strict_scaled_epsilon"]/data["K"]-H0sse+H1sse)/(s*s))
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
        kappa=1., epsilon_scaled=data["strict_scaled_epsilon"],
        epsilon_physical_MSE=data["strict_scaled_epsilon"]/data["K"],
        remaining_error_assumption="independent Gaussian, fixed per-node validation RMS or meter reference",
        power_error_assumption="fixed channel SD from declared forecast fraction; same-node P/Q shared, independent nodes, declared temporal AR",
        zero_control_screen_only=not full, nonzero_effect_resources=active.copy())
    return MSEGate(offset, factor, active, h, 1., delta, float(beta), s,
                   data["K"], 0., (float(H0sse/(s*s)), float(H1sse/(s*s))),
                   int(data["history_drop"].size))
