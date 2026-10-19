"""Complete fixed-candidate AC Gaussian measurement likelihood (research only).

All electrical quantities, including separate edge R and X, are per unit.
P/Q at terminals are total consumptions including EV. A freely varying common
source has exactly one physical location: node 1 after the full-support edge.
Its P and Q must be anchored by master P/Q or independent source telemetry.
The root voltage is known from calibration, not estimated in this first model.

This module adapts the canonical tree and batched impedance formulation in
ac_general_v9; it does not use that module's known X/R, error boxes or outer LP.
The production fit_ac_pqv implementation supplies the joint latent-P/Q modeling
pattern but fixes source amplitudes and uses ohm inputs, so is not called here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Mapping
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

_LEGACY = Path(__file__).resolve().parents[1] / "mobile_ev_probing"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))
from ac_general_v9 import RadialCandidate


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class Observations:
    """Independent time-by-terminal metering with a calibrated known root.

    master_p/q measure actual slack power including losses and the common
    source. source_p/q, when present, independently meter the common source.
    No measured voltage-drop channel is additionally included.
    """
    p: np.ndarray
    q: np.ndarray
    v: np.ndarray
    root_v: np.ndarray | float = 1.0
    master_p: np.ndarray | None = None
    master_q: np.ndarray | None = None
    source_p: np.ndarray | None = None
    source_q: np.ndarray | None = None

    def __post_init__(self):
        p = np.asarray(self.p, dtype=float)
        if p.ndim != 2 or not p.shape[0] or not p.shape[1]:
            raise ValueError("p must have shape (time, terminal) with positive dimensions")
        for name in ("p", "q", "v"):
            object.__setattr__(self, name, _array(getattr(self, name), p.shape, name))
        root = _array(self.root_v, (len(p),), "root_v")
        if (root <= 0).any() or (self.v <= 0).any():
            raise ValueError("observed voltage magnitudes must be positive")
        object.__setattr__(self, "root_v", root)
        for name in ("master_p", "master_q", "source_p", "source_q"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(value, (len(p),), name))

    @property
    def t(self):
        return len(self.p)

    @property
    def n(self):
        return self.p.shape[1]


@dataclass(frozen=True)
class Calibration:
    """Independent offset readings y_cal=b+epsilon for every P/Q/V meter.

    Each channel may have one or more rows K by n. Readings must originate
    independently of the operating records; zero-input or known-reference
    differences are examples. Their likelihood is included exactly once.
    """
    p: np.ndarray
    q: np.ndarray
    v: np.ndarray
    p_sigma: float | np.ndarray
    q_sigma: float | np.ndarray
    v_sigma: float | np.ndarray


@dataclass(frozen=True)
class MeasurementModel:
    """Frozen known independent Gaussian standard deviations.

    Diagonal covariance and independence across time/channel are explicit
    synthetic assumptions, not a field claim. The supplied bias corrections
    are externally calibrated. fit_bias=True instead estimates one shared
    bias per terminal/channel, and requires independent Calibration readings.
    """
    p_sigma: float | np.ndarray
    q_sigma: float | np.ndarray
    v_sigma: float | np.ndarray
    master_p_sigma: float | np.ndarray | None = None
    master_q_sigma: float | np.ndarray | None = None
    source_p_sigma: float | np.ndarray | None = None
    source_q_sigma: float | np.ndarray | None = None
    bias_p: float | np.ndarray = 0.0
    bias_q: float | np.ndarray = 0.0
    bias_v: float | np.ndarray = 0.0
    fit_bias: bool = False
    calibration: Calibration | None = None


@dataclass
class FitResult:
    candidate: str
    r: np.ndarray
    x: np.ndarray
    p: np.ndarray
    q: np.ndarray
    source_p: np.ndarray
    source_q: np.ndarray
    bias_p: np.ndarray
    bias_q: np.ndarray
    bias_v: np.ndarray
    loss: float
    loglik: float
    success: bool
    status: int
    message: str
    nfev: int
    optimality: float
    predicted: dict
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        """Small JSON-safe result; states remain accessible on the object."""
        return dict(candidate=self.candidate, r=self.r.tolist(), x=self.x.tolist(),
                    loss=float(self.loss), loglik=float(self.loglik),
                    success=bool(self.success), status=int(self.status),
                    message=self.message, nfev=int(self.nfev),
                    optimality=float(self.optimality), diagnostics=self.diagnostics)


def concat_observations(*items):
    if not items or len({x.n for x in items}) != 1:
        raise ValueError("at least one observation with consistent terminal count is required")
    values = {}
    for name in ("p", "q", "v", "root_v", "master_p", "master_q", "source_p", "source_q"):
        blocks = [getattr(item, name) for item in items]
        if all(b is None for b in blocks):
            values[name] = None
        elif any(b is None for b in blocks):
            raise ValueError(f"{name} channel must be present in all concatenated blocks")
        else:
            values[name] = np.concatenate(blocks)
    return Observations(**values)


def ac_predict(candidate, r, x, p, q, source_p=0.0, source_q=0.0,
               root_v=1.0, *, tol=2e-13, max_iter=400):
    """Exact radial constant-power AC sweep, R and X independently specified.

    Returns unbiased physical responses. Every call verifies the nodal AC
    equation and complex-power balance; a failure raises RuntimeError.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != candidate.n or not len(p):
        raise ValueError("p must have shape (time, candidate.n)")
    p = _array(p, p.shape, "p")
    q = _array(q, p.shape, "q")
    t, m = len(p), candidate.m
    r, x = _array(r, (m,), "r"), _array(x, (m,), "x")
    if (r < 0).any() or (x < 0).any():
        raise ValueError("nonnegative R and X required")
    sp, sq = _array(source_p, (t,), "source_p"), _array(source_q, (t,), "source_q")
    root = _array(root_v, (t,), "root_v")
    if (root <= 0).any():
        raise ValueError("root magnitude must be positive")
    loads = np.zeros((t, m), complex)
    loads[:, 0] = sp + 1j * sq
    loads[:, list(candidate.terminal_edges)] += p + 1j * q
    z = r + 1j * x
    zbus = (candidate.B * z) @ candidate.B.T
    voltage = np.broadcast_to(root[:, None], loads.shape).astype(complex).copy()
    for iteration in range(1, max_iter + 1):
        if np.min(abs(voltage)) < .1:
            raise RuntimeError("AC low-voltage iterate; restart is invalid")
        new = root[:, None] - np.conj(loads / voltage) @ zbus.T
        if not np.isfinite(new).all():
            raise RuntimeError("nonfinite AC iterate; restart is invalid")
        if np.max(abs(new - voltage)) <= tol:
            voltage = new
            break
        voltage = new
    else:
        raise RuntimeError("AC sweep failed to converge; restart is invalid")
    currents = np.conj(loads / voltage)
    branch_current = currents @ candidate.B
    master = root * np.conj(currents.sum(axis=1))
    losses = (abs(branch_current)**2 * z).sum(axis=1)
    balance = master - loads.sum(axis=1) - losses
    equation = voltage - root[:, None] + currents @ zbus.T
    max_equation = float(np.max(abs(equation)))
    max_balance = float(np.max(abs(balance)))
    if max_equation > 2e-10 or max_balance > 2e-10:
        raise RuntimeError("AC equation or power balance failed")
    return dict(p=p, q=q, v=abs(voltage[:, list(candidate.terminal_edges)]),
                root_v=root, master_p=master.real, master_q=master.imag,
                source_p=sp, source_q=sq, voltage=voltage, loads=loads,
                branch_current=branch_current, zbus=zbus,
                ac_equation_residual=max_equation, power_balance_residual=max_balance,
                p_balance_max_abs=float(np.max(abs(balance.real))),
                q_balance_max_abs=float(np.max(abs(balance.imag))), iterations=iteration)


def _response_derivatives(candidate, physical, estimate_source=True):
    """Implicit real AC derivatives: solves dV-Z diag(c) conj(dV)=RHS.

    These derivatives are exact for the converged nonsingular local branch;
    complex-step is inappropriate because the equations contain conjugation.
    Columns are shared R/X then local terminal P/Q and optional source P/Q.
    """
    voltage, loads, zbus = (physical[k] for k in ("voltage", "loads", "zbus"))
    t, m = voltage.shape
    n, local = candidate.n, 2 * candidate.n + (2 if estimate_source else 0)
    width = 2 * m + local
    c = np.conj(loads) / np.conj(voltage)**2
    k = zbus[None, :, :] * c[:, None, :]
    eye = np.eye(m)[None, :, :]
    matrix = np.concatenate((np.concatenate((eye - k.real, -k.imag), axis=2),
                             np.concatenate((-k.imag, eye + k.real), axis=2)), axis=1)
    rhs = np.zeros((t, m, width), complex)
    # dZ @ I = B[:,e] * branch_current_e.
    edge = -candidate.B[None, :, :] * physical["branch_current"][:, None, :]
    rhs[:, :, :m], rhs[:, :, m:2*m] = edge, 1j * edge
    ds = np.zeros((m, local), complex)
    ds[list(candidate.terminal_edges), np.arange(n)] = 1
    ds[list(candidate.terminal_edges), n + np.arange(n)] = 1j
    if estimate_source:
        ds[0, 2*n], ds[0, 2*n+1] = 1, 1j
    direct_current = np.conj(ds)[None, :, :] / np.conj(voltage)[:, :, None]
    rhs[:, :, 2*m:] = -np.einsum("ij,tjk->tik", zbus, direct_current)
    dv_real = np.linalg.solve(matrix, np.concatenate((rhs.real, rhs.imag), axis=1))
    dv = dv_real[:, :m] + 1j * dv_real[:, m:]
    di = -c[:, :, None] * np.conj(dv)
    di[:, :, 2*m:] += direct_current
    dmaster = physical["root_v"][:, None] * np.conj(di.sum(axis=1))
    ids = list(candidate.terminal_edges)
    dv_mag = (np.conj(voltage[:, ids, None]) * dv[:, ids]).real / abs(voltage[:, ids, None])
    return dict(v=dv_mag, master_p=dmaster.real, master_q=dmaster.imag)


def _bounds(pair, shape, name):
    try:
        low, high = pair
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a (lower, upper) pair") from exc
    low, high = _array(low, shape, name), _array(high, shape, name)
    if np.any(low >= high):
        raise ValueError(f"{name} requires finite lower < upper")
    return low, high


def fit_candidate(candidate, observations, model, *, initial=None, multistart=2,
                  seed=0, max_nfev=150, r_bounds=(.000001, .1), x_bounds=(.000001, .1),
                  p_bounds=(-1., 1.), q_bounds=(-1., 1.), source_bounds=(-.5, .5),
                  voltage_bounds=(.85, 1.15), estimate_source=True, tol=1e-8, linear_solver="lsmr"):
    """Best retained local complete Gaussian fit, with explicit diagnostics.

    Warm starts may be shorter than observations: their shared parameters and
    historical states are retained, then future states initialize from meters.
    Optimizer success and restart agreement never certify a global MLE.
    """
    began = perf_counter()
    if not isinstance(observations, Observations) or not isinstance(model, MeasurementModel):
        raise TypeError("Observations and MeasurementModel are required")
    if observations.n != candidate.n:
        raise ValueError("candidate terminal count and observations differ")
    if int(multistart) != multistart or multistart < 1:
        raise ValueError("multistart must be a positive integer")
    if int(max_nfev) != max_nfev or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if linear_solver not in ("lsmr","exact"):
        raise ValueError("linear_solver must be lsmr or exact")
    t, n, m = observations.t, candidate.n, candidate.m
    channels = ["p", "q", "v"]
    sigmas = {}
    for name in ("p", "q", "v", "master_p", "master_q", "source_p", "source_q"):
        observed, sigma = getattr(observations, name), getattr(model, name + "_sigma")
        if name in channels or observed is not None:
            if sigma is None:
                raise ValueError(f"{name}_sigma is required for the observed {name} channel")
            shape = (t, n) if name in ("p", "q", "v") else (t,)
            sigmas[name] = _array(sigma, shape, name + "_sigma")
            if (sigmas[name] <= 0).any():
                raise ValueError("all included standard deviations must be positive")
            if name not in channels:
                channels.append(name)
        elif sigma is not None:
            raise ValueError(f"{name} observation is required when sigma is supplied")
    anchored = (all(k in channels for k in ("master_p", "master_q")) or
                all(k in channels for k in ("source_p", "source_q")))
    if estimate_source and not anchored:
        raise ValueError("free common source P/Q requires master P/Q or independent source P/Q telemetry")
    bias0 = [_array(getattr(model, "bias_" + k), (n,), "bias_" + k) for k in ("p", "q", "v")]
    calibration = []
    if model.fit_bias:
        if model.calibration is None:
            raise ValueError("free shared meter biases require independent calibration readings")
        for name in ("p", "q", "v"):
            values = np.asarray(getattr(model.calibration, name), dtype=float)
            if values.ndim == 1:
                values = values[None, :]
            if values.ndim != 2 or values.shape[1] != n or not len(values):
                raise ValueError("calibration readings must have shape (K,n) for all P/Q/V channels")
            values = _array(values, values.shape, "calibration." + name)
            sigma = _array(getattr(model.calibration, name + "_sigma"), values.shape,
                           "calibration." + name + "_sigma")
            if (sigma <= 0).any():
                raise ValueError("calibration standard deviations must be positive")
            calibration.append((name, values, sigma))
            bias0[("p", "q", "v").index(name)] = values.mean(axis=0)
    elif model.calibration is not None:
        raise ValueError("calibration likelihood is used only when fit_bias=True")
    bias_width = 3 * n if model.fit_bias else 0
    shared, state_width = 2*m + bias_width, 2*n + (2 if estimate_source else 0)
    residual_width = sum(n if k in ("p", "q", "v") else 1 for k in channels)
    calibration_size = sum(v.size for _, v, _ in calibration)
    count = shared + t * state_width
    rlo, rhi = _bounds(r_bounds, (m,), "r_bounds")
    xlo, xhi = _bounds(x_bounds, (m,), "x_bounds")
    if (rlo < 0).any() or (xlo < 0).any():
        raise ValueError("R/X bounds must be nonnegative")
    plo, phi = _bounds(p_bounds, (t, n), "p_bounds")
    qlo, qhi = _bounds(q_bounds, (t, n), "q_bounds")
    slo, shi = _bounds(source_bounds, (t, 2), "source_bounds")
    lower = np.r_[rlo, xlo, np.full(bias_width, -.2),
                   np.c_[plo, qlo, slo if estimate_source else np.empty((t,0))].ravel()]
    upper = np.r_[rhi, xhi, np.full(bias_width, .2),
                   np.c_[phi, qhi, shi if estimate_source else np.empty((t,0))].ravel()]
    if np.any(np.concatenate(bias0) < -.2) or np.any(np.concatenate(bias0) > .2):
        if model.fit_bias:
            raise ValueError("initial calibrated bias exceeds finite +/- .2 pu bias bounds")
    p0, q0 = observations.p - bias0[0], observations.q - bias0[1]
    if estimate_source:
        sp0 = (observations.source_p if observations.source_p is not None else
               observations.master_p - p0.sum(axis=1))
        sq0 = (observations.source_q if observations.source_q is not None else
               observations.master_q - q0.sum(axis=1))
        states = np.c_[p0, q0, sp0, sq0]
    else:
        states = np.c_[p0, q0]
    r0, x0 = np.clip(np.full(m,.012), rlo,rhi), np.clip(np.full(m,.006),xlo,xhi)
    if initial is not None:
        get = initial.get if isinstance(initial, Mapping) else lambda k,default=None: getattr(initial,k,default)
        r0, x0 = _array(get("r",r0),(m,),"initial.r"), _array(get("x",x0),(m,),"initial.x")
        for name, sl in (("p",slice(0,n)),("q",slice(n,2*n))):
            old = get(name)
            if old is not None:
                old = np.asarray(old,float)
                if old.ndim != 2 or old.shape[1] != n:
                    raise ValueError("warm-start terminal states have incompatible shape")
                kept = min(t,len(old))
                states[:kept,sl] = old[:kept]
        if estimate_source:
            for name,j in (("source_p",2*n),("source_q",2*n+1)):
                old = get(name)
                if old is not None:
                    kept = min(t,len(old))
                    states[:kept,j] = np.asarray(old,float)[:kept]
        if model.fit_bias:
            bias0 = [_array(get("bias_"+k,bias0[j]),(n,),"initial.bias_"+k)
                     for j,k in enumerate(("p","q","v"))]
    start = np.r_[r0,x0,np.concatenate(bias0) if model.fit_bias else np.empty(0),states.ravel()]
    margin = 1e-10 * np.maximum(upper-lower,1)
    start = np.clip(start,lower+margin,upper-margin)
    scale = np.r_[np.maximum((rhi-rlo)*.1,1e-4),np.maximum((xhi-xlo)*.1,1e-4),
                  np.full(bias_width,.001),
                  np.c_[sigmas["p"],sigmas["q"],
                        np.full((t,2),.01) if estimate_source else np.empty((t,0))].ravel()]
    offsets, row_start = {}, 0
    for name in channels:
        width = n if name in ("p","q","v") else 1
        offsets[name] = slice(row_start,row_start+width)
        row_start += width
    normalizer = sum(float(np.sum(np.log(2*np.pi*s**2))) for s in sigmas.values())
    normalizer += sum(float(np.sum(np.log(2*np.pi*s**2))) for _,_,s in calibration)
    cache = {}
    ac_evaluations = 0

    def evaluate(z):
        nonlocal ac_evaluations
        if "z" in cache and np.array_equal(cache["z"],z):
            return cache["residual"]
        state = z[shared:].reshape(t,state_width)
        bp,bq,bv = (z[2*m:shared].reshape(3,n) if model.fit_bias else bias0)
        physical = ac_predict(candidate,z[:m],z[m:2*m],state[:,:n],state[:,n:2*n],
                              state[:,2*n] if estimate_source else 0.,
                              state[:,2*n+1] if estimate_source else 0.,observations.root_v)
        ac_evaluations += 1
        means = dict(physical,p=physical["p"]+bp,q=physical["q"]+bq,
                     v=physical["v"]+bv,physical_v=physical["v"])
        blocks = []
        for name in channels:
            value = (means[name] - getattr(observations,name))/sigmas[name]
            blocks.append(value.reshape(t,-1))
        residual = np.concatenate(blocks,axis=1).ravel()
        if calibration:
            biases = {"p":bp,"q":bq,"v":bv}
            residual = np.r_[residual,*[((biases[k]-values)/s).ravel()
                                        for k,values,s in calibration]]
        cache.clear()
        cache.update(z=z.copy(),residual=residual,physical=physical,means=means,
                     biases=(bp.copy(),bq.copy(),bv.copy()))
        return residual

    def jacobian(z):
        evaluate(z)
        derivatives = _response_derivatives(candidate,cache["physical"],estimate_source)
        rows,cols,values = [],[],[]
        shared_e = np.arange(2*m)
        local_e = shared + np.arange(t)[:,None]*state_width + np.arange(state_width)
        for name in channels:
            sl = offsets[name]
            width = sl.stop-sl.start
            rr = np.arange(t)[:,None]*residual_width + np.arange(sl.start,sl.stop)
            ss = sigmas[name].reshape(t,width)
            if name in ("p","q"):
                j = np.arange(n)+(0 if name=="p" else n)
                rows.extend(rr.ravel());cols.extend(local_e[:,j].ravel());values.extend((1/ss).ravel())
            elif name in ("source_p","source_q"):
                if estimate_source:
                    j = 2*n+(0 if name=="source_p" else 1)
                    rows.extend(rr.ravel());cols.extend(local_e[:,[j]].ravel());values.extend((1/ss).ravel())
            else:
                d = derivatives[name].reshape(t,width,2*m+state_width)/ss[:,:,None]
                rows.extend(np.broadcast_to(rr[:,:,None],(t,width,2*m)).ravel())
                cols.extend(np.broadcast_to(shared_e,(t,width,2*m)).ravel())
                values.extend(d[:,:,:2*m].ravel())
                rows.extend(np.broadcast_to(rr[:,:,None],(t,width,state_width)).ravel())
                cols.extend(np.broadcast_to(local_e[:,None,:],(t,width,state_width)).ravel())
                values.extend(d[:,:,2*m:].ravel())
            if model.fit_bias and name in ("p","q","v"):
                block = ("p","q","v").index(name)
                rows.extend(rr.ravel())
                cols.extend(np.broadcast_to(2*m+block*n+np.arange(n),(t,n)).ravel())
                values.extend((1/ss).ravel())
        last = t*residual_width
        for block,(name,reading,sigma) in enumerate(calibration):
            rows.extend(np.arange(last,last+reading.size))
            cols.extend(np.tile(2*m+block*n+np.arange(n),len(reading)))
            values.extend((1/sigma).ravel())
            last += reading.size
        return csr_matrix((values,(rows,cols)),shape=(t*residual_width+calibration_size,count))

    rng = np.random.default_rng(seed)
    starts = [start]
    for j in range(1,int(multistart)):
        z = start.copy()
        # Impedance restarts change genuine physical coordinates, not states
        # by an implausibly large meter-noise perturbation.
        z[:2*m] *= np.exp(rng.normal(0,.5,2*m))
        z = np.clip(z,lower+margin,upper-margin)
        starts.append(z)
    restarts, retained = [], []
    initial_loss = None
    for index,z in enumerate(starts):
        try:
            loss0 = float(evaluate(z) @ evaluate(z))
            if initial_loss is None:
                initial_loss = loss0
            solver_jac=jacobian if linear_solver=="lsmr" else lambda point:jacobian(point).toarray()
            options={"atol":1e-10,"btol":1e-10,"maxiter":500} if linear_solver=="lsmr" else {}
            solution = least_squares(evaluate,z,jac=solver_jac,bounds=(lower,upper),
                                     x_scale=scale,ftol=tol,xtol=tol,gtol=tol,
                                     max_nfev=int(max_nfev),tr_solver=linear_solver,tr_options=options)
            residual = evaluate(solution.x)
            loss = float(residual@residual)
            physical = cache["physical"]
            vlo,vhi = voltage_bounds
            state_feasible = (np.min(abs(physical["voltage"])) >= vlo and
                              np.max(abs(physical["voltage"])) <= vhi)
            valid = bool(solution.success and state_feasible and np.isfinite(loss))
            record = dict(index=index,loss=loss,success=valid,solver_success=bool(solution.success),
                          status=int(solution.status),message=str(solution.message),
                          nfev=int(solution.nfev),optimality=float(solution.optimality),
                          physical_voltage_feasible=bool(state_feasible))
            restarts.append(record)
            retained.append((valid,loss,solution,dict(cache),record))
        except (RuntimeError,FloatingPointError,np.linalg.LinAlgError,ValueError) as error:
            restarts.append(dict(index=index,success=False,status=-99,message=str(error),loss=None))
    if not retained:
        return FitResult(candidate.name,np.full(m,np.nan),np.full(m,np.nan),p0,q0,
                         np.full(t,np.nan),np.full(t,np.nan),*bias0,np.inf,-np.inf,False,-99,
                         "all numerical restarts failed",0,np.inf,{},
                         dict(restarts=restarts,global_optimality_certified=False,
                              seconds=perf_counter()-began,measurement_channels=channels,
                              failure_kind="computation"))
    valid,loss,solution,best,record = min(retained,key=lambda item:(not item[0],item[1]))
    evaluate(solution.x)
    full_jac = jacobian(solution.x).toarray()
    # Column scaling gives a readable local diagnostic across mixed units.
    singular = np.linalg.svd(full_jac*scale[None,:],compute_uv=False)
    rank_tol = 1e-7*singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular>rank_tol))
    # Project local nuisance columns before diagnosing shared impedance/bias.
    qlocal,_ = np.linalg.qr(full_jac[:,shared:],mode="complete")
    projected = qlocal[:,t*state_width:].T @ (full_jac[:,:shared]*scale[None,:shared])
    shared_singular = np.linalg.svd(projected,compute_uv=False)
    shared_tol = 1e-7*shared_singular[0] if len(shared_singular) else 0.
    shared_rank = int(np.count_nonzero(shared_singular>shared_tol))
    z,states = solution.x,solution.x[shared:].reshape(t,state_width)
    physical = cache["physical"]
    finite_losses = [x["loss"] for x in restarts if x.get("loss") is not None]
    diagnostics = dict(
        objective="complete_known_diagonal_Gaussian",initial_loss=initial_loss,linear_solver=linear_solver,
        measurement_channels=channels,source_node=1,source_spatial_location_shared=True,
        source_anchored=bool(anchored),root_voltage_known=True,
        source_amplitude_ratio_fixed=False,source_sign="positive_consumption",
        ac_solution_branch="root_initialized_fixed_point; other equilibria not enumerated",
        impedance_units="pu",known_x_over_r=False,
        joint_parameter_count=count,residual_count=len(best["residual"]),
        local_jacobian_rank=rank,local_shared_rank=shared_rank,shared_parameter_count=shared,
        local_jacobian_rank_tolerance=float(rank_tol),shared_rank_tolerance=float(shared_tol),
        local_scaled_singular_values=singular.tolist(),
        local_projected_shared_singular_values=shared_singular.tolist(),
        local_identifiability_full_column_rank=bool(rank==count),
        rank_is_global_identifiability_certificate=False,
        active_bound_count=int(np.count_nonzero(solution.active_mask)),
        restart_loss_spread=float(max(finite_losses)-min(finite_losses)) if finite_losses else None,
        restarts=restarts,ac_evaluations=ac_evaluations,
        ac_equation_residual=physical["ac_equation_residual"],
        power_balance_residual=physical["power_balance_residual"],
        min_voltage=float(np.min(abs(physical["voltage"]))),
        max_voltage=float(np.max(abs(physical["voltage"]))),
        physical_voltage_feasible=record["physical_voltage_feasible"],
        global_optimality_certified=False,global_loss_lower_bound=None,
        loglikelihood_is_topology_probability=False,seconds=perf_counter()-began,
        failure_kind=None if valid else ("physical_domain" if not record["physical_voltage_feasible"] else "computation"),
        bias_policy="independent_calibration_joint" if model.fit_bias else "known_calibrated")
    return FitResult(candidate.name,z[:m].copy(),z[m:2*m].copy(),
                     states[:,:n].copy(),states[:,n:2*n].copy(),
                     states[:,2*n].copy() if estimate_source else np.zeros(t),
                     states[:,2*n+1].copy() if estimate_source else np.zeros(t),
                     *cache["biases"],loss,-.5*(loss+normalizer),valid,int(solution.status),
                     str(solution.message),int(solution.nfev),float(solution.optimality),
                     cache["means"],diagnostics)


__all__ = ["RadialCandidate","Observations","Calibration","MeasurementModel","FitResult",
           "ac_predict","fit_candidate","concat_observations"]
