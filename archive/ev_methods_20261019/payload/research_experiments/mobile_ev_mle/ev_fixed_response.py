"""Fixed-line, unwhitened V/root-P/root-Q responses and local input derivatives.

No fitting, control selection, or parameter updates occur here. Inputs are total
terminal P/Q in per unit; an EV command must already be included exactly once.
Input columns are terminal P, terminal Q, source P, source Q. The source enters
physical node 1 after the full-support root edge, with positive demand sign.

``linear`` means lossless branch flow (linear squared voltage), ``i2`` uses
lossless subtree flow squared divided by known root voltage squared, and
``anchored_i2`` adds the I2 remainder to the exact AC value and Jacobian at its
fixed anchor. The last two keep complete downstream losses in voltage/root P/Q.
"""
from __future__ import annotations

import numpy as np

from . import likelihood as lk


METHODS = ("linear", "i2", "anchored_i2", "ac")


def _inputs(candidate, p, q, source_p, source_q, root_v):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != candidate.n or not len(p):
        raise ValueError("p must be a nonempty time-by-terminal array")
    if not np.isfinite(p).all():
        raise ValueError("p must be finite")
    q = lk._array(q, p.shape, "q")
    t = len(p)
    sp = lk._array(source_p, (t,), "source_p")
    sq = lk._array(source_q, (t,), "source_q")
    root = lk._array(root_v, (t,), "root_v")
    if np.any(root <= 0):
        raise ValueError("known root voltage must be positive")
    return tuple(np.array(a, copy=True) for a in (p, q, sp, sq, root))


def _lines(candidate, r, x):
    r, x = lk._array(r, (candidate.m,), "r"), lk._array(x, (candidate.m,), "x")
    if np.any(r < 0) or np.any(x < 0):
        raise ValueError("fixed R/X must be nonnegative")
    r, x = r.copy(), x.copy()
    r.setflags(write=False); x.setflags(write=False)
    return r, x


def _loads(candidate, inputs):
    p, q, sp, sq, _ = inputs
    demand = np.zeros((len(p), candidate.m), dtype=complex)
    demand[:, 0] = sp + 1j*sq
    demand[:, list(candidate.terminal_edges)] += p + 1j*q
    return demand


def _loss_coefficients(candidate, z):
    # H[e,g] multiplies ell_g in voltage drop on e. B[g,e] means g is below e.
    h = 2*np.real(np.conj(z[:, None])*z[None, :])*np.asarray(candidate.B).T
    h[np.diag_indices(candidate.m)] -= np.abs(z)**2
    return h


def _readonly(a):
    value = np.array(a, copy=True)
    value.setflags(write=False)
    return value


class FixedResponse:
    """One immutable candidate/line snapshot; responses are unwhitened and in pu.

    ``evaluate(..., derivatives=False)`` returns ``jac=None``. ``ac`` caches its
    latest identical inputs, so a value request followed by a Jacobian request
    needs only one forward AC call. Counter values include anchor construction.
    Root voltage is given, not differentiated. An anchored evaluation requires
    exactly the anchor's batch shape and root-voltage path.
    """
    def __init__(self, candidate, r, x, *, method="ac", anchor=None):
        if method not in METHODS:
            raise ValueError("unknown response method "+repr(method))
        self.candidate, self.method = candidate, method
        self.r, self.x = _lines(candidate, r, x)
        self.z = _readonly(self.r+1j*self.x)
        self.B = _readonly(candidate.B)
        self.path = _readonly(candidate.M)
        self.H = _readonly(_loss_coefficients(candidate, self.z))
        n, m = candidate.n, candidate.m
        direct = np.zeros((m, 2*n+2), complex)
        direct[list(candidate.terminal_edges), np.arange(n)] = 1.
        direct[list(candidate.terminal_edges), n+np.arange(n)] = 1j
        direct[0, 2*n:] += (1., 1j)
        self.flow_jac = _readonly(self.B.T@direct)
        self.total_jac = _readonly(direct.sum(axis=0))
        self.linear_u_jac = _readonly(-2*np.real(np.conj(self.z)[:, None]*self.flow_jac))
        self.counter = dict(ac_value_calls=0, ac_jacobian_calls=0, evaluate_calls=0,
                            cache_hits=0, approximate_value_calls=0, approximate_jacobian_calls=0)
        self._cache = None
        self.anchor = None if anchor is None else tuple(_readonly(a) for a in _inputs(candidate, *anchor))
        if method == "anchored_i2":
            if self.anchor is None:
                raise ValueError("anchored_i2 requires (p,q,source_p,source_q,root_v)")
            exact = self._ac(self.anchor, True)
            approximate = self._approximate(self.anchor, True, loss=True)
            self._anchor_value_correction = _readonly(exact["values"]-approximate["values"])
            self._anchor_jac_correction = _readonly(exact["jac"]-approximate["jac"])
            self._anchor_state = _readonly(self._state(self.anchor))

    @property
    def counters(self):
        return dict(self.counter)

    @staticmethod
    def _state(inputs):
        p, q, sp, sq, _ = inputs
        return np.column_stack((p, q, sp, sq))

    def _approximate(self, inputs, derivatives, *, loss):
        self.counter["approximate_value_calls"] += 1
        demand = _loads(self.candidate, inputs)
        root = inputs[-1]
        a = demand@self.B
        ell = np.abs(a)**2/root[:, None]**2 if loss else np.zeros_like(a.real)
        edge_drop = 2*np.real(np.conj(self.z)[None, :]*a)+ell@self.H.T
        squared = root[:, None]**2-edge_drop@self.B.T
        if not np.isfinite(squared).all() or np.any(squared <= 0):
            raise ValueError("approximate non-slack squared voltage must be positive")
        voltage = np.sqrt(squared)
        ids = list(self.candidate.terminal_edges)
        master = demand.sum(axis=1)+ell@self.z
        values = np.column_stack((voltage[:, ids], master.real, master.imag))
        jac = None
        if derivatives:
            self.counter["approximate_jacobian_calls"] += 1
            dell = (2*np.real(np.conj(a)[:, :, None]*self.flow_jac[None, :, :])/
                    root[:, None, None]**2 if loss else
                    np.zeros((len(a), self.candidate.m, 2*self.candidate.n+2)))
            du = self.linear_u_jac[None, :, :]-np.einsum("eg,tgk->tek", self.H, dell)
            node_du = np.einsum("je,tek->tjk", self.B, du)
            terminal_jac = node_du[:, ids]/(2*voltage[:, ids, None])
            master_jac = self.total_jac[None, :]+np.einsum("e,tek->tk", self.z, dell)
            jac = np.concatenate((terminal_jac, master_jac.real[:, None, :],
                                  master_jac.imag[:, None, :]), axis=1)
        return dict(values=values, jac=jac, all_voltage=voltage,
                    current_squared=ell, squared_voltage=squared)

    def _ac(self, inputs, derivatives):
        cached = self._cache
        same = cached is not None and all(np.array_equal(a, b) for a, b in zip(inputs, cached["inputs"]))
        if same:
            self.counter["cache_hits"] += 1
        else:
            self.counter["ac_value_calls"] += 1
            physical = lk.ac_predict(self.candidate, self.r, self.x, *inputs)
            cached = dict(inputs=tuple(a.copy() for a in inputs), physical=physical, jac=None,
                values=np.column_stack((physical["v"], physical["master_p"], physical["master_q"])))
            self._cache = cached
        if derivatives and cached["jac"] is None:
            self.counter["ac_jacobian_calls"] += 1
            derivative = lk._response_derivatives(self.candidate, cached["physical"], True)
            offset = 2*self.candidate.m
            cached["jac"] = np.concatenate((derivative["v"][:, :, offset:],
                derivative["master_p"][:, None, offset:], derivative["master_q"][:, None, offset:]), axis=1)
        return dict(values=cached["values"].copy(), jac=cached["jac"].copy() if derivatives else None,
                    all_voltage=np.abs(cached["physical"]["voltage"]), physical=cached["physical"])

    def evaluate(self, p, q, source_p, source_q, root_v, *, derivatives=True):
        if not isinstance(derivatives, (bool, np.bool_)):
            raise TypeError("derivatives must be boolean")
        inputs = _inputs(self.candidate, p, q, source_p, source_q, root_v)
        self.counter["evaluate_calls"] += 1
        if self.method == "ac":
            result = self._ac(inputs, derivatives)
        else:
            if self.method == "anchored_i2" and (inputs[0].shape != self.anchor[0].shape or
                                                  not np.array_equal(inputs[-1], self.anchor[-1])):
                raise ValueError("anchored_i2 requires the anchor batch shape and known root-voltage path")
            result = self._approximate(inputs, derivatives, loss=self.method != "linear")
            if self.method == "anchored_i2":
                delta = self._state(inputs)-self._anchor_state
                result["values"] += self._anchor_value_correction+np.einsum("tjk,tk->tj", self._anchor_jac_correction, delta)
                if derivatives:
                    result["jac"] += self._anchor_jac_correction
                # Only measured voltage is corrected by the output anchor formula.
                result["all_voltage"] = result["values"][:, :self.candidate.n]
                if not np.isfinite(result["all_voltage"]).all() or np.any(result["all_voltage"] <= 0):
                    raise ValueError("anchored terminal voltage must be finite and positive")
        all_voltage = result["all_voltage"]
        diagnostics = dict(min_voltage=float(np.min(all_voltage)), max_voltage=float(np.max(all_voltage)),
            voltage_scope="terminal_nodes" if self.method == "anchored_i2" else "all_non_slack_nodes",
            root_voltage_min=float(np.min(inputs[-1])), root_voltage_max=float(np.max(inputs[-1])),
            source_interface=1, line_parameters_fixed=True, PQ_inputs_total=True,
            EV_added_by_response=False, physical_acceptance_certified=False,
            counters=self.counters, anchor_value_and_jacobian_exact=self.method == "anchored_i2")
        if self.method == "ac":
            diagnostics.update(ac_equation_residual=result["physical"]["ac_equation_residual"],
                               power_balance_residual=result["physical"]["power_balance_residual"])
        return dict(values=result["values"], jac=result["jac"], method=self.method, diagnostics=diagnostics)


def exact_current_reconstruction(candidate, r, x, p, q, source_p, source_q, root_v, *, physical=None):
    """Reconstruct branch flows and every squared voltage using exact AC ell.

    ``physical`` may be a saved same-input ``lk.ac_predict`` result; then this
    function makes no AC call. Residuals audit sending/receiving definitions and
    full downstream loss identity, rather than certifying an approximate model.
    """
    inputs = _inputs(candidate, p, q, source_p, source_q, root_v)
    r, x = _lines(candidate, r, x); z = r+1j*x
    ac_calls = int(physical is None)
    if physical is None:
        physical = lk.ac_predict(candidate, r, x, *inputs)
    for name, value in zip(("p", "q", "source_p", "source_q", "root_v"), inputs):
        if name not in physical or not np.array_equal(np.asarray(physical[name]), value):
            raise ValueError("provided physical state has different "+name)
    voltage = np.asarray(physical["voltage"])
    current = np.asarray(physical["branch_current"])
    if voltage.shape != (len(inputs[0]), candidate.m) or current.shape != voltage.shape:
        raise ValueError("provided AC voltage/current shape mismatch")
    if not np.isfinite(voltage).all() or not np.isfinite(current).all():
        raise ValueError("provided AC voltage/current must be finite")
    root = inputs[-1]
    parent_voltage = np.column_stack([root if parent < 0 else voltage[:, parent]
                                     for parent in candidate.parents])
    line_error = np.max(np.abs(parent_voltage-voltage-current*z[None, :]))
    if line_error > 2e-9:
        raise ValueError("provided AC state differs from fixed R/X or tree")
    ell = np.abs(current)**2
    demand = _loads(candidate, inputs); a = demand@candidate.B
    sending = parent_voltage*np.conj(current)
    receiving = voltage*np.conj(current)
    reconstructed_sending = a+(ell*z[None, :])@candidate.B
    eta = ell@_loss_coefficients(candidate, z).T
    reconstructed_u = root[:, None]**2-(2*np.real(np.conj(z)[None, :]*a)+eta)@candidate.B.T
    u = np.abs(voltage)**2
    master = demand.sum(axis=1)+ell@z
    def error(a, b):
        return float(np.max(np.abs(a-b)))
    residuals = dict(sending_subtree=error(sending, reconstructed_sending),
        sending_receiving_loss=error(sending, receiving+z[None, :]*ell),
        voltage_reconstruction=error(u, reconstructed_u),
        current_sending=error(np.abs(parent_voltage)**2*ell, np.abs(sending)**2),
        current_receiving=error(u*ell, np.abs(receiving)**2),
        root_power_balance=error(master, physical["master_p"]+1j*np.asarray(physical["master_q"])))
    return dict(current_squared=ell, sending_power=sending, receiving_power=receiving,
        subtree_no_loss=a, reconstructed_sending_power=reconstructed_sending,
        squared_voltage=u, reconstructed_squared_voltage=reconstructed_u,
        eta=eta, reconstructed_root_power=master, residuals=residuals, ac_value_calls=ac_calls,
        source_interface=1, line_parameters_fixed=True)


__all__ = ["METHODS", "FixedResponse", "exact_current_reconstruction"]
