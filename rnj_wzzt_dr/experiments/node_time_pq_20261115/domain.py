"""Linear node/hour action-band wrapper; physical execution stays unchanged."""
from __future__ import annotations

from copy import deepcopy
from math import fsum

import numpy as np

from .bandwidth import array_hash


BAND_RELATIVE_BUFFER = 1e-7


class NodeTimePQCase:
    """Intersect a physical pure-P/pure-Q contract with training-only bands.

    The public A_ub/b_ub include both domains for relaxation builders.  attach
    delegates the unchanged physical case once and then adds only band rows.
    Actual delivery uses the original physical case; its band violation is a
    separately reported delivery diagnostic, not a device-contract violation.
    """

    def __init__(self, physical, arrays, metadata):
        self.physical = physical
        self.shape, self.dimension = tuple(physical.shape), int(physical.dimension)
        self.H, self.R, self.N = physical.H, physical.R, physical.N
        if self.H != 24 or self.shape != (24, self.R) or self.dimension != 24*self.R:
            raise ValueError('Node/hour band expects the original 24 hourly controls')
        self.bandwidth_metadata = deepcopy(metadata)
        self.labels = tuple(physical.labels)
        if tuple(np.asarray(arrays['labels']).tolist()) != self.labels:
            raise ValueError('Band terminal order differs from the physical case')
        self.kappa = np.asarray(arrays['kappa'], float).copy()
        self.band_kvar = np.asarray(arrays['band_kvar'], float).copy()
        self.decision_hours = np.asarray(arrays['decision_hours'], int).copy()
        if (self.kappa.shape != (self.N,) or self.band_kvar.shape != (24, self.N)
                or not np.isfinite(self.kappa).all() or not np.isfinite(self.band_kvar).all()
                or np.any(self.band_kvar < 0)):
            raise ValueError('Finite node slopes and nonnegative 24-by-node bands required')
        clock = np.asarray(physical.clock, float)
        if (clock.shape != (24,) or not np.isfinite(clock).all()
                or self.decision_hours.shape != (24,)
                or len(set(self.decision_hours.tolist())) != 24
                or not np.array_equal(np.floor(clock % 24).astype(int), self.decision_hours)):
            raise ValueError('Band local-hour order differs from physical decision clock')
        if metadata.get('decision_hours') != self.decision_hours.tolist():
            raise ValueError('Bandwidth metadata clock does not match its array')
        expected_hashes = metadata.get('array_content_sha256', {})
        for key, value in arrays.items():
            if key not in expected_hashes or array_hash(value,
                    integer=key in ('labels', 'counts', 'decision_hours')) != expected_hashes[key]:
                raise ValueError(f'Bandwidth array content hash mismatch: {key}')
        if physical.P_R+physical.Q_R != self.R or len(physical.resources) != self.R:
            raise ValueError('Expected original P-resource prefix followed by pure-Q resources')
        # Independent resource catalog map for direct audit; never infer Q from a
        # q/p ratio, and aggregate multiple contracts at the same terminal.
        self._node = np.asarray([self.labels.index(int(row['node_label']))
            for row in physical.resources], dtype=int)
        expected_B = np.zeros((2*self.N, self.R))
        for resource, node in enumerate(self._node):
            expected_B[node if resource < physical.P_R else self.N+node, resource] = 1.
        B = np.asarray(physical.B, float)
        if B.shape != expected_B.shape or not np.array_equal(B, expected_B):
            raise ValueError('Physical B must match declared pure-P and pure-Q resource catalog')
        C = B[self.N:, :]-self.kappa[:, None]*B[:self.N, :]
        positive = np.kron(np.eye(24), C)
        self.band_A_ub = np.concatenate((positive, -positive), axis=0)
        self.band_b_ub = np.tile(self.band_kvar.ravel(order='C'), 2)
        scale = np.where(self.band_b_ub > 0, self.band_b_ub, 1.)
        self.encoded_band_A_ub = self.band_A_ub/scale[:, None]
        self.encoded_band_b_ub = np.where(self.band_b_ub > 0, 1.-BAND_RELATIVE_BUFFER, 0.)
        self._positive_encoded_rows = self.band_b_ub > 0
        zero = self.band_kvar.ravel(order='C') == 0
        self.encoded_band_A_eq = positive[zero].copy()
        self.encoded_band_b_eq = np.zeros(np.count_nonzero(zero))
        original_A = np.asarray(physical.A_ub, float)
        original_b = np.asarray(physical.b_ub, float)
        if original_A.shape != (len(original_b), self.dimension):
            raise ValueError('Physical inequality matrix dimension mismatch')
        self.A_ub = np.concatenate((original_A, self.band_A_ub), axis=0)
        self.b_ub = np.concatenate((original_b, self.band_b_ub))
        # The exposed encoded inequalities retain both zero-band signs, so a
        # relaxation using only this interface still enforces every zero band.
        self.encoded_A_ub = np.concatenate((original_A, self.encoded_band_A_ub), axis=0)
        self.encoded_b_ub = np.concatenate((original_b, self.encoded_band_b_ub))
        self.encoded_A_eq = np.concatenate((np.asarray(physical.A_eq, float), self.encoded_band_A_eq), axis=0)
        self.encoded_b_eq = np.concatenate((np.asarray(physical.b_eq, float), self.encoded_band_b_eq))
        self.band_encoding = dict(relative_buffer=BAND_RELATIVE_BUFFER,
            positive_band='abs(departure)/original_band <= 1-relative_buffer',
            zero_band='departure == 0, without a bandwidth floor',
            cost_bound_scope='Activation cost for the encoded QCQP with positive bands reduced by the declared numerical buffer; not a lower bound for the unbuffered original band domain')
        for value in (self.kappa, self.band_kvar, self.decision_hours,
                      self.band_A_ub, self.band_b_ub, self.A_ub, self.b_ub,
                      self.encoded_band_A_ub, self.encoded_band_b_ub,
                      self.encoded_band_A_eq, self.encoded_band_b_eq,
                      self.encoded_A_ub, self.encoded_b_ub,
                      self.encoded_A_eq, self.encoded_b_eq):
            value.setflags(write=False)

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def band_audit(self, u):
        """Sum contracts by node directly, independent of attached matrix rows."""
        action = np.asarray(self.physical.array(u), float)
        p, q = np.zeros((24, self.N)), np.zeros((24, self.N))
        for hour in range(24):
            for node in range(self.N):
                p[hour, node] = fsum(float(action[hour, resource]) for resource in range(self.P_R)
                    if self._node[resource] == node)
                q[hour, node] = fsum(float(action[hour, resource]) for resource in range(self.P_R, self.R)
                    if self._node[resource] == node)
        deviation = q-p*self.kappa[None, :]
        excess = np.abs(deviation)-self.band_kvar
        # Only floating-point summation/multiplication roundoff is allowed here.
        # No physical-domain tolerance, artificial band floor or hidden buffer.
        magnitude = np.abs(q)+np.abs(p*self.kappa[None, :])+self.band_kvar
        roundoff = 64*np.finfo(float).eps*magnitude
        encoded_band = self.band_kvar*(1.-BAND_RELATIVE_BUFFER)
        encoded_excess = np.abs(deviation)-encoded_band
        return dict(passed=bool(np.all(excess <= roundoff)),
            encoded_passed=bool(np.all(encoded_excess <= roundoff)),
            encoded_max_violation_kvar=float(np.maximum(encoded_excess, 0.).max(initial=0.)),
            relative_model_buffer=BAND_RELATIVE_BUFFER,
            max_violation_kvar=float(np.maximum(excess, 0.).max(initial=0.)),
            minimum_slack_kvar=float((-excess).min(initial=np.inf)),
            maximum_roundoff_tolerance_kvar=float(roundoff.max(initial=0.)),
            delta_p_kw=p.tolist(), delta_q_kvar=q.tolist(),
            deviation_kvar=deviation.tolist(), slack_kvar=(-excess).tolist(),
            violation_kvar=np.maximum(excess, 0.).tolist(),
            passed_cells=(excess <= roundoff).tolist(),
            criterion='abs(delta_q-kappa*delta_p) <= frozen node/hour band plus arithmetic roundoff only',
            fit_receipt_sha256=self.bandwidth_metadata.get('fit_receipt_sha256'))

    def audit(self, u, tolerance=1e-7):
        result = dict(self.physical.audit(u, tolerance=tolerance))
        physical_passed = bool(result['passed'])
        band = self.band_audit(u)
        result.update(passed=bool(physical_passed and band['passed']),
            physical_passed=physical_passed, node_time_band=band,
            feasibility_scope='Original physical contracts AND frozen training node/hour linear P/Q bands; actual delivery uses the physical case')
        return result

    def attach(self, model, name=None):
        u, fee = self.physical.attach(model) if name is None else self.physical.attach(model, name=name)
        flat = u.reshape(self.dimension)
        if np.any(self._positive_encoded_rows):
            model.addConstr(self.encoded_band_A_ub[self._positive_encoded_rows]@flat
                <= self.encoded_band_b_ub[self._positive_encoded_rows], name='training_node_time_pq_positive_band')
        if len(self.encoded_band_b_eq):
            model.addConstr(self.encoded_band_A_eq@flat == self.encoded_band_b_eq,
                name='training_node_time_pq_zero_band')
        return u, fee
