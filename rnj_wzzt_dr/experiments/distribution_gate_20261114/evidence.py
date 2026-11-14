"""Frozen-native-bank selection bounds under a declared Gaussian error law.

Only historical observations and a planning forecast enter this module.  The
scoring matrices stay frozen.  Planning means use the existing historical
row-scale correction; its accuracy on the real network is not guaranteed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..absolute_identification_20261114.solver import SymmetricRecovery


METER_RELATIVE_SD = .0005
FORECAST_RELATIVE_SD = .05


def symmetric(a):
    return (a + np.swapaxes(a, -1, -2))/2


def gaussian_power_integral(A, power_sd):
    """E exp(-x'Ax), x~N(m,diag(sd^2)), as m'Qm+logdet/2.

    Returns Q, half logdet G, and the minimum eigenvalue of G.  A may be
    indefinite.  A failed positive-definiteness check raises LinAlgError;
    negative eigenvalues of A or Q are never discarded.
    """
    A, sd = symmetric(np.asarray(A, float)), np.asarray(power_sd, float)
    if A.ndim != 2 or sd.ndim != 2 or A.shape != (sd.shape[1],)*2:
        raise ValueError('Expected one symmetric matrix and time-by-channel SD')
    F_A_F = sd[:, :, None]*A[None]*sd[:, None, :]
    G = symmetric(np.eye(A.shape[0])[None]+2*F_A_F)
    chol = np.linalg.cholesky(G)
    AF = A[None]*sd[:, None, :]
    solved = np.linalg.solve(G, np.swapaxes(AF, 1, 2))
    Q = symmetric(A[None]-2*np.matmul(AF, solved))
    half_logdet = np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    minimum = float(np.linalg.eigvalsh(G).min())
    return Q, half_logdet, minimum


def quadratic_box_upper(q, a, c, lower, upper):
    """Valid (possibly loose) bound over the full componentwise action box."""
    center, radius = (upper+lower)/2, (upper-lower)/2
    at_center = np.einsum('hi,khij,hj->k', center, q, center, optimize=True)
    at_center += 2*np.einsum('khi,hi->k', a, center)+c
    linear = np.einsum('khij,hj->khi', q, center)+a
    return at_center+2*np.einsum('khi,hi->k', np.abs(linear), radius) + np.einsum(
        'khij,hi,hj->k', np.abs(q), radius, radius, optimize=True)


class DistributionEvidence:
    """Full-native different-topology gates and unchanged representative guards.

    q/a/c arrays represent sum_h u_h' q_h u_h + 2 a_h' u_h + c.
    ``direct`` is an audit of the evidence constraints, not of resource limits.
    Solver numerical buffers are deliberately outside these original formulas.
    """

    def __init__(self, case, score, validation, forecast, base_kw,
                 mode='mgf_forecast', target_mass=.99, lambda_start=.5):
        if mode not in ('mgf_forecast', 'mgf_fixed', 'normal_fixed'):
            raise ValueError(mode)
        if not np.isfinite(lambda_start) or lambda_start <= 0:
            raise ValueError('Positive finite starting lambda required')
        self.case, self.score, self.mode = case, score, mode
        self.base_kw, self.target_mass = float(base_kw), float(target_mass)
        self.forecast = np.asarray(forecast, float)
        self.T, self.N, self.R = case.H*case.repeat, case.N, case.R
        if (self.forecast.shape != (self.T, 2*self.N)
                or not np.isfinite(self.forecast).all()
                or not np.isfinite(base_kw) or base_kw <= 0):
            raise ValueError('Finite forecast and positive power base required')
        if validation.voltage is None or validation.root is None:
            raise ValueError('Historical validation voltage and root are required; no unit-voltage fallback')
        voltage, root = np.asarray(validation.voltage, float), np.asarray(validation.root, float)
        if (voltage.shape != validation.y.shape or root.shape != (len(voltage),)
                or not np.isfinite(voltage).all() or not np.isfinite(root).all()
                or np.any(voltage <= 0) or np.any(root <= 0)):
            raise ValueError('Finite positive historical voltage/root with aligned time required')
        if hasattr(case, 'labels') and tuple(case.labels) != tuple(validation.labels):
            raise ValueError('Validation terminal labels differ from resource labels')
        self.root_median, self.voltage_median = float(np.median(root)), float(np.median(voltage))
        self.root_variance = (2*METER_RELATIVE_SD*self.root_median**2)**2
        self.individual_variance = (2*METER_RELATIVE_SD*self.voltage_median**2)**2
        self.sigma_e = self.root_variance*np.ones((self.N, self.N))+self.individual_variance*np.eye(self.N)
        self.power_sd = (FORECAST_RELATIVE_SD*np.abs(self.forecast)
                         if mode == 'mgf_forecast' else np.zeros_like(self.forecast))
        self.sigma_p_diag = self.power_sd**2

        old = SymmetricRecovery(case, score, validation, self.forecast, base_kw,
                                mode='symmetric_guard', target_mass=target_mass)
        self.H, self.history, self.sigma2 = old.H, old.history, old.sigma2
        self.K, self.targets, self.weights, self.tie = old.K, old.targets, old.weights, old.tie
        self.guards, self.pairs = old.guards, old.pairs
        from ..flexibility_sweep_20261113.fitting import calibrated_means
        self.means = calibrated_means(score, validation)
        bank = score.get('native_bank')
        if not bank:
            raise ValueError('Complete frozen native_bank required')
        self.native_H = np.asarray([p['H'] for p in bank], float)
        self.native_history = np.asarray([p['history_sse'] for p in bank], float)
        self.native_topology = [p['topology_id'] for p in bank]
        if (self.native_H.shape != (len(bank), self.N, 2*self.N)
                or not np.isfinite(self.native_H).all()
                or not np.isfinite(self.native_history).all()
                or not np.isfinite(self.means).all()):
            raise ValueError('Native bank or planning mean has invalid dimensions/values')
        self.native_indices = [int(p['index']) for p in score['parameters']]
        rep_topologies = []
        for i, k in enumerate(self.native_indices):
            if not 0 <= k < len(bank):
                raise ValueError('Representative native index is outside bank')
            if (not np.array_equal(self.H[i], self.native_H[k])
                    or self.history[i] != self.native_history[k]
                    or score['models'][i]['topology_id'] != self.native_topology[k]):
                raise ValueError('Representative/native scoring point or topology mismatch')
            rep_topologies.append(self.native_topology[k])
        if len(set(rep_topologies)) != len(rep_topologies):
            raise ValueError('Representative topologies must be unique')

        # The full P/Q action map matters even when D has zero columns: E can
        # couple those columns to other nonzero differences in the mean term.
        self.control_map = np.asarray(case.response_effect(np.eye(2*self.N), base_kw), float)
        if self.control_map.shape != (2*self.N, case.R):
            raise ValueError('Resource response_effect must preserve arbitrary output rows')
        probe = np.arange(case.H*case.R, dtype=float).reshape(case.shape)/(case.H*case.R+1)
        mapped = np.repeat(probe@self.control_map.T, case.repeat, axis=0)
        if not np.allclose(mapped, case.shift_pq_pu(probe, base_kw), atol=1e-13, rtol=1e-12):
            raise ValueError('Resource shift and hourly action map disagree')

        self.rows, self.mean_matrices, self.variance_matrices = [], [], []
        self.effective_matrices, self.logdet_terms = [], []
        self.mu_q, self.mu_a, self.mu_c = [], [], []
        self.variance_q, self.variance_a, self.variance_c = [], [], []
        self.gate_q, self.gate_a, self.gate_c = [], [], []
        for i in self.targets:
            native_i = self.native_indices[i]
            competitors = [j for j, topology in enumerate(self.native_topology)
                           if topology != self.native_topology[native_i]]
            alpha = .1/len(competitors) if competitors else None
            for j in competitors:
                D, E = self.H[i]-self.native_H[j], self.means[i]-self.H[i]
                B = symmetric((D.T@D+E.T@D+D.T@E)/(2*self.sigma2))
                V = symmetric(D.T@self.sigma_e@D/self.sigma2**2)
                h = float((self.native_history[j]-self.history[i])/(2*self.sigma2))
                row = dict(target_rep=int(i), target_native=int(native_i), competitor_native=int(j),
                    target_topology_id=self.native_topology[native_i], competitor_topology_id=self.native_topology[j],
                    alpha=float(alpha), h=h, tie=self.tie, z=float(norm.isf(alpha)),
                    lambda_start=float(lambda_start), lambda_selected=None, lambda_halvings=0)
                self.mean_matrices.append(B); self.variance_matrices.append(V)
                mq, ma, mc = self.expand(np.broadcast_to(B, (self.T,)+B.shape))
                vq, va, vc = self.expand(np.broadcast_to(V, (self.T,)+V.shape))
                self.mu_q.append(mq); self.mu_a.append(ma); self.mu_c.append(mc+h-self.tie)
                self.variance_q.append(vq); self.variance_a.append(va); self.variance_c.append(vc)
                J0 = float(np.square(self.forecast@D.T).sum()/(2*self.sigma2))
                row.update(baseline_J=J0, baseline_mu=h-self.tie+mc, baseline_variance=vc,
                    baseline_variance_over_2J=None if J0 == 0 else vc/(2*J0),
                    baseline_mu_over_sd=None if vc == 0 else (h-self.tie+mc)/np.sqrt(vc))
                if mode != 'normal_fixed':
                    lam = float(lambda_start)
                    for halvings in range(80):
                        A = symmetric(lam*B-lam*lam*V/2)
                        try:
                            Q, logdet, mineig = gaussian_power_integral(A, self.power_sd)
                            break
                        except np.linalg.LinAlgError:
                            lam *= .5
                    else:
                        raise ArithmeticError('No valid fixed lambda after 80 halvings')
                    q, a, c = self.expand(Q)
                    c += float(logdet.sum())+lam*(h-self.tie)-np.log(1/alpha)
                    self.gate_q.append(q); self.gate_a.append(a); self.gate_c.append(c)
                    self.effective_matrices.append(Q); self.logdet_terms.append(float(logdet.sum()))
                    row.update(lambda_selected=lam, lambda_=lam, lambda_halvings=halvings,
                        gaussian_domain_positive_definite=True, minimum_G_eigenvalue=mineig,
                        effective_matrix_minimum_eigenvalue=float(np.linalg.eigvalsh(Q).min()),
                        effective_matrix_maximum_eigenvalue=float(np.linalg.eigvalsh(Q).max()))
                    row['lambda'] = lam
                self.rows.append(row)

        count = len(self.rows)
        for name in ('mu', 'variance', 'gate'):
            k = count if name != 'gate' or mode != 'normal_fixed' else 0
            for suffix, shape in [('q', (k, case.H, case.R, case.R)), ('a', (k, case.H, case.R)), ('c', (k,))]:
                key = name+'_'+suffix
                setattr(self, key, np.asarray(getattr(self, key), float).reshape(shape))
        self.mean_matrices = np.asarray(self.mean_matrices, float).reshape(count, 2*self.N, 2*self.N)
        self.variance_matrices = np.asarray(self.variance_matrices, float).reshape(count, 2*self.N, 2*self.N)
        self.alphas = np.asarray([r['alpha'] for r in self.rows])
        self.z = np.asarray([r['z'] for r in self.rows])
        self.variance_upper = np.maximum(0., quadratic_box_upper(self.variance_q, self.variance_a,
            self.variance_c, case.lower, case.upper))
        self.gate_upper = (quadratic_box_upper(self.gate_q, self.gate_a, self.gate_c,
            case.lower, case.upper) if mode != 'normal_fixed' else None)
        self.metadata = dict(mode=mode, target_mass=target_mass, targets=list(self.targets),
            target_native_indices=[self.native_indices[i] for i in self.targets],
            representative_count=self.K, native_count=len(bank), pair_count=count,
            guard_count=len(self.guards), score_sigma2=self.sigma2, meter_relative_sd=METER_RELATIVE_SD,
            root_voltage_median=self.root_median, terminal_voltage_median=self.voltage_median,
            root_squared_voltage_variance=self.root_variance, individual_squared_voltage_variance=self.individual_variance,
            voltage_error='Independent across times; shared root scalar plus independent terminal scalar Gaussians',
            power_error='Independent across times and P/Q channels; SD=0.05*abs(frozen forecast); zero in fixed modes',
            mean_source='Existing validation row-scale calibrated_means; frozen H used for all scores',
            constrained_bank='Every native point of different topology for each selected representative target',
            confidence_scope='Conditional on declared K mean and fixed Gaussian errors; not a true-network or posterior calibration guarantee',
            guard_scope='Original representative bidirectional mean-increment guards; no nominal absolute gate retained',
            diagnostic_moment_scope='mu and variance are conditional on power at the shifted forecast; MGF modes integrate power uncertainty separately',
            log_bound_serialization='Stable normal log-CDF; a nonfinite log bound is JSON null with its explicit row/reason record',
            lambda_rule='Start .5 (or explicit lambda_start); halve only for non-positive-definite Gaussian domain',
            box_bound_scope='Relaxed componentwise original resource box; energy/time coupling omitted; upper<0 proves gate infeasible',
            gate_box_upper=self.gate_upper, variance_box_upper=self.variance_upper)

    def expand(self, matrices):
        matrices = np.asarray(matrices, float)
        F = self.control_map
        q_t = np.einsum('ci,tcd,dj->tij', F, matrices, F, optimize=True)
        a_t = np.einsum('ci,tcd,td->ti', F, matrices, self.forecast, optimize=True)
        c = float(np.einsum('tc,tcd,td->', self.forecast, matrices, self.forecast, optimize=True))
        return symmetric(q_t.reshape(self.case.H, self.case.repeat, self.R, self.R).sum(1)), a_t.reshape(self.case.H, self.case.repeat, self.R).sum(1), c

    def direct(self, u):
        u = self.case.array(u)
        power = self.forecast+self.case.shift_pq_pu(u, self.base_kw)
        mean = np.einsum('tc,kcd,td->k', power, self.mean_matrices, power, optimize=True)
        mu = mean+np.asarray([r['h']-r['tie'] for r in self.rows])
        variance = np.einsum('tc,kcd,td->k', power, self.variance_matrices, power, optimize=True)
        if np.any(variance < -1e-8):
            raise ArithmeticError('Negative Gaussian variance beyond rounding tolerance')
        variance = np.maximum(variance, 0.)
        if self.mode == 'normal_fixed':
            margins = mu-self.z*np.sqrt(variance)
            standardized = np.divide(-mu, np.sqrt(variance),
                out=np.full_like(mu, -np.inf), where=variance > 0)
            pair_failure = norm.cdf(standardized)
            pair_failure[(variance == 0)&(mu <= 0)] = 1.
            log_failure = norm.logcdf(standardized)
            log_failure[(variance == 0)&(mu <= 0)] = 0.
        else:
            T = np.asarray([float(np.einsum('tc,tcd,td->', power, Q, power, optimize=True))+c
                            for Q, c in zip(self.effective_matrices, self.logdet_terms)])
            log_failure = np.asarray([r['lambda_selected']*(r['tie']-r['h']) for r in self.rows])-T
            margins = np.log(self.alphas)-log_failure
            pair_failure = np.exp(np.minimum(0., log_failure))
        gains = []
        for i, j, K, Q, a in self.guards:
            D, E = self.H[i]-self.H[j], K-self.H[i]
            def moment(p):
                d, e = p@D.T, p@E.T
                return float((d*d+2*e*d).sum()/(2*self.sigma2))
            gains.append(moment(power)-moment(self.forecast))
        sums = {str(i): 0. for i in self.targets}
        for row, failure in zip(self.rows, pair_failure):
            sums[str(row['target_rep'])] += float(failure)
        feasible = bool(np.all(margins >= -1e-7) and np.all(pair_failure <= self.alphas+1e-9)
                        and all(g >= -1e-7 for g in gains))
        log_records = [dict(row=int(k), value=str(value),
            reason='zero_variance_deterministic_success' if variance[k] == 0 and mu[k] > 0
            else 'floating_point_log_tail_limit') for k, value in enumerate(log_failure) if not np.isfinite(value)]
        return dict(feasible=feasible, margins=margins, minimum_pair_margin=float(margins.min()) if len(margins) else None,
            mu=mu, variance=variance, pair_failure=pair_failure, pair_failure_bound=pair_failure,
            mean_log_score_minus_tie_under_declared_law=mu+np.einsum('kcc,tc->k', self.mean_matrices, self.sigma_p_diag),
            log_pair_failure_bound=[float(v) if np.isfinite(v) else None for v in log_failure],
            log_pair_failure_nonfinite=log_records,
            pair_failure_numerical_underflow=[int(k) for k, (p, logp) in enumerate(zip(pair_failure, log_failure))
                                               if p == 0 and np.isfinite(logp)],
            target_failure_sums=sums,
            worst_recovery_lower=max(0., 1-max(sums.values(), default=0.)), guard_gains=gains,
            minimum_guard_gain=min(gains) if gains else None)

    def verify_expansion(self, u):
        u = self.case.array(u)
        info = self.direct(u)
        errors = {}
        for name, actual in [('mu', info['mu']), ('variance', info['variance'])]:
            q, a, c = [getattr(self, name+'_'+suffix) for suffix in ('q', 'a', 'c')]
            expanded = np.einsum('hi,khij,hj->k', u, q, u)+2*np.einsum('khi,hi->k', a, u)+c
            errors[name] = float(np.max(np.abs(expanded-actual), initial=0.))
            if not np.allclose(expanded, actual, atol=1e-8, rtol=1e-8):
                raise ArithmeticError(name+' quadratic expansion differs from direct full prediction')
        if self.mode != 'normal_fixed':
            expanded = np.einsum('hi,khij,hj->k', u, self.gate_q, u)+2*np.einsum('khi,hi->k', self.gate_a, u)+self.gate_c
            errors['gate'] = float(np.max(np.abs(expanded-info['margins']), initial=0.))
            if not np.allclose(expanded, info['margins'], atol=1e-8, rtol=1e-8):
                raise ArithmeticError('MGF gate expansion differs from full Gaussian integral')
        for gain, (_, _, _, Q, a) in zip(info['guard_gains'], self.guards):
            expanded = np.einsum('hi,ij,hj', u, Q, u)+2*np.sum(a*u)
            if not np.isclose(expanded, gain, atol=1e-8, rtol=1e-8):
                raise ArithmeticError('Existing directional guard expansion differs')
        info['expansion_max_absolute_errors'] = errors
        return info
