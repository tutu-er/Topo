"""Fixed-covariance Gaussian quadratic loss difference, with no solver dependency.

Let w = mean + control @ u + cov_factor @ xi, xi ~ N(0, I), and
L = w.T @ A @ w + 2*a.T @ w + c. All parameters except u are frozen.
``build_quadratic_gate`` expands -log E[exp(-lambda_*L)] into a quadratic
T(u) = u.T @ Q @ u + q.T @ u + const. T(u) >= h is sufficient for
Pr(L > delta) >= beta under this declared Gaussian model. It is not an exact
probability or a guarantee about which topology is true. A may be indefinite.
The covariance factor can be rectangular, including zero columns.

For a general reference voltage, stack w=(P,e), form the two frozen residual
maps r_k=(Hbar-Hk)P+b+e, and subtract their squared norms to obtain A,a,c.
This module does not fit Hbar, choose candidates, or modify a device model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _finite(value, name, shape=None):
    value = np.array(value, dtype=float, copy=True)
    if not np.isfinite(value).all() or (shape is not None and value.shape != shape):
        raise ValueError(f"Invalid {name}: shape {value.shape}, expected {shape}")
    return value


@dataclass
class QuadraticGate:
    """Q/q/const/h are optimizer inputs; evaluate and bound take control u."""
    Q: np.ndarray
    q: np.ndarray
    const: float
    h: float
    lambda_: float
    delta: float
    beta: float
    A: np.ndarray
    a: np.ndarray
    c: float
    mean: np.ndarray
    control: np.ndarray
    cov_factor: np.ndarray
    cholesky: np.ndarray
    logdet_J: float

    def _u(self, u):
        return _finite(u, "control vector u", (len(self.q),))

    def evaluate(self, u):
        """Return the expanded deterministic score T(u), not a probability."""
        u = self._u(u)
        return float(u @ self.Q @ u + self.q @ u + self.const)

    def log_mgf(self, u):
        """Exact log E exp(-lambda_*L), computed before polynomial expansion."""
        m = self.mean + self.control @ self._u(u)
        v = self.cov_factor.T @ (self.A @ m + self.a)
        z = np.linalg.solve(self.cholesky, v)
        return float(-self.lambda_*(m @ self.A @ m + 2*self.a @ m + self.c)
                     + 2*self.lambda_**2*(z @ z) - .5*self.logdet_J)

    def bound(self, u):
        """Exponential failure upper bound, capped at 1 for probability output."""
        score = -self.log_mgf(u)
        log_failure = self.lambda_*self.delta-score
        return dict(score=score, h=self.h, margin=score-self.h,
            log_failure_upper=log_failure,
            failure_upper=math.exp(min(0., log_failure)),
            success_lower=-math.expm1(min(0., log_failure)),
            sufficient=bool(log_failure <= math.log1p(-self.beta)))


def build_quadratic_gate(A, a, c, mean, control, cov_factor, delta, beta, lambda_):
    """Build a Gaussian quadratic gate; reject a non-positive integration matrix.

    Shapes: A(n,n), a(n,), mean(n,), control(n,k), cov_factor(n,r).
    ``u`` has k coordinates; r independent standard normal factors generate
    the fixed covariance cov_factor @ cov_factor.T. Positive definiteness of
    J=I+2*lambda_*cov_factor.T@A@cov_factor is required, without eigenvalue
    clipping. h=lambda_*delta+log(1/(1-beta)). No covariance is estimated here.
    """
    mean = _finite(mean, "mean")
    if mean.ndim != 1 or not len(mean):
        raise ValueError("mean must be a nonempty vector")
    n = len(mean)
    A, a = _finite(A, "A", (n, n)), _finite(a, "a", (n,))
    if not np.allclose(A, A.T, rtol=1e-12, atol=1e-12):
        raise ValueError("A must be symmetric")
    A = (A+A.T)/2
    B, U = _finite(control, "control"), _finite(cov_factor, "cov_factor")
    if B.ndim != 2 or U.ndim != 2 or B.shape[0] != n or U.shape[0] != n:
        raise ValueError("control and cov_factor need shape (n,k) and (n,r)")
    c, delta, beta, lam = map(float, (c, delta, beta, lambda_))
    if not np.isfinite([c, delta, beta, lam]).all() or not 0 < beta < 1 or lam <= 0:
        raise ValueError("Finite c/delta, 0<beta<1 and lambda_>0 are required")
    J = np.eye(U.shape[1]) + 2*lam*(U.T @ A @ U)
    J = (J+J.T)/2
    try:
        chol = np.linalg.cholesky(J)
    except np.linalg.LinAlgError as exc:
        raise ValueError("I+2*lambda_*U.T@A@U is not positive definite") from exc
    v0, D = U.T @ (A @ mean+a), U.T @ A @ B
    v0, D = np.linalg.solve(chol, v0), np.linalg.solve(chol, D)
    logdet = float(2*np.log(np.diag(chol)).sum())
    Q = lam*(B.T @ A @ B) - 2*lam**2*(D.T @ D)
    q = 2*lam*(B.T @ (A @ mean+a)) - 4*lam**2*(D.T @ v0)
    const = float(lam*(mean @ A @ mean+2*a @ mean+c)
                  - 2*lam**2*(v0 @ v0) + .5*logdet)
    h = float(lam*delta-math.log1p(-beta))
    return QuadraticGate((Q+Q.T)/2, q, const, h, lam, delta, beta,
                         A, a, c, mean, B, U, chol, logdet)


def _sanity():
    """Deterministic scalar integration, old-reference reduction, and rejection."""
    gate = build_quadratic_gate([[.8]], [-.2], .4, [1.1], [[.7]], [[.6]], .3, .9, .3)
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    for u in (-1., 0., .8):
        w = 1.1+.7*u+.6*np.sqrt(2)*nodes
        integral = np.sum(weights*np.exp(-.3*(.8*w*w-.4*w+.4)))/np.sqrt(np.pi)
        assert abs(gate.log_mgf([u])-math.log(integral)) < 1e-12
        assert abs(gate.evaluate([u])+gate.log_mgf([u])) < 1e-12
    # If Hbar=H1, L=(e+d*P)^2-e^2 reproduces the old kappa=1 score.
    d, sp, se, lam = .7, .3, 1., .2
    old = build_quadratic_gate([[d*d, d], [d, 0]], [0, 0], 0,
        [1.2, 0], [[.5], [0]], [[sp, 0], [0, se]], .2, .9, lam)
    t = lam-2*lam*lam*se*se
    for u in (-1., 0., 1.):
        j = 1+2*t*d*d*sp*sp
        expected = t*d*d*(1.2+.5*u)**2/j+.5*math.log(j)
        assert abs(old.evaluate([u])-expected) < 1e-12
    try:
        build_quadratic_gate([[-1]], [0], 0, [0], [[1]], [[1]], 0, .9, 1.)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-positive Gaussian integration matrix accepted")
    print("quadratic.py: deterministic sanity checks passed")


if __name__ == "__main__":
    _sanity()
