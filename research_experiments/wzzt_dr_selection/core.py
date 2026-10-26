"""Small-system mathematical reference for sequential WZZT block selection.

This module is independent of the production WZZT implementation. It minimizes
the *sum* of absolute errors, jointly refitting every selected block's nonnegative
R/X coefficients. Zero coefficients are allowed. Arrays P, Q and Y have shape
(n, T); flattened observations use NumPy C order and beta is (r0, x0, r1, x1, ...).
LP certificates are floating-point checks, not statistical reliability claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from operator import index
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linprog

Block = tuple[int, ...]
Family = tuple[Block, ...]


def _block(block: Iterable[int], n: int | None = None) -> Block:
    values = tuple(index(i) for i in block)
    if not values or len(set(values)) != len(values):
        raise ValueError("A block must contain distinct node indices and be nonempty")
    if min(values) < 0 or (n is not None and max(values) >= n):
        raise ValueError("Block node index is out of range")
    return tuple(sorted(values))


def _family(blocks: Iterable[Iterable[int]], n: int | None = None) -> Family:
    result = tuple(_block(b, n) for b in blocks)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate blocks are not allowed")
    return result


def is_laminar(blocks: Iterable[Iterable[int]]) -> bool:
    """Whether each pair is disjoint or one contains the other; empty is valid."""
    family = _family(blocks)
    sets = [set(b) for b in family]
    return all(not (a & b) or a <= b or b <= a
               for a, b in combinations(sets, 2))


def all_blocks(n: int, *, include_singletons: bool = True) -> Family:
    n = index(n)
    if n < 1:
        raise ValueError("n must be positive")
    return tuple(b for size in range(1 if include_singletons else 2, n + 1)
                 for b in combinations(range(n), size))


def enumerate_laminar_families(n: int) -> tuple[Family, ...]:
    """All n<=4 laminar families containing singletons, with every other block optional.

    'Complete' here means exhaustive over this finite mathematical domain; it
    does not assert that any family has statistically identified positive edges.
    The all-node block is optional (except n=1, where it is a fixed singleton).
    """
    n = index(n)
    if not 1 <= n <= 4:
        raise ValueError("Exhaustive enumeration is restricted to 1 <= n <= 4")
    singletons = tuple((i,) for i in range(n))
    optional = all_blocks(n, include_singletons=False)
    families = []
    for mask in range(1 << len(optional)):
        extra = tuple(b for i, b in enumerate(optional) if mask & (1 << i))
        if is_laminar(extra):
            families.append(singletons + extra)
    return tuple(families)


def compatible_next_candidates(
    n: int, selected: Iterable[Iterable[int]], *, include_singletons: bool = False
) -> Family:
    """All unselected blocks whose addition preserves laminarity."""
    selected = _family(selected, n)
    if not is_laminar(selected):
        raise ValueError("Selected blocks must form a laminar family")
    return tuple(b for b in all_blocks(n, include_singletons=include_singletons)
                 if b not in selected and is_laminar(selected + (b,)))


def _inputs(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    if P.ndim != 2 or Q.shape != P.shape or min(P.shape) < 1:
        raise ValueError("P and Q must have the same nonempty (n, T) shape")
    if not np.isfinite(P).all() or not np.isfinite(Q).all():
        raise ValueError("P and Q must be finite")
    return P, Q


def block_features(
    P: np.ndarray, Q: np.ndarray, blocks: Iterable[Iterable[int]]
) -> np.ndarray:
    """Return F of shape (n*T, 2*k), with columns vec(z z.T P), vec(z z.T Q)."""
    P, Q = _inputs(P, Q)
    blocks = _family(blocks, P.shape[0])
    F = np.empty((P.size, 2 * len(blocks)), dtype=float)
    for k, block in enumerate(blocks):
        z = np.zeros(P.shape[0])
        z[list(block)] = 1.0
        F[:, 2 * k] = np.outer(z, z @ P).ravel(order="C")
        F[:, 2 * k + 1] = np.outer(z, z @ Q).ravel(order="C")
    return F


def predict(
    P: np.ndarray, Q: np.ndarray, blocks: Iterable[Iterable[int]], beta: np.ndarray
) -> np.ndarray:
    P, Q = _inputs(P, Q)
    F = block_features(P, Q, blocks)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (F.shape[1],) or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite vector with two entries per block")
    return (F @ beta).reshape(P.shape, order="C")


def effective_blocks(
    blocks: Iterable[Iterable[int]], beta: np.ndarray, tol: float = 1e-8
) -> Family:
    """Blocks with at least one coefficient above tol; a descriptive threshold."""
    blocks = _family(blocks)
    beta = np.asarray(beta, dtype=float)
    if tol < 0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and nonnegative")
    if beta.shape != (2 * len(blocks),) or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite vector with two entries per block")
    return tuple(b for k, b in enumerate(blocks) if np.max(beta[2*k:2*k+2]) > tol)


@dataclass
class FitResult:
    blocks: Family
    beta: np.ndarray
    loss: float
    status: str
    dual_bound: float
    gap: float
    constraint_violation: float
    dual_violation: float
    dual: np.ndarray
    lp_objective: float
    solver_status: int
    message: str

    @property
    def success(self) -> bool:
        return self.status == "optimal"

    def predict(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        return predict(P, Q, self.blocks, self.beta)

    def effective_blocks(self, tol: float = 1e-8) -> Family:
        return effective_blocks(self.blocks, self.beta, tol)


def fit_l1(
    P: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    blocks: Iterable[Iterable[int]],
    *,
    certificate_tol: float = 1e-7,
    options: Mapping | None = None,
    require_laminar: bool = True,
) -> FitResult:
    """Jointly refit all weights by min_{beta>=0} ||vec(Y)-F beta||_1.

    The dual is max y.T q subject to |q|<=1 and F.T q<=0. Both primal
    and dual residuals and their gap are recomputed independently of solver.fun.
    A nonoptimal or unverified solve is retained with success=False.
    """
    P, Q = _inputs(P, Q)
    Y = np.asarray(Y, dtype=float)
    if Y.shape != P.shape or not np.isfinite(Y).all():
        raise ValueError("Y must be finite and have the same shape as P")
    if certificate_tol <= 0 or not np.isfinite(certificate_tol):
        raise ValueError("certificate_tol must be finite and positive")
    blocks = _family(blocks, P.shape[0])
    if require_laminar and not is_laminar(blocks):
        raise ValueError("Fixed support must be laminar")
    F, y = block_features(P, Q, blocks), Y.ravel(order="C")
    m, k = F.shape
    A = np.block([[F, -np.eye(m)], [-F, -np.eye(m)]])
    b = np.concatenate([y, -y])
    objective = np.concatenate([np.zeros(k), np.ones(m)])
    result = linprog(objective, A_ub=A, b_ub=b, bounds=(0, None),
                     method="highs", options=None if options is None else dict(options))
    beta = np.full(k, np.nan)
    loss = lp_objective = dual_bound = gap = float("nan")
    primal_violation = dual_violation = float("inf")
    dual = np.full(m, np.nan)
    if result.x is not None and np.isfinite(result.x).all():
        beta = np.asarray(result.x[:k]).copy()
        loss = float(np.abs(y - F @ beta).sum())
        lp_objective = float(objective @ result.x)
        primal_violation = float(max(0., np.max(-result.x, initial=0.),
                                     np.max(A @ result.x - b, initial=0.)))
    marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
    if marginals is not None and np.isfinite(marginals).all():
        dual = np.asarray(marginals[:m] - marginals[m:]).copy()
        dual_violation = float(max(0., np.max(np.abs(dual) - 1., initial=0.),
                                   np.max(F.T @ dual, initial=0.)))
        # Only a numerically dual-feasible value is exposed as a dual bound.
        if dual_violation <= certificate_tol:
            dual_bound = float(y @ dual)
            gap = loss - dual_bound
    status_names = {1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    status = status_names.get(result.status, "numerical_warning")
    objective_scale = 1. + abs(loss) if np.isfinite(loss) else 1.
    if (result.success and primal_violation <= certificate_tol
            and dual_violation <= certificate_tol and np.isfinite(gap)
            and abs(gap) <= certificate_tol * objective_scale
            and abs(loss - lp_objective) <= certificate_tol * objective_scale):
        status = "optimal"
    return FitResult(blocks, beta, loss, status, dual_bound, gap,
                     primal_violation, dual_violation, dual, lp_objective,
                     int(result.status), str(result.message))


@dataclass
class SelectionResult:
    base_fit: FitResult
    fits: dict[Block, FitResult]
    best_candidates: Family
    best_loss: float
    improvement: float
    stopped: bool | None
    status: str
    failed_candidates: Family = ()


def select_next_block(
    P: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    selected: Iterable[Iterable[int]],
    candidates: Iterable[Iterable[int]] | None = None,
    *,
    tol: float = 1e-8,
    tie_tol: float = 1e-8,
    fit_options: Mapping | None = None,
) -> SelectionResult:
    """Evaluate one-block additions, with a joint refit and every best tie returned.

    stopped=True when the best improvement is <=tol or no candidates exist.
    Any failed fit makes stopped=None: an incomplete comparison cannot certify
    a stopping decision. best_candidates is then only the observed best set.
    No arbitrary tie winner is selected or interpreted as reliable.
    """
    P, Q = _inputs(P, Q)
    if min(tol, tie_tol) < 0 or not np.isfinite([tol, tie_tol]).all():
        raise ValueError("Selection tolerances must be finite and nonnegative")
    selected = _family(selected, P.shape[0])
    if not is_laminar(selected):
        raise ValueError("Selected blocks must form a laminar family")
    if candidates is None:
        candidates = compatible_next_candidates(P.shape[0], selected)
    else:
        candidates = _family(candidates, P.shape[0])
        for block in candidates:
            if block in selected or not is_laminar(selected + (block,)):
                raise ValueError("Candidate must be unselected and laminar-compatible")
    base = fit_l1(P, Q, Y, selected, options=fit_options)
    if not base.success:
        return SelectionResult(base, {}, (), float("nan"), float("nan"),
                               None, "base_fit_failed")
    if not candidates:
        return SelectionResult(base, {}, (), base.loss, 0., True, "no_candidates")
    fits = {block: fit_l1(P, Q, Y, selected + (block,), options=fit_options)
            for block in candidates}
    failed = tuple(block for block, fit in fits.items() if not fit.success)
    good = {block: fit for block, fit in fits.items() if fit.success}
    if not good:
        return SelectionResult(base, fits, (), float("nan"), float("nan"),
                               None, "incomplete", failed)
    best_loss = min(fit.loss for fit in good.values())
    best = tuple(block for block, fit in good.items()
                 if fit.loss <= best_loss + tie_tol)
    improvement = base.loss - best_loss
    if failed:
        return SelectionResult(base, fits, best, best_loss, improvement,
                               None, "incomplete", failed)
    # A zero new coefficient makes every addition at least as good as the base.
    if improvement < -max(tol, 1e-7 * (1. + abs(base.loss))):
        return SelectionResult(base, fits, best, best_loss, improvement,
                               None, "numerical_warning")
    stopped = improvement <= tol
    return SelectionResult(base, fits, best, best_loss, improvement, stopped,
                           "stopped" if stopped else "selected")
