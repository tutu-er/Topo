"""Finite response-ambiguity risk calculations; no topology identifier inside.

Each binary coordinate is a pre-price potential participation query, for example
('node201:event0', price1). Menus project from ONE domain, not separate copulas.
This assumes own-price/fixed-task response, not peer-price interference.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Moment:
    coordinates: tuple[int, ...]
    pattern: tuple[int, ...]
    lower: float
    upper: float


@dataclass(frozen=True)
class BinaryDomain:
    names: tuple[str, ...]
    atoms: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray
    a_ub: np.ndarray | None
    b_ub: np.ndarray | None

    def __post_init__(self):
        # Own readonly snapshots: narrowing probabilities requires a NEW LP.
        for name in ('atoms', 'a_eq', 'b_eq', 'a_ub', 'b_ub'):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)


def _interval(lower, upper):
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower <= upper <= 1):
        raise ValueError('probability bounds must satisfy 0 <= lower <= upper <= 1')
    return float(lower), float(upper)


def binary_domain(names: Sequence[str], marginal_bounds, moments=(), *, max_atoms=65536):
    """Allow all binary types consistent with stated marginal/joint intervals.

    No independence or individual monotonicity is imposed. Number of atoms is
    exponential in the UNION of quoted node-event-price queries across menus.
    An explicit size gate prevents accidental large allocations.
    """
    names = tuple(names)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('unique nonempty coordinate names are required')
    if len(marginal_bounds) != len(names):
        raise ValueError('one marginal interval per coordinate is required')
    count = 1 << len(names)
    if count > max_atoms or len(names) > 30:
        raise ValueError('explicit response-type size gate exceeded')
    atoms = ((np.arange(count, dtype=np.uint64)[:, None] >> np.arange(len(names), dtype=np.uint64)) & 1).astype(np.uint8)
    eq, beq, ub, bub = [np.ones(count)], [1.], [], []

    def append(row, lower, upper):
        lower, upper = _interval(lower, upper)
        row = np.asarray(row, dtype=float)
        if lower == upper:
            eq.append(row); beq.append(lower)
        else:
            ub.extend((row, -row)); bub.extend((upper, -lower))

    for i, pair in enumerate(marginal_bounds):
        append(atoms[:, i], *pair)
    for moment in moments:
        ids, pattern = moment.coordinates, moment.pattern
        if not ids or len(set(ids)) != len(ids) or len(ids) != len(pattern):
            raise ValueError('joint moment needs distinct coordinates and aligned pattern')
        if any(not isinstance(i, int) or i < 0 or i >= len(names) for i in ids) or any(v not in (0, 1) for v in pattern):
            raise ValueError('invalid joint moment coordinates or pattern')
        append(np.all(atoms[:, ids] == pattern, axis=1), moment.lower, moment.upper)
    return BinaryDomain(names, atoms, np.asarray(eq), np.asarray(beq),
                        np.asarray(ub) if ub else None, np.asarray(bub) if ub else None)


def projected_keys(domain: BinaryDomain, coordinates):
    """Project a menu to global node-event slots; -1 is an unpriced zero task.

    Slot order must be identical across all menus, including different sites.
    A coordinate may be reused only deliberately (same potential response).
    """
    coordinates = tuple(coordinates)
    if not coordinates or len(coordinates) > 62:
        raise ValueError('invalid number of global node-event slots')
    result = np.zeros(len(domain.atoms), dtype=np.uint64)
    for slot, coordinate in enumerate(coordinates):
        if not isinstance(coordinate, int) or coordinate < -1 or coordinate >= len(domain.names):
            raise ValueError('invalid potential-response coordinate')
        if coordinate >= 0:
            result |= domain.atoms[:, coordinate].astype(np.uint64) << np.uint64(slot)
    return result


def marginal_projection(all_query_bounds, retained_queries, *, extra_moments=(), max_atoms=65536):
    """Exact pairwise projection ONLY when the global law has marginal bounds.

    Any retained joint law can be extended by independent omitted coordinates
    whose marginals lie in their intervals. Therefore an objective depending
    only on these queries has exactly the same supremum in this projected LP.
    This does not impose independence ON retained response variables. Additional
    global moments invalidate the simple extension argument and are rejected.
    """
    if extra_moments:
        raise ValueError('global joint moments require an explicit extension check; cannot discard coordinates')
    for name, pair in all_query_bounds.items():
        if not isinstance(name, str) or not name:
            raise ValueError('nonempty global query names required')
        _interval(*pair)
    retained_queries = tuple(retained_queries)
    if any(name not in all_query_bounds for name in retained_queries):
        raise ValueError('retained query missing from common global mechanism')
    return binary_domain(retained_queries, [all_query_bounds[name] for name in retained_queries], max_atoms=max_atoms)


class ExpectationLP:
    """Maximize finite expectation with checked numerical primal/dual receipts.

    Receipts are floating-point audits, not exact arithmetic certificates.
    Limits apply to this explicit probability LP, not to continuous impedances.
    """
    def __init__(self, domain, *, max_solves=500, time_limit=15., tolerance=1e-8):
        self.domain = domain
        self.max_solves = int(max_solves)
        self.time_limit = float(time_limit)
        self.tolerance = float(tolerance)
        self.receipts = []
        self.cache = {}

    def maximize(self, coefficients):
        v = np.asarray(coefficients, dtype=float)
        if v.shape != (len(self.domain.atoms),) or not np.isfinite(v).all():
            raise ValueError('one finite coefficient per response type is required')
        key = v.tobytes()
        if key in self.cache:
            return self.cache[key]
        if len(self.receipts) >= self.max_solves:
            raise RuntimeError('predeclared probability-LP count exhausted')
        d = self.domain
        res = linprog(-v, A_eq=d.a_eq, b_eq=d.b_eq, A_ub=d.a_ub, b_ub=d.b_ub,
                      bounds=(0, None), method='highs', options={'time_limit': self.time_limit})
        receipt = dict(status=int(res.status), success=bool(res.success), message=str(res.message),
                       variables=len(v), value=None)
        self.receipts.append(receipt)
        if res.status != 0 or not res.success:
            raise RuntimeError(f'probability LP unresolved: {res.status}: {res.message}')
        w = np.asarray(res.x)
        y = np.asarray(res.eqlin.marginals)
        z = np.asarray(res.ineqlin.marginals) if d.a_ub is not None else np.empty(0)
        reduced = -v-d.a_eq.T@y
        dual_min = float(d.b_eq@y)
        inequality_violation = 0.
        sign_violation = 0.
        if d.a_ub is not None:
            reduced -= d.a_ub.T@z
            dual_min += float(d.b_ub@z)
            inequality_violation = float(np.maximum(d.a_ub@w-d.b_ub, 0).max(initial=0))
            sign_violation = float(np.maximum(z, 0).max(initial=0))
        primal = float(v@w)
        dual = -dual_min
        residuals = dict(equality=float(abs(d.a_eq@w-d.b_eq).max(initial=0)),
                         inequality=inequality_violation, nonnegative=float(np.maximum(-w, 0).max(initial=0)),
                         dual_feasibility=float(np.maximum(-reduced, 0).max(initial=0)),
                         dual_sign=sign_violation, objective_gap=abs(dual-primal))
        scale = max(1., float(abs(v).max(initial=0)))
        if max(residuals.values()) > self.tolerance*scale:
            receipt.update(residuals=residuals, value=primal, dual_value=dual)
            raise RuntimeError('numerical primal/dual audit failed')
        receipt.update(value=primal, dual_value=dual, residuals=residuals)
        self.cache[key] = primal
        return primal


def evaluate_regret(lp: ExpectationLP, lower, upper, identities=None):
    """Risk and regret intervals on a declared FINITE history-reference domain.

    Arrays have shape (history explanations, menus, common response types).
    A shared identity denotes the same observation, algorithm/configuration and
    evaluation target. Its loss interval must agree globally. Missing cells are
    bounded explicitly, never filled with zero or dropped/renormalized.
    Partial-table bounds can be loose because shared unknown values across types
    are relaxed; complete tables give numerical finite-domain values.
    """
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.ndim != 3 or lo.shape != hi.shape or lo.shape[2] != len(lp.domain.atoms):
        raise ValueError('expected aligned history x menu x type arrays')
    if lo.shape[0] == 0 or lo.shape[1] == 0 or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo < 0) or np.any(lo > hi):
        raise ValueError('finite nonnegative ordered loss intervals are required')
    ids = None if identities is None else np.asarray(identities, dtype=object)
    if ids is not None:
        if ids.shape != lo.shape or any(not isinstance(s, str) or not s for s in ids.flat):
            raise ValueError('identities must be nonempty strings with the same shape')
        seen = {}
        for name, a, b in zip(ids.flat, lo.flat, hi.flat, strict=True):
            if name in seen and seen[name] != (a, b):
                raise ValueError('shared cache identity has inconsistent loss bounds')
            seen[name] = (a, b)
    histories, menus, _ = lo.shape
    risk = []
    for h in range(histories):
        risk.append([dict(lower=-lp.maximize(-lo[h, m]), upper=lp.maximize(hi[h, m])) for m in range(menus)])
    lower_g, upper_g, pairs = np.zeros(menus), np.zeros(menus), []
    for m in range(menus):
        for h in range(histories):
            for comparator in range(menus):
                if m == comparator:
                    continue
                a = lo[h, m]-hi[h, comparator]
                b = hi[h, m]-lo[h, comparator]
                if ids is not None:
                    same = ids[h, m] == ids[h, comparator]
                    a = np.where(same, 0., a); b = np.where(same, 0., b)
                left, right = lp.maximize(a), lp.maximize(b)
                lower_g[m] = max(lower_g[m], left)
                upper_g[m] = max(upper_g[m], right)
                pairs.append(dict(menu=m, history=h, comparator=comparator, lower=left, upper=right))
    return dict(scope='declared_finite_history_domain_and_response_constraints',
                full_parameter_coverage=False, topology_probability=False,
                complete_table=bool(np.array_equal(lo, hi)), partial_bounds_may_be_loose=bool(np.any(lo != hi)),
                risk=risk, regret_lower=lower_g.tolist(), regret_upper=upper_g.tolist(), comparisons=pairs)


def select_menu(result, expected_fees, *, seed=944231, tolerance=1e-9):
    """Report upper-bound minimizer, all ties, and its honest suboptimality bound.

    An unresolved candidate is not an authorized deployment recommendation.
    Hard budget/device checks belong to the existing feasible menu constructor.
    """
    lower = np.asarray(result['regret_lower']); upper = np.asarray(result['regret_upper'])
    fees = np.asarray(expected_fees, dtype=float)
    if fees.shape != upper.shape or not np.isfinite(fees).all() or np.any(fees < 0):
        raise ValueError('one nonnegative expected fee per feasible menu is required')
    tied = np.flatnonzero(upper <= upper.min()+tolerance)
    tied = [int(m) for m in tied if fees[m] <= fees[tied].min()+tolerance]
    candidate = Random(seed).choice(tied)
    gap = max(0., float(upper[candidate]-lower.min()))
    return dict(candidate_index=candidate, all_cost_and_upper_bound_ties=tied, tie_seed=seed,
                finite_domain_suboptimality_upper=gap, selection_resolved=bool(gap <= tolerance),
                certified_true_topology_or_global_parameter_optimum=False)
