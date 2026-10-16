"""Four nested EV operation domains and score-preserving plan inheritance.

Only two switches vary: permission to move and permission to discharge.
The battery, connection windows, initial state and terminal service are shared.
Full paths are lifted by matching arcs; no route is reconstructed from powers.
The score cache is for one fixed AC objective depending on net powers only.
If route costs enter the objective, use a separate objective and cache key.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from .ev_mobile_service_domain import (
    MobileServiceConfig, MobileServiceDomain, build_mobile_service_domain,
)


DOMAIN_FLAGS = {
    "fixed_charge": (False, False),
    "fixed_bidir": (False, True),
    "mobile_charge": (True, False),
    "mobile_bidir": (True, True),
}
PARENTS = {
    "fixed_charge": (),
    "fixed_bidir": ("fixed_charge",),
    "mobile_charge": ("fixed_charge",),
    "mobile_bidir": ("fixed_bidir", "mobile_charge"),
}


def build_nested_domains(config: MobileServiceConfig):
    """Build F+, F±, M+, M± with precisely the same public physical inputs.

    Both permission flags in ``config`` are replaced deliberately. ``max_moves``
    remains an upper bound; mobile domains always include a zero-move path when
    that path meets the common public initial/terminal/access constraints.
    A common terminal site different from the initial site may make both fixed
    domains empty; the builder does not silently change the task to avoid it.
    """
    if not isinstance(config, MobileServiceConfig):
        raise TypeError("config must be MobileServiceConfig")
    return {
        name: build_mobile_service_domain(replace(
            config, mobile=mobile, allow_discharge=discharge))
        for name, (mobile, discharge) in DOMAIN_FLAGS.items()
    }


def _same_value(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def assert_nested(parent: MobileServiceDomain, child: MobileServiceDomain):
    """Check the simple inclusion contract before lifting a feasible plan.

    Every input except the two permission flags must be equal. A child may
    add a permission; it cannot remove one. This intentionally rejects some
    coincidental inclusions (for example zero-capacity discharge), rather than
    inferring a physical inclusion from net powers alone.
    """
    for field in fields(MobileServiceConfig):
        name = field.name
        if name not in ("mobile", "allow_discharge"):
            if not _same_value(getattr(parent.config, name), getattr(child.config, name)):
                raise ValueError("different public physical input: " + name)
    for name in ("mobile", "allow_discharge"):
        if getattr(parent.config, name) and not getattr(child.config, name):
            raise ValueError("child removes permission: " + name)
    return True


def lift_operation(parent: MobileServiceDomain, child: MobileServiceDomain, x,
                   *, tolerance=1e-7):
    """Copy a complete feasible operation into a containing domain, without solve.

    The full ServiceArc value is its key, including origin connection state,
    duration, driving energy and worker cost. Newly available travel arcs stay
    zero. Continuous variables and charge/discharge mode are copied exactly.
    Both the input and lifted operation receive independent physical checks.
    """
    assert_nested(parent, child)
    audit = parent.check(x, tolerance=tolerance)
    if not audit["feasible"]:
        raise ValueError("parent operation infeasible: " + ",".join(audit["issues"]))
    source = np.asarray(x, float)
    lifted = np.zeros(child.model.n_variables)
    for name in ("u", "charge", "discharge", "energy", "mode"):
        a, b = getattr(parent.indices, name), getattr(child.indices, name)
        if a.shape != b.shape:
            raise ValueError("variable shape changed: " + name)
        lifted[b] = source[a]
    lookup = {arc: int(index) for arc, index in zip(child.indices.arcs, child.indices.arc)}
    if len(lookup) != len(child.indices.arcs):
        raise ValueError("child has duplicate operation arc keys")
    for arc, index in zip(parent.indices.arcs, parent.indices.arc):
        if arc not in lookup:
            raise ValueError("child does not contain parent operation arc")
        lifted[lookup[arc]] = source[int(index)]
    audit = child.check(lifted, tolerance=tolerance)
    if not audit["feasible"]:
        raise ValueError("lifted operation infeasible: " + ",".join(audit["issues"]))
    return lifted


def action_key(power):
    """Exact time-major power identity, with signed zero normalized."""
    power = np.array(power, dtype="<f8", order="C", copy=True)
    if power.ndim != 2 or not np.isfinite(power).all():
        raise ValueError("power must be finite [time,site]")
    power[power == 0.] = 0.
    payload = np.asarray(power.shape, dtype="<i8").tobytes() + power.tobytes()
    return hashlib.sha256(payload).hexdigest()


class ACScoreCache:
    """Reuse one deterministic full-AC score across containing physical domains.

    ``objective_id`` must identify the complete information set, fitted
    references, forecast, weights, fitting options and seeds. The evaluator
    receives only a copied net-power array and returns ``success`` and ``score``
    plus its complete evidence. Failed evaluations are retained in this cache.
    Its numerical local-fit result is reused, never recomputed for a new domain.
    """

    def __init__(self, objective_id: str, evaluator: Callable):
        if not isinstance(objective_id, str) or not objective_id or not callable(evaluator):
            raise ValueError("nonempty objective_id and callable evaluator required")
        self.objective_id = objective_id
        self.evaluator = evaluator
        self._records = {}
        self.evaluations = 0

    @staticmethod
    def _validate(record):
        record = deepcopy(dict(record))
        if not isinstance(record.get("success"), (bool, np.bool_)):
            raise ValueError("score evaluator must return boolean success")
        if record["success"]:
            score = float(record["score"])
            if not np.isfinite(score):
                raise ValueError("successful score must be finite")
            record["score"] = score
        return record

    def remember(self, power, record):
        """Import a known score witness for this explicitly shared objective.

        Intended for inheritance from a serialized parent. The caller supplies
        its matching objective_id through ``inherit_plan``; no new AC fit runs.
        A conflicting successful score for the exact same powers is rejected.
        """
        key, record = action_key(power), self._validate(record)
        if key in self._records:
            old = self._records[key]
            if old["success"] != record["success"] or (
                    old["success"] and old["score"] != record["score"]):
                raise ValueError("conflicting cached score for the same AC objective and action")
        else:
            self._records[key] = record
        return key

    def evaluate(self, power):
        power = np.array(power, float, copy=True)
        key = action_key(power)
        if key not in self._records:
            record = self._validate(self.evaluator(power))
            self.evaluations += 1
            self._records[key] = record
        return key, deepcopy(self._records[key])

    def records(self):
        return deepcopy(self._records)


@dataclass(frozen=True)
class ScoredPlan:
    domain: str
    objective_id: str
    action_key: str
    x: np.ndarray
    score: float
    score_record: dict
    origin: str


def score_operation(domain_name: str, domain: MobileServiceDomain, x,
                    cache: ACScoreCache, *, origin: str, tolerance=1e-7):
    """Validate an executable full path, then retain its common AC score."""
    x = np.array(x, float, copy=True)
    audit = domain.check(x, tolerance=tolerance)
    if not audit["feasible"]:
        raise ValueError("operation infeasible: " + ",".join(audit["issues"]))
    key, record = cache.evaluate(domain.decode(x)["u_kw"])
    if not record["success"]:
        raise ValueError("full AC scoring failed: " + str(record.get("failure", "unspecified")))
    x.setflags(write=False)
    return ScoredPlan(domain_name, cache.objective_id, key, x, record["score"], record, origin)


def inherit_plan(plan: ScoredPlan, domains: Mapping[str, MobileServiceDomain],
                 child_name: str, cache: ACScoreCache, *, tolerance=1e-7):
    """Lift a parent's actual path and reuse exactly the same score record."""
    if plan.objective_id != cache.objective_id:
        raise ValueError("parent uses a different AC objective")
    x = lift_operation(domains[plan.domain], domains[child_name], plan.x, tolerance=tolerance)
    power = domains[child_name].decode(x)["u_kw"]
    if action_key(power) != plan.action_key:
        raise ValueError("parent action changed during inheritance")
    cache.remember(power, plan.score_record)
    result = score_operation(child_name, domains[child_name], x, cache,
                             origin="parent:" + plan.domain, tolerance=tolerance)
    if result.action_key != plan.action_key or result.score != plan.score:
        raise ValueError("parent action or common objective score changed during inheritance")
    return result


def select_with_inheritance(child_name: str, domains: Mapping[str, MobileServiceDomain],
                            parents: Sequence[ScoredPlan], proposals: Sequence,
                            cache: ACScoreCache, *, tolerance=1e-7):
    """Keep the best full-AC plan, with parents first and exact ties unchanged.

    Proposal failure or a finite MILP time budget cannot remove a scored parent.
    A larger-domain solver may improve a surrogate while its full-AC score falls;
    this selector records that proposal and keeps the parent. Monotonicity is for
    this fixed cached planning objective, not for future noisy observations.
    """
    candidates = [inherit_plan(p, domains, child_name, cache, tolerance=tolerance)
                  for p in parents]
    failures = []
    for index, x in enumerate(proposals):
        try:
            candidates.append(score_operation(child_name, domains[child_name], x, cache,
                origin="proposal:" + str(index), tolerance=tolerance))
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(dict(proposal=index, failure=str(exc)))
    if not candidates:
        return None, dict(success=False, failure="no scored feasible plan", failures=failures,
                          objective_id=cache.objective_id)
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate.score > best.score:
            best = candidate
    parent_best = max((p.score for p in parents), default=None)
    if parent_best is not None and best.score < parent_best:
        raise AssertionError("common planning objective degraded below a parent")
    return best, dict(success=True, objective_id=cache.objective_id,
        selected_origin=best.origin, selected_score=best.score,
        parent_best_score=parent_best, nondecreasing_from_parents=True,
        candidates=[dict(origin=p.origin, action_key=p.action_key, score=p.score)
                    for p in candidates], failures=failures,
        proposal_scoring_complete=not failures,
        selected_parent=best.origin.startswith("parent:"),
        total_cached_ac_evaluations=cache.evaluations)
