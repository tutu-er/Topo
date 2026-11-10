"""Small solver-control checks, independent of the private paper15 case."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from . import solve
except ImportError:
    import solve


class SmallProblem:
    def __init__(self, target=.9, probability=None, gradient=None):
        self.target = target
        self.zero, self.warm = np.array([0.]), np.array([.2])
        self.lower, self.upper = np.array([0.]), np.array([1.])
        self.bounds = [(0., 1.)]
        self.probability = probability or (lambda x:.5+.5*x)
        self.gradient = gradient or (lambda x:.5)

    def evaluate(self, values):
        values = np.asarray(values, float)
        x = float(values[0])
        return dict(values=values.copy(), probability=self.probability(x),
                    gradient=np.array([self.gradient(x)]), fee=x*x,
                    fee_gradient=2*values, moments={}, analytic_tail_bound=0.)


def test_infeasible_start_still_solves_minimum_fee(tmp_path):
    problem = SmallProblem()
    best, summary = solve.optimize(problem, tmp_path, maxiter=30, seconds=10.)
    assert summary["statuses"][0]["stage"] == "cost"
    assert summary["qualified_by_search"] and best["probability"] >= problem.target
    assert abs(best["fee"]-.64) < 1e-4
    records = json.loads((tmp_path/"trace.json").read_text())
    assert summary["evaluations"] == len(records)
    with np.load(tmp_path/"checkpoint.npz", allow_pickle=False) as saved:
        assert saved["values"].shape == (len(records), 1)


def test_target_is_case_specific_not_hard_coded(tmp_path):
    best, summary = solve.optimize(SmallProblem(target=.8), tmp_path,
                                   maxiter=30, seconds=10.)
    assert summary["target"] == .8
    assert best["probability"] >= .8
    assert abs(best["fee"]-.36) < 1e-4


def test_solver_failure_is_retained_before_restoration(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Deliberate synthetic failure")
    monkeypatch.setattr(solve, "minimize", fail)
    best, summary = solve.optimize(SmallProblem(), tmp_path, restore_seconds=10.)
    assert (tmp_path/"cost_failure.json").exists()
    assert summary["statuses"][0]["iterations"] is None
    assert summary["statuses"][1]["stage"] == "restore"
    assert best["probability"] >= .9


def test_numerical_tolerance_does_not_relax_target(tmp_path, monkeypatch):
    monkeypatch.setattr(solve, "minimize", lambda *args, **kwargs:
        SimpleNamespace(x=args[1], nit=1, success=True, message="synthetic return"))
    problem = SmallProblem(probability=lambda x:.9-1e-8, gradient=lambda x:0.)
    best, summary = solve.optimize(problem, tmp_path)
    assert best["probability"] < problem.target
    assert not summary["qualified_by_search"]
    assert summary["statuses"][-1]["status"] == "zero_projected_gradient"


def test_legacy_case_replay_lock_prevents_new_optimization(tmp_path, monkeypatch):
    monkeypatch.setattr(solve, "load_problem", lambda path:
        SimpleNamespace(config={"approved_for_new_optimization":False}))
    output = tmp_path/"blocked"
    args = SimpleNamespace(case=tmp_path, out=output, quote=None, check_only=False)
    with pytest.raises(ValueError, match="locked for replay"):
        solve.main(args)
    assert not output.exists()
    # Replay mode passes the legacy lock and retains its ordinary quote check.
    args.check_only = True
    with pytest.raises(ValueError, match="--check-only requires --quote"):
        solve.main(args)
