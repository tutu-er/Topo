"""Small solver-control checks; no private paper15 data or AC is read."""
import json
from types import SimpleNamespace

import numpy as np

from research_experiments.wzzt_dr_selection.dr_continuous_independent import full_price


class SmallProblem:
    def __init__(self, probability=None, gradient=None):
        self.zero, self.old = np.array([0.]), np.array([.2])
        self.lower, self.upper = np.array([0.]), np.array([1.])
        self.bounds = [(0., 1.)]
        self.probability = probability or (lambda x:.5+.5*x)
        self.gradient = gradient or (lambda x:.5)

    def evaluate(self, values):
        values = np.asarray(values, float)
        x = float(values[0])
        return dict(values=values.copy(), x=values.reshape(1, 1), mean_u=values.reshape(1, 1),
                    probability=self.probability(x), gradient=np.array([self.gradient(x)]),
                    fee=x*x, fee_gradient=2*values, seconds=0., moments={}, analytic_tail_bound=0.)


def test_slsqp_runs_from_infeasible_start_and_minimizes_original_fee(tmp_path):
    problem = SmallProblem()
    assert problem.evaluate(problem.old)["probability"] < full_price.TARGET
    best, summary = full_price.optimize(problem, tmp_path, maxiter=30, seconds=10.)
    assert summary["statuses"][0]["stage"] == "cost"
    assert summary["qualified_by_search"]
    assert best["probability"] >= .9
    assert abs(best["fee"]-.64) < 1e-4
    assert summary["fixed_cost_scale"] == .2**2
    trace = json.loads((tmp_path/"optimization_trace.json").read_text())
    assert summary["price_evaluations"] == len(trace)
    assert [row["index"] for row in trace] == list(range(len(trace)))


def test_failed_cost_stage_is_saved_and_original_constraint_can_restore(tmp_path, monkeypatch):
    def failed_solver(*args, **kwargs):
        raise RuntimeError("Synthetic cost-stage failure")
    monkeypatch.setattr(full_price, "minimize", failed_solver)
    best, summary = full_price.optimize(SmallProblem(), tmp_path, restore_seconds=10.)
    assert (tmp_path/"cost_exception.json").exists()
    assert summary["statuses"][0]["solver_success"] is False
    assert summary["statuses"][0]["iterations"] is None
    assert summary["statuses"][1]["stage"] == "restore"
    assert best["probability"] >= .9 and summary["qualified_by_search"]


def test_near_target_is_not_silently_qualified_by_numerical_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(full_price, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(x=args[1], nit=1, message="synthetic return", success=True))
    problem = SmallProblem(probability=lambda x:.9-1e-8, gradient=lambda x:0.)
    best, summary = full_price.optimize(problem, tmp_path)
    assert best["probability"] < .9
    assert not summary["qualified_by_search"]
    assert summary["statuses"][-1]["status"] == "zero_projected_gradient"


def test_finalize_forces_all_orders_and_rejects_inconsistent_probability(tmp_path, monkeypatch):
    (tmp_path/"request.json").write_text("{}", encoding="utf-8")
    orders = []
    def probability(model, threshold, **kwargs):
        order = kwargs["order"]
        orders.append(order)
        return dict(probability=.9+order*1e-5, analytic_tail_bound=0., certified_error=None)
    monkeypatch.setattr(full_price, "gauss_probability", probability)
    device = SimpleNamespace(R=1, dt=.25, delta_upper_kw=np.ones((1, 1)),
        delta_lower_kw=-np.ones((1, 1)), cost_full_activation=np.ones((1, 1)),
        audit=lambda u:dict(passed=True))
    problem = SimpleNamespace(device=device, active=np.array([0]), normalization=(0., 1.),
        data=dict(sigma=np.array([1.]), packet=dict(threshold=.1), inputs={}),
        nominal=lambda values:dict(x=np.array([[.1]]), mean_u=np.array([[.1]]),
            a_C=np.array([1.]), a_F=np.array([0.]), fee=.0025))
    summary = full_price.finalize_quote(problem, dict(values=np.array([.1])),
        dict(target_scope="synthetic"), tmp_path, {})
    assert orders == [16, 32, 64]
    assert not summary["qualified"]
    assert (tmp_path/"price_freeze.json").exists()
    assert not summary["refined"]["numerical_pass"]
