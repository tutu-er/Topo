from types import SimpleNamespace
import numpy as np
import pytest
from .gradient_price_search import SearchSettings, optimize_price_curves, search_record
from .single_site_milp import SingleSiteConstraints, check_single_site_plan


class AnalyticEvaluator:
    horizon = 2
    base_kva = 20.
    q_over_p = 0.
    history = SimpleNamespace(n=2)

    def __init__(self, fail_site=None, pq_reverse=False):
        self.fail_site, self.pq_reverse = fail_site, pq_reverse
        self.calls = 0

    def evaluate(self, site, power_kw):
        self.calls += 1
        u = np.asarray(power_kw)/20.
        if site == self.fail_site:
            return dict(success=False, score=None, pairs=[], failures=["retained_failure"],
                        fit_calls=2, total_ac_calls=3)
        score = float(u@u)+(site*.001)
        g = 2*u/20.
        pair = dict(score=score, gradient_kw=g, pq_only_gradient_kw=-g if self.pq_reverse else g)
        return dict(success=True, score=score, pairs=[pair, dict(pair)], failures=[],
                    fit_calls=2, total_ac_calls=3, fit_results={"memory_only":object()})


def constraints():
    return SingleSiteConstraints(energy=.075, initial_site=0,
        travel_slots=np.zeros((2,2),int), travel_energy=np.zeros((2,2)))


def test_zero_equal_energy_first_order_escapes_with_optimized_seeds():
    evaluator=AnalyticEvaluator()
    result=optimize_price_curves(evaluator,constraints(),settings=SearchSettings(max_iterations=2))
    assert result["success"]
    assert result["selected_site"] == 1
    assert result["score"] == pytest.approx(.091)
    assert result["gain_over_best_constant"] == pytest.approx(.045)
    assert not result["global_design_optimality_certified"]
    assert check_single_site_plan(result["plan"],constraints())["feasible"]
    assert result["fit_calls"] == 2*evaluator.calls
    assert result["total_ac_calls"] == 3*evaluator.calls
    assert any(r["stop_reasons"] for r in result["site_records"])
    record=search_record(result)
    assert "fit_results" not in record["evaluation"]
    assert isinstance(record["site_plans"]["1"],dict)


def test_failed_site_is_kept_and_other_site_remains_available():
    result=optimize_price_curves(AnalyticEvaluator(fail_site=1),constraints())
    assert result["success"] and result["selected_site"] == 0
    assert result["failures"] and len(result["site_records"]) == 2
    assert 1 not in result["site_plans"]


def test_all_failed_comparisons_do_not_become_high_score():
    class Fail(AnalyticEvaluator):
        def evaluate(self,site,power_kw):
            return dict(success=False,score=None,pairs=[],failures=["failure"])
    result=optimize_price_curves(Fail(),constraints())
    assert not result["success"] and result["score"] is None
    assert result["plan"] is None and len(result["failures"]) >= 2


def test_pq_only_uses_same_full_objective_guard():
    class Concave(AnalyticEvaluator):
        def evaluate(self,site,power_kw):
            result=super().evaluate(site,power_kw)
            result["score"]=-result["score"]
            for pair in result["pairs"]:
                pair["score"]=-pair["score"]
                pair["gradient_kw"]=-pair["gradient_kw"]
                pair["pq_only_gradient_kw"]=-pair["gradient_kw"]
            return result
    result=optimize_price_curves(Concave(),constraints(),
        settings=SearchSettings(max_iterations=2,gradient_mode="pq_only",max_starts=1))
    assert result["success"] and result["gain_over_best_constant"] >= -1e-12
    assert all(not step["accepted"] for r in result["site_records"] for step in r["steps"])


def test_affine_bound_is_reported_for_its_own_subproblem():
    result=optimize_price_curves(AnalyticEvaluator(),constraints(),
        settings=SearchSettings(max_iterations=1,max_starts=1))
    steps=[s for r in result["site_records"] for s in r["steps"]]
    assert steps
    assert all("best_bound" in s["milp"] for s in steps)
    assert result["bound_scope"] == "affine operational subproblem only"


@pytest.mark.parametrize("setting", [SearchSettings(max_starts=4),
    SearchSettings(max_iterations=-1), SearchSettings(gradient_mode="bad")])
def test_invalid_settings(setting):
    with pytest.raises(ValueError):
        optimize_price_curves(AnalyticEvaluator(),constraints(),settings=setting)


def test_time_varying_caps_do_not_require_constant_feasibility():
    c=SingleSiteConstraints(energy=.1,initial_site=0,
        p_charge_max=np.array([[.1,.1],[.3,.3]]),
        travel_slots=np.zeros((2,2),int),travel_energy=np.zeros((2,2)))
    result=optimize_price_curves(AnalyticEvaluator(),c,settings=SearchSettings(max_iterations=2))
    assert result["success"]
    assert result["best_constant_plan"] is None
    assert result["gain_over_best_constant"] is None
    assert result["plan"].p[:,result["selected_site"]] == pytest.approx([.1,.3])
    assert all(not r["constant_operation"]["success"] for r in result["site_records"])
    assert check_single_site_plan(result["plan"],c)["feasible"]


def test_failed_constant_AC_still_tests_nonconstant_feasible_seeds():
    class ConstantFails(AnalyticEvaluator):
        def evaluate(self,site,power_kw):
            if np.ptp(power_kw) < 1e-8:
                return dict(success=False,score=None,pairs=[],failures=["constant_only_failure"])
            return super().evaluate(site,power_kw)
    result=optimize_price_curves(ConstantFails(),constraints())
    assert result["success"] and result["best_constant_plan"] is None
    assert result["failures"]
    assert result["score"] > .08