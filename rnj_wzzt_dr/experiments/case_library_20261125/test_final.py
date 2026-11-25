"""Tests of the method for the sealed test (identification/final.py). Development cases only."""
import numpy as np
import pytest

from rnj_wzzt_dr.experiments.case_library_20261125 import build
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c import WZZT_C
from rnj_wzzt_dr.identification import combo, final, revised as rv

needs_cases = pytest.mark.skipif(not (build.OUT/"dev").exists(), reason="the development cases are not on this machine")


def series(name):
    inp = build.to_input(build.load("dev", name))
    return inp, p2.series(inp)


@needs_cases
@pytest.mark.parametrize("name", ["soumalas11_street_l1_none", "dev_small_003_none", "paper15_star_l2_none"])
def test_without_prior_or_warm_start_the_tree_is_combo_fit(name):
    inp, (tr, va) = series(name)
    a = combo.fit(tr, va, **WZZT_C)["selected"]
    st = final.Settings(lam=0., warm=False, pool_seconds=240., root_seconds=240.)
    b, rec = final.tree(final.priced(combo.setup(tr, va), 0.), st)
    assert {C for C in a if 1 < len(C) < inp.n} == {C for C in b if 1 < len(C) < inp.n}
    assert rec["certified"]


@needs_cases
def test_the_prior_lowers_the_price_and_stays_positive():
    inp, (tr, va) = series("paper15_street_l1_none")
    S = combo.setup(tr, va)
    P = final.priced(S, final.LAMBDA)
    assert 0 < P["alpha"] < S["alpha"]
    assert S["alpha"]-P["alpha"] == pytest.approx(2*final.LAMBDA/S["N"])
    with pytest.raises(ValueError):
        final.priced(S, 100.)


@needs_cases
def test_the_full_method_runs_with_and_without_a_source():
    inp, (tr, va) = series("soumalas11_street_l1_end")
    r = final.fit(tr, va, inp.hist["Ppcc"], inp.hist["Qpcc"])
    assert r["path"] and len(r["path"][0]) == inp.n and r["source_evidence"] > 4.6
    assert r["record"]["certified"]
    blind = final.fit(tr, va)
    assert "located" not in blind and blind["record"]["certified"]


@needs_cases
def test_warm_start_does_not_change_the_optimum():
    """The warm-started pool MIQP returns the same criterion value as the cold one (within the MIQP gap)."""
    inp, (tr, va) = series("dev_medium_005_none")
    S = final.priced(combo.setup(tr, va), final.LAMBDA)
    cold, _ = final.tree(S, final.Settings(warm=False, pool_seconds=240.))
    warm, _ = final.tree(S, final.Settings(warm=True))
    crit = lambda f: rv.exact_fit(S["design"], f, 2.)["objective"]+S["alpha"]*rv.internal_count(f, inp.n)
    assert .5*S["N"]*(crit(warm)-crit(cold)) < .5
    assert np.isfinite(crit(warm))


@needs_cases
def test_a_numerical_failure_of_the_matrix_qp_falls_back(monkeypatch):
    """If the Gurobi QP ends with a numerical error, the set-up is repeated with the SciPy solver and still returns a tree."""
    inp, (tr, va) = series("soumalas11_star_l1_none")
    real = rv.ordered_sensitivity

    def broken(design, solver="slsqp"):
        if solver == "gurobi":
            raise RuntimeError("Ordered QP (Gurobi) ended with status 12")
        return real(design, solver)
    monkeypatch.setattr(rv, "ordered_sensitivity", broken)
    r = final.fit(tr, va)
    assert r["record"]["certified"]
    assert rv.ordered_sensitivity is broken                    # the patch of the fallback is undone
