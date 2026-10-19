"""Independent physics, analytic minimization and score-identity tests."""
import numpy as np
import pytest
if __package__:
    from . import experiment as e
else:
    import experiment as e


@pytest.fixture(scope="module")
def ctx():
    return e.make_context()


def test_original_case33_and_pv_accounting(ctx):
    assert len(ctx["network"]["branches"])==32
    assert ctx["ids"]==list(range(2,34))
    assert ctx["p0"].sum()==3715
    assert ctx["q0"].sum()==2300
    for spec in e.CONFIG["states"]:
        g=e.generate_state(ctx,spec)
        expected=spec["p_load"]*3715-.6*3715*spec["pv_factor"]
        assert g["net_p_kw"]==pytest.approx(expected)
        assert g["generator_AC_agreement"]<1e-10
        assert g["physics"]["power_balance_residual"]<1e-10
    reverse=e.generate_state(ctx,e.CONFIG["states"][2])
    assert reverse["root_p_kw"]<0
    assert np.min(reverse["z_true"][:32])<0


def test_all_backend_derivatives_by_central_differences(ctx):
    states=[e.generate_state(ctx,s) for s in e.CONFIG["states"]]
    rows=e.derivative_checks(ctx,states)
    assert len(rows)==9
    assert max(r["max_relative_error"] for r in rows)<2e-5


def test_scalar_linear_EIV_matches_closed_form(monkeypatch):
    def toy(ctx,backend,z,derivatives=False):
        return np.array([2*z[0]+1,0.,0.]),np.array([[2.],[0.],[0.]]) if derivatives else None
    monkeypatch.setattr(e,"response",toy)
    zobs=np.array([3.]);yobs=np.array([6.,0.,0.])
    sz=np.array([.4]);sy=np.array([.3,1.,1.])
    fit=e.fit_method({},"toy","EIV",zobs,yobs,sz,sy)
    exact=(zobs[0]/.4**2+2*(yobs[0]-1)/.3**2)/(1/.4**2+4/.3**2)
    assert fit["success"]
    assert fit["z"][0]==pytest.approx(exact,abs=1e-11)
    assert fit["gradient_inf"]<1e-9
    expected=.5*(6.-(2*3.+1))**2/(.3**2+4*.4**2)
    assert fit["score"]==pytest.approx(expected,abs=1e-11)


def test_noise_free_AC_is_minimum_and_approximation_retained(ctx):
    g=e.generate_state(ctx,e.CONFIG["states"][0])
    sample=e.run_sample(ctx,g,0,.002,e.CONFIG["seeds"][0],clean=True)
    assert sample["reference_qualified"]
    assert sample["methods"]["ac/EIV"]["score"]<1e-14
    assert sample["methods"]["ac/FIX"]["score"]<1e-14
    assert sample["methods"]["linear/EIV"]["score"]>1.
    assert sample["methods"]["i2/EIV"]["score"]>1.
    for item in sample["methods"].values():
        assert item["independent_AC_max_error"]<1e-9


def test_paired_noisy_score_decomposition_and_reference(ctx):
    g=e.generate_state(ctx,e.CONFIG["states"][1])
    sample=e.run_sample(ctx,g,1,.01,e.CONFIG["seeds"][1])
    assert not sample["failures"]
    assert sample["reference_qualified"]
    for item in sample["methods"].values():
        assert item["signed_score_error"]==pytest.approx(
            item["model_score_bias"]+item["state_fitting_loss"],abs=1e-9)
        assert item["state_fitting_loss"]>=-1e-7
        assert item["AC_back_score"]>=sample["methods"]["ac/EIV"]["score"]-1e-7
    assert sample["oracle_true_state_noise_score"]>=sample["methods"]["ac/EIV"]["score"]-1e-7
