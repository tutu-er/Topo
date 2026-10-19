"""Closed-form profile/curvature checks and a tiny independent MILP audit."""
import numpy as np
import pytest

from .ev_fixed_train_scalar_profile_checks import quadratic_profile,box_profile,chord_bounds,two_slot_milp


def test_source_profile_schur_curvature_and_time_separation():
    p=np.array([-.8,.2,1.4]);source=p/2.
    f=.5*((p-source)**2+source**2)
    np.testing.assert_allclose(f,quadratic_profile(p),atol=1e-15,rtol=0)
    fpp,fps,fss=1.,-1.,2.
    assert fpp-fps/fss*fps==.5
    step=1e-3
    second=(quadratic_profile(p+step)-2*quadratic_profile(p)+quadratic_profile(p-step))/step**2
    np.testing.assert_allclose(second,.5,atol=2e-10,rtol=0)
    # The independent minimizers [.5,.5] produce total .5 for powers [1,1].
    # Adding an intertime source sum=0 instead forces the best source [0,0]
    # and total 1: the decomposition premise is substantive.
    assert float(quadratic_profile([1.,1.]).sum())==.5
    assert .5*((1.-0.)**2+0.**2)*2==1.
    assert .5*(1.-0.)**2 > float(quadratic_profile(1.))  # feasible source gives upper value


def test_both_curvature_signs_require_correct_upper_and_lower_envelopes():
    a,b=-.5,1.5;p=np.linspace(a,b,101)
    for function in (lambda x:x*x,lambda x:3.-x*x):
        low,high,interp,epsilon=chord_bounds(function,a,b,p,2.)
        actual=function(p)
        assert np.all(low<=actual+1e-14) and np.all(actual<=high+1e-14)
        assert max(abs(interp-actual))==pytest.approx(epsilon)
    _,_,convex,_=chord_bounds(lambda x:x*x,a,b,p,2.)
    _,_,concave,_=chord_bounds(lambda x:3.-x*x,a,b,p,2.)
    assert np.all(convex>=p*p-1e-14)
    assert np.all(concave<=3.-p*p+1e-14)


def test_source_active_boundary_switch_retains_global_lipschitz_derivative_bound():
    # d'=p-proj_[0,1](p), slope 1/0/1; d'' changes at 0 and 1.
    p=np.linspace(-.5,1.5,201);derivative=p-np.clip(p,0.,1.)
    assert np.all(abs(np.diff(derivative))<=np.diff(p)+1e-14)
    np.testing.assert_allclose(box_profile([-.5,.3,1.5]),[.125,0.,.125],atol=1e-15,rtol=0)
    for a,b in ((-.5,.4),(.4,1.5),(-.5,1.5)):
        points=np.linspace(a,b,101)
        lower,upper,_,_=chord_bounds(box_profile,a,b,points,1.)
        assert np.all(lower<=box_profile(points)+1e-14) and np.all(box_profile(points)<=upper+1e-14)


def test_two_slot_maximization_bound_direction_and_refinement():
    gaps=[]
    for grid in ([0.,1.,2.],[0.,.5,1.,1.5,2.]):
        lower=two_slot_milp(grid,side="lower");upper=two_slot_milp(grid,side="upper")
        assert lower["success"] and upper["success"]
        assert lower["model_incumbent"]<=1.+1e-9
        assert upper["maximization_model_best_bound"]>=1.-1e-9
        for result in (lower,upper):
            assert sum(result["powers"])==pytest.approx(2.,abs=1e-9)
            assert result["exact_physical_score"]==pytest.approx(1.,abs=1e-9)
            assert result["exact_physical_score"]>result["constant_score"]
        gaps.append(upper["maximization_model_best_bound"]-lower["model_incumbent"])
    assert gaps[0]==pytest.approx(.25,abs=1e-9)
    assert gaps[1]==pytest.approx(.0625,abs=1e-9)
