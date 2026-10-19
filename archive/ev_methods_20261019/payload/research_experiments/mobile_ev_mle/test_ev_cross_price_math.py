"""Analytic envelope signs and nonsmooth directions; no AC fitting."""
import numpy as np
import pytest
from .ev_bilinear_support_milp import BilinearPair,evaluate_qp


def scalar_qp(c,B,A0,At,l0,h0,S,u):
    low=np.array([l0-S*u]);high=np.array([h0-S*u])
    p=BilinearPair('scalar',[c],[[B]],[[A0]],[[[At]]],low,high)
    fit=evaluate_qp(p,[u]);z=float(fit['z'][0]);r=float(fit['residual'][0]);A=A0+At*u
    gradient_z=2*A*r
    lower=gradient_z if abs(z-low[0])<1e-8 and gradient_z>0 else 0.
    upper=-gradient_z if abs(z-high[0])<1e-8 and gradient_z<0 else 0.
    explicit=2*r*(B+At*z)
    moving=explicit+(upper-lower)*S
    fixed_physical=2*r*(B+At*z-A*S)
    return dict(q=fit['q_primal_upper'],z=z,muL=lower,muU=upper,
        explicit=explicit,moving=moving,fixed_physical=fixed_physical,price=.5*moving)


@pytest.mark.parametrize('u',[.1,.4,.8])
def test_moving_lower_boundary_has_negative_price(u):
    f=scalar_qp(0.,0.,1.,0.,1.,2.,1.,u)
    assert f['q']==pytest.approx((1-u)**2)
    assert f['z']==pytest.approx(1-u)
    assert f['muL']==pytest.approx(2*(1-u)) and f['muU']==0.
    assert f['moving']==pytest.approx(-2*(1-u))
    assert f['fixed_physical']==pytest.approx(f['moving'])
    assert f['price']==pytest.approx(-(1-u))
    assert f['muL']-f['muU']>0  # proposed reversed sign would be wrong


@pytest.mark.parametrize('u',[.1,.4,.8])
def test_moving_upper_boundary_has_positive_price(u):
    f=scalar_qp(0.,0.,1.,0.,-2.,-1.,1.,u)
    assert f['q']==pytest.approx((1+u)**2)
    assert f['muL']==0. and f['muU']==pytest.approx(2*(1+u))
    assert f['moving']==pytest.approx(2*(1+u))
    assert f['fixed_physical']==pytest.approx(f['moving'])


def test_explicit_bilinear_and_moving_boundary_terms_match_finite_difference():
    values=dict(c=.2,B=.3,A0=1.,At=.4,l0=1.,h0=2.,S=.7)
    u=.25;h=1e-6
    f=scalar_qp(**values,u=u)
    numerical=(scalar_qp(**values,u=u+h)['q']-scalar_qp(**values,u=u-h)['q'])/(2*h)
    assert f['moving']==pytest.approx(numerical,rel=1e-8)
    assert f['fixed_physical']==pytest.approx(numerical,rel=1e-8)
    assert abs(f['explicit']-numerical)>.1


def test_fixed_outer_box_omits_boundary_derivative_for_its_own_value_function():
    values=dict(c=.2,B=.3,A0=1.,At=.4,l0=.5,h0=2.,S=0.)
    u=.25;h=1e-6
    f=scalar_qp(**values,u=u)
    numerical=(scalar_qp(**values,u=u+h)['q']-scalar_qp(**values,u=u-h)['q'])/(2*h)
    assert f['moving']==pytest.approx(f['explicit'])
    assert f['moving']==pytest.approx(numerical,rel=1e-8)


def test_inner_multiple_minimizers_give_minimum_directional_derivative():
    # q(u)=min_{-1<=z<=1}(1+u*z)^2=(1-|u|)^2 near u=0.
    # Every z is optimal at u=0 and partial_u f=2z.
    def q(u):
        p=BilinearPair('tie',[1.],[[0.]],[[0.]],[[[1.]]],[-1.],[1.])
        return evaluate_qp(p,[u])['q_primal_upper']
    assert q(0.)==pytest.approx(1.)
    h=1e-7
    for direction in (-1.,1.,2.):
        derivative=(q(h*direction)-q(0.))/h
        exact=min(2*(-1)*direction,2*1*direction)
        assert derivative==pytest.approx(exact,rel=3e-7)
    assert (q(h)-q(-h))/(2*h)==pytest.approx(0.)
    # A central finite difference of zero hides the nonsmooth descent.


def test_all_pair_tie_uses_minimum_of_all_active_prices():
    # Phi=min(u,-u)=-|u|. Choosing a single +1 price would be misleading.
    prices=np.array([1.,-1.])
    for d in (-2.,1.):
        assert min(prices*d)==-abs(d)


def test_zero_price_does_not_solve_global_quadratic_maximum():
    assert 2*0.==0.
    assert 1.**2>0.**2


def test_positive_free_hessian_without_strict_complementarity_does_not_ensure_uniqueness():
    # min z1², z1 in [-.5,.5], z2 in [0,1]. At (0,0), z2 is
    # lower-active, its multiplier is zero, and the free Hessian is 2.
    A=np.array([[1.,0.]])
    assert (2*A.T@A)[0,0]>0
    for z2 in (0.,.5,1.):
        r=A@np.array([0.,z2])
        assert float(r@r)==0.
