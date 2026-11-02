import math
import numpy as np
import pytest
from scipy.special import ndtr
from scipy.optimize import brentq
from delivery_check import log_delivery_probability


def test_zero_and_deterministic_limits():
    assert log_delivery_probability([0.], [.1], [-1.], [1.])==0.
    assert log_delivery_probability([1.], [0.], [-1.], [1.])==0.
    assert log_delivery_probability([1.1], [0.], [-1.], [1.])==-math.inf


def test_independent_all_day_probability_is_product_not_marginal():
    u=np.ones(5);cv=.1
    single=ndtr(0)-ndtr(-20)
    assert math.exp(log_delivery_probability(u,cv,-1.,1.))==pytest.approx(single**5)
    assert math.exp(log_delivery_probability(-u,cv,-1.,1.))==pytest.approx(single**5)


def test_root_derating_and_direct_monte_carlo():
    u=np.array([1.,-.9,.7,-.5]);cv=.15
    f=lambda a: log_delivery_probability(a*u,cv,-1.,1.)-math.log(.95)
    scale=brentq(f,0.,1.)
    assert 0<scale<1
    assert math.exp(log_delivery_probability(scale*u,cv,-1.,1.))==pytest.approx(.95,abs=1e-10)
    z=np.random.default_rng(61102).normal(size=(200000,4))
    delivery=scale*u*(1+cv*z)
    actual=np.mean(np.all(np.abs(delivery)<=1,axis=1))
    assert abs(actual-.95)<.002
