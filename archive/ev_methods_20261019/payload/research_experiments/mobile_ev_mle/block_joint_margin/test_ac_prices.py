"""Independent finite differences for physical AC price ingredients."""
import numpy as np
import pytest

from .. import likelihood as lk
from .. import ev_mobile_cases_v2 as cases
from .ac_objective import _power_derivative, _output, _future_observation, _sigmas


@pytest.mark.parametrize('amplitude',[-5.,0.,5.])
def test_full_ac_injection_derivative_at_charge_and_discharge(amplitude):
    candidate=cases.candidate_bank('remote8')[1]
    r=np.linspace(.001,.004,candidate.m);x=.22*r
    p=np.full((3,3),.12);q=.32*p;power=np.zeros_like(p);power[:,1]=amplitude
    args=(candidate,r,x)
    physical=lk.ac_predict(*args,p+power/20.,q,.1,.035)
    exact=_power_derivative(candidate,physical,20.)
    step=1e-3
    for t in range(3):
        for j in range(3):
            delta=np.zeros_like(p);delta[t,j]=step/20.
            plus=lk.ac_predict(*args,p+power/20.+delta,q,.1,.035)
            minus=lk.ac_predict(*args,p+power/20.-delta,q,.1,.035)
            numerical=(_output(plus)-_output(minus))/(2*step)
            target=np.zeros_like(numerical);target[t]=exact[t,:,j]
            np.testing.assert_allclose(numerical,target,rtol=2e-6,atol=2e-9)
    assert physical['ac_equation_residual']<2e-10
    assert physical['power_balance_residual']<2e-10


def test_measurement_bias_applied_once_and_output_noise_not_line_noise():
    candidate=cases.candidate_bank('remote8')[0]
    physical=lk.ac_predict(candidate,np.full(candidate.m,.002),
        np.full(candidate.m,.0005),np.full((2,3),.1),np.full((2,3),.03),.1,.04)
    meter=lk.MeasurementModel(.01,.01,.002,master_p_sigma=.03,master_q_sigma=.02,
        bias_p=np.array([.001,.002,.003]),bias_q=.004,bias_v=-.0007)
    obs=_future_observation(physical,meter)
    np.testing.assert_allclose(obs.p-meter.bias_p,physical['p'])
    np.testing.assert_allclose(obs.q-meter.bias_q,physical['q'])
    np.testing.assert_allclose(obs.v-meter.bias_v,physical['v'])
    sigma=_sigmas(meter,2,3)
    np.testing.assert_allclose(sigma,np.tile([.002,.002,.002,.03,.02],(2,1)))


def test_residual_price_sign_and_kw_units_at_fixed_explanations():
    a,b=cases.candidate_bank('remote8')[:2]
    p=np.full((2,3),.13);q=.3*p;u=np.zeros_like(p)
    weights=np.tile([1e6,1e6,1e6,10000.,10000.],(2,1))
    def value(power,with_derivative=False):
        ya=lk.ac_predict(a,np.full(a.m,.002),np.full(a.m,.0006),p+power/20.,q,.1,.03)
        yb=lk.ac_predict(b,np.full(b.m,.002),np.full(b.m,.0006),p+power/20.,q,.1,.03)
        e=_output(ya)-_output(yb)
        v=.5*np.sum(weights*e*e)
        if with_derivative:
            g=np.einsum('to,toj->tj',weights*e,
                _power_derivative(a,ya,20.)-_power_derivative(b,yb,20.))
            return v,g
        return v
    _,gradient=value(u,True)
    for t,j in [(0,0),(1,1),(0,2)]:
        step=np.zeros_like(u);step[t,j]=.001
        numerical=(value(u+step)-value(u-step))/.002
        assert gradient[t,j]==pytest.approx(numerical,rel=2e-6,abs=1e-8)
