import numpy as np
import pytest
from .information_power_link_checks import power_interval,small_signal_coefficient,check_power_link

@pytest.mark.parametrize('rank',[1,2,3])
@pytest.mark.parametrize('signal',[0.,.0001,1.,10.,100.])
def test_poisson_mixture_independent_tail_with_numeric_omitted_mass(rank,signal):
    r=power_interval(rank,.1,signal)
    assert r['lower']-3e-13 <= r['independent_scipy_tail'] <= r['upper']+3e-13
    assert r['omitted_poisson_mass']<1e-13


def test_monotone_power_and_local_derivative():
    for rank in (1,2,3):
        values=[power_interval(rank,.1,x)['independent_scipy_tail'] for x in (0,.0001,1,10,100)]
        assert np.all(np.diff(values)>0)
        slope=(values[1]-values[0])/.0001
        assert slope==pytest.approx(small_signal_coefficient(rank,.1),rel=1e-4)


def test_small_signal_average_information_expansion_and_uniform_remainder():
    r=check_power_link()
    assert r['success'] and r['engineering_average_half_distance']==1.25
    assert all(x['within_remainder_bound'] for x in r['scales'])
    assert abs(r['scales'][-1]['power_increment_ratio']-1)<.002
    c=r['same_average_signal_counterexample']
    assert abs(c['difference'])>.02  # equal average lambda is not equal average power at finite signal.

@pytest.mark.parametrize('rank,alpha,lam',[(0,.1,1),(2,0,1),(2,.1,-1),(2,.1,np.inf)])
def test_invalid_fixed_power_parameters_fail(rank,alpha,lam):
    with pytest.raises(ValueError):power_interval(rank,alpha,lam)