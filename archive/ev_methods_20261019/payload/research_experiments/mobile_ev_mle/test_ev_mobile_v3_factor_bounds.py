from itertools import product
import numpy as np
from .ev_mobile_v3_domain import fractional_absolute_bound,factor_bound_provider,tighten_operation_bounds
from .ev_mobile_v3_cases import service_domain


def test_fractional_bound_exhaustive_tiny_controls():
    weights=np.array([.5,2.,1.]);caps=np.ones(3);bound,a=fractional_absolute_bound(weights,caps,2.)
    assert bound==3. and np.sum(a)==2.
    actual=max(weights@np.asarray(x) for x in product((0,.5,1.),repeat=3) if sum(x)<=2.)
    assert bound==actual


def test_single_site_bound_covers_all_tiny_actions():
    beta=np.array([[2.,-1.],[-3.,4.],[.2,.4]])
    bound,_=fractional_absolute_bound(np.max(abs(beta),axis=1),np.ones(3),2.)
    actions=[np.zeros(2),np.array([1.,0]),np.array([-1.,0]),np.array([0,1.]),np.array([0,-1.])]
    maximum=0.
    for states in product(actions,repeat=3):
        u=np.array(states)
        if np.sum(abs(u))<=2.:maximum=max(maximum,abs(np.sum(beta*u)))
    assert abs(maximum-bound)<1e-12


def test_provider_is_tighter_than_independent_coordinate_box_and_safe():
    domain=service_domain(12,4,mobile=True);op=tighten_operation_bounds(domain)
    beta=np.ones(48);provider=factor_bound_provider(domain,op)
    result=provider(None,0,dict(coefficient=beta,intercept=2.))
    assert result['lower']<=2.-30 and result['upper']>=2.+30
    box=float(np.sum(np.asarray(op['control_upper'])))
    assert result['upper']-2.<box
    rng=np.random.default_rng(5)
    caps=result['metadata']['slot_absolute_caps_kw']
    for _ in range(100):
        u=np.zeros((12,4));site=rng.integers(4,size=12)
        amp=rng.uniform(-1,1,12)*caps
        amp*=min(1.,30./max(1e-15,np.sum(abs(amp))))
        u[np.arange(12),site]=amp
        value=2.+np.sum(u)
        assert result['lower']<=value<=result['upper']
