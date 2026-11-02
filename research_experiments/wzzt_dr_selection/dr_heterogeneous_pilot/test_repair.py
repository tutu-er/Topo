import numpy as np
import pytest
from test_pilot import dev
from planner import Probability
from planning_repair import MeanDifference


def test_same_L1_mean_gradient_through_both_controllers():
    d=dev();rng=np.random.default_rng(51);m=192
    packet=dict(mu_C=rng.normal(size=m),mu_F=rng.normal(size=m)*.8,
        L_C=rng.normal(size=(m,3)),L_F=rng.normal(size=(m,3))*.8,
        effect_C=rng.normal(size=(2,d.R))*.2,effect_F=rng.normal(size=(2,d.R))*.1,threshold=.25)
    p=Probability(packet,d,4,22,23);mean=MeanDifference(p)
    x=rng.normal(size=(d.R,24))*.05;out=mean.evaluate(x);v=rng.normal(size=x.shape);eps=1e-6
    num=(mean.evaluate(x+eps*v)['value']-mean.evaluate(x-eps*v)['value'])/(2*eps)
    assert out['gradient']@v.ravel()==pytest.approx(num,rel=5e-4,abs=3e-6)


def test_same_L1_mean_matches_gaussian_quadrature():
    d=dev();rng=np.random.default_rng(52);m=192
    packet=dict(mu_C=rng.normal(size=m),mu_F=rng.normal(size=m)*.8,
        L_C=rng.normal(size=(m,3)),L_F=rng.normal(size=(m,3))*.8,
        effect_C=np.zeros((2,d.R)),effect_F=np.zeros((2,d.R)),threshold=.25)
    p=Probability(packet,d,3,22,23);out=MeanDifference(p).evaluate(np.zeros((d.R,24)))
    from scipy.integrate import quad
    reference=0.
    for a in p.a0:
        roots=-a[p.b!=0]/p.b[p.b!=0];roots=np.unique(roots[(roots>-10)&(roots<10)])
        val,_=quad(lambda z:float(np.abs(a+p.b*z)@p.w)*np.exp(-.5*z*z)/np.sqrt(2*np.pi),-10,10,epsabs=1e-6,limit=1000,points=roots)
        reference+=val/p.samples
    assert out['value']==pytest.approx(reference,abs=2e-6)
