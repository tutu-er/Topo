import numpy as np
from .geometry import expand


def test_hourly_expansion_equals_full_multichannel_response():
    rng=np.random.default_rng(20261115)
    H=rng.normal(size=(3,6));X=rng.normal(size=(8,6));B=rng.normal(size=(6,4));u=rng.normal(size=(2,4))
    q,a,c=expand(H,X,B,2,4,.03)
    expanded=np.einsum('hr,hrs,hs->',u,q,u)+np.sum(a*u)+c
    response=(X+np.repeat(u@B.T,4,axis=0))@H.T
    np.testing.assert_allclose(expanded,np.sum(response**2)/.06,rtol=1e-12)


def test_error_decrease_and_distance_increase_can_still_flip_score():
    Hc=np.array([[2.,2.]]);Hj=np.array([[1.,.9]]);Hstar=np.array([[3.,1.1]])
    values=[]
    for x in (np.array([1.,0.]),np.array([0.,1.])):
        e=float(np.sum(((Hstar-Hc)@x)**2)/2);J=float(np.sum(((Hc-Hj)@x)**2)/2)
        margin=float(np.sum(((Hstar-Hj)@x)**2)-np.sum(((Hstar-Hc)@x)**2))
        values.append((e,J,margin))
    assert values[1][0]<values[0][0] and values[1][1]>values[0][1]
    assert values[0][2]>0 and values[1][2]<0
