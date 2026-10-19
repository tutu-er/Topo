"""Independent physical derivatives and an electrically irrelevant site."""
import numpy as np
import pytest
from .. import likelihood as lk
from ..candidates import RadialCandidate
from .context import output,power_jacobian
from .coefficients import build_tasks_from_responses


def test_kw_derivatives_match_full_ac_finite_difference():
    c=RadialCandidate.from_supports('cherry',3,((0,1),))
    r=np.linspace(.001,.003,c.m); x=.25*r
    p=np.full((2,3),.12);q=.3*p
    physical=lk.ac_predict(c,r,x,p,q,.1,.04)
    J=power_jacobian(c,physical,20.)
    for t,j in [(0,0),(1,2)]:
        step=np.zeros_like(p);step[t,j]=.001/20.
        numerical=(output(lk.ac_predict(c,r,x,p+step,q,.1,.04))
                  -output(lk.ac_predict(c,r,x,p-step,q,.1,.04)))/.002
        target=np.zeros_like(numerical);target[t]=J[t,:,j]
        np.testing.assert_allclose(numerical,target,rtol=3e-6,atol=2e-9)


def test_complete_ac_irrelevant_root_branch_has_zero_g():
    # Zero shared stem means the two branches meet only at a fixed-voltage root.
    # Known parameters here are a mechanism diagnostic, never evaluation inputs.
    a=RadialCandidate.from_supports('star',3,())
    b=RadialCandidate.from_supports('cherry01',3,((0,1),))
    p=np.zeros((2,3));p[:,2]=.12;q=.3*p
    responses={}
    for c in (a,b):
        supports=[tuple(s) for s in c.supports]
        r=np.array([0. if len(s)==3 else .002 for s in supports]);x=.25*r
        physical=lk.ac_predict(c,r,x,p,q,0.,0.)
        responses[c.name]=dict(mean=output(physical),jacobian=power_jacobian(c,physical,20.))
    g=responses[a.name]['jacobian']-responses[b.name]['jacobian']
    np.testing.assert_allclose(g[:,:,2],0.,atol=1e-13)
    assert np.linalg.norm(g[:,:,0])>1e-6
    task=build_tasks_from_responses(responses,np.ones((2,5)),include_intercept=False)[0]
    irrelevant=np.zeros((2,3));irrelevant[:,2]=5.
    useful=np.zeros((2,3));useful[:,0]=5.
    assert task.value(irrelevant)<1e-22
    assert task.value(useful)>1e-8
    assert sum(irrelevant.ravel()**2)==sum(useful.ravel()**2)
