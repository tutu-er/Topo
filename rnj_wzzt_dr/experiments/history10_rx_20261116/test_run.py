"""Controlled fixed-support recovery and bookkeeping, without a new AC batch."""
import numpy as np
import pytest

from ...observations import Observation
from ...identification.optimization import QPFitter, response_matrices
from .data import indices, observations, slice_observations, standard_noise, POINTS
from .run import training_bounds


def test_contiguous_indices_are_not_misread_as_two_endpoints():
    world=indices()
    assert training_bounds(world,2)==(576,768)
    assert training_bounds(world,4)==(384,768)
    assert training_bounds(world,8)==(0,768)
    world['train_indices']['2']=[1,3]
    with pytest.raises(ValueError):training_bounds(world,2)


def test_exact_linear_fixed_support_recovers_r_and_x_without_regularization():
    rng=np.random.default_rng(482)
    p=rng.uniform(.01,.1,size=(192,2));q=rng.normal(0,.02,size=p.shape)
    family=((0,),(1,),(0,1))
    r=np.array([.03,.07,.02]);x=np.array([.04,.05,.01])
    z=np.array([[1.,0.],[0.,1.],[1.,1.]])
    R=(z.T*r)@z;X=(z.T*x)@z
    obs=Observation(p,q,p@R.T+q@X.T,(101,102))
    fit=QPFitter(obs,reference_mse=1e-6).fit(family)
    fitted=response_matrices(fit,2)
    np.testing.assert_allclose(fitted[0],R,atol=2e-12,rtol=0)
    np.testing.assert_allclose(fitted[1],X,atol=2e-12,rtol=0)


def test_future_changes_do_not_modify_either_training_observation():
    n=2;p=np.ones((POINTS,n))*.02;q=p*.2
    clean=dict(P=p,Q=q,V=np.full_like(p,1.01),root=np.full(POINTS,1.02),labels=[101,102])
    noise=standard_noise(n,'soumalas11',0)
    original=observations(clean,noise)
    changed={k:np.copy(v) for k,v in clean.items()}
    changed['P'][768:]*=10;changed['V'][768:]*=.99
    other=observations(changed,noise)
    for days in (2,8):
        a=slice_observations(original,indices(),days)['train']
        b=slice_observations(other,indices(),days)['train']
        assert np.array_equal(a.p,b.p)
        assert np.array_equal(a.y,b.y)
