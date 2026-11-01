"""Pure shape and convex-distance checks, without case or future data."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from design import contrast_shape,warm_quotes
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem,evaluate_plan


def test_guided_shape_is_energy_neutral_with_two_kwh_prefix_limit():
    shape=contrast_shape(np.linspace(-1,1,96))
    assert np.sum(shape)==0
    assert np.count_nonzero(shape==-1)==8
    assert np.count_nonzero(shape==1)==8
    assert .25*np.abs(shape).sum()/2==2
    assert np.max(np.abs(.25*np.cumsum(shape)))<=2


def test_guided_shape_exactly_maximizes_linear_equal_duration_action():
    rng=np.random.default_rng(731)
    g=rng.normal(size=96)
    shape=contrast_shape(g)
    optimum=np.sort(g)[-8:].sum()-np.sort(g)[:8].sum()
    assert g@shape==pytest.approx(optimum)
    assert g@shape>=0


def test_true_norm_gain_has_convex_subgradient_lower_bound():
    rng=np.random.default_rng(922)
    b=rng.normal(size=(96,3));column=rng.normal(size=3)
    marginal=np.sign(b)@column
    u=contrast_shape(marginal)
    for a in (0,.2,1):
        gain=np.abs(b+a*u[:,None]*column).sum()-np.abs(b).sum()
        assert gain+1e-10>=a*(marginal@u)


def test_constant_sign_energy_neutral_change_can_give_zero_distance_gain():
    # DR creates a nonzero voltage difference but adds zero to the norm over
    # the day. A norm of only the DR response is not the objective.
    b=np.ones((96,3))*10
    u=contrast_shape(np.ones(96))
    effect=u[:,None]*np.array([.1,.2,.3])
    assert np.abs(effect).sum()>0
    assert np.abs(b+effect).sum()-np.abs(b).sum()==pytest.approx(0)


@pytest.mark.parametrize('shape',([1,2],[np.nan]*96,[1]*95))
def test_invalid_guidance_is_rejected(shape):
    with pytest.raises(ValueError):contrast_shape(shape)


def test_warm_quotes_keep_same_continuous_domain_and_allow_single_point():
    problem=PriceProblem(mu0=[0.],background_factors=np.zeros((1,1)),profiles=np.zeros((1,10)),
        response_factors=np.zeros((10,1)),elasticities=np.ones(10),reduction_hours=np.ones(10),
        price_upper=np.ones(10),actual_amplitude_upper=np.ones(10),benefit_directions=[[0.]],
        benefit_offsets=[0.],risk_allocations=[.1],budget=1.,exclusive_groups=[list(range(5)),list(range(5,10))])
    seeds=warm_quotes(problem)
    assert len(seeds)==451
    assert not np.any(seeds[0])
    assert any(np.count_nonzero(s)==1 for s in seeds)
    for seed in seeds:
        assert evaluate_plan(problem,seed)['numerical_feasible']
