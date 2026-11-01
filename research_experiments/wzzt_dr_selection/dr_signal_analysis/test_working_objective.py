from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_working_objective import finite_stats,replay_draws,gains_for_samples


def test_lower90_attainment_uses_900th_largest_not_90th_percentile():
    x=np.arange(1000.)
    s=finite_stats(x,100.)
    assert s['sample_90pct_lower_gain']==100.
    assert s['standard_p10']==pytest.approx(99.9)
    assert s['target_count']==900
    assert s['required_scenarios']==900


def test_draw_stream_interleaves_day_then_factors_exactly():
    day,z=replay_draws(789,4)
    rng=np.random.default_rng(789)
    for i in range(4):
        assert day[i]==rng.integers(6)
        assert np.array_equal(z[i],rng.uniform(-1,1,2))


def test_batch_paired_gain_matches_direct_nonlinear_absolute_arithmetic():
    bases=np.array([[1.,-1.],[2.,1.]])
    col=np.array([[1.,2.],[-2.,1.]])
    mean=np.array([.5,.2]);D=np.array([[.1,.2],[.1,-.2]])
    days=np.array([0,1]);z=np.array([[1.,-1.],[-1.,1.]])
    q=gains_for_samples(bases,col,mean,D,days,z)
    for i in range(2):
        u=mean*(1+D@z[i]);expect=(np.abs(bases[days[i]]+col@u)-np.abs(bases[days[i]])).mean()
        assert q[i]==pytest.approx(expect)


def test_zero_response_gain_is_exactly_zero():
    bases=np.array([[1.,-1.]])
    q=gains_for_samples(bases,np.array([[2.],[3.]]),np.array([0.]),np.zeros((1,2)),np.array([0,0]),np.ones((2,2)))
    assert np.array_equal(q,np.zeros(2))
