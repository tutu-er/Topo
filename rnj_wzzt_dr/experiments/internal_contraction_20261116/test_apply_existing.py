from unittest.mock import patch

import numpy as np
import pytest

from ...observations import Observation
from .apply_existing import CacheOnlyFitter


def test_missing_cache_cannot_start_an_optimizer():
    obs = Observation(np.eye(2), np.zeros((2,2)), np.eye(2), (1,2))
    fitter = CacheOnlyFitter(obs)
    with patch('rnj_wzzt_dr.identification.optimization.lsq_linear', side_effect=AssertionError('No new solve allowed')):
        with pytest.raises(RuntimeError, match='new QP forbidden'):
            fitter.fit([(0,), (1,)])
    assert fitter.records == []


def test_cache_replay_returns_the_sealed_point_without_optimization():
    obs = Observation(np.eye(2), np.zeros((2,2)), np.eye(2), (1,2))
    fitter = CacheOnlyFitter(obs)
    point = dict(supports=[[0],[1]], r=[1.,1.], x=[0.,0.])
    fitter.cache[((0,),(1,))] = point
    with patch('rnj_wzzt_dr.identification.optimization.lsq_linear', side_effect=AssertionError('No new solve allowed')):
        assert fitter.fit([(1,), (0,)]) is point
    assert fitter.records == []
