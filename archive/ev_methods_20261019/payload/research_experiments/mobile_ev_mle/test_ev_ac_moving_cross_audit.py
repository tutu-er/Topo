"""Independent channel layout and same-witness error bookkeeping checks."""
import numpy as np
import pytest
from .ev_ac_moving_cross import channel_slices, channel_norms
from .ev_ac_moving_cross_audit import compare_errors


def test_channel_layout_keeps_root_and_optional_source_separate():
    channels=('p','q','v','master_p','master_q','source_p','source_q')
    slices=channel_slices(channels,2,3)
    assert slices==dict(p=slice(0,6),q=slice(6,12),v=slice(12,18),
        master_p=slice(18,20),master_q=slice(20,22),source_p=slice(22,24),source_q=slice(24,26))
    norms=channel_norms(np.ones(26),channels,2,3)
    assert norms['v']['norm']==pytest.approx(np.sqrt(6))
    assert norms['source_p']['rows']==2


def test_cost_error_and_RMS_gate_use_same_white_vector():
    channels=('p','q','v','master_p','master_q'); n=1; h=2; t=2
    hrows=10; exact=np.full(20,2.); prediction=exact.copy(); prediction[14:16]+=.3
    metrics=compare_errors(exact,dict(proxy=prediction),hrows,h,t,n,channels)['proxy']
    assert metrics['future_channel_RMS']['v']==pytest.approx(.3)
    assert metrics['future_channel_RMS_gate_pass'] is False
    assert metrics['history_error_norm']==0.
    error=prediction-exact
    assert metrics['sse_error']==pytest.approx(2*exact@error+error@error)


def test_bad_channel_vector_length_is_rejected():
    with pytest.raises(ValueError): channel_norms(np.zeros(7),('p','q','v'),2,3)
