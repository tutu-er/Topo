import numpy as np
from research_experiments.mobile_ev_mle.ac_history_refresh.evidence.fixed_response_audit import voltage_errors


def test_same_background_cancels_and_physical_units_are_retained():
    part=dict(indices=np.array([1,2]),sigma=np.array([.01,.02]),physical_scale=400.)
    # Same background bias cancels in the response, but remains in absolute V.
    result=voltage_errors(np.array([9.,4.,5.]),np.array([7.,2.,3.]),
        np.array([0.,2.,3.]),np.array([0.,0.,1.]),part)
    assert result['increment_error_V']==0
    assert result['relative_increment_error']==0
    assert np.isclose(result['absolute_voltage_error_V'],np.sqrt((8**2+16**2)/2))


def test_zero_signal_is_not_given_an_invented_relative_error():
    result=voltage_errors(np.ones(2),np.zeros(2),np.zeros(2),np.zeros(2),
        dict(indices=np.array([0,1]),sigma=np.ones(2),physical_scale=1.))
    assert result['relative_increment_error'] is None
    assert result['increment_error_V']==1.
