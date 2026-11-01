import math
import numpy as np
import pytest
from .driver import cp_interval,exact_margin,rotation_curve

def test_rotation_zero_energy_whole_day_and_matching_cost():
    u,c=rotation_curve([],0.2744976172777763,1)
    assert u.shape==(2,96) and np.all(u!=0.)
    assert np.max(abs(u.sum(axis=1)))<1e-12
    assert np.max(abs(u[0]+u[1]))==0.
    fee=float(c@(.25*np.maximum(-u,0.).sum(axis=1)))
    assert fee*1.2==pytest.approx(.2744976172777763)
    other,_=rotation_curve([],0.2744976172777763,-1)
    assert other==pytest.approx(-u)

def test_rotation_rejects_fake_phase():
    with pytest.raises(ValueError):rotation_curve([],1.,0)

def test_binomial_interval_denominator_is_days():
    assert cp_interval(12,12)==pytest.approx([.025**(1/12),1.])
    assert cp_interval(0,12)==pytest.approx([0.,1-.025**(1/12)])

def test_exact_margin_is_independent_of_selected_reference_switch():
    d={'loss_rows':[dict(candidate_id='coarse',historical_validation_mae=.0012,new_validation_mae=.0011),
        dict(candidate_id='refined',historical_validation_mae=.001,new_validation_mae=.001)]}
    assert exact_margin(d)==pytest.approx(.6*(.0012-1.03*.001)+.4*(.0011-1.02*.001)-1e-12)
