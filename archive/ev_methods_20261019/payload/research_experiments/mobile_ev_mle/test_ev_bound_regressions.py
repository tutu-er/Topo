"""Regression audits for distinct controls, frozen data and strict feasible witnesses."""
import numpy as np
import pytest
from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair,adaptive_search,upper_master
from research_experiments.mobile_ev_mle.ev_operations_milp import EVConfig

def test_frozen_arrays_own_data_and_reject_mutation():
    source=np.ones((1,1))
    pair=FrozenPair('freeze',np.zeros(1),source,np.ones((1,1)),np.zeros((0,1)),np.array([-1.]),np.array([1.]))
    initial=pair.fingerprint();source[:]=99
    assert pair.B[0,0]==1 and pair.fingerprint()==initial
    for array in (pair.c,pair.B,pair.A,pair.F,pair.lower,pair.upper,pair.scale):
        if array.size:
            with pytest.raises(ValueError):array.flat[0]=0
        assert not array.flags.writeable

def test_outside_witness_even_rounding_size_cannot_supply_global_bound():
    pair=FrozenPair('strict',np.zeros(1),np.ones((1,1)),np.ones((1,1)),np.zeros((0,1)),np.array([-1.]),np.array([1.]))
    cfg=EVConfig(n_slots=1,n_sites=1,net_grid_energy_kwh=.75)
    with pytest.raises(ValueError,match='infeasible'):
        upper_master(cfg,[pair],[[np.array([np.nextafter(1.,2.)])]])

def test_close_distinct_controls_are_independently_evaluated():
    pair=FrozenPair('amplified',np.zeros(1),np.array([[1.e10]]),np.zeros((1,0)),np.zeros((0,0)),np.zeros(0),np.zeros(0))
    cfg=EVConfig(n_slots=1,n_sites=1,net_grid_energy_kwh=.75)
    result=adaptive_search(cfg,[pair],initial_us=([3.+1e-10],[3.+2e-10]),max_iterations=0)
    rows=[r for r in result['records'] if r['origin']=='declared_initial_vector']
    assert len(rows)==2
    assert rows[0]['u_kw']!=rows[1]['u_kw']
    assert rows[0]['q_upper']!=rows[1]['q_upper']
