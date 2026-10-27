from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_dr_selection.end_to_end.physical_case import (build_reference_case, reference_terminal_profiles,
                           simulate_observations, synthetic_q_from_reference_pf,
                           effective_truth_supports, _physical_checks)
from rnj_wzzt.data.paper_style_terminal_lv import build_paper_style_terminal_lv_case


def loads(net, count=3, value=4.0):
    columns=net.buses.loc[net.buses.bus_type.eq('observed_terminal'),'bus_id'].astype(int).tolist()
    return pd.DataFrame(value,index=pd.RangeIndex(count),columns=columns)


def test_reference_case_preserves_graph_and_removes_legacy_scale():
    net=build_reference_case();old=build_paper_style_terminal_lv_case()
    assert len(net.buses)==22 and len(net.branches)==21
    assert len(net.hidden_buses())==6 and len(net.get_leaf_buses())==15
    assert net.metadata['impedance_scale']==1.0
    np.testing.assert_allclose(net.branches[['r_ohm','x_ohm']],old.branches[['r_ohm','x_ohm']]/5)
    assert net.closed_edges()==old.closed_edges()
    assert net.check_terminal_load_only()['violations']==[]
    supports = effective_truth_supports(net)
    assert len(supports) == 21 and [101] in supports and [115] in supports


def test_external_units_zero_load_and_observed_truth_separation():
    net=build_reference_case();p=loads(net,value=0.0)
    root=pd.Series([1.01,1.02,1.03],index=p.index)
    out=simulate_observations(net,p,p,seed=5,root_true=root,pq_noise_rel=0,v_noise_rel=0,root_meter_noise_rel=0)
    assert out['checks']['usable_ac_solution']
    assert out['checks']['voltage_limits_pass']
    assert out['checks']['physical_outputs_finite']
    assert 'truth_supports' not in out['observed']
    assert out['truth']['truth_supports'] == effective_truth_supports(net)
    np.testing.assert_allclose(out['observed']['drop_target'],0,atol=1e-15)
    assert not any('true' in key.lower() or key=='ac' for key in out['observed'])
    p=loads(net,value=4.0);q=p*.3
    out=simulate_observations(net,p,q,seed=5,root_true=root,pq_noise_rel=0,v_noise_rel=0,root_meter_noise_rel=0)
    np.testing.assert_allclose(out['observed']['P_terminal'],.02)
    np.testing.assert_allclose(out['observed']['Q_terminal'],.006)
    assert out['checks']['max_system_p_balance_residual_pu']<1e-8
    assert out['checks']['max_nodal_q_residual_pu']<1e-8
    assert out['checks']['max_distflow_identity_error_pu2']<1e-12
    np.testing.assert_allclose(out['observed']['drop_target'],root.to_numpy()[:,None]**2-out['observed']['V_terminal'].to_numpy()**2)


def test_meter_randomness_does_not_change_physical_inputs_with_fixed_root():
    net=build_reference_case();p=loads(net,count=24);q=p*.3;root=pd.Series(1.02,index=p.index)
    a=simulate_observations(net,p,q,seed=7,root_true=root)
    b=simulate_observations(net,p,q,seed=8,root_true=root)
    np.testing.assert_array_equal(a['truth']['V_terminal_true'],b['truth']['V_terminal_true'])
    assert not np.array_equal(a['observed']['P_terminal'],b['observed']['P_terminal'])
    c=simulate_observations(net,p,q,seed=7,root_true=root)
    np.testing.assert_array_equal(a['observed']['V_terminal'],c['observed']['V_terminal'])


def test_nonconvergence_retained_and_blocks_estimator_payload():
    net=build_reference_case();p=loads(net,value=8.0)
    out=simulate_observations(net,p,p*.4,seed=1,max_iter=1)
    assert out['observed'] is None
    assert out['checks']['nonconverged_samples']==len(p)
    assert len(out['truth']['ac']['converged'])==len(p)


def test_bad_terminal_and_time_axes_rejected():
    net=build_reference_case();p=loads(net)
    with pytest.raises(ValueError,match='terminal order'):
        simulate_observations(net,p.iloc[:,:-1],p.iloc[:,:-1],seed=1)
    q=p.copy();q.index=[1,2,3]
    with pytest.raises(ValueError,match='identical timestamps'):
        simulate_observations(net,p,q,seed=1)
    q=p.copy();q.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        simulate_observations(net,p,q,seed=1)


def test_reference_hourly_averaging_preserves_daily_energy_and_synthetic_q():
    p96,q96,_=reference_terminal_profiles(hourly=False)
    p24,q24,meta=reference_terminal_profiles(hourly=True)
    np.testing.assert_allclose(p96.sum()*0.25,p24.sum(),rtol=1e-13)
    np.testing.assert_allclose(q96.sum()*0.25,q24.sum(),rtol=1e-13)
    assert meta['output_count']==24 and meta['measured_load'] is False
    p=loads(build_reference_case(),count=48)
    q=synthetic_q_from_reference_pf(p)
    assert q.index.equals(p.index) and list(q.columns)==list(p.columns)
    assert np.isfinite(q).all().all() and (q>0).all().all()
    np.testing.assert_allclose(q.iloc[:24],q.iloc[24:])


def test_synthetic_pf_uses_datetime_hour_instead_of_starting_at_midnight():
    net=build_reference_case();p=loads(net,count=24)
    full=synthetic_q_from_reference_pf(p)
    subset=p.iloc[:4].copy()
    subset.index=pd.date_range('2026-10-27 12:00',periods=4,freq='h')
    q=synthetic_q_from_reference_pf(subset)
    np.testing.assert_allclose(q.to_numpy(),full.iloc[12:16].to_numpy())


def test_independent_balance_check_detects_tampered_receiving_branch_flow():
    """A converged flag and plausible V cannot hide an inconsistent power flow."""
    net=build_reference_case();p=loads(net,count=4);q=p*.3
    result=simulate_observations(net,p,q,seed=9)
    assert result['checks']['electrical_equations_pass']
    altered=deepcopy(result['truth']['ac'])
    label=altered['metadata']['branch_edges'][0]['label']
    altered['branch_p_to_pu'].loc[0,label] += .01
    assert altered['converged'].all()
    check=_physical_checks(net,result['truth']['P_true'],result['truth']['Q_true'],
                           altered,(.9,1.1),1e-8)
    assert check['converged_samples']==len(p)
    assert check['physical_outputs_finite'] and check['voltage_limits_pass']
    assert check['max_nodal_p_residual_pu'] > .009
    assert not check['electrical_equations_pass']
