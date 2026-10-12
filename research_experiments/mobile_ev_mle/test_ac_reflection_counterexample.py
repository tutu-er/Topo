from pathlib import Path

import numpy as np
import pytest

try:
    from .ac_reflection_counterexample import (reflect_stem, stem_voltage_squared,
        stem_invariant_jacobian, construct_example, gaussian_point_comparison,
        summarize_example, _apis)
except ImportError:
    from ac_reflection_counterexample import (reflect_stem, stem_voltage_squared,
        stem_invariant_jacobian, construct_example, gaussian_point_comparison,
        summarize_example, _apis)


ROOT = Path(__file__).resolve().parents[2]
if not (ROOT/'research_experiments/mobile_ev_mle/likelihood.py').exists():
    ROOT = Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo')


def test_reflection_is_distinct_positive_involution_and_preserves_invariants():
    theta=np.array([.005,.0035]);kappa=1/3
    reflected=reflect_stem(theta,kappa)
    assert np.allclose(reflected,[.0061,.0002],rtol=0,atol=1e-17)
    assert np.all(reflected>1e-6) and not np.allclose(theta,reflected)
    assert np.allclose(reflect_stem(reflected,kappa),theta,rtol=0,atol=1e-17)
    a=np.array([1.,kappa])
    assert a@reflected == pytest.approx(a@theta,abs=1e-17)
    assert reflected@reflected == pytest.approx(theta@theta,abs=5e-20)
    assert abs(np.linalg.det(stem_invariant_jacobian(theta,kappa)))>1e-3


def test_exact_scalar_voltage_identity_and_phase_difference():
    theta=np.array([.005,.0035]);reflected=reflect_stem(theta,1/3)
    p=np.array([.48,.58,.68]);q=p/3;v0=1.02
    direct=abs(v0-complex(*theta)*np.conj(p+1j*q)/v0)**2
    first=stem_voltage_squared(theta,p,q,v0)
    second=stem_voltage_squared(reflected,p,q,v0)
    assert np.max(abs(direct-first))<3e-16
    assert np.max(abs(first-second))<3e-16
    assert np.max(abs(v0-complex(*theta)*np.conj(p+1j*q)/v0-
        (v0-complex(*reflected)*np.conj(p+1j*q)/v0)))>1e-3


@pytest.mark.parametrize('topology',['star','cherry_01'])
def test_full_AC_all_meter_means_equal_shared_lines_sources_and_EV_once(topology):
    example=construct_example(topology,root=ROOT)
    a,b=example['first'],example['second'];summary=summarize_example(example)
    assert max(summary['channel_mean_max_absolute_differences'].values())<2e-13
    assert summary['original_target_master_error']<2e-13
    assert summary['max_AC_equation_residual']<2e-13
    assert summary['max_power_balance_residual']<2e-13
    assert summary['voltage_min']>.9 and summary['voltage_max']<1.1
    assert summary['both_source_paths_within_declared_bounds'] and summary['final_parameters_within_declared_line_bounds']
    assert not summary['reflected_stem_inside_single_30pct_trust']
    assert summary['root_X_relative_change']>.94
    assert np.array_equal(example['p_total'],example['base_p']+example['ev_kw']/20.)
    assert np.count_nonzero(example['ev_kw'][:example['history_points']])==0
    assert summary['EV_future_grid_energy_kwh']==6.
    assert len(example['r'])==example['candidate'].m and example['source_location']==1
    assert np.array_equal(example['r'][1:],example['reflected_r'][1:])
    assert np.array_equal(example['x'][1:],example['reflected_x'][1:])
    dz=complex(*example['reflected_theta'])-complex(*example['theta'])
    assert np.max(abs(example['reflected_source']-example['source']+dz*abs(example['target_master'])**2))<1e-16
    assert summary['all_node_complex_voltage_difference']>1e-3


@pytest.mark.parametrize('topology',['star','cherry_01'])
def test_nonzero_noise_point_likelihoods_equal_without_MLE_claim(topology):
    example=construct_example(topology,root=ROOT)
    result=gaussian_point_comparison(example,noise_seed=981007)
    assert result['whitened_noise_squared_norm']>0
    assert result['loss_at_feasible_points'][0]>0
    assert abs(result['loss_at_feasible_points'][0]-result['loss_at_feasible_points'][1])<1e-8
    assert result['loglik_absolute_difference']<1e-8
    assert not result['parameters_optimized'] and not result['either_point_claimed_global_MLE']


def test_complete_white_profile_rank_two_at_both_global_explanations():
    example=construct_example('cherry_01',root=ROOT)
    for geometry in example['geometries']:
        assert geometry['projected_stem_rank']==2
        assert min(geometry['projected_stem_singular_values'])>1.
        assert geometry['nuisance_columns']==8*example['points']
        assert geometry['all_other_line_parameters_fixed']
        assert not geometry['is_full_line_beta_profile']
        assert not geometry['local_profile_rank_is_global_identifiability']
    for geometry in example['historical_geometries']:
        assert geometry['projected_stem_rank']==2
        assert min(geometry['projected_stem_singular_values'])>1.
        assert geometry['nuisance_columns']==8*example['history_points']
        assert geometry['observation_scope']=='history only'
        assert not geometry['is_full_line_beta_profile']
    _,_,ac_predict,meter_model,mean_and_jacobian,_=_apis(ROOT)
    candidate=example['candidate'];r,x=example['r'],example['x'];s=example['source'];meter=meter_model(3)
    response=mean_and_jacobian(candidate,r,x,example['p_total'],example['q_total'],s.real,s.imag,1.,meter)
    for index,(which,column) in enumerate([('r',0),('x',candidate.m)]):
        rr,xx=r.copy(),x.copy();eps=1e-7
        target=rr if which=='r' else xx;target[0]+=eps
        plus=mean_and_jacobian(candidate,rr,xx,example['p_total'],example['q_total'],s.real,s.imag,1.,meter,with_jacobian=False)['mean']
        target[0]-=2*eps
        minus=mean_and_jacobian(candidate,rr,xx,example['p_total'],example['q_total'],s.real,s.imag,1.,meter,with_jacobian=False)['mean']
        finite=(plus-minus)/(2*eps)
        assert np.max(abs(finite-response['line_jacobian'][:,column]))<1e-4


def test_varying_master_power_factor_breaks_this_reflection_equivalence():
    theta=np.array([.005,.0035]);reflected=reflect_stem(theta,1/3)
    p=np.array([.48,.58,.68]);delta=np.array([.02,-.03,.04]);q=p/3+delta
    first=stem_voltage_squared(theta,p,q);second=stem_voltage_squared(reflected,p,q)
    assert np.max(abs((second-first)+2*delta*(reflected[1]-theta[1])))<3e-16
    assert np.max(abs(second-first))>1e-4
    example=construct_example('cherry_01',root=ROOT,varying_pf=True)
    summary=summarize_example(example)
    assert summary['channel_mean_max_absolute_differences']['v']>1e-5
    assert summary['both_source_paths_within_declared_bounds']
