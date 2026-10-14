"""Saved-record domain diagnostics; no generation, selection or AC fitting."""
from copy import deepcopy
import pytest
from . import ev_original_domain_contrast as d
from .test_ev_original_analysis import bundle


def pair(short=False):
    _,trials,_,_=bundle();asset=deepcopy(trials[0]);asset['short_edge']=short;asset['source_interface']=1
    asset['seeds']={'asset':1,'background':2,'source':3,'noise':4,'policy':5}
    asset['history']={'p':[[.12]*3],'v':[[1.]*3]};asset['measurement_models']={'fixed':'identical'}
    asset['config'].update(asset_bounds=dict(r_bounds=[.00065,.012],x_bounds=[.00025,.0016]),device={'power_kw':6.})
    family=[[0,1]] if short else [];supports=[[0,1,2]]+family+[[0],[1],[2]]
    r=[.005]*len(supports);x=[.001]*len(supports)
    if short:r[1]=.00005;x[1]=.00002
    asset['truth'].update(family=family,r=r,x=x,p=[[.12]*3]*2,q=[[.04]*3]*2,source_p=[.1]*2,source_q=[.03]*2,
        root_v=[1.]*2,white={'v':[[0.]*3]*2},assets=[{'support':s} for s in supports])
    asset['selection']['context']={'line_bounds':asset['config']['asset_bounds']}
    numeric=deepcopy(asset);numeric['domain']='numeric';numeric['selection']['context']['line_bounds']=d.NUMERIC_BOUNDS
    return asset,numeric


def change_margin(trial,method,value):
    row=next(r for r in trial['rows'] if r['method']==method)
    for fit in row['fits']:fit['loss']=1. if fit['name']=='star' else 1.+value


def test_same_history_domain_effect_and_within_domain_EV_increment_are_separate():
    asset,numeric=pair();change_margin(asset,'constant',1.);change_margin(asset,'charge_fixed',2.)
    change_margin(numeric,'constant',.5);change_margin(numeric,'charge_fixed',2.5)
    result=d.compare_pair(asset,numeric);row=next(r for r in result['paired_rows'] if r['method']=='charge_fixed')
    assert result['audit']['all_immutable_inputs_equal'] and len(result['method_rows'])==18
    assert row['asset_actual_M_minus_constant']==pytest.approx(1.)
    assert row['numeric_actual_M_minus_constant']==pytest.approx(2.)
    assert row['asset_minus_numeric_actual_M_minus_constant']==pytest.approx(-1.)


@pytest.mark.parametrize('field',['truth','history','source_sha256','config','measurement_models'])
def test_truth_white_device_meter_or_source_differences_reject_pair(field):
    asset,numeric=pair();numeric[field]=deepcopy(numeric[field]);numeric[field]['changed']=True
    with pytest.raises(ValueError,match='immutable inputs differ'):d.validate_pair(asset,numeric)


def test_both_missing_immutable_fields_do_not_count_as_verified_equality():
    asset,numeric=pair();del asset['history'];del numeric['history']
    with pytest.raises(ValueError,match='inputs missing'):d.validate_pair(asset,numeric)


def test_short_edge_structure_is_retained_but_true_RX_is_outside_asset_box():
    asset,numeric=pair(short=True);a,n=map(d.parameter_coverage,(asset,numeric))
    assert a['structural_candidate_covered'] and n['structural_candidate_covered']
    assert not a['true_RX_in_domain'] and a['domain_status']=='outside_RX_domain'
    assert a['outside_edge_count']==1 and n['true_RX_in_domain']
    assert a['edges'][1]['R_violation']==pytest.approx(.0006)
    described=d.describe_domain(asset,'short_edge_domain',.01)[1]
    assert all(r['domain_status']=='outside_RX_domain' and 'outside' in r['margin_scope'] for r in described)


def test_failed_method_and_candidate_stay_in_method_and_nested_audit_denominators():
    asset,numeric=pair();row=next(r for r in numeric['rows'] if r['method']=='charge_fixed')
    row.update(status='fit_failure',fits=[],failures={'star':'failed'})
    result=d.compare_pair(asset,numeric)
    assert len(result['method_rows'])==18 and len(result['paired_rows'])==9
    compared=next(r for r in result['paired_rows'] if r['method']=='charge_fixed')
    assert compared['numeric_actual_M'] is None and compared['numeric_actual_M_minus_constant'] is None
    audit=[r for r in result['nested_cost_audits'] if r['method']=='charge_fixed']
    assert len(audit)==4 and all(not r['both_costs_available'] for r in audit)


def test_wide_domain_local_cost_order_violation_is_reported_as_computational():
    asset,numeric=pair();change_margin(numeric,'constant',3.)
    result=d.compare_pair(asset,numeric)
    audit=[r for r in result['nested_cost_audits'] if r['method']=='constant']
    assert len(audit)==4 and sum(r['local_attainment_order_violation'] for r in audit)==3
    assert all('local attained' in r['interpretation'] for r in audit)


def test_recorded_proxy_bounds_must_match_declared_fit_domain():
    asset,numeric=pair();numeric['selection']['context']['line_bounds']={'r_bounds':[.01,.1],'x_bounds':[1e-6,.1]}
    with pytest.raises(ValueError,match='saved context domain differs'):d.parameter_coverage(numeric)


def test_numeric_unresolved_is_not_a_statistical_test_and_zero_error_is_not_imputed():
    asset,_=pair();change_margin(asset,'constant',.005)
    coverage,rows,edges=d.describe_domain(asset,'pilot_domain',.01)
    constant=next(r for r in rows if r['method']=='constant')
    assert constant['numerical_actual_unresolved'] and constant['predicted_unresolved']==12
    assert all(e['R_relative_error'] is None for e in edges) # absent estimates, not zero errors.
