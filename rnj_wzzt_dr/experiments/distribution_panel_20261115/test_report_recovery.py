"""Administrative recovery must not change comparison or cost denominators."""
from collections import Counter

import pytest

from . import report


def row(input_id,forecast,policy,repeat=0,correct=True):
    zero_cost={key:0. for key in ('total_yuan','activation_yuan','availability_yuan',
        'active_activation_yuan','reactive_activation_yuan','reserved_P_kw','reserved_Q_kvar')}
    return dict(id=input_id,case=input_id,date_cluster='2024-01-04',season='winter',
        forecast=forecast,repeat=repeat,strategy=policy,audit_status='passed',original_status='complete',
        evaluation=dict(native_correct=correct,native_full_support_exact=correct,wrong_high=False,true_high=correct),
        voltage_screen_passed=True,recovered_and_voltage_screen_passed=correct,
        planned_cost=zero_cost,settled_cost=zero_cost,requested=[],delivered=[],
        electrical=dict(active_loss_kwh=0.,reactive_loss_kvarh=0.,root_signed_kwh=0.,
            root_import_kwh=0.,root_export_kwh=0.,maximum_branch_current_A=0.,
            all_bus_voltage=dict(minimum=1.,maximum=1.)))


def quote(input_id='input00',forecast=0,policy='normal_fixed',recovery=None):
    return dict(id=input_id,forecast=forecast,policy=policy,new_gate_feasible=True,
        nonzero_executed=True,optimizer_invoked=False,zero_action_sufficient=False,
        anchor_fallback=False,zero_fallback=False,native_status=None,status='saved_seed_after_interruption',
        source='saved_seed_after_interruption',relative_gap=.9,cost_yuan=30.,lower_bound_yuan=3.,
        charged_total_seconds=751.2,executed_cost=dict(activation_yuan=30.),recovery=recovery)


def test_four_interrupted_forecasts_exclude_all_eight_policies_together():
    # 46 inputs x 2 forecasts; 8 recovered quotes affect exactly 4 forecasts.
    records=[row(f'input{i:02d}',f,p,k) for i in range(46) for f in range(2)
        for k in range(2) for p in report.POLICIES]
    quotes=[quote('input00',f,recovery={'kind':'interrupted_search'}) for f in range(2)]
    quotes += [quote('input01',f,p,recovery={'kind':'action_lost'}) for f in range(2) for p in report.MODES]
    marked,forecasts=report.interruption_panel(records,quotes)
    assert len(forecasts)==92 and sum(f['interruption_affected'] for f in forecasts)==4
    assert all('interruption_affected' not in r for r in records)  # do not mutate the audit source
    assert len(marked)==len(records)  # primary result retains all interrupted rows
    affected=[r for r in marked if r['interruption_affected']]
    unaffected=[r for r in marked if not r['interruption_affected']]
    assert Counter(r['strategy'] for r in affected)=={p:8 for p in report.POLICIES}
    assert Counter(r['strategy'] for r in unaffected)=={p:176 for p in report.POLICIES}
    for p in report.POLICIES:
        assert report.paired_comparison(marked,p,'no_dfs')['n']==184
        assert report.paired_comparison(unaffected,p,'no_dfs')['n']==176


def test_subsets_keep_both_adverse_and_favorable_interrupted_changes():
    records=[]
    for f in range(3):
        for p in report.POLICIES:
            # Interrupted forecast0: method gains. Forecast1: method loses.
            # Uninterrupted forecast2: both recover. Neither outcome chooses exclusion.
            correct=(p=='mgf_fixed') if f==0 else (p!='mgf_fixed') if f==1 else True
            records.append(row('input00',f,p,correct=correct))
    quotes=[quote(forecast=f,recovery={'kind':'interrupted_search'}) for f in (0,1)]
    marked,_=report.interruption_panel(records,quotes)
    primary=report.paired_comparison(marked,'mgf_fixed','no_dfs')
    assert (primary['n'],primary['gained'],primary['lost'])==(3,1,1)
    clean=report.interruption_subset(marked,quotes,affected=False)
    interrupted=report.interruption_subset(marked,quotes,affected=True)
    assert clean['forecasts']==1 and interrupted['forecasts']==2
    assert set(clean['paired_vs_no_dfs'])==set(report.POLICIES)
    assert all(g['n']==1 for g in clean['groups'].values())
    assert (interrupted['paired_vs_no_dfs']['mgf_fixed']['gained'],
        interrupted['paired_vs_no_dfs']['mgf_fixed']['lost'])==(1,1)
    assert clean['comparisons']['mgf_fixed']['no_dfs']['inferential_role']=='administrative_interruption_sensitivity'


def test_lost_old_incumbent_never_replaces_current_point_cost_or_gap(tmp_path):
    old=tmp_path/'old.json'
    report.base.write(old,dict(cost_yuan=2.1,lower_bound_yuan=1.5,quote_wall_seconds=600.03))
    recovery=dict(kind='completed_quote_action_lost',original_quote=report.base.ref(old),
        original_action_unavailable=True,prior_budget_charge_seconds=600.,
        prior_measured_wall_lower_bound=600.03,remaining_native_budget_seconds=0.,
        recovery_verification_seconds=.2,additional_quote_wall_seconds=0.,timing_scope='Budget accounting')
    q=quote(recovery=recovery)
    historical=report.historical_recovery_summary([q])
    current=report.solver_summary([q],'normal_fixed')
    assert historical['rows'][0]['historical_reported_cost_yuan']==2.1
    assert historical['rows'][0]['current_point_cost_yuan']==30.
    assert historical['distributions']['historical_quote_wall_seconds']['mean']==600.03
    assert historical['distributions']['prior_budget_charge_seconds']['mean']==600.
    assert current['nonzero_feasible_quotes']['activation_upper_bound_yuan']['mean']==30.
    assert current['nonzero_feasible_quotes']['activation_lower_bound_yuan']['mean']==3.
    assert current['nonzero_feasible_quotes']['gap']['mean']==.9
    assert current['nonzero_feasible_optimizer_invoked']['n']==0
    old.write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='Changed audit input'):
        report.historical_recovery_summary([q])


def test_empty_recovery_does_not_exclude_and_unknown_identity_fails_closed():
    records=[row('input00',0,p) for p in report.POLICIES]
    marked,forecasts=report.interruption_panel(records,[quote(recovery={})])
    assert not any(r['interruption_affected'] for r in marked)
    assert not forecasts[0]['interruption_affected']
    assert report.historical_recovery_summary([quote(recovery=None)])['n']==0
    with pytest.raises(ValueError,match='outside'):
        report.interruption_panel(records,[quote('absent',recovery={'kind':'interrupted'})])
    with pytest.raises(ValueError,match='dictionary'):
        report.interruption_panel(records,[quote(recovery=True)])


def test_all_interrupted_common_subset_is_explicitly_empty():
    records=[row('input00',0,p) for p in report.POLICIES]
    quotes=[quote(recovery={'kind':'interrupted'})]
    marked,_=report.interruption_panel(records,quotes)
    clean=report.interruption_subset(marked,quotes,affected=False)
    assert clean['forecasts']==0 and clean['execution_rows']==0
    assert all(g['n']==0 and g['recovery_rate'] is None for g in clean['groups'].values())
    assert all(c['n']==0 for c in clean['paired_vs_no_dfs'].values())
