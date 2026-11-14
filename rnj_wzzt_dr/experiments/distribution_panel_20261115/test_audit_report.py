"""Meaningful audit/denominator regressions; no network or optimization runs."""
from pathlib import Path

import numpy as np
import pytest

from . import audit, report


def entry(name='paper15',day='2024-02-05'):
    return dict(id=name+'__'+day,case=name,start_date=day,date_cluster=day)


def fake_row(e,policy,correct,forecast=0,repeat=0):
    return dict(**audit.identity(e),forecast=forecast,repeat=repeat,strategy=policy,
        audit_status='passed',original_status='complete',evaluation={'native_correct':correct})


def test_data_valid_fit_failure_stays_in_all_eight_denominators():
    good,failed,missing=entry(),entry('flynn16'),entry('soumalas11')
    ledger=[dict(task=good,status='complete'),dict(task=failed,status='failed',stage='original_101'),
        dict(task=missing,status='data_incomplete')]
    records=[fake_row(good,p,True) for p in audit.POLICIES]
    rows,qualification=report.complete_panel(records,ledger,[],dict(forecasts=1,repeats=1))
    assert len(rows)==16
    for p in audit.POLICIES:
        group=[r for r in rows if r['strategy']==p]
        assert len(group)==2 and sum(report.recovered(r) for r in group)==1
    assert sum(q['data_valid'] for q in qualification)==2


def test_missing_audited_success_fails_closed():
    good=entry();ledger=[dict(task=good,status='complete')]
    records=[fake_row(good,p,True) for p in audit.POLICIES[:-1]]
    with pytest.raises(ValueError,match='Missing audited'):
        report.complete_panel(records,ledger,[],dict(forecasts=1,repeats=1))


def test_duplicate_identity_fails_closed():
    e=entry();row=fake_row(e,'no_dfs',True)
    with pytest.raises(ValueError,match='Duplicate'):
        report.complete_panel([row,row],[],[],dict(forecasts=1,repeats=1))


def test_zero_valid_inputs_is_reportable():
    rows,qualification=report.complete_panel([], [dict(task=entry(),status='data_incomplete')],[],dict(forecasts=2,repeats=16))
    assert rows==[] and len(qualification)==1
    assert report.summarize(rows)['recovery_rate'] is None
    assert report.paired_comparison(rows,'mgf_fixed','no_dfs')['n']==0
    assert report.cluster_interval([])['interval'] is None


def test_networks_first_then_equal_dates_not_pooled_paths():
    rows=[]
    # Day1 has four networks; day2 has one. Four +1 changes and one -1
    # give a pooled +.6, but equal-date comparison must be exactly zero.
    for name in ('paper15','flynn16','pengwah18','soumalas11'):
        e=entry(name,'2024-02-05')
        rows += [fake_row(e,'mgf_fixed',True),fake_row(e,'no_dfs',False)]
    e=entry('paper15','2024-05-20')
    rows += [fake_row(e,'mgf_fixed',False),fake_row(e,'no_dfs',True)]
    result=report.paired_comparison(rows,'mgf_fixed','no_dfs')
    assert result['gained']==4 and result['lost']==1
    assert result['pooled_path_difference']==.6
    assert result['date_equal_weight_difference']==0.
    assert report.cluster_interval(result['dates'])['interval'] is None


def test_cluster_bootstrap_ignores_number_of_paths():
    dates=[dict(date_cluster=str(i),season='winter',difference=d) for i,d in enumerate((-.5,.2,.3,.7))]
    a=report.cluster_interval(dates,draws=1000)
    b=report.cluster_interval([dict(d,n_paths=99999) for d in dates],draws=1000)
    assert a==b and a['date_clusters']==4


def test_reference_hash_detects_corruption(tmp_path):
    path=tmp_path/'value.json';path.write_text('{}',encoding='utf8');ref=audit.base.ref(path)
    path.write_text('{"changed":true}',encoding='utf8')
    with pytest.raises(ValueError,match='Changed audit input'):report.checked(ref)


def test_PQ_map_and_energy_window_are_independently_checked():
    config=dict(action_units=['kW','kvar'],resources=[
        dict(id='pv1',kind='pv',node_label=101,contract_kw=2.,q_over_p=.2,
            fee_positive_yuan_per_kwh=.1,fee_negative_yuan_per_kwh=.1),
        dict(id='q1',kind='reactive',node_label=101,capacity_kvar=3.,
            fee_positive_yuan_per_kvarh=.05,fee_negative_yuan_per_kvarh=.05)])
    action=np.zeros((24,2));action[2,0]=1.;action[2,1]=-2.
    devices,violation,dp,dq=audit.resource_values(config,action,[101],np.full((96,1),-.1),100.)
    np.testing.assert_allclose(dp[8:12,0],.01)
    np.testing.assert_allclose(dq[8:12,0],-.018)
    assert devices[0]['absolute_energy']==1 and max(violation.values())==0
    action[0,0]=1.
    with pytest.raises(ArithmeticError,match='Resource domain violation'):
        audit.resource_values(config,action,[101],np.full((96,1),-.1),100.)


def test_single_family_vacuous_gate_is_explicit_not_empirical_truth():
    model=dict(families=['same','same'],targets=[0],score=dict(cells=96,reference_mse=1.,validation_tie=1e-12),n=1,sigma2=1.)
    result=audit.exact_gates(model,np.zeros((96,2)),np.zeros((96,2)),'normal_fixed')
    assert result['feasible'] and result['rows']==[] and result['worst_recovery_lower']==1
    assert 'truth coverage' in result['probability_scope']


def test_reservation_uses_requested_peak_after_partial_delivery():
    config=dict(action_units=['kvar'],resources=[dict(id='q',kind='reactive',node_label=1,
        capacity_kvar=5.,fee_positive_yuan_per_kvarh=.05,fee_negative_yuan_per_kvarh=.05)])
    request=np.zeros((24,1));request[0,0]=4;request[1,0]=-2
    bill=audit.base.procurement(config,request,request*.9,.5,.5)
    assert bill['planned']['availability_yuan']==bill['settled']['availability_yuan']==2.
    assert bill['planned']['activation_yuan']==pytest.approx(.3)
    assert bill['settled']['activation_yuan']==pytest.approx(.27)


def test_boundary_rejects_actual_context_before_seal(tmp_path):
    path=tmp_path/'run.py'
    def script(before):
        context='    ctx = old.context(entry)\n'
        seal="    io.write(directory / 'seal.json', {})\n"
        solve="    solver.plan(case, planning['score'], planning['val'], prediction, planning['base'])\n"
        return 'def run_forecast(task):\n'+solve+(context+seal if before else seal+context)
    path.write_text(script(True),encoding='utf8')
    assert not audit.planning_boundary(path)['passed']
    path.write_text(script(False),encoding='utf8')
    assert audit.planning_boundary(path)['passed']


def test_zero_certificates_do_not_dilute_nonzero_solver_gap():
    def quote(gap,nonzero):
        return dict(policy='mgf_fixed',new_gate_feasible=True,nonzero_executed=nonzero,
            optimizer_invoked=nonzero,zero_action_sufficient=not nonzero,anchor_fallback=False,zero_fallback=False,
            native_status=9 if nonzero else None,status='time_limit' if nonzero else 'zero_sufficient',
            relative_gap=gap,cost_yuan=10. if nonzero else 0.,lower_bound_yuan=2. if nonzero else 0.,
            charged_total_seconds=600. if nonzero else 1.)
    result=report.solver_summary([quote(0.,False)]*9+[quote(.8,True)],'mgf_fixed')
    assert result['gap_mean_feasible']==pytest.approx(.08)
    assert result['nonzero_feasible_optimizer_invoked']['gap']['mean']==pytest.approx(.8)
    assert result['zero_cost_certificates']['n']==9
