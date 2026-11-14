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


def recovery_quote(tmp_path, **changes):
    path=tmp_path/'original_plan.json'
    path.write_text('{"quote_wall_seconds": 600.03, "cost_yuan": 2.1}',encoding='utf8')
    recovery=dict(kind='completed_quote_action_lost',original_quote=audit.base.ref(path),
        original_action_unavailable=True,prior_budget_charge_seconds=600.,
        prior_measured_wall_lower_bound=600.03,remaining_native_budget_seconds=0.,
        recovery_verification_seconds=.2,additional_quote_wall_seconds=0.,
        timing_scope='Conservative original budget plus measured verification, not exact total wall')
    recovery.update(changes)
    return dict(recovery=recovery,shared_preparation_seconds=151.,
        charged_total_seconds=751.2,optimizer_invoked=False)


def test_recovery_keeps_lost_point_separate_and_budget_exhausted(tmp_path):
    quote=recovery_quote(tmp_path)
    checker=audit.Audit();checker.recovery_check(quote)
    assert checker.counts['recovered_quotes']==1
    assert 'quote_wall_seconds' not in quote


def test_recovery_rejects_budget_reset_and_post_exhaustion_search(tmp_path):
    quote=recovery_quote(tmp_path,remaining_native_budget_seconds=600.)
    with pytest.raises(AssertionError,match='resets'):
        audit.Audit().recovery_check(quote)
    quote=recovery_quote(tmp_path);quote['optimizer_invoked']=True
    with pytest.raises(AssertionError,match='exhausted'):
        audit.Audit().recovery_check(quote)


def test_recovery_rejects_changed_old_receipt_and_wrong_accounted_time(tmp_path):
    quote=recovery_quote(tmp_path)
    Path(quote['recovery']['original_quote']['path']).write_text('{}',encoding='utf8')
    with pytest.raises(AssertionError,match='hash differs'):
        audit.Audit().recovery_check(quote)
    quote=recovery_quote(tmp_path);quote['charged_total_seconds']+=600.
    with pytest.raises(AssertionError,match='recovery_charged_time'):
        audit.Audit().recovery_check(quote)


def test_continuation_boundary_preserves_arguments_and_original_runner(tmp_path):
    path=tmp_path/'continuation.py'
    text=('ORIGINAL_PLAN = original.solver.plan\n'
          'def checkpoint_plan(*args, **kwargs):\n'
          '    receipt, point = ORIGINAL_PLAN(*args, **kwargs)\n'
          '    return receipt, point\n'
          'def run_stage(out, p):\n'
          '    pool.submit(original.run_forecast, task)\n')
    path.write_text(text,encoding='utf8')
    assert audit.continuation_boundary(path)['passed']
    path.write_text(text.replace('ORIGINAL_PLAN(*args, **kwargs)',
        'ORIGINAL_PLAN(*args, seconds=600, **kwargs)'),encoding='utf8')
    assert not audit.continuation_boundary(path)['passed']


def test_recovery_boundary_checks_helper_inputs_and_seal_order(tmp_path):
    path=tmp_path/'recover_partial.py'
    text=('def recover_quote(case, planning, prediction):\n'
          "    return solver.plan(case, planning['score'], planning['val'], prediction, planning['base'])\n"
          'def run_forecast(task):\n'
          '    q = recover_quote(case, planning, prediction)\n'
          "    io.write(directory / 'seal.json', {})\n"
          '    ctx = old.context(entry)\n')
    path.write_text(text,encoding='utf8')
    assert audit.recovery_boundary(path)['passed']
    path.write_text(text.replace("planning['val']",'actual_voltage'),encoding='utf8')
    assert not audit.recovery_boundary(path)['passed']


def test_recovery_bound_current_cost_and_original_problem_are_independent(tmp_path):
    e=entry();root=tmp_path/'run';folder=root/'fraction0.10'/e['id']/'forecast00'
    folder.mkdir(parents=True)
    def saved(name,data):
        path=folder/name;audit.base.write(path,data);return audit.base.ref(path)
    score=saved('score.json',{});val=saved('val.json',{});forecast=saved('forecast.json',{})
    resource=saved('resource_case.json',{});seeds=saved('seeds.json',{})
    audit.base.write(root/'protocol.json',{'solver_seconds':600.})
    original=saved('mgf_fixed_plan.json',dict(mode='mgf_fixed',bound_cost_object='activation_yuan',
        lower_bound_yuan=.25,quote_wall_seconds=600.,cost_yuan=2.))
    pause=saved('pause.json',dict(partial_tasks=[dict(id=e['id'],forecast='forecast00',
        saved_plans=[original],native_logs=[])]))
    problem=dict(mode='mgf_fixed',original_protocol=audit.base.ref(root/'protocol.json'),
        score=score,validation=val,forecast=forecast,resource_case=resource,
        shared_initial_points=seeds,target_mass=.99,buffer=1e-8)
    digest=audit.hashlib.sha256(audit.json.dumps(problem,sort_keys=True).encode()).hexdigest()
    recovery=dict(kind='completed_quote_action_unavailable',task_id=e['id'],forecast=0,
        same_problem_inputs=problem,same_problem_fingerprint=digest,pause_manifest=pause,original_quote=original,
        prior_budget_charge_seconds=600.,prior_measured_wall_lower_bound=600.,remaining_native_budget_seconds=0.)
    action=np.zeros((24,1));np.savez(folder/'action.npz',u=action)
    quote=dict(recovery=recovery,initial_points=seeds,requested_quote_budget_seconds=600.,
        bound_sources=[dict(reference=original,lower_bound_yuan=.25,same_problem_fingerprint=digest,origin='prior_same_mode')],
        cost_yuan=3.,lower_bound_yuan=.25,relative_gap=2.75/3.,lower_bound_withdrawn=False,
        executed_action_checkpoint=audit.base.ref(folder/'action.npz'))
    e.update(score=score,files={'validation.npz':val},forecast_ref=forecast)
    actual=audit.Audit().recovery_provenance_check(e,0,'mgf_fixed',quote)
    np.testing.assert_array_equal(actual,action)
    quote['relative_gap']=(2.-.25)/2.  # copied old incumbent gap is incorrect for the replacement
    with pytest.raises(AssertionError,match='current_recovery_gap'):
        audit.Audit().recovery_provenance_check(e,0,'mgf_fixed',quote)
    quote['relative_gap']=2.75/3.
    quote['bound_sources'][0]['same_problem_fingerprint']='another_problem'
    with pytest.raises(AssertionError,match='bound problem differs'):
        audit.Audit().recovery_provenance_check(e,0,'mgf_fixed',quote)
