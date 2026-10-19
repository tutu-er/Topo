"""Analytical domain nesting and input-isolation checks, without a big study."""
import json

import numpy as np
import pytest

try:
    from . import adaptive_domain_audit as domain
except ImportError:
    import adaptive_domain_audit as domain


def analytical_pair():
    metadata=dict(candidate_names=['h'],centre_indices=[0],
        variable_layout=[dict(beta=(0,1),history_state=(1,2),future_source=(2,4))],
        free_indices=[0,1,2,3],fixed_values=[0.,0.,0.,0.],future_row_start=1,
        line_trust_fraction=.6,source_delta_bound=.15)
    raw=dict(name='analytical',c=np.array([3.,2.]),B=np.array([[0.],[.5]]),
        A=np.array([[0.,0.,0.,0.],[1.,0.,1.,.5]]),F=np.zeros((0,4)),
        lower=np.array([-.6,-100.,-.15,-.15]),upper=np.array([.6,100.,.15,.15]),
        scale=np.ones(4),metadata=metadata)
    context=dict(centres={'h':[dict(beta=np.array([1.]))]})
    return context,raw,domain.FrozenPair(**raw)


def test_nested_boxes_exact_scalar_cost_and_preserved_history():
    context,raw,broad=analytical_pair()
    values=[]
    for radius in (.02,.05,.15):
        small,pair=domain.restrict_pair(context,raw,broad,line_radius=.1,source_radius=radius)
        answer=domain.solve_pair(pair,np.array([0.]))
        # Historical residual 3 survives; future residual can decrease by .1
        # from the line, radius from P, and .5*radius from Q at the same source.
        exact=9.+(2.-.1-1.5*radius)**2
        assert answer['q_upper']==pytest.approx(exact,abs=1e-10)
        assert answer['residual'][0]**2==pytest.approx(9.)
        assert pair.lower[1]==-100. and pair.upper[1]==100.
        assert domain.matrix_hash(pair)==domain.matrix_hash(broad)
        values.append(answer['q_upper'])
    assert values[0]>values[1]>values[2]


def test_source_and_line_bounds_intersect_broad_physical_box():
    context,raw,broad=analytical_pair()
    raw['lower'][2]=-.01
    broad=domain.FrozenPair(**raw)
    small,pair=domain.restrict_pair(context,raw,broad,line_radius=.3,source_radius=.05)
    assert pair.lower[2]==-.01
    assert pair.upper[2]==.05
    assert pair.lower[0]==-.3 and pair.upper[0]==.3
    with pytest.raises(ValueError,match='nested'):
        domain.restrict_pair(context,raw,broad,line_radius=.7,source_radius=.05)


def test_control_report_retains_historical_and_future_terms():
    context,raw,pair=analytical_pair()
    rows=domain.solve_controls(context,[raw],[pair],{'none':{'u_kw':np.array([0.])}},rollback_active=False)
    active=rows[0]['active_affine']
    assert active['affine_history_cost']==9.
    assert rows[0]['q_upper']==pytest.approx(9.+active['affine_future_distance_squared'])
    compensation=active['source_compensation'][0]
    assert compensation['source_node']==1
    assert compensation['source_step_linf']==pytest.approx(.15)
    assert compensation['source_bound_active_count']==2
    assert not rows[0]['history_removed_from_objective']


def test_failed_competitor_preserved_instead_of_deleted(monkeypatch):
    context,raw,pair=analytical_pair()
    good=domain.solve_pair
    bad=domain.FrozenPair(**{**raw,'name':'bad'})
    def solve(item,u):
        if item.name=='bad':raise RuntimeError('required numerical failure')
        return good(item,u)
    monkeypatch.setattr(domain,'solve_pair',solve)
    rows=domain.solve_controls(context,[raw,{**raw,'name':'bad'}],[pair,bad],
        {'none':{'u_kw':np.array([0.])}},rollback_active=False)
    assert rows[0]['status']=='required_QP_failure'
    assert rows[0]['q_lower'] is None
    assert rows[0]['failed_pairs']==['bad']
    assert rows[0]['required_pair_count']==2 and len(rows[0]['pairs'])==2


def test_whitelist_loader_never_accesses_future_arrays_or_truth(tmp_path):
    actions=[dict(method=method,action=dict(p=np.zeros((8,3)).tolist(),
        q=np.zeros((8,3)).tolist(),sites=[0]*8)) for method in domain.METHODS]
    file=tmp_path/'trial_1.json'
    raw=dict(records=actions,seed={'poison':'must not be used'},truth={'secret':[1,2,3]},
        truth_topology='poison',selection={'uses_future_noise':True})
    file.write_text(json.dumps(raw))
    values={"history_"+k:np.ones((2,3))*.1 for k in ('p','q')}
    values['history_v']=np.ones((2,3))
    values.update({"history_"+k:np.ones(2) for k in ('master_p','master_q','root_v')})
    values['refit_milp_future_noise']=np.array([{'cannot_unpickle':True}],dtype=object)
    np.savez(tmp_path/'trial_1_observations.npz',**values)
    history,planned=domain.load_history_actions(file)
    expected_seed=domain.history_seed(history)
    raw['seed']=999999;raw['truth_topology']=0;raw['truth']={'different':True}
    file.write_text(json.dumps(raw))
    second,again=domain.load_history_actions(file)
    assert domain.history_seed(second)==expected_seed
    assert set(planned)==set(domain.METHODS)
    assert all(np.array_equal(planned[k]['u_kw'],again[k]['u_kw']) for k in planned)


def test_forecast_variants_share_centres_but_change_declared_paths():
    h=domain.Observations(p=np.array([[.1,.1,.1],[.2,.2,.2],[.3,.3,.3]]),
        q=np.full((3,3),.04),v=np.ones((3,3)),master_p=np.ones(3),master_q=np.ones(3))
    original=dict(history=h,model=domain.model(3),config={'horizon':2},
        summary={},centres={'h':[object()]})
    mean=domain.scenario_context(original,[dict(name='mean')])
    trend=domain.scenario_context(original,[dict(name='trend',baseline='ols_trend')])
    assert mean['centres'] is original['centres'] and trend['centres'] is original['centres']
    assert np.all(trend['scenarios'][0]['p0']>mean['scenarios'][0]['p0'])
    assert not trend['summary']['forecast_is_coverage_guarantee']


def test_nested_direction_checked_only_for_identical_model_family():
    rows=[dict(domain=dict(centres=1,line_radius=.1,source_radius=.02),
        controls=[dict(method='none',status='complete',q_lower=10.,q_upper=10.)]),
        dict(domain=dict(centres=2,line_radius=.6,source_radius=.15),
        controls=[dict(method='none',status='complete',q_lower=8.,q_upper=8.)])]
    checked=domain.nested_checks(rows)
    assert len(checked)==1 and not checked[0]['invalid']
    rows[1]['controls'][0].update(q_lower=12.,q_upper=12.)
    assert domain.nested_checks(rows)[0]['invalid']


def test_adding_feasible_centre_cannot_increase_global_pair_minimum():
    context,raw,pair=analytical_pair()
    another=domain.FrozenPair(**{**raw,'name':'second_center','c':np.array([1.,.5])})
    first=domain.solve_pair(pair,np.array([0.]))['q_upper']
    other=domain.solve_pair(another,np.array([0.]))['q_upper']
    assert min(first,other)<=first
    assert other==pytest.approx(1.)


def write_input(folder,identifier,*,negative_p=0.,history_shift=0.):
    p=np.zeros((8,3));p[0,0]=negative_p
    actions=[dict(method=method,action=dict(p=p.tolist(),q=np.zeros_like(p).tolist(),sites=[0]*8))
             for method in domain.METHODS]
    file=folder/f'trial_{identifier}.json';file.write_text(json.dumps(dict(records=actions)))
    values={"history_"+k:np.ones((2,3))*(.1+history_shift) for k in ('p','q')}
    values['history_v']=np.ones((2,3))
    values.update({"history_"+k:np.ones(2) for k in ('master_p','master_q','root_v')})
    np.savez(folder/f'trial_{identifier}_observations.npz',**values)
    return file


def test_tiny_negative_retained_raw_with_explicit_AC_correction(tmp_path):
    file=write_input(tmp_path,1,negative_p=-5e-8)
    history,actions=domain.load_history_actions(file)
    raw=actions['refit_milp']
    assert raw['u_kw'][0,0]==-1e-6
    assert raw['raw_u_kw'][0,0]==-1e-6
    assert raw['canonical_AC_u_kw'][0,0]==0.
    warning=raw['numerical_input_audit']
    assert warning['power_tolerance_pu']==1e-7
    assert warning['AC_correction_linf_kw']==pytest.approx(1e-6)
    assert warning['raw_control_hash']!=warning['AC_canonical_control_hash']
    assert warning['proxy_uses_raw_control'] and not warning['original_EV_trial_modified']
    saved=json.loads(file.read_text())
    assert saved['records'][0]['action']['p'][0][0]==-5e-8


def test_real_discharge_is_not_erased_by_input_tolerance(tmp_path):
    file=write_input(tmp_path,1,negative_p=-.001)
    with pytest.raises(ValueError,match='charging-only'):
        domain.load_history_actions(file)


def test_AC_replay_only_canonicalizes_tolerated_power(monkeypatch):
    context,raw,pair=analytical_pair()
    checked=[]
    def actual(context,raw,u,z):
        checked.append(np.array(u))
        return dict(actual_soft_cost=1.,affine_soft_cost=1.,physically_feasible=True,
                    nonlinear_infimum_computed=False)
    monkeypatch.setattr(domain,'evaluate_frozen_pair_actual',actual)
    answer=domain.solve_controls(context,[raw],[pair],{'rounded':{'u_kw':np.array([-1e-6])}})[0]
    assert checked[0][0]==0.
    replay=answer['active_AC_witness']
    assert replay['AC_control_correction_linf_kw']==pytest.approx(1e-6)
    assert replay['affine_response_correction_norm']==pytest.approx(.5e-6)
    assert replay['raw_control_affine_cost']==answer['q_upper']
    assert not replay['original_EV_trial_modified']


def test_input_failure_keeps_history_denominator_and_next_runs(tmp_path,monkeypatch):
    input=tmp_path/'input';input.mkdir();out=tmp_path/'out'
    write_input(input,1,negative_p=-.001)
    write_input(input,2,history_shift=.01)
    called=[]
    def cheap(history,actions,**kwargs):
        called.append(domain.history_input_hash(history))
        return dict(history_hash=called[-1],status='complete',seconds=.01,domains=[],models=[],milp={})
    monkeypatch.setattr(domain,'audit_history',cheap)
    summary=domain.run_audit(input,out,limit=2,milp_histories=0)
    assert summary['history_count']==2 and summary['selected_file_count']==2
    assert summary['failure_history_count']==1 and len(called)==1
    failed=json.loads((out/'failed_input_trial_1.json').read_text())
    assert failed['failure_stage']=='input_loading'
    assert failed['independent_history_denominator_preserved']
    assert summary['status_counts']=={'complete':1,'required_input_or_audit_failure':1}
    assert (out/'summary.json').exists()


def test_saved_domain_evidence_cannot_be_overwritten(tmp_path):
    input=tmp_path/'input';input.mkdir();out=tmp_path/'out';out.mkdir()
    write_input(input,1)
    evidence=out/'history_previous.json';evidence.write_text('{"original":true}')
    with pytest.raises(ValueError,match='fresh output'):
        domain.run_audit(input,out,limit=1,milp_histories=0)
    assert evidence.read_text()=='{"original":true}'
