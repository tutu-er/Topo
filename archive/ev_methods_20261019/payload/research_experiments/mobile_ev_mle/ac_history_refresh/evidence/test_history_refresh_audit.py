"""Hand-calculated synthetic trajectories and byte provenance, no optimizers."""
from copy import deepcopy
import base64
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256,sha1
import numpy as np
import pytest
from research_experiments.mobile_ev_mle.ac_history_refresh.optimizer import GaussianHistoryProblem,DEFAULTS
from research_experiments.mobile_ev_mle.ac_history_refresh.evidence.history_refresh_audit import (
    audit_candidate,audit_record,audit_freeze,audit_bindings,audit_run,canonical,digest,
    prepare_without_integrals,_prior_without_integration,Checks,audit_runtime_dependencies)


def problem(physical=False):
    def evaluate(w,with_jacobian=True):
        valid=not physical or abs(w[1])<.4
        out=dict(mean=np.array([w[0]+w[1]]),physical_ok=valid,
                 physical_failures=[] if valid else ['declared_state_domain'])
        if with_jacobian:out.update(J_eta=np.ones((1,1)),J_state=np.ones((1,1)))
        return out
    prior=dict(mean=np.array([0.]),cov=np.eye(1),A=np.array([[1.],[-1.]]),b=np.array([2.,2.]))
    return GaussianHistoryProblem(evaluate,[3.],prior,np.eye(1))


def fit_record(zero=True):
    # Exact quadratic: Phi(0,0)=4.5; Phi(1,1)=1.5; g0 dot p=-6.
    trace=[dict(iteration=0,objective_before=4.5,terms_before=dict(measurement=4.5,line_prior=0.,process_prior=0.),
        direction_eta=[1.],direction_white_state=[1.],directional_derivative=-6.,alpha=1.,objective_after=1.5,
        relative_objective_change=3/5.5,trials=[dict(alpha=1.,backtrack=0,accepted=True,
            physical_ok=True,physical_failures=[],objective=1.5)])]
    if zero:trace.append(dict(iteration=1,objective_before=1.5,direction_eta=[0.],direction_white_state=[0.],
        directional_derivative=0.,alpha=0.,objective_after=1.5,relative_objective_change=0.,trials=[]))
    return dict(status='converged_first_order' if zero else 'maximum_iterations',success=zero,
        initial_eta=[0.],initial_latent=[0.],initial_objective=4.5,map_w=[1.,1.],objective=1.5,
        options=dict(DEFAULTS),trace=trace,relative_objective_change=3/5.5,
        kkt=dict(scaled=0.,line_scaled=0.,state_scaled=0.,complementarity_scaled=0.,
                 eta_primal_violation=0.,dual=[0.,0.]),counters=dict(accepted_steps=1),
        prior_info=dict(process_covariance_sha256=sha256(np.eye(1).astype('<f8').tobytes()).hexdigest(),
                        original_prior_reused_once=True,history_likelihood_count=1))


def test_complete_replay_zero_alpha_has_valid_effective_change_without_new_optimization():
    result=audit_candidate(problem(),fit_record())
    assert result['passed'],result
    assert result['effective_relative_objective_change']==0
    assert result['saved_relative_objective_change']>1e-8
    assert result['final_KKT']['scaled']==0
    assert result['costs']['Jacobian_calls']==3
    assert result['costs']['new_optimizations']==result['costs']['new_GHK_calls']==0


def test_original_maximum_budget_failure_is_valid_evidence_not_audit_failure():
    result=audit_candidate(problem(),fit_record(False))
    assert result['passed'],result
    assert not result['scientific_success']
    assert result['scientific_status']=='maximum_iterations'
    assert result['effective_relative_objective_change']>1e-8


@pytest.mark.parametrize('tamper',('direction','objective','dual','Armijo'))
def test_independent_forward_and_saved_dual_checks_detect_tampering(tamper):
    fit=fit_record()
    if tamper=='direction':fit['trace'][0]['direction_eta']=[.9]
    elif tamper=='objective':fit['trace'][0]['trials'][0]['objective']=1.2
    elif tamper=='dual':fit['kkt']['dual']=[-1.,0.]
    else:fit['trace'][0]['directional_derivative']=-1e8
    result=audit_candidate(problem(),fit)
    assert not result['passed']


def test_rejected_physical_trial_retained_as_legitimate_scientific_failure():
    fit=fit_record(False);fit.update(status='physical_boundary_blocked',map_w=[0.,0.],objective=4.5,
        kkt=None,counters=dict(accepted_steps=0));fit['trace'][0].pop('alpha');fit['trace'][0].pop('objective_after')
    fit['trace'][0]['trials']=[dict(alpha=1.,backtrack=0,accepted=False,physical_ok=False,
        physical_failures=['declared_state_domain'])]
    result=audit_candidate(problem(True),fit)
    assert result['passed'],result
    assert result['trajectory'][0]['trials'][0]['physical']['physical_ok'] is False
    assert result['scientific_status']=='physical_boundary_blocked'


def test_minimal_process_failure_preserves_denominator_without_building_problem():
    def forbidden(*args):raise AssertionError('failed worker has no problem to rebuild')
    result=audit_record({},dict(status='process_failure',failures={'worker':'timeout'},
        keep_all_on_failure=True),{},prepare=forbidden)
    assert result['passed'] and result['candidates']=={}
    assert result['scientific_status']=='process_failure'
    assert result['costs']['AC_calls']==0


def test_declared_input_failure_and_unavailable_forward_remain_explicit():
    result=audit_record({},dict(status='input_failure',failures={'input':'missing historical meters'}),{})
    assert result['passed'] and result['scientific_status']=='input_failure'
    p=problem()
    def failed(*args,**kwargs):raise RuntimeError('saved AC computation failure')
    p.evaluator=failed
    fit=fit_record(False);fit.update(status='numerical_failure',objective=None,
        map_w=[0.,0.],trace=[dict(iteration=0,error='RuntimeError: saved AC computation failure')])
    replay=audit_candidate(p,fit)
    assert replay['passed'],replay
    assert replay['final_objective'] is None and replay['final_KKT'] is None


def test_GHK_failure_keeps_valid_MAP_numerical_replay():
    fit=fit_record();prior=problem().line_prior
    p=dict(names=['k'],candidates=[SimpleNamespace(name='k')],plans={},history=None,meter=None,
        options=DEFAULTS,K_D=np.eye(1),line_priors={'k':prior},
        initializations={'k':dict(eta=[0.],latent=[0.])})
    B=dict(status='partial_failure',failures={'GHK':'precision_failure'},candidate_names=['k'],plans={},actions={},
        history_refresh=dict(context=None,candidates={'k':dict(MAP=fit,posterior={'status':'precision_failure'})}))
    result=audit_record({'actions':{}},B,{},prepare=lambda *args:p,problem_factory=lambda *args,**kwargs:problem())
    assert result['passed'],result
    assert result['candidates']['k']['posterior_scientific_status']=='precision_failure'
    assert result['candidates']['k']['scientific_success']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture_freeze(tmp_path):
    Aroot=tmp_path/'A';Broot=tmp_path/'B';Aroot.mkdir();Broot.mkdir()
    files={'A/mod.py':tmp_path/'mod.py','B/optimizer.py':tmp_path/'optimizer.py'}
    for name,path in files.items():path.write_bytes(('frozen '+name).encode())
    specs=dict(calibration=[dict(index=0)],evaluation=[dict(index=0)])
    A=dict(source_sha256={'mod.py':digest(files['A/mod.py'])},specifications=specs)
    A['policy_id']=sha256(json.dumps(A,sort_keys=True,ensure_ascii=False).encode()).hexdigest();write(Aroot/'freeze.json',A)
    B=dict(input_A_root=str(Aroot),input_A_freeze_sha256=digest(Aroot/'freeze.json'),input_A_policy_id=A['policy_id'],
        source_sha256={name:digest(path) for name,path in files.items()},source_paths={name:str(path) for name,path in files.items()},
        specifications=specs,A_source_count=1,B_source_count=1)
    B['policy_id']=canonical(B);write(Broot/'freeze.json',B)
    snapshot={name:base64.b64encode(path.read_bytes()).decode() for name,path in files.items()}
    with gzip.open(Broot/'source_snapshot.json.gz','wt',encoding='utf-8') as stream:json.dump(snapshot,stream)
    return Aroot,Broot,A,B,snapshot


def test_full_source_and_decoded_snapshot_hash_audit_detects_tampering(tmp_path):
    Aroot,Broot,A,B,snapshot=fixture_freeze(tmp_path)
    _,result=audit_freeze(Broot,expected_sources=2);assert result['passed'],result
    snapshot['A/mod.py']=base64.b64encode(b'altered snapshot').decode()
    with gzip.open(Broot/'source_snapshot.json.gz','wt',encoding='utf-8') as stream:json.dump(snapshot,stream)
    _,bad=audit_freeze(Broot,expected_sources=2)
    assert not bad['passed'] and any(r['check']=='snapshot/A/mod.py' for r in bad['failures'])


def test_supplementary_dependencies_have_distinct_snapshot_live_and_git_blob_scope(tmp_path):
    payload=b'unchanged dependency bytes\r\n';name='research_experiments/extra.py';live=tmp_path/name
    live.parent.mkdir();live.write_bytes(payload);root=tmp_path/'B';root.mkdir()
    stored=payload.replace(b'\r\n',b'\n');blob=sha1(b'blob '+str(len(stored)).encode()+b'\0'+stored).hexdigest()
    write(root/'runtime_dependency_git_audit.json',dict(rows=[dict(path=name,sha256=sha256(payload).hexdigest(),
        git_tree_at_A_freeze='100644 blob '+blob+'\t'+name)],primary_source_manifests_changed=False))
    with gzip.open(root/'runtime_dependency_snapshot.json.gz','wt',encoding='utf-8') as f:
        json.dump({name:base64.b64encode(payload).decode()},f)
    good=audit_runtime_dependencies(root,{},execution_root=tmp_path,expected_count=1)
    assert good['passed'] and good['outside_original_main_manifest']
    assert good['retroactive_freeze_claim'] is False
    assert good['rows'][0]['Git_blob_comparison']=='CRLF-to-LF text equivalence'
    live.write_bytes(b'changed live bytes')
    bad=audit_runtime_dependencies(root,{},execution_root=tmp_path,expected_count=1)
    assert not bad['passed']


def test_calibration_input_hashes_and_original_A_binding_are_checked(tmp_path):
    Aroot,Broot,A,B,_=fixture_freeze(tmp_path);name='history_000.json.gz'
    for path in (Aroot/'calibration'/name,Broot/'calibration'/name):write(path,{'raw':'same immutable data'})
    acal=dict(policy_id=A['policy_id'],input_sha256={'calibration\\'+name:digest(Aroot/'calibration'/name)})
    write(Aroot/'calibration.json',acal);write(Aroot/'evaluation_calibration_binding.json',
        dict(policy_id=A['policy_id'],calibration_sha256=digest(Aroot/'calibration.json')))
    cal=dict(policy_id=B['policy_id'],input_A_policy_id=A['policy_id'],old_A_calibration_used=False,
        fixed_before_evaluation=True,histories=1,G11_multiplier=4,input_B_sha256={name:digest(Broot/'calibration'/name)},
        input_A_sha256={name:digest(Aroot/'calibration'/name)})
    write(Broot/'calibration.json',cal);write(Broot/'evaluation_calibration_binding.json',
        dict(policy_id=B['policy_id'],calibration_sha256=digest(Broot/'calibration.json'),
             G11_multiplier=4,input_A_policy_id=A['policy_id']))
    _,good=audit_bindings(Broot,B);assert good['passed'],good
    cal['input_B_sha256'][name]='tampered';write(Broot/'calibration.json',cal)
    _,bad=audit_bindings(Broot,B);assert not bad['passed']


def test_main_guard_opens_no_main_history_before_statistical_complete(tmp_path):
    write(tmp_path/'controller.json',{'status':'running'})
    with pytest.raises(RuntimeError,match='no main raw was opened'):audit_run(tmp_path)


def test_observed_reconstruction_never_calls_GHK_or_normalization_or_future_truth(monkeypatch):
    from research_experiments.mobile_ev_mle.ac_history_refresh import io
    from research_experiments.mobile_ev_mle.ac_history_refresh.test_io_replay import _raw,Guarded
    from research_experiments.mobile_ev_mle import ev_bayes_response_truncation as trunc
    A=Guarded(_raw(),forbidden=('actions','true_name','true_index','prediction','posterior_centers',
                              'actual_white_noise','true_lines_evaluation_only'))
    options=io._public_options(A)
    def saved_prior(m,*args):return dict(_prior_without_integration(m,options),log_mass=-.2)
    # Supply a saved normalizer, never numerically integrate even in this fixture.
    monkeypatch.setattr(trunc,'line_prior',saved_prior)
    original=io.prepare_history(A)
    B=dict(history_refresh=dict(candidates={c.name:dict(MAP=dict(prior_info=dict(
        line_prior=original['line_priors'][c.name]))) for c in original['candidates']}),
        process_hyperparameters=io._serial(original['hyperparameters']),
        observed_input_sha256=original['observed_input_sha256'])
    def forbidden(*args,**kwargs):raise AssertionError('integration is forbidden in audit')
    for name in ('line_prior','line_prior_mass','bounded_line_moments'):monkeypatch.setattr(trunc,name,forbidden)
    checks=Checks();rebuilt=prepare_without_integrals(A,B,dict(base_kw=20.,future_root_v=1.),checks)
    assert checks.result()['passed'],checks.result()
    np.testing.assert_array_equal(rebuilt['K_D'],original['K_D'])
    assert set(rebuilt['initializations'])==set(original['initializations'])
