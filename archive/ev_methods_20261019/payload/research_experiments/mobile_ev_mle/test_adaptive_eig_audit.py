"""Independent finite-prior, physical replay and no-oracle input checks."""
from collections.abc import Mapping
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

try:
    from . import adaptive_eig_audit as audit
except ImportError:
    import adaptive_eig_audit as audit


@pytest.fixture(scope="module")
def bank():
    return audit.build_finite_bank(3, 2)


def _observation(bank, atom_index=0):
    return audit.audit_common.unvectorize(bank.history_means[atom_index], bank.meter, bank.history_count)


def _actions(bank):
    zero = np.zeros((bank.future_count, 3))
    constant = zero.copy(); constant[:, 0] = .10
    chosen = zero.copy(); chosen[0, 1] = .20
    return {name:audit.PlannedAction(name, p, np.zeros_like(p)) for name,p in
            (("none",zero),("constant_fixed",constant),("refit_milp",chosen))}


def test_all_four_atom_counts_normalization_and_no_repeated_star_strength(bank):
    assert len(bank.atoms) == 80
    assert np.bincount(bank.labels).tolist() == [8, 24, 24, 24]
    prior = audit.engineering_prior(bank)
    assert prior.sum() == pytest.approx(1, abs=1e-15)
    assert np.all(prior > 0)
    for label in range(4):
        assert prior[bank.labels == label].sum() == pytest.approx(.25)
    assert all(atom["strength"] is None for atom in bank.atoms[:8])
    assert len({(a["rscale"],a["xscale"],a["path_id"]) for a in bank.atoms[:8]}) == 8
    assert len({json.dumps(row,sort_keys=True) for row in bank.diagnostics["atom_manifest"]}) == 80
    assert not bank.labels.flags.writeable


@pytest.mark.parametrize("star,weak",audit.SENSITIVITY_GRID)
def test_topology_and_within_cherry_prior_mass(bank,star,weak):
    prior = audit.engineering_prior(bank,star_mass=star,weak_mass=weak)
    assert prior[bank.labels==0].sum() == pytest.approx(star)
    for h in range(1,4):
        indices = bank.labels==h
        assert prior[indices].sum() == pytest.approx((1-star)/3)
        internal = np.array([a["strength"]==0 for a in bank.atoms])
        assert prior[indices&internal].sum()/prior[indices].sum() == pytest.approx(weak)


@pytest.mark.parametrize("star,weak",[(0,.5),(1,.5),(.5,0),(.5,1),(np.nan,.5),(.5,np.inf)])
def test_improper_prior_masses_rejected(bank,star,weak):
    with pytest.raises(ValueError,match="proper"):
        audit.engineering_prior(bank,star_mass=star,weak_mass=weak)


def test_known_historical_root_used_future_root_predeclared(bank):
    other = audit.build_finite_bank(3,2,history_root_v=np.full(3,1.01))
    assert not np.array_equal(other.history_means,bank.history_means)
    # Only the historical known input changed; future root remains declared 1.
    action = _actions(bank)["constant_fixed"]
    m0,_ = audit.replay_action(bank,action)
    m1,_ = audit.replay_action(other,action)
    np.testing.assert_array_equal(m0,m1)
    with pytest.raises(ValueError,match="root inputs"):
        audit.audit_history(_observation(bank),_actions(bank),bank=other,selection_samples=16,
            validation_samples=(16,),validation_repeats=1)


def test_ev_injected_once_in_p_q_and_independent_direct_ac(bank):
    action = _actions(bank)["refit_milp"]
    zero = _actions(bank)["none"]
    future,physical = audit.replay_action(bank,action)
    baseline,_ = audit.replay_action(bank,zero)
    width = bank.future_count*3
    np.testing.assert_allclose(future[:,:width]-baseline[:,:width],
        np.tile((action.p/bank.meter.p_sigma).ravel(),(80,1)),atol=1e-12)
    np.testing.assert_allclose(future[:,width:2*width]-baseline[:,width:2*width],
        np.tile((action.q/bank.meter.q_sigma).ravel(),(80,1)),atol=1e-12)
    atom = bank.atoms[31]; label = int(bank.labels[31]); rows = slice(3,5)
    prediction = audit.likelihood.ac_predict(bank.candidates[label],atom["r"],atom["x"],
        atom["p"][rows]+action.p,atom["q"][rows]+action.q,
        atom["source_p"][rows],atom["source_q"][rows],root_v=1.)
    start_master = 3*width
    np.testing.assert_allclose(future[31,start_master:start_master+2],
        prediction["master_p"]/bank.meter.master_p_sigma,atol=1e-12)
    assert physical["ev_added_once"]
    assert physical["ac_equation_residual"] < 1e-10


class WhitelistOnly(Mapping):
    """Any attempt to traverse or fetch a forbidden evaluation field fails."""
    def __init__(self,allowed):
        self.allowed = allowed
        self.accessed = []

    def __getitem__(self,key):
        self.accessed.append(key)
        if key not in self.allowed:
            raise AssertionError("forbidden evaluation field accessed: "+str(key))
        return self.allowed[key]

    def __iter__(self):
        raise AssertionError("entire mixed evaluation object was traversed")

    def __len__(self):
        raise AssertionError("mixed evaluation object length inspected")


def test_action_loader_accesses_only_whitelist_no_truth_or_future(bank):
    p = np.zeros((2,3)).tolist()
    action = WhitelistOnly({"p":p,"q":p})
    record = WhitelistOnly({"method":"none","action":action})
    trial = WhitelistOnly({"records":[record]})
    result = audit.extract_planned_actions(trial)
    assert list(result) == ["none"]
    assert trial.accessed == ["records"]
    assert record.accessed == ["method","action"]
    assert action.accessed == ["p","q"]


def test_npz_loader_does_not_decode_forbidden_object_future_noise(tmp_path,bank):
    historical = _observation(bank)
    arrays = {"history_"+name:getattr(historical,name) for name in (*audit.CHANNELS,"root_v")}
    arrays.update(truth=np.array({"forbidden":object()},dtype=object),
                  noise=np.array([object()],dtype=object),
                  refit_milp_v=np.full((2,3),np.nan))
    path = tmp_path/"mixed.npz"
    np.savez(path,**arrays)
    result = audit.read_history(path)
    for name in (*audit.CHANNELS,"root_v"):
        np.testing.assert_array_equal(getattr(result,name),getattr(historical,name))
    # A direct read of the poison entry is forbidden with allow_pickle=False.
    with np.load(path,allow_pickle=False) as stored:
        with pytest.raises(ValueError,match="Object arrays"):
            stored["truth"]


def test_array_loader_does_not_traverse_forbidden_npz_entries(bank):
    obs = _observation(bank)
    arrays = WhitelistOnly({"history_"+name:getattr(obs,name) for name in (*audit.CHANNELS,"root_v")})
    result = audit.history_from_arrays(arrays)
    assert arrays.accessed == list(audit.HISTORY_KEYS)
    assert result.t == 3


def _fake_scorer(calls,selection_seed):
    def score(posterior,means,*,samples_per_topology,seed,batch_size):
        calls.append((samples_per_topology,seed,tuple(means)))
        # Deliberately reverse the winner in validation: the audit must not
        # reselect after seeing a validation stream.
        values = {name:float((name=="refit_milp") if seed==selection_seed else (name=="none"))
                  for name in means}
        return {name:dict(eig=value,eig_se=.01,identity_difference=0.,
            paired_comparisons={base:dict(eig_gain=value-values[base],eig_gain_se=.02)
                                for base in means}) for name,value in values.items()}
    return score


def test_selection_2048_frozen_validation_2048_8192_three_fresh_streams(bank):
    calls = []
    result = audit.audit_history(_observation(bank),_actions(bank),bank=bank,seed=99,
                                scorer=_fake_scorer(calls,99))
    assert result["selected_action"] == "refit_milp"
    assert calls[0][:2] == (2048,99)
    assert [c[0] for c in calls[1:]] == [2048]*3+[8192]*3
    assert len({c[1] for c in calls}) == 7
    for level in result["validation"]:
        assert level["frozen_winner_comparisons"]["none"]["eig_gain"] == -1
    assert not result["original_pilot_actions_modified"]
    assert not result["selector_reads_future_observations"]
    assert result["engineering_prior_matches_pilot_population"] is False
    assert result["history_fit"]["loss_is_not_calibrated_goodness_of_fit_pvalue"]


def test_action_order_invariance_duplicate_mean_groups_and_exact_cache(bank):
    actions = _actions(bank)
    actions["duplicate_constant"] = audit.PlannedAction("duplicate_constant",actions["constant_fixed"].p,actions["constant_fixed"].q)
    call1,call2 = [],[]
    cache = {}
    first = audit.audit_history(_observation(bank),actions,bank=bank,selection_samples=8,
        validation_samples=(8,),validation_repeats=1,seed=77,scorer=_fake_scorer(call1,77),replay_cache=cache)
    second = audit.audit_history(_observation(bank),dict(reversed(list(actions.items()))),bank=bank,
        selection_samples=8,validation_samples=(8,),validation_repeats=1,seed=77,
        scorer=_fake_scorer(call2,77),replay_cache=cache)
    assert first == second
    assert first["identical_predictive_mean_action_groups"] == [["constant_fixed","duplicate_constant"]]
    assert len(cache) == 3
    changed = actions["refit_milp"].p.copy();changed[0,1]+=4e-10
    actions["refit_milp"] = audit.PlannedAction("refit_milp",changed,actions["refit_milp"].q)
    audit.audit_history(_observation(bank),actions,bank=bank,selection_samples=8,
        validation_samples=(8,),validation_repeats=1,seed=77,scorer=_fake_scorer([],77),replay_cache=cache)
    assert len(cache) == 4


def test_forbidden_fields_changes_do_not_change_allowed_inputs_or_results(bank):
    allowed = {"records":[{"method":name,"action":{"p":action.p.tolist(),"q":action.q.tolist()}}
                          for name,action in _actions(bank).items()]}
    left = {**allowed,"truth":{"r":[1,2,3]},"case":"strong","seed":42,"noise":[99]}
    right = {**allowed,"truth":"poisoned","case":"different","seed":-7,"noise":None,
             "future":{"p":float("nan")},"records_extra":None}
    a,b = audit.extract_planned_actions(left),audit.extract_planned_actions(right)
    first = audit.audit_history(_observation(bank),a,bank=bank,selection_samples=8,
        validation_samples=(8,),validation_repeats=1,seed=71,scorer=_fake_scorer([],71))
    second = audit.audit_history(_observation(bank),b,bank=bank,selection_samples=8,
        validation_samples=(8,),validation_repeats=1,seed=71,scorer=_fake_scorer([],71))
    assert first == second


def test_binary_gaussian_information_matches_independent_quadrature():
    prior = np.array([.25,.75])
    posterior = audit.bayes.posterior_weights([0.],np.zeros((2,1)),np.array([0,1]),prior)
    means = np.array([[-.5],[.5]])
    def integrand(y):
        log_density = norm.logpdf(y,loc=means[:,0])
        log_mix = logsumexp(np.log(prior)+log_density)
        return float(np.sum(prior*np.exp(log_density)*(log_density-log_mix)))
    exact,error = quad(integrand,-12,12,epsabs=1e-11)
    assert error < 1e-9
    row = audit.bayes_rb.rb_eig_scores(posterior,{"actual":means},samples_per_topology=4096,seed=37)["actual"]
    assert abs(row["eig"]-exact) < 6*row["eig_se"]+1e-4
    assert abs(row["identity_difference"]) < 6*row["identity_difference_se"]+1e-10
    assert 0 <= exact <= posterior.topology_entropy


def test_identical_predictive_distributions_have_zero_information():
    posterior = audit.bayes.posterior_weights([0.],np.zeros((4,1)),np.arange(4),np.full(4,.25))
    row = audit.bayes_rb.rb_eig_scores(posterior,{"same":np.zeros((4,1))},samples_per_topology=128,seed=19)["same"]
    assert abs(row["eig"]) < 1e-12
    assert row["eig_se"] < 1e-12


def test_underflow_keeps_log_mass_and_log_entropy_diagnostic():
    posterior = audit.bayes.posterior_weights([0.],np.array([[0.],[100.],[101.],[102.]]),
                                            np.arange(4),np.full(4,.25))
    assert posterior.topology_entropy == 0
    diagnostics = audit._entropy_diagnostics(posterior,1e-10)
    assert diagnostics["saturated"]
    assert diagnostics["zero_displayed_topology_mass_count"] == 3
    assert np.isfinite(diagnostics["log_entropy_from_preserved_double_log_masses"])
    assert np.all(np.isfinite(posterior.log_topology_probabilities))
    assert "not exact zero" in diagnostics["interpretation"]


def test_minimal_real_eig_replay_and_all_four_posterior_labels(bank):
    result = audit.audit_history(_observation(bank),_actions(bank),bank=bank,
        selection_samples=16,validation_samples=(32,),validation_repeats=2,seed=81,batch_size=16)
    assert set(result["posterior"]["topology_probabilities"]) == {0,1,2,3}
    assert len(result["posterior_log_atom_weights"]) == 80
    assert len(result["validation"][0]["scores"]) == 2
    assert result["future_mean_hashes"]["none"] != result["future_mean_hashes"]["constant_fixed"]
    assert result["maximum_abs_identity_difference"] >= 0


def test_cli_batch_preserves_only_allowed_input_hashes(tmp_path,bank,capsys):
    input_dir = tmp_path/"inputs"; output_dir = tmp_path/"outputs"
    input_dir.mkdir()
    obs = _observation(bank)
    values = {"history_"+key:getattr(obs,key) for key in (*audit.CHANNELS,"root_v")}
    values["future_noise"] = np.array([object()],dtype=object)
    np.savez(input_dir/"trial_999_observations.npz",**values)
    trial = {"records":[{"method":name,"action":action.payload()} for name,action in _actions(bank).items()],
             "truth":{"forbidden":True},"case":"not used","seed":999}
    (input_dir/"trial_999.json").write_text(json.dumps(trial),encoding="utf-8")
    result = audit.run_audit(input_dir,output_dir,limit=1,sensitivity_limit=0,
        selection_samples=8,validation_samples=(8,),validation_repeats=1,seed=83,batch_size=8)
    assert result["complete_history_count"] == 1
    assert result["config"]["input_contract"]["truth_case_seed_noise_and_future_fields_accessed"] is False
    stored = json.loads((output_dir/"trial_999_eig.json").read_text(encoding="utf-8"))
    assert stored["finite_bank"]["atom_count"] == 80
    assert "truth" not in stored and "seed" not in stored
    assert stored["baseline"]["history_input_hash"] == audit.array_hash(*[getattr(obs,k) for k in (*audit.CHANNELS,"root_v")])


def test_invalid_duplicate_or_mismatched_actions_rejected(bank):
    zero = np.zeros((2,3))
    with pytest.raises(ValueError,match="duplicate"):
        audit.extract_planned_actions({"records":[{"method":"none","action":{"p":zero,"q":zero}}]*2})
    with pytest.raises(ValueError,match="P/Q"):
        audit.PlannedAction("bad",zero,np.zeros((3,3)))
    with pytest.raises(ValueError,match="positive integer"):
        audit.build_finite_bank(0,2)


def test_meter_cannot_change_under_frozen_history_means(bank):
    meter = replace(bank.meter,p_sigma=bank.meter.p_sigma*1.01)
    changed = replace(bank,meter=meter)
    with pytest.raises(ValueError,match="meter changed"):
        audit.audit_history(_observation(bank),_actions(bank),bank=changed,
            selection_samples=8,validation_samples=(8,),validation_repeats=1)


def test_changed_allowed_history_cannot_mix_existing_output_configuration(tmp_path,bank,capsys):
    input_dir = tmp_path/"inputs"; output_dir = tmp_path/"outputs"
    input_dir.mkdir()
    obs = _observation(bank)
    arrays = {"history_"+key:getattr(obs,key) for key in (*audit.CHANNELS,"root_v")}
    npz = input_dir/"trial_55_observations.npz"
    np.savez(npz,**arrays)
    payload = {"records":[{"method":name,"action":action.payload()} for name,action in _actions(bank).items()]}
    (input_dir/"trial_55.json").write_text(json.dumps(payload),encoding="utf-8")
    kwargs = dict(limit=1,sensitivity_limit=0,selection_samples=8,validation_samples=(8,),validation_repeats=1,batch_size=8)
    audit.run_audit(input_dir,output_dir,**kwargs)
    arrays["history_p"] = arrays["history_p"].copy()+1e-5
    np.savez(npz,**arrays)
    with pytest.raises(ValueError,match="mix audit configurations"):
        audit.run_audit(input_dir,output_dir,**kwargs)
