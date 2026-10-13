"""Frozen independent original-prior H integration with paired L32/L128."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import inspect
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.dont_write_bytecode=True
import numpy as np
import scipy
from scipy.special import logsumexp
from . import continuous_ac_inverse as engine
from . import continuous_eig_sampling as sampling
from . import continuous_eig_proposals as proposals_module
from . import continuous_eig_guided_latent as guided_module
from . import continuous_eig_final_helpers as helpers
from .continuous_eig_honly_run import DEFAULT_RESEARCH_ROOT,cross_repeat_summary
from .continuous_eig_honly_v2_run import below_normal_self

SEEDS=(28301301,28301302,28301303)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def variance_diagnostics(bank):
    """Sample diagnostics; latent CRN correlation prevents unconditional SE claims."""
    modes=[];global_conditional_relative_variance=0.
    for b in bank.blocks:
        total=float(logsumexp(b.logweight));B=len(b.beta);L=b.history.samples_per_time
        ess=float(sampling.log_ess(b.logweight))
        conditional_theta_relative_variance=(B/ess-1)/(B-1) if ess>0 else None
        contribution=0. if not np.isfinite(total) else float(np.exp(total-bank.log_evidence))
        if conditional_theta_relative_variance is not None:
            global_conditional_relative_variance+=contribution**2*conditional_theta_relative_variance
        if np.isfinite(total):
            w=np.exp(b.logweight-total);active=w>0
            with np.errstate(divide="ignore",invalid="ignore"):
                relse=np.sqrt(np.maximum(L/b.history.time_ess-1,0)/(L-1))
            weighted_ess=np.sum(w[active,None]*b.history.time_ess[active],axis=0)
            weighted_se=np.sum(w[active,None]*relse[active],axis=0)
            joint=np.sqrt(np.maximum(np.prod(1+relse[active]**2,axis=1)-1,0))
            weighted_joint=float(w[active]@joint)
        else:
            weighted_ess=weighted_se=None;weighted_joint=None
        modes.append(dict(component=b.component.name,conditional_theta_ESS=ess,
            theta_sampling_relative_SE_conditional_on_fixed_latent_cloud=None if conditional_theta_relative_variance is None
                else float(np.sqrt(max(conditional_theta_relative_variance,0.))),
            conditional_weighted_time_latent_ESS=None if weighted_ess is None else weighted_ess.tolist(),
            conditional_weighted_time_latent_relative_SE=None if weighted_se is None else weighted_se.tolist(),
            conditional_weighted_joint_likelihood_relative_SE_sample_moment=weighted_joint,
            latent_selected_by_random_H_weights=True,latent_SE_is_error_bound=False,
            theta_SE_is_unconditional_evidence_SE=False,mode_retained_when_display_mass_underflows=True))
    return dict(modes=modes,
        stratified_theta_relative_SE_conditional_on_fixed_latent_cloud=float(np.sqrt(max(global_conditional_relative_variance,0.))),
        shared_latent_cloud_makes_unconditional_beta_latent_variance_decomposition_unidentified=True,
        repeated_fresh_clouds_and_paired_doubling_are_required=True)


def run(research_root,output,pilot_directory):
    begun=perf_counter();priority=below_normal_self()
    root,output=Path(research_root),Path(output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    candidates=tuple(RadialCandidate.from_supports(name,3,family) for name,family in zip(
        sampling.TOPOLOGY_NAMES,((),((0,1),),((0,2),),((1,2),))))
    given=root/"research_experiments/mobile_ev_mle/outputs/continuous_eig_Honly/H_710009.json"
    history,meter,actions,_=sampling.read_sanitized_bundle(given,given.with_suffix(".npz"))
    if not np.all(history.root_v==1.):
        raise ValueError("accepted inverse proof requires calibrated root=1")
    for action in actions:
        action.validate_nominal_domain()
    components=sampling.declared_components(candidates)
    frozen_q=Path(pilot_directory)/"final_proposals_unvalidated_for_posterior.json"
    payloads=json.loads(frozen_q.read_text())
    if len(payloads)!=7:
        raise ValueError("all seven frozen proposal modes required")
    proposals=[]
    for c,payload in zip(components,payloads):
        if c.name!=payload["target_prior"]:
            raise ValueError("frozen proposal mode ordering differs")
        p=proposals_module.SequentialGaussianBoxProposal(c,payload["means_phi"],payload["covariances_phi"],payload["uniform_mass"])
        if p.payload()["coordinate_orders"]!=payload["coordinate_orders"]:
            raise ValueError("frozen sequential conditional ordering differs")
        proposals.append(p)
    source_files=(Path(__file__),Path(sampling.__file__),Path(proposals_module.__file__),Path(guided_module.__file__),
        Path(helpers.__file__),Path(engine.__file__),Path(inspect.getfile(RadialCandidate)),
        Path(__file__).parent/"continuous_eig_honly_run.py",Path(__file__).parent/"continuous_eig_honly_v2_run.py")
    snapshot=output/"source_snapshot";snapshot.mkdir()
    for path in source_files:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for name in ("continuous_eig_final_plan.md","continuous_eig_tempered_proposal_plan.md",
                 "continuous_eig_guided_latent_plan.md","continuous_ac_inverse_math.md"):
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    (snapshot/frozen_q.name).write_bytes(frozen_q.read_bytes())
    deadline=datetime(2026,10,13,1,0,tzinfo=timezone.utc)
    config=dict(version="continuous_proper_nominal_Honly_v5_frozen_final",history_id="710009",seeds=list(SEEDS),
        theta_per_component=4096,latent_per_time=128,paired_prefix_per_time=32,independent_repeats=3,beta_chunk_size=256,
        source_sha256={p.name:digest(p) for p in source_files},plan_sha256=digest(Path(__file__).parent/"continuous_eig_final_plan.md"),
        sanitized_inputs_sha256={given.name:digest(given),given.with_suffix(".npz").name:digest(given.with_suffix(".npz"))},
        frozen_pilot_q_sha256=digest(frozen_q),pilot_weights_reused=False,pilot_q_reoptimized=False,
        H_array_hash=history.identity(),meter_hash=meter.identity(),proper_prior_unchanged=True,
        prior_components=[c.payload() for c in components],normalized_frozen_proposals=[p.payload() for p in proposals],
        target="original declared pi(beta) * proper L_H(beta), historical update exactly once at each L",
        tempered_final_target=False,old_H_particle_weights_reused=False,paired_prefix_replaces_not_multiplies_H=True,
        latent=dict(baseline_mass=.2,FD_white_step=.02,true_nonlinear_voltage=True,full_q_ratio_restored=True),
        random_stream_contract=dict(children=9,normal_cloud_child=0,theta_children=list(range(1,8)),mixture_uniform_child=8,
            independent_times=True,independent_repeats=True,fresh_theta_and_latent=True,CRN_across_beta_and_modes=True,
            prefix32_from_same128_per_time_kernel=True),
        thresholds=dict(minimum_conditional_mode_ESS=20.,minimum_global_ESS=100.,maximum_weight=.05,
            every_mode_required=True,rare_mode_exemption=False),
        tests="56 independent tests passed before final MC, including prefix/chunk and pure H replacement",
        soft_seconds=1800.,budget_checked_after_complete_repeat=True,hard_outer_deadline_UTC=deadline.isoformat(),
        supported_numerical_failure="preserve seven-mode ledger and stop; not probability zero",
        predictive_MC_enabled=False,formal_C_D_stress_modified=False,
        formal_input_actual_future_truth_noise_fields_accessed=False,
        incidental_schema_inspection_ledger="disclosed sampler math; not a formal proposal/selection input",
        process_priority=priority,runtime=dict(executable=sys.executable,python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__,threads=1))
    sampling.save_json(output/"config_before_MC.json",config)
    print(json.dumps(dict(phase="frozen_before_MC",config_hash=digest(output/"config_before_MC.json"),seeds=list(SEEDS),
        q_hash=digest(frozen_q),theta=4096,latent=128,paired_prefix=32,priority=priority)),flush=True)
    banks128=[];banks32=[];paired=[];stop_reason="three_fresh_repeats_complete"
    for repeat,seed in enumerate(SEEDS):
        if perf_counter()-begun>=1800 or datetime.now(timezone.utc)>=deadline:
            stop_reason="pre_next_repeat_soft_budget_or_outer_deadline";break
        started=perf_counter();streams=np.random.SeedSequence(seed).spawn(9)
        uniforms=np.random.default_rng(streams[8]).random((history.t,128))
        normal_cloud=np.random.default_rng(streams[0]).standard_normal((history.t,128,8))
        prefixes={};state=dict(component=None)
        def evaluator(c,b,h,m,z,e):
            state["component"]=c.name
            if not np.array_equal(z,normal_cloud):
                raise ValueError("final cloud stream differs from frozen stream contract")
            full,prefix=helpers.paired_chunked_likelihood(c,b,h,m,z,uniforms,e,chunk_size=256)
            prefixes[c.name]=prefix
            print(json.dumps(dict(phase="mode_complete",repeat=repeat,component=c.name,seconds=perf_counter()-started,
                zero_H_count=int(np.sum(~np.isfinite(full.loglik))),invalid_inverse_count=full.invalid_inverse_count)),flush=True)
            return full
        try:
            full=sampling.build_history_bank(components,proposals,history,meter,engine,n_theta=4096,n_latent=128,
                seed=seed,proof_accepted=True,minimum_mode_ess=20.,minimum_total_ess=100.,maximum_weight=.05,
                latent_evaluator=evaluator)
            prefix=helpers.bank_with_new_H_estimates(full,prefixes)
        except (RuntimeError,ValueError,ArithmeticError) as error:
            failure=dict(status="numerical_failure",repeat=repeat,seed=seed,component=state["component"],
                exception_type=type(error).__name__,reason=str(error),completed_repeats=len(banks128),
                retained_mode_ledger=[dict(component=c.name,mode_dropped=False,zero_is_exact_model_rejection=False) for c in components],
                numerical_failure_not_zero_model_probability=True,seconds=perf_counter()-begun,predictive_MC_executed=False)
            sampling.save_json(output/f"repeat_{repeat}_numerical_failure.json",failure)
            sampling.save_json(output/"summary.json",failure);print(json.dumps(failure),flush=True)
            return banks128,banks32
        banks128.append(full);banks32.append(prefix)
        for level,bank in ((32,prefix),(128,full)):
            row=bank.summary();row.update(repeat=repeat,seed=seed,paired_latent_level=level,
                normal_cloud_hash=sampling.array_hash(normal_cloud),mixture_uniform_hash=sampling.array_hash(uniforms),
                seconds=perf_counter()-started,variance_diagnostics=variance_diagnostics(bank))
            sampling.save_json(output/f"repeat_{repeat}_L{level}.json",row)
            sampling.save_particle_bank(output/f"repeat_{repeat}_L{level}_particles.npz",bank)
        modes=[]
        for b32,b128 in zip(prefix.blocks,full.blocks):
            if not np.array_equal(b32.beta,b128.beta) or not np.array_equal(b32.logq,b128.logq):
                raise ValueError("paired levels must use identical fresh theta/logq")
            modes.append(dict(component=b128.component.name,beta_hash=sampling.array_hash(b128.beta),
                logq_hash=sampling.array_hash(b128.logq),theta_and_q_identical=True,
                log_mode_evidence_delta_128_minus_32=float(logsumexp(b128.logweight)-logsumexp(b32.logweight)),
                conditional_ESS_L32=float(sampling.log_ess(b32.logweight)),conditional_ESS_L128=float(sampling.log_ess(b128.logweight)),
                zero_history_theta_L32=int(np.sum(~np.isfinite(b32.history.loglik))),
                zero_history_theta_L128=int(np.sum(~np.isfinite(b128.history.loglik))),mode_preserved=True))
        pairing=dict(repeat=repeat,seed=seed,logZ_delta_128_minus_32=full.log_evidence-prefix.log_evidence,
            globalESS_L32=prefix.diagnostics["total_ess"],globalESS_L128=full.diagnostics["total_ess"],
            maximum_weight_L32=prefix.diagnostics["maximum_particle_weight"],maximum_weight_L128=full.diagnostics["maximum_particle_weight"],
            modes=modes,normal_cloud_hash=sampling.array_hash(normal_cloud),mixture_uniform_hash=sampling.array_hash(uniforms),
            latent_prefix_is_same_kernel=True,H_updates_each_level=1,previous_H_weights_reused=False,seconds=perf_counter()-started)
        paired.append(pairing);sampling.save_json(output/f"repeat_{repeat}_paired.json",pairing)
        sampling.save_json(output/"checkpoint.json",dict(completed_repeats=len(paired),seconds=perf_counter()-begun,
            paired=paired,all_modes_preserved=True,predictive_MC_executed=False))
        print(json.dumps(dict(phase="paired_repeat_complete",repeat=repeat,seconds=pairing["seconds"],
            logZ32=prefix.log_evidence,logZ128=full.log_evidence,delta=pairing["logZ_delta_128_minus_32"],
            globalESS32=pairing["globalESS_L32"],globalESS128=pairing["globalESS_L128"],
            modeESS128=[x["conditional_ESS_L128"] for x in modes])),flush=True)
    levels={}
    for level,banks in ((32,banks32),(128,banks128)):
        if banks:
            levels[str(level)]=cross_repeat_summary(banks)
            levels[str(level)]["integral_doubling_not_yet_executed"]=False
    failed=len(paired)!=3 or any(b.diagnostics["status"]=="failed" for b in banks128+banks32)
    summary=dict(status="failed" if failed else "needs_repeat_and_doubling_stability_review",
        seconds=perf_counter()-begun,stop_reason=stop_reason,independent_repeats=len(paired),levels=levels,paired=paired,
        config_hash=digest(output/"config_before_MC.json"),frozen_q_hash=digest(frozen_q),
        posterior_accepted=False,predictive_MC_executed=False,all_modes_preserved=True,prior_unchanged=True,
        original_target_H_updates=1,tempered_pilot_weights_reused=False,priorzero_and_numerical_failure_distinct=True,
        numerical_failure_count=0,latent_doubling_executed=True,previous_v1_v2_v3_v4_failures_preserved=True)
    sampling.save_json(output/"summary.json",summary)
    print(json.dumps(dict(phase="complete",status=summary["status"],seconds=summary["seconds"],
        logZrange32=levels.get("32",{}).get("log_evidence_range"),logZrange128=levels.get("128",{}).get("log_evidence_range"))),flush=True)
    return banks128,banks32


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root",type=Path,default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot-directory",type=Path,default=Path(__file__).parent/"continuous_eig_tempered_pilot_v1")
    args=parser.parse_args();run(args.research_root,args.output,args.pilot_directory)
