"""Frozen H-only guided-latent v4; no beta fitting or predictive run."""
from __future__ import annotations
import argparse
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
from . import continuous_ac_inverse as engine
from . import continuous_eig_sampling as sampling
from . import continuous_eig_proposals as proposals_module
from . import continuous_eig_guided_latent as guided_module
from .continuous_eig_honly_run import cross_repeat_summary,DEFAULT_RESEARCH_ROOT
from .continuous_eig_honly_v2_run import below_normal_self

SEEDS=(27101301,27101302,27101303)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run(research_root,output,v2_directory):
    begun=perf_counter();priority=below_normal_self()
    root,output=Path(research_root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    candidates=tuple(RadialCandidate.from_supports(name,3,family) for name,family in zip(
        sampling.TOPOLOGY_NAMES,((),((0,1),),((0,2),),((1,2),))))
    given=root/"research_experiments/mobile_ev_mle/outputs/continuous_eig_Honly/H_710009.json"
    history,meter,actions,_=sampling.read_sanitized_bundle(given,given.with_suffix(".npz"))
    if not np.all(history.root_v==1.):
        raise ValueError("accepted proof requires root=1")
    for action in actions:
        action.validate_nominal_domain()
    components=sampling.declared_components(candidates)
    v2config_path=Path(v2_directory)/"config_before_MC.json"
    v2config=json.loads(v2config_path.read_text())
    proposals=[]
    for c,payload in zip(components,v2config["proposals"]):
        if c.name!=payload["target_prior"]:
            raise ValueError("frozen prior component ordering differs")
        gaussian=sampling.GaussianUniformProposal(c,np.asarray(payload["means_phi"]),
            np.asarray(payload["covariances_phi"]),.2)
        proposals.append(proposals_module.SequentialGaussianBoxProposal.from_gaussian(gaussian))
    source_files=(Path(__file__),Path(sampling.__file__),Path(proposals_module.__file__),
        Path(guided_module.__file__),Path(engine.__file__),Path(inspect.getfile(RadialCandidate)),
        Path(__file__).parent/"continuous_eig_honly_run.py",Path(__file__).parent/"continuous_eig_honly_v2_run.py")
    snapshot=output/"source_snapshot";snapshot.mkdir()
    for path in source_files:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for name in ("continuous_eig_guided_latent_plan.md","continuous_eig_Honly_v3_plan.md","continuous_ac_inverse_math.md"):
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    config=dict(version="continuous_proper_nominal_Honly_v4_frozen",history_id="710009",seeds=list(SEEDS),
        theta_per_component=2048,latent_per_time=32,independent_repeats=3,
        source_sha256={p.name:digest(p) for p in source_files},
        plan_sha256=digest(Path(__file__).parent/"continuous_eig_guided_latent_plan.md"),
        sanitized_inputs_sha256={given.name:digest(given),given.with_suffix(".npz").name:digest(given.with_suffix(".npz"))},
        v2_frozen_proposal_config_sha256=digest(v2config_path),H_array_hash=history.identity(),meter_hash=meter.identity(),
        beta_means_profile_covariance_reoptimized=False,prior_unchanged=True,
        prior_components=[c.payload() for c in components],beta_proposals=[p.payload() for p in proposals],
        latent_proposal=dict(baseline_mass=.2,guided_mass=.8,space="white w=(x-m0)/sigma_meter",
            H="diag(1/sigmaV)*dV/dwhite",r="(yVcal-V0)/sigmaV",precision="I+H.T@H",
            FD_white_step=.02,FD_shapes_only_q=True,linear_voltage_in_target=False,
            full_mixture_density=True,meter_density_ratio_restored=True,
            unavailable_guidance_outside_target="normalized p0 fallback, mode retained",
            invalid_in_true_target_support="numerical_failure stops experiment",
            nonfinite_supported_logtarget="numerical_failure, never probability zero"),
        random_stream_contract=dict(seed_sequence_children=9,normal_cloud_child=0,
            theta_proposal_children=list(range(1,8)),mixture_uniform_child=8,
            time_clouds_independent=True,normal_and_uniform_streams_independent=True,CRN_across_beta=True),
        tests="48 independent combined tests passed before v4 MC",
        thresholds=dict(minimum_conditional_mode_ESS=20.,minimum_global_ESS=100.,maximum_weight=.05),
        H_updates=1,WZZT_extra_likelihood=False,old_pilot_particle_weights_reused=False,
        predictive_MC_enabled=False,latent_doubling_enabled=False,
        formal_C_D_stress_modified=False,actual_evaluation_future_truth_noise_accessed=False,
        incidental_earlier_schema_inspection="Disclosed in sampler math; not a formal input or proposal/filter selector",
        process_priority=priority,runtime=dict(executable=sys.executable,python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__,threads=1))
    sampling.save_json(output/"config_before_MC.json",config)
    print(json.dumps(dict(phase="frozen_before_MC",config_hash=digest(output/"config_before_MC.json"),
        seeds=list(SEEDS),priority=priority)),flush=True)
    banks=[]
    for repeat,seed in enumerate(SEEDS):
        t=perf_counter()
        stream=np.random.SeedSequence(seed).spawn(9)[8]
        uniforms=np.random.default_rng(stream).random((history.t,32))
        state=dict(current_component=None)
        def evaluator(c,b,h,m,z,e):
            state["current_component"]=c.name
            return guided_module.guided_latent_likelihood(c,b,h,m,z,uniforms,e,
                baseline_mass=.2,fd_white_step=.02)
        try:
            bank=sampling.build_history_bank(components,proposals,history,meter,engine,n_theta=2048,
                n_latent=32,seed=seed,proof_accepted=True,minimum_mode_ess=20.,minimum_total_ess=100.,
                maximum_weight=.05,latent_evaluator=evaluator)
        except (RuntimeError,ValueError,ArithmeticError) as error:
            failure=dict(status="numerical_failure",repeat=repeat,seed=seed,component=state["current_component"],
                exception_type=type(error).__name__,reason=str(error),
                retained_component_ledger=[dict(component=c.name,status="failed_or_not_completed_preserved",
                    mode_dropped=False,zero_is_exact_rejection=False) for c in components],
                completed_repeats=len(banks),seconds=perf_counter()-begun,predictive_MC_executed=False,
                experiment_stopped_not_conditioned_on_success=True)
            sampling.save_json(output/f"repeat_{repeat}_numerical_failure.json",failure)
            sampling.save_json(output/"summary.json",failure)
            print(json.dumps(failure),flush=True)
            return banks
        banks.append(bank)
        row=bank.summary();row.update(repeat=repeat,seconds=perf_counter()-t,mixture_uniform_hash=sampling.array_hash(uniforms))
        sampling.save_json(output/f"repeat_{repeat}.json",row)
        sampling.save_particle_bank(output/f"repeat_{repeat}_particles.npz",bank)
        print(json.dumps(dict(phase="H_repeat_complete",repeat=repeat,seconds=row["seconds"],
            status=bank.diagnostics["status"],total_ESS=bank.diagnostics["total_ess"],
            maximum_weight=bank.diagnostics["maximum_particle_weight"],
            conditional_mode_ESS=[r["conditional_particle_ESS"] for r in bank.diagnostics["modes"]],
            guidance_fallback_counts=[r["guided_latent_proposal"]["guidance_fallback_outside_target_support_count"]
                for r in bank.diagnostics["modes"]])),flush=True)
    summary=cross_repeat_summary(banks)
    summary.update(seconds=perf_counter()-begun,config_hash=digest(output/"config_before_MC.json"),
        v1_v2_v3_failure_evidence_retained=True,beta_proposal_reoptimized=False,prior_unchanged=True,
        true_nonlinear_voltage_target=True,meter_ratio_restored=True,predictive_MC_executed=False,
        latent_doubling_executed=False,numerical_failure_count=0)
    sampling.save_json(output/"summary.json",summary)
    print(json.dumps(dict(phase="complete",status=summary["status"],seconds=summary["seconds"],
        log_evidence_range=summary["log_evidence_range"])),flush=True)
    return banks


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root",type=Path,default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--v2-directory",type=Path,default=Path(__file__).parent/"continuous_eig_Honly_v2")
    args=parser.parse_args()
    run(args.research_root,args.output,args.v2_directory)
