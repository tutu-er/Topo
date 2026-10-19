"""Fixed-schedule H-only proposal engineering, no posterior/EIG delivery."""
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
from . import continuous_eig_tempered_proposal as fit_module
from .continuous_eig_honly_run import DEFAULT_RESEARCH_ROOT
from .continuous_eig_honly_v2_run import below_normal_self

KAPPAS=(.001,.003,.01,.03,.1,.3,1.)
SEEDS=tuple(range(28201301,28201308))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run(research_root,output,v2_directory):
    begun=perf_counter();priority=below_normal_self()
    root,output=Path(research_root),Path(output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    candidates=tuple(RadialCandidate.from_supports(name,3,family) for name,family in zip(
        sampling.TOPOLOGY_NAMES,((),((0,1),),((0,2),),((1,2),))))
    given=root/"research_experiments/mobile_ev_mle/outputs/continuous_eig_Honly/H_710009.json"
    history,meter,actions,_=sampling.read_sanitized_bundle(given,given.with_suffix(".npz"))
    if not np.all(history.root_v==1.):
        raise ValueError("known root=1 required by the accepted proof")
    for action in actions:
        action.validate_nominal_domain()
    components=sampling.declared_components(candidates)
    v2config_path=Path(v2_directory)/"config_before_MC.json"
    v2config=json.loads(v2config_path.read_text())
    proposals=[]
    for c,payload in zip(components,v2config["proposals"]):
        if c.name!=payload["target_prior"]:
            raise ValueError("prior mode ordering differs from frozen v2")
        original=sampling.GaussianUniformProposal(c,payload["means_phi"],payload["covariances_phi"],.2)
        proposals.append(proposals_module.SequentialGaussianBoxProposal.from_gaussian(original))
    files=(Path(__file__),Path(sampling.__file__),Path(proposals_module.__file__),Path(guided_module.__file__),
        Path(fit_module.__file__),Path(engine.__file__),Path(inspect.getfile(RadialCandidate)),
        Path(__file__).parent/"continuous_eig_honly_v2_run.py")
    snapshot=output/"source_snapshot";snapshot.mkdir()
    for path in files:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for name in ("continuous_eig_tempered_proposal_plan.md","continuous_eig_guided_latent_plan.md","continuous_ac_inverse_math.md"):
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    deadline=datetime(2026,10,13,1,0,tzinfo=timezone.utc)
    config=dict(version="Honly_tempered_proposal_engineering_fixed_v1",history_id="710009",
        kappas=list(KAPPAS),stage_seeds=list(SEEDS),theta_per_mode=2048,latent_per_time=32,
        soft_seconds=1200.,check_budget_after_complete_stage=True,hard_outer_deadline_UTC=deadline.isoformat(),
        source_sha256={p.name:digest(p) for p in files},
        plan_sha256=digest(Path(__file__).parent/"continuous_eig_tempered_proposal_plan.md"),
        sanitized_inputs_sha256={given.name:digest(given),given.with_suffix(".npz").name:digest(given.with_suffix(".npz"))},
        initial_v2_frozen_centers_and_profiles_hash=digest(v2config_path),
        H_array_hash=history.identity(),meter_hash=meter.identity(),
        proper_prior_unchanged=True,prior_components=[c.payload() for c in components],
        initial_normalized_proposals=[p.payload() for p in proposals],
        engineering_weight="pi(beta)/q(beta) * Lhat_H(beta)^kappa",
        random_Lhat_power_is_exact_tempered_posterior=False,pilot_probabilities_or_EIG_reported=False,
        fit=dict(minimum_ESS="dimension+2",previous_covariance="mean of engineering Gaussian covariance matrices",
            new_covariance=".9 Cweighted + .1 Cprevious + 1e-6 I",new_component_count=1,
            uniform_coverage_mass=.2,failure="retain exact last valid normalized proposal and mode"),
        latent=dict(guidance_baseline_mass=.2,FD_white_step=.02,true_nonlinear_target=True,meter_ratio_restored=True),
        stream_contract=dict(children=9,normal_child=0,theta_children=list(range(1,8)),mixture_uniform_child=8,
            time_clouds_independent=True,every_stage_fresh=True,old_weights_reused=False),
        final_independent_posterior_integration_authorized=False,predictive_MC_authorized=False,
        mathematical_H_prior_update_count_in_any_future_final_estimator=1,
        tests="53 independent sampler/proposal/guidance/weightedfit tests passed before pilot",
        process_priority=priority,runtime=dict(executable=sys.executable,python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__,threads=1))
    sampling.save_json(output/"config_before_pilot.json",config)
    print(json.dumps(dict(phase="frozen_before_pilot",hash=digest(output/"config_before_pilot.json"),
                         kappas=list(KAPPAS),priority=priority)),flush=True)
    stages=[];stop_reason="fixed_schedule_complete"
    for stage,(kappa,seed) in enumerate(zip(KAPPAS,SEEDS)):
        if perf_counter()-begun>=1200 or datetime.now(timezone.utc)>=deadline:
            stop_reason="pre_next_stage_soft_budget_or_outer_deadline";break
        stage_dir=output/f"stage_{stage}";stage_dir.mkdir()
        sampling.save_json(stage_dir/"proposals_before.json",[p.payload() for p in proposals])
        streams=np.random.SeedSequence(seed).spawn(9)
        z=np.random.default_rng(streams[0]).standard_normal((history.t,32,8))
        uniforms=np.random.default_rng(streams[8]).random((history.t,32))
        stage_rows=[];arrays={};next_proposals=[];t=perf_counter()
        for index,(c,p,stream) in enumerate(zip(components,proposals,streams[1:8])):
            try:
                beta,logq=p.sample(2048,np.random.default_rng(stream))
                likelihood=guided_module.guided_latent_likelihood(c,beta,history,meter,z,uniforms,engine,
                    baseline_mass=.2,fd_white_step=.02)
                fit=fit_module.fit_tempered_proposal(c,p,beta,logq,likelihood.loglik,kappa,
                    minimum_ess=2*c.candidate.m+2,old_covariance_mass=.1,diagonal_floor=1e-6)
            except (RuntimeError,ValueError,ArithmeticError) as error:
                failure=dict(status="numerical_failure",stage=stage,kappa=kappa,seed=seed,component=c.name,
                    exception=type(error).__name__,reason=str(error),completed_stages=stages,
                    all_modes_preserved=[c.name for c in components],mode_dropped=False,
                    target_or_proposal_failure_not_zero_model_probability=True,seconds=perf_counter()-begun,
                    final_posterior_integration_executed=False,predictive_MC_executed=False)
                sampling.save_json(stage_dir/"numerical_failure.json",failure)
                sampling.save_json(output/"summary.json",failure)
                print(json.dumps(dict(phase="numerical_failure",stage=stage,component=c.name,reason=str(error))),flush=True)
                return proposals
            next_proposals.append(fit.proposal)
            logw=c.prior_logpdf(beta)-logq+kappa*likelihood.loglik
            normalizer=float(logsumexp(logw));w=np.exp(logw-normalizer) if np.isfinite(normalizer) else np.zeros(len(beta))
            active=w>0
            with np.errstate(divide="ignore",invalid="ignore"):
                relse=np.sqrt(np.maximum(32/likelihood.time_ess-1.,0.)/31)
            weighted_ess=None if not np.any(active) else np.sum(w[active,None]*likelihood.time_ess[active],axis=0)
            weighted_se=None if not np.any(active) else np.sum(w[active,None]*relse[active],axis=0)
            row=dict(component=c.name,stage=stage,kappa=kappa,seed=seed,
                proposal_fit=fit.diagnostics,latent_likelihood_diagnostics=likelihood.diagnostics(),
                random_tempered_weights_are_posterior=False,
                engineering_weighted_time_latent_ESS=None if weighted_ess is None else weighted_ess.tolist(),
                engineering_weighted_time_relative_SE=None if weighted_se is None else weighted_se.tolist(),
                q_before=p.payload(),q_after=fit.proposal.payload(),mode_preserved=True)
            stage_rows.append(row)
            for key,value in (("beta",beta),("logq",logq),("loghistory",likelihood.loglik),
                ("latent_ess",likelihood.time_ess),("log_engineering_weight",logw)):
                arrays[f"{key}_{index}"]=value
            sampling.save_json(stage_dir/f"mode_{index}.json",row)
        proposals=next_proposals
        np.savez_compressed(stage_dir/"pilot_particles.npz",**arrays)
        sampling.save_json(stage_dir/"proposals_after.json",[p.payload() for p in proposals])
        row=dict(stage=stage,kappa=kappa,seed=seed,seconds=perf_counter()-t,
            conditional_engineering_ESS=[r["proposal_fit"]["conditional_ESS"] for r in stage_rows],
            fit_status=[r["proposal_fit"]["status"] for r in stage_rows],
            failed_or_fallback_components=[r["component"] for r in stage_rows if r["proposal_fit"]["status"]!="proposal_updated"],
            numerical_failure_count=0,all_modes_preserved=True,
            proposals_after_hash=digest(stage_dir/"proposals_after.json"))
        stages.append(row);sampling.save_json(stage_dir/"summary.json",row)
        sampling.save_json(output/"checkpoint.json",dict(completed_stages=stages,
            seconds=perf_counter()-begun,latest_normalized_proposals=[p.payload() for p in proposals],
            final_H_weights_or_posterior_created=False,predictive_MC_executed=False))
        print(json.dumps(dict(phase="stage_complete",**row)),flush=True)
    final=dict(status="engineering_pilot_complete" if len(stages)==len(KAPPAS) else "engineering_pilot_budget_checkpoint",
        stop_reason=stop_reason,seconds=perf_counter()-begun,stages=stages,
        final_normalized_proposals=[p.payload() for p in proposals],
        numerical_failure_count=0,all_modes_preserved=True,prior_unchanged=True,
        pilot_weight_is_exact_tempered_posterior=False,pilot_weights_enter_final_estimator=False,
        final_independent_H_integration_executed=False,predictive_MC_executed=False)
    sampling.save_json(output/"summary.json",final)
    sampling.save_json(output/"final_proposals_unvalidated_for_posterior.json",[p.payload() for p in proposals])
    print(json.dumps(dict(phase="complete",status=final["status"],seconds=final["seconds"],
                         final_fit_status=[] if not stages else stages[-1]["fit_status"])),flush=True)
    return proposals


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root",type=Path,default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--v2-directory",type=Path,default=Path(__file__).parent/"continuous_eig_Honly_v2")
    args=parser.parse_args();run(args.research_root,args.output,args.v2_directory)
