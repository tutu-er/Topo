"""Parent-authorized H-only local proposal repair and fresh final importance.

Target/priors are unchanged. The bounded fixed-cloud local optimizer defines
only q's mean; its result is neither a prior nor a global MLE. Final weights
use completely fresh theta and latent streams, excluding all old pilot weights.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.optimize import minimize

from . import continuous_ac_inverse as engine
from . import continuous_eig_sampling as sampling
from .continuous_eig_honly_run import cross_repeat_summary, DEFAULT_RESEARCH_ROOT

OPT_SEEDS = tuple(range(24101301,24101308))
FINAL_SEEDS = (25101301,25101302,25101303)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def below_normal_self():
    if sys.platform != "win32":
        return dict(attempted=False,reason="non-Windows host")
    try:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE,wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        result = bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000))
        return dict(attempted=True,applied=result,priority="BELOW_NORMAL_PRIORITY_CLASS",
                    only_current_process=True)
    except Exception as error:
        return dict(attempted=True,applied=False,error=str(error),only_current_process=True)


class OptimizationStop(RuntimeError):
    pass


def best_v1_start(component,component_index,v1_directory):
    best = None
    for repeat in range(3):
        path = Path(v1_directory)/f"repeat_{repeat}_particles.npz"
        with np.load(path,allow_pickle=False) as arrays:
            beta = np.asarray(arrays[f"beta_{component_index}"],float)
            values = np.asarray(arrays[f"loghistory_{component_index}"],float)
        allowed = np.isfinite(component.prior_logpdf(beta)) & np.isfinite(values)
        if np.any(allowed):
            index = int(np.argmax(np.where(allowed,values,-np.inf)))
            candidate = dict(repeat=repeat,index=index,beta=beta[index],stored_log_H_likelihood=float(values[index]),
                             particle_file=str(path),particle_file_sha256=digest(path))
            if best is None or candidate["stored_log_H_likelihood"] > best["stored_log_H_likelihood"]:
                best = candidate
    if best is None:
        raise RuntimeError("all v1 H estimates zero for this retained mode; no permitted start")
    return best


def local_center(component,start,history,meter,seed,*,seconds=60.,maxfev=2000):
    begun = perf_counter()
    z = np.random.default_rng(seed).standard_normal((history.t,32,8))
    ledger, best = [], None
    stop_reason = None
    numeric_error = None
    result = None

    def objective(phi):
        nonlocal best
        if perf_counter()-begun >= seconds:
            raise OptimizationStop("wall_budget_before_next_evaluation")
        if len(ledger) >= maxfev:
            raise OptimizationStop("maxfev_before_next_evaluation")
        phi = np.asarray(phi,float)
        beta = phi*component.nominal_beta
        estimate = sampling.latent_likelihood(component,beta[None],history,meter,z,engine)
        loglik = float(estimate.loglik[0])
        row = dict(index=len(ledger),seconds=perf_counter()-begun,phi=phi.tolist(),
                   log_H_likelihood=loglik,finite_joint=bool(np.isfinite(loglik)),
                   physical_inverse_failure=False)
        ledger.append(row)
        if np.isfinite(loglik) and (best is None or loglik > best["loglik"]):
            best = dict(phi=phi.copy(),beta=beta.copy(),loglik=loglik,
                        evaluation_index=len(ledger)-1,diagnostics=estimate.diagnostics())
        return 1e100 if not np.isfinite(loglik) else -loglik

    phi0 = np.clip(np.asarray(start["beta"])/component.nominal_beta,.8,1.2)
    try:
        objective(phi0)
        result = minimize(objective,phi0,method="Powell",bounds=[(.8,1.2)]*len(phi0),
            options=dict(xtol=.0005,ftol=.001,maxfev=maxfev-1))
    except OptimizationStop as error:
        stop_reason = str(error)
    except (RuntimeError,ValueError,ArithmeticError) as error:
        numeric_error = str(error)
    elapsed = perf_counter()-begun
    selected = np.asarray(start["beta"],float) if best is None else best["beta"]
    optimized = dict(component=component.name,seed=seed,latent_per_time=32,
        latent_cloud_hash=sampling.array_hash(z),latent_time_clouds_independent=True,
        optimization_cloud_reused_by_final_IS=False,budget_seconds=seconds,maxfev=maxfev,
        actual_seconds=elapsed,evaluation_count=len(ledger),method="bounded_Powell_scaled_phi",
        starting_v1_record=start,starting_phi=phi0.tolist(),
        selected_beta=selected.tolist(),selected_phi=(selected/component.nominal_beta).tolist(),
        best_log_H_likelihood=None if best is None else best["loglik"],
        best_evaluation_index=None if best is None else best["evaluation_index"],
        selected_point_local_only=True,selected_point_is_prior_or_global_MLE=False,
        optimizer_success=False if result is None else bool(result.success),
        optimizer_message=None if result is None else str(result.message),
        stop_reason=stop_reason,numerical_failure=numeric_error,
        status="numerical_failure" if numeric_error else "no_finite_fixed_cloud_point" if best is None else
               "budget_stopped_best_retained" if stop_reason else "completed_local_engineering_search",
        best_point_latent_diagnostics=None if best is None else best["diagnostics"],
        evaluation_ledger=ledger)
    return selected,optimized


def run(research_root,output,v1_directory):
    started = perf_counter()
    priority = below_normal_self()
    root,output = Path(research_root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    candidates = tuple(RadialCandidate.from_supports(name,3,family) for name,family in zip(
        sampling.TOPOLOGY_NAMES,((),((0,1),),((0,2),),((1,2),))))
    given = root/"research_experiments/mobile_ev_mle/outputs/continuous_eig_Honly/H_710009.json"
    history,meter,actions,centers = sampling.read_sanitized_bundle(given,given.with_suffix(".npz"))
    if not np.all(history.root_v == 1.):
        raise ValueError("accepted proof requires known root=1")
    for action in actions:
        action.validate_nominal_domain()
    components = sampling.declared_components(candidates)
    original_proposals = tuple(sampling.GaussianUniformProposal.from_history_centers(c,centers[c.candidate.name],
                              ridge=75.,uniform_mass=.2) for c in components)
    starts = [best_v1_start(c,i,v1_directory) for i,c in enumerate(components)]
    source_files = (Path(__file__),Path(sampling.__file__),Path(engine.__file__),
                    Path(inspect.getfile(RadialCandidate)),Path(__file__).parent/"continuous_eig_honly_run.py")
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for path in source_files:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for name in ("continuous_eig_sampling_math.md","continuous_ac_inverse_math.md","continuous_eig_Honly_v2_plan.md"):
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    config = dict(version="continuous_proper_nominal_Honly_v2_frozen",history_id="710009",
        input_contract="root-sanitized H-only observations and H-only center summaries; frozen action declarations",
        sanitized_input_sha256={given.name:digest(given),given.with_suffix(".npz").name:digest(given.with_suffix(".npz"))},
        H_array_hash=history.identity(),meter_hash=meter.identity(),
        source_sha256={p.name:digest(p) for p in source_files},
        plan_sha256=digest(Path(__file__).parent/"continuous_eig_Honly_v2_plan.md"),
        sampler_tests="24 independent tests passed immediately before v2 execution",
        optimization_seeds=list(OPT_SEEDS),final_IS_seeds=list(FINAL_SEEDS),
        optimization=dict(method="bounded_Powell",theta_space="phi=beta/canonical in [.8,1.2]",
            initial_points="each mode's maximum stored finite H likelihood supported particle across all v1 repeats",
            seconds_per_mode=60.,maxfev_per_mode=2000,latent_per_time=32,xtol=.0005,ftol=.001,
            fixed_cloud_only_shapes_proposal=True,global_optimization_certificate=False),
        final_IS=dict(theta_per_component=2048,latent_per_time=32,repeats=3,
            old_v1_weights_enter_final_estimator=False,optimization_cloud_enters_final_likelihood=False,
            fresh_theta_and_latent_streams=True),
        proposal=dict(uniform_mass=.2,Gaussian_mass=.8,Gaussian_split="equally over original H profile covariances",
            Gaussian_center="best local fixed-cloud point",scaled_covariance="(D Iprofile D+75I)^-1",
            untruncated=True,outside_prior_retained_zero_target=True,exact_q_in_weights=True,
            engineering_profile_is_prior_or_CI=False),
        thresholds=dict(minimum_conditional_mode_ESS=20.,minimum_global_ESS=100.,maximum_weight=.05),
        proper_prior_unchanged=True,prior_components=[c.payload() for c in components],
        v1_starts=starts,original_proposals=[p.payload() for p in original_proposals],
        H_updates=1,WZZT_extra_likelihood=False,predictive_MC_enabled=False,
        actual_evaluation_future_or_truth_used=False,formal_C_D_stress_modified=False,
        incidental_schema_inspection_ledger="Initial prior task inspection exposed top-level truth fields; unused in formal inputs or selection; see sampler math",
        process_priority=priority,runtime=dict(executable=sys.executable,python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__,threads=1))
    sampling.save_json(output/"config_before_optimization.json",config)
    print(json.dumps(dict(phase="frozen_before_optimization",config_hash=digest(output/"config_before_optimization.json"),
                          priority=priority)),flush=True)
    new_proposals,optimizations = [],[]
    for i,(c,old,start,seed) in enumerate(zip(components,original_proposals,starts,OPT_SEEDS)):
        beta,row = local_center(c,start,history,meter,seed)
        # Every old H-only covariance remains, centered at the same selected
        # local point. The full untruncated mixture density remains explicit.
        new = sampling.GaussianUniformProposal(c,np.broadcast_to(beta/c.nominal_beta,old.means_phi.shape).copy(),
                                               old.covariances_phi,.2)
        new_proposals.append(new);optimizations.append(row)
        sampling.save_json(output/f"optimization_{i}.json",row)
        print(json.dumps(dict(phase="optimization_complete",mode=c.name,status=row["status"],
            seconds=row["actual_seconds"],nfev=row["evaluation_count"],
            best_log_H_likelihood=row["best_log_H_likelihood"],numerical_failure=row["numerical_failure"])),flush=True)
    pre_mc = dict(parent_config_hash=digest(output/"config_before_optimization.json"),
        final_IS_seeds=list(FINAL_SEEDS),theta_per_component=2048,latent_per_time=32,
        independent_repeats=3,proposals=[p.payload() for p in new_proposals],
        optimization_status=[r["status"] for r in optimizations],prior_unchanged=True,
        old_particle_weights_reused=False,optimization_MC_cloud_reused=False,
        selected_means_define_only_q=True,proof_and_domain_contract_required=True,
        current_source_sha256={p.name:digest(p) for p in source_files},
        source_unchanged_since_optimization_freeze=all(digest(p)==config["source_sha256"][p.name] for p in source_files))
    sampling.save_json(output/"config_before_MC.json",pre_mc)
    if any(r["numerical_failure"] is not None for r in optimizations):
        summary = dict(status="numerical_failure_in_supported_inverse_domain",optimizations=optimizations,
            final_IS_executed=False,predictive_MC_executed=False,seconds=perf_counter()-started)
        sampling.save_json(output/"summary.json",summary)
        return []
    banks = []
    for repeat,seed in enumerate(FINAL_SEEDS):
        t = perf_counter()
        bank = sampling.build_history_bank(components,new_proposals,history,meter,engine,n_theta=2048,
            n_latent=32,seed=seed,proof_accepted=True,minimum_mode_ess=20.,minimum_total_ess=100.,maximum_weight=.05)
        banks.append(bank)
        row = bank.summary();row.update(repeat=repeat,seconds=perf_counter()-t)
        sampling.save_json(output/f"repeat_{repeat}.json",row)
        sampling.save_particle_bank(output/f"repeat_{repeat}_particles.npz",bank)
        print(json.dumps(dict(phase="fresh_H_repeat_complete",repeat=repeat,seconds=row["seconds"],
            status=bank.diagnostics["status"],total_ESS=bank.diagnostics["total_ess"],
            maximum_weight=bank.diagnostics["maximum_particle_weight"],
            conditional_mode_ESS=[r["conditional_particle_ESS"] for r in bank.diagnostics["modes"]])),flush=True)
    summary = cross_repeat_summary(banks)
    summary.update(seconds=perf_counter()-started,optimization_status=[r["status"] for r in optimizations],
        optimization_seconds=[r["actual_seconds"] for r in optimizations],
        optimization_nfev=[r["evaluation_count"] for r in optimizations],
        config_before_optimization_hash=digest(output/"config_before_optimization.json"),
        config_before_MC_hash=digest(output/"config_before_MC.json"),
        v1_failure_evidence_retained=True,prior_unchanged=True,old_pilot_weights_reused=False)
    sampling.save_json(output/"summary.json",summary)
    print(json.dumps(dict(phase="complete",status=summary["status"],seconds=summary["seconds"],
                         log_evidence_range=summary["log_evidence_range"])),flush=True)
    return banks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root",type=Path,default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--v1-directory",type=Path,default=Path(__file__).parent/"continuous_eig_Honly_v1")
    args = parser.parse_args()
    run(args.research_root,args.output,args.v1_directory)
