"""Frozen H-only integration run. Predictive MC is deliberately absent."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.dont_write_bytecode = True
import numpy as np
import scipy

from . import continuous_ac_inverse as engine
from . import continuous_eig_sampling as sampling

DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SEEDS = (22101301,22101302,22101303)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def cross_repeat_summary(banks):
    logz = np.array([b.log_evidence for b in banks])
    posterior = np.stack([b.topology_mass for b in banks])
    mode = []
    for index in range(7):
        rows = [b.diagnostics["modes"][index] for b in banks]
        logs = np.array([r["log_mode_evidence"] for r in rows])
        times = np.array([r["log_single_time_marginal_evidence"] for r in rows])
        finite_logs = np.isfinite(logs)
        finite_times = np.isfinite(times).all(axis=0)
        ranges = np.full(times.shape[1],np.nan)
        ranges[finite_times] = np.ptp(times[:,finite_times],axis=0)
        mode.append(dict(component=rows[0]["component"],
            repeated_log_mode_evidence=logs.tolist(), all_repeats_numerically_nonzero=bool(finite_logs.all()),
            log_mode_evidence_range=None if not finite_logs.all() else float(np.ptp(logs)),
            repeated_conditional_mode_ESS=[r["conditional_particle_ESS"] for r in rows],
            repeated_numerical_posterior_mode_mass=[r["numerical_posterior_mode_mass"] for r in rows],
            repeated_log_single_time_marginal_evidence=times.tolist(),
            log_single_time_marginal_evidence_range=ranges.tolist(),
            fluctuation_includes_beta_and_latent_MC=True,
            mode_retained=True, zero_is_exact_model_rejection=False))
    return dict(status="failed" if any(b.diagnostics["status"] == "failed" for b in banks) else "needs_integral_doubling",
        independent_repeats=len(banks), repeated_log_evidence=logz.tolist(),
        log_evidence_range=None if not np.isfinite(logz).all() else float(np.ptp(logz)),
        numerical_topology_mass_by_repeat=posterior.tolist(),
        topology_mass_range=np.ptp(posterior,axis=0).tolist(),
        posterior_values_credible=False, topology_ordering_accepted=False,
        repeated_global_ESS=[b.diagnostics["total_ess"] for b in banks],
        repeated_maximum_weight=[b.diagnostics["maximum_particle_weight"] for b in banks],
        modes=mode, integral_doubling_not_yet_executed=True,
        predictive_MC_executed=False, actual_evaluation_future_accessed=False)


def run(research_root, output):
    started = perf_counter()
    root = Path(research_root)
    sanitized_json = root/"research_experiments/mobile_ev_mle/outputs/continuous_eig_Honly/H_710009.json"
    sanitized_npz = sanitized_json.with_suffix(".npz")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    import inspect
    candidates = tuple(RadialCandidate.from_supports(name,3,family) for name,family in zip(
        sampling.TOPOLOGY_NAMES,((),((0,1),),((0,2),),((1,2),))))
    history,meter,actions,centers = sampling.read_sanitized_bundle(sanitized_json,sanitized_npz)
    if not np.all(history.root_v == 1.):
        raise ValueError("uniform inverse proof requires calibrated root=1")
    for action in actions:
        if np.any(action.q != 0) or np.any(action.p < -1e-14) or np.max(action.p) > .3+1e-14:
            raise ValueError("frozen nominal proof requires QEV=0 and 0<=P<=.3")
        if np.max(action.p.sum(axis=1)) > .3+1e-14 or np.any((action.p > 1e-12).sum(axis=1) > 1):
            raise ValueError("frozen inverse envelope requires one EV terminal per time")
    components = sampling.declared_components(candidates)
    proposals = tuple(sampling.GaussianUniformProposal.from_history_centers(c,centers[c.candidate.name],
                       ridge=75.,uniform_mass=.2) for c in components)
    source_files = (Path(__file__),Path(sampling.__file__),Path(engine.__file__),
                    Path(inspect.getfile(RadialCandidate)))
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for path in source_files:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for name in ("continuous_eig_sampling_math.md","continuous_ac_inverse_math.md"):
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    config = dict(version="continuous_proper_nominal_Honly_v1_frozen",history_id="710009",
        sanitized_json=str(sanitized_json),sanitized_npz=str(sanitized_npz),
        sanitized_input_sha256={sanitized_json.name:digest(sanitized_json),sanitized_npz.name:digest(sanitized_npz)},
        history_array_hash=history.identity(),meter_hash=meter.identity(),
        source_sha256={p.name:digest(p) for p in source_files},
        math_sha256={name:digest(Path(__file__).parent/name) for name in (
            "continuous_eig_sampling_math.md","continuous_ac_inverse_math.md")},
        inverse_proof_accepted=True, inverse_tests="agent-reported 20 independent tests passed; root acceptance before MC",
        sampler_tests="24 independent current tests passed; original v1 snapshot recorded its 15 pre-MC tests", seeds=list(FROZEN_SEEDS),
        theta_per_component=2048,latent_per_time=32,independent_repeats=3,
        ESS_thresholds=dict(minimum_mode_ESS=20.,minimum_total_ESS=100.,maximum_weight=.05),
        prior_components=[c.payload() for c in components],proposals=[p.payload() for p in proposals],
        proposal_centers_Honly=True,center_clipping_changes_only_proposal=True,
        normalized_Gaussian_is_untruncated=True,outside_prior_samples_are_not_redrawn=True,
        H_updates=1,WZZT_extra_likelihood=False,
        frozen_planned_actions={a.name:dict(p=a.p.tolist(),q=a.q.tolist(),hash=a.identity()) for a in actions},
        future_actions_used_for_current_H_integral=False,predictive_MC_enabled=False,
        NPZ_allowed_keys=list(sampling.HISTORY_KEYS),NPZ_future_or_latent_keys_indexed=False,
        original_trial_accessed_by_this_driver=False,
        incidental_schema_inspection="Separate sampler agent initial inspection exposed case/truth index; not formal inputs or used in selection; see math ledger",
        runtime=dict(executable=sys.executable,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threads=1))
    sampling.save_json(output/"config_before_MC.json",config)
    print(json.dumps(dict(phase="frozen_before_MC",config_hash=digest(output/"config_before_MC.json"),seeds=list(FROZEN_SEEDS))),flush=True)
    banks = []
    for repeat,seed in enumerate(FROZEN_SEEDS):
        t = perf_counter()
        bank = sampling.build_history_bank(components,proposals,history,meter,engine,
            n_theta=2048,n_latent=32,seed=seed,proof_accepted=True,
            minimum_mode_ess=20.,minimum_total_ess=100.,maximum_weight=.05)
        banks.append(bank)
        row = bank.summary()
        row.update(repeat=repeat,seconds=perf_counter()-t)
        sampling.save_json(output/f"repeat_{repeat}.json",row)
        sampling.save_particle_bank(output/f"repeat_{repeat}_particles.npz",bank)
        print(json.dumps(dict(phase="H_repeat_complete",repeat=repeat,seconds=row["seconds"],
            status=bank.diagnostics["status"],total_ESS=bank.diagnostics["total_ess"],
            maximum_weight=bank.diagnostics["maximum_particle_weight"],
            conditional_mode_ESS=[r["conditional_particle_ESS"] for r in bank.diagnostics["modes"]],
            numerical_zero_modes=[r["component"] for r in bank.diagnostics["modes"] if r["numerical_zero_mode_estimate"]])),flush=True)
    summary = cross_repeat_summary(banks)
    summary.update(config_hash=digest(output/"config_before_MC.json"),seconds=perf_counter()-started,
                   repeat_artifacts=[f"repeat_{i}.json" for i in range(3)])
    sampling.save_json(output/"summary.json",summary)
    print(json.dumps(dict(phase="complete",status=summary["status"],seconds=summary["seconds"],
                         log_evidence_range=summary["log_evidence_range"])),flush=True)
    return banks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root",type=Path,default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--output",type=Path,required=True)
    given = parser.parse_args()
    run(given.research_root,given.output)
