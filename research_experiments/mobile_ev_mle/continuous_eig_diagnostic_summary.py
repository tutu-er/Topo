"""Read-only postprocessing of generated H-only banks, no new Monte Carlo."""
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import logsumexp
from .continuous_eig_sampling import save_json


def analyze(directory):
    directory = Path(directory)
    summary = json.loads((directory/"summary.json").read_text())
    runs = []
    for repeat in range(3):
        given = json.loads((directory/f"repeat_{repeat}.json").read_text())
        modes = []
        with np.load(directory/f"repeat_{repeat}_particles.npz",allow_pickle=False) as arrays:
            for index,row in enumerate(given["diagnostics"]["modes"]):
                lw = np.asarray(arrays[f"logweight_{index}"],float)
                ess = np.asarray(arrays[f"latent_ess_{index}"],float)
                total = float(logsumexp(lw))
                w = np.exp(lw-total) if np.isfinite(total) else np.zeros(len(lw))
                active = w > 0
                L = row["latent_per_time"]
                with np.errstate(divide="ignore",invalid="ignore"):
                    se = np.sqrt(np.maximum(L/ess-1.,0.)/(L-1))
                if np.any(active):
                    weighted_ess = np.sum(w[active,None]*ess[active],axis=0)
                    weighted_se = np.sum(w[active,None]*se[active],axis=0)
                    theta_joint_se = np.sqrt(np.maximum(np.prod(1+se[active]**2,axis=1)-1,0))
                    weighted_joint = float(w[active]@theta_joint_se)
                else:
                    weighted_ess=weighted_se=None;weighted_joint=None
                modes.append(dict(component=row["component"],conditional_theta_ESS=row["conditional_particle_ESS"],
                    supported_theta_count=row["theta_count"]-row["prior_outside_theta_count"],
                    conditional_weighted_time_latent_ESS=None if weighted_ess is None else weighted_ess.tolist(),
                    mean_conditional_weighted_time_latent_ESS=None if weighted_ess is None else float(weighted_ess.mean()),
                    conditional_weighted_time_latent_relative_SE=None if weighted_se is None else weighted_se.tolist(),
                    mean_conditional_weighted_time_latent_relative_SE=None if weighted_se is None else float(weighted_se.mean()),
                    conditional_weighted_joint_likelihood_relative_SE_sample_moment=weighted_joint,
                    relative_SE_is_error_bound=False,relative_SE_selected_by_random_H_weights=True,
                    inverse_invalid_count=row["invalid_inverse_count"],
                    proposal_roundoff=row["proposal"].get("sampling_roundoff_diagnostics"),
                    global_display_mass_underflow_does_not_remove_this_mode=True))
        runs.append(dict(repeat=repeat,status=given["diagnostics"]["status"],
            global_ESS=given["diagnostics"]["total_ess"],maximum_particle_weight=given["diagnostics"]["maximum_particle_weight"],
            log_evidence=given["log_evidence"],modes=modes))
    roundoff = {}
    for run in runs:
        for mode in run["modes"]:
            for key,row in (mode["proposal_roundoff"] or {}).items():
                target = roundoff.setdefault(key,dict(maximum_absolute_correction=0.,total_clipped_value_count=0))
                target["maximum_absolute_correction"] = max(target["maximum_absolute_correction"],row["maximum_absolute_correction"])
                target["total_clipped_value_count"] += row["clipped_value_count"]
    return dict(directory=str(directory),status=summary["status"],log_evidence_range=summary["log_evidence_range"],
        evidence_ratio_over_repeats=float(np.exp(summary["log_evidence_range"])),
        runs=runs,roundoff_maxima=roundoff,postprocessing_only=True,new_MC_executed=False,
        posterior_accepted=False,predictive_MC_executed=False)


if __name__ == "__main__":
    stage = Path(__file__).parent
    results = {version:analyze(stage/f"continuous_eig_Honly_{version}") for version in ("v1","v2","v3","v4")}
    save_json(stage/"continuous_eig_Honly_comparison.json",results)
    print(json.dumps({k:dict(status=v["status"],logZ_range=v["log_evidence_range"],
         global_ESS=[r["global_ESS"] for r in v["runs"]],roundoff=v["roundoff_maxima"]) for k,v in results.items()}))
