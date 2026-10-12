"""Model-level historical screening before spending EV discrimination effort.

The alpha budget applies only to the fixed finite composite family. This is
not pruning nuisance atoms by an uncalibrated loss cutoff.
"""
import numpy as np
from .power import historical_losses,joint_gaps,finite_rank_pvalue


def historical_candidate_pvalues(history_y,history_means,labels,*,B=199,seed=0):
    means=np.asarray(history_means,float);labels=np.asarray(labels)
    historical_losses(history_y,means)  # validates the full historical bank
    if int(B)!=B or B<1:
        raise ValueError("positive integer calibration count required")
    unique=np.unique(labels)
    zeros=np.zeros(len(means))
    observed=joint_gaps(history_y,zeros,means,labels)[0]
    rng=np.random.default_rng(seed)
    pvalues=np.zeros(len(unique))
    for i,h in enumerate(labels):
        col=int(np.flatnonzero(unique==h)[0])
        ys=means[i]+rng.normal(size=(B,means.shape[1]))
        calibration=joint_gaps(ys,zeros,means,labels)[:,col]
        pvalues[col]=max(pvalues[col],float(finite_rank_pvalue(observed[col],calibration)))
    return dict(topology_labels=unique.tolist(),pvalues=pvalues.tolist(),
                observed_gaps=observed.tolist(),B=int(B),seed=int(seed),
                full_finite_null=True,unconditional_history_level=True)


def compatible_history(result,alpha_history=.05):
    if not 0<alpha_history<1:
        raise ValueError("historical level must lie between zero and one")
    retained=[h for h,p in zip(result["topology_labels"],result["pvalues"]) if p>alpha_history]
    status=("historical_model_failure" if not retained else
            "already_resolved_in_finite_family" if len(retained)==1 else
            "active_discrimination_needed")
    return dict(retained=retained,status=status,alpha_history=float(alpha_history))

