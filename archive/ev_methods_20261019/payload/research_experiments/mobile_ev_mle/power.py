"""Direct discrimination power with conservative finite-composite rank tests.

Means have already been whitened. Inner minimization reselects a complete atom
shared across history and future. Exact finite-family tests are not tests of
the larger continuous AC model.
"""
from __future__ import annotations
import numpy as np
import hashlib
from scipy.stats import beta


def _fingerprint(*arrays):
    digest=hashlib.sha256()
    for array in arrays:
        array=np.asarray(array,dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def historical_losses(history_y, history_means):
    y, means = np.asarray(history_y,float), np.asarray(history_means,float)
    if means.ndim!=2 or y.shape!=(means.shape[1],) or not np.isfinite(means).all() or not np.isfinite(y).all():
        raise ValueError("finite compatible historical means and observations required")
    return np.sum((means-y)**2,axis=1)


def joint_gaps(future_y, history_loss, future_means, labels):
    """Return joint best-loss gaps for each topology and every observed row."""
    y = np.atleast_2d(np.asarray(future_y,float))
    means, history_loss, labels = (np.asarray(future_means,float),
                                   np.asarray(history_loss,float),np.asarray(labels))
    if means.ndim!=2 or y.shape[1]!=means.shape[1] or history_loss.shape!=(len(means),) or labels.shape!=(len(means),):
        raise ValueError("joint mean/label/loss dimensions differ")
    if not all(np.isfinite(a).all() for a in (y,means,history_loss)):
        raise ValueError("failed means cannot be silently excluded")
    # Centering prevents subtracting large squared voltage means (~10^8).
    center = means[0]
    yy, mm = y-center, means-center
    distance = np.maximum(0.,np.sum(yy*yy,axis=1)[:,None]+np.sum(mm*mm,axis=1)[None,:]-2*yy@mm.T)
    losses = distance+history_loss[None,:]
    unique = np.unique(labels)
    minima = np.column_stack([np.min(losses[:,labels==h],axis=1) for h in unique])
    return minima-np.min(minima,axis=1)[:,None]


def finite_rank_pvalue(statistic, calibration_values):
    """Upper-tail +1 rank; >= keeps ties conservative."""
    values = np.sort(np.asarray(calibration_values,float))
    statistic = np.asarray(statistic,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(statistic).all():
        raise ValueError("finite nonempty calibration and statistic required")
    return (1+len(values)-np.searchsorted(values,statistic,side="left"))/(len(values)+1)


def calibrate_rank_nulls(history_y, history_means, future_means, labels, *, B=99, seed=0):
    """For every null atom calibrate the max gap over ALL supplied actions.

    Conditional on H, the full atom and the independent noise model, a tested
    selected-action gap is dominated by the all-action maximum. The maximum of
    the atom p-values protects the finite composite null without historical
    parameter pruning. No future evaluation data enter this construction.
    """
    labels = np.asarray(labels)
    h_loss = historical_losses(history_y,history_means)
    unique = np.unique(labels)
    if int(B)!=B or B<1 or not future_means:
        raise ValueError("positive calibration size and nonempty actions required")
    shape = np.asarray(next(iter(future_means.values()))).shape
    if any(np.asarray(m).shape!=shape for m in future_means.values()) or shape[0]!=len(labels):
        raise ValueError("action dimensions must coincide")
    if any(not np.isfinite(m).all() for m in future_means.values()):
        raise ValueError("all finite null means are required")
    rng = np.random.default_rng(seed)
    maxima = np.empty((len(labels),B))
    for i,h in enumerate(labels):
        noise = rng.normal(size=(B,shape[1]))
        values = np.zeros(B)
        column = int(np.flatnonzero(unique==h)[0])
        for means in future_means.values():
            z = joint_gaps(means[i]+noise,h_loss,means,labels)[:,column]
            values = np.maximum(values,z)
        maxima[i] = np.sort(values)
    return dict(values=maxima, labels=labels.copy(),B=int(B),seed=int(seed),
                action_names=list(future_means),full_null=True,conditional_history=True,
                history_fingerprint=_fingerprint(history_y,history_means,labels),
                action_fingerprints={name:_fingerprint(means) for name,means in future_means.items()})


def candidate_pvalues(history_y, history_means, selected_means, labels, future_y, calibration):
    labels = np.asarray(labels)
    if not np.array_equal(labels,calibration["labels"]):
        raise ValueError("calibration does not contain the same complete null family")
    if _fingerprint(history_y,history_means,labels)!=calibration["history_fingerprint"]:
        raise ValueError("calibration belongs to different history or atom order")
    if _fingerprint(selected_means) not in calibration["action_fingerprints"].values():
        raise ValueError("selected action was not included in null calibration")
    gaps = joint_gaps(future_y,historical_losses(history_y,history_means),selected_means,labels)
    pvalues = np.ones_like(gaps)
    for col,h in enumerate(np.unique(labels)):
        ranks = [finite_rank_pvalue(gaps[:,col],row) for row in calibration["values"][labels==h]]
        pvalues[:,col] = np.max(ranks,axis=0)
    return pvalues


def power_scores(history_y, history_means, future_means, labels, *, cutoff=12.,alpha=.1,
                 training_count=64,calibration_count=99,seed=0):
    """Training-only, two-sided success probabilities; final calibration is fresh."""
    if not 0<alpha<1 or training_count<2 or cutoff<0:
        raise ValueError("invalid power design configuration")
    labels = np.asarray(labels)
    unique = np.unique(labels)
    loss = historical_losses(history_y,history_means)
    calibration = calibrate_rank_nulls(history_y,history_means,future_means,labels,
                                      B=calibration_count,seed=seed+10)
    rng = np.random.default_rng(seed+20)
    dim = np.asarray(next(iter(future_means.values()))).shape[1]
    strata = []
    for h in unique:
        indices = np.flatnonzero((labels==h)&(loss<=np.min(loss[labels==h])+cutoff+1e-10))
        generators = rng.choice(indices,size=training_count,replace=True)
        noise = rng.normal(size=(training_count,dim))
        strata.append((h,indices,generators,noise))
    result = {}
    for name,means in future_means.items():
        rows = []
        for col,(h,indices,generators,noise) in enumerate(strata):
            pv = candidate_pvalues(history_y,history_means,means,labels,means[generators]+noise,calibration)
            retained = pv>alpha
            success = retained[:,col] & (np.sum(retained,axis=1)==1)
            wrong_removed = np.mean(~np.delete(retained,col,axis=1),axis=1)
            atom_powers = [dict(atom=int(i),trials=int(np.sum(generators==i)),
                              success=float(np.mean(success[generators==i])) if np.any(generators==i) else None)
                           for i in indices]
            rows.append(dict(topology=int(h),power=float(np.mean(success)),
                power_se=float(np.std(success,ddof=1)/np.sqrt(training_count)),
                false_rejection=float(np.mean(~retained[:,col])),wrong_removal=float(np.mean(wrong_removed)),
                trials=int(training_count),support_size=len(indices),per_atom=atom_powers,
                worst_sampled_atom_power=min(r["success"] for r in atom_powers if r["success"] is not None)))
        result[name] = dict(score=min(r["power"] for r in rows),strata=rows,
            criterion="minimum topology-specific training-average correct unique retention",
            null_scope="complete finite atom family",training_seed=int(seed),alpha=float(alpha),
            calibration_count=int(calibration_count),training_count=int(training_count))
    return result


def binomial_interval(count,total,confidence=.95):
    if total<1:
        return [None,None]
    tail=(1-confidence)/2
    return [0. if count==0 else float(beta.ppf(tail,count,total-count+1)),
            1. if count==total else float(beta.ppf(1-tail,count+1,total-count))]


def evaluation_summary(pvalues,true_label,unique_labels,alpha=.1):
    """Evaluation-only summaries; truth is never a selector parameter."""
    pv=np.asarray(pvalues,float)
    col=int(np.flatnonzero(np.asarray(unique_labels)==true_label)[0])
    kept=pv>alpha
    events=dict(false_rejection=~kept[:,col],
                correct_unique=kept[:,col]&(kept.sum(axis=1)==1),
                unresolved=kept.sum(axis=1)>1,
                empty=kept.sum(axis=1)==0,
                wrong_unique=(~kept[:,col])&(kept.sum(axis=1)==1))
    result={}
    for name,event in events.items():
        k,n=int(event.sum()),len(event)
        result[name]=dict(count=k,total=n,rate=k/n,interval95=binomial_interval(k,n))
    return result
