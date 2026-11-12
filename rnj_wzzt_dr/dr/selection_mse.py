"""Final validation selection using physical unweighted MSE.

The input losses must describe the actual output models. In an end-to-end
MSE pipeline, contract a family then QP-fit it on training before validation.
Re-scoring old fixed L1 points is a scoring ablation, not a new MSE fit.

Strict is the primary rule, matching the existing K*MSE strict view. The
optional relative band is only a separately reported sensitivity check.
No truth, test observations or topology reliability probability is used.
"""
import math
import numpy as np


def choose_mse(losses, internal_counts, *, K, candidate_ids=None,
               relative_band=0., epsilon_scaled=1e-12):
    """Select minimum K*MSE; numerical ties by internal count, score, order.

    losses: physical p.u.^4 MSE values, same validation cells and units.
    internal_counts: proper internal supports of each output model.
    K>0: one frozen loss scale for the whole input, common to candidates.
    epsilon_physical=epsilon_scaled/K. No separate 3% band by default.
    """
    losses=np.asarray(losses,float);counts=np.asarray(internal_counts)
    if (losses.ndim!=1 or not losses.size or counts.shape!=losses.shape
            or not np.isfinite(losses).all() or (losses<0).any()
            or not np.isfinite(counts).all() or (counts<0).any()
            or not np.equal(counts,counts.astype(int)).all()):
        raise ValueError('Finite nonnegative MSE and integer support counts required')
    if not math.isfinite(K) or K<=0 or not math.isfinite(relative_band) or relative_band<0:
        raise ValueError('Require K>0 and relative_band>=0')
    if not math.isfinite(epsilon_scaled) or epsilon_scaled<0:
        raise ValueError('Invalid numerical tolerance')
    ids=list(range(len(losses))) if candidate_ids is None else list(candidate_ids)
    if len(ids)!=len(losses) or len(set(ids))!=len(ids):
        raise ValueError('Unique candidate IDs required')
    scaled=K*losses;threshold=(1+relative_band)*float(scaled.min())+epsilon_scaled
    eligible=np.flatnonzero(scaled<=threshold)
    selected=min(eligible,key=lambda i:(counts[i],scaled[i],i))
    return dict(selected=ids[selected],eligible=[ids[i] for i in eligible],
        mse={k:float(v) for k,v in zip(ids,losses)},
        rule='strict_mse' if relative_band==0 else 'relative_band_mse_sensitivity',
        K=K,epsilon_scaled=epsilon_scaled,epsilon_physical=epsilon_scaled/K,
        relative_band=relative_band,threshold_physical=threshold/K)


def required_new_sse_gap(history_sse_current,history_sse_reference,
                         history_cells,new_cells,*,K,epsilon_scaled=1e-12):
    """New-day SSE_current-SSE_reference needed to beat a strict-MSE tie.

    SSE is a physical p.u.^4 sum over equal cell sets. Divide this returned
    number by a common s^2 only when the probability model uses that scale.
    This is a frozen pair's sufficient exclusion condition, not recovery
    probability over an unknown set of all trees.
    """
    vals=[history_sse_current,history_sse_reference,K,epsilon_scaled]
    if not all(math.isfinite(v) for v in vals) or min(vals[:2])<0 or K<=0 or epsilon_scaled<0:
        raise ValueError('Invalid history SSE, K or numerical tolerance')
    if any(isinstance(n,bool) or int(n)!=n or n<=0 for n in [history_cells,new_cells]):
        raise ValueError('Positive integer observation-cell counts required')
    return ((history_cells+new_cells)*epsilon_scaled/K
            -(history_sse_current-history_sse_reference))
