"""Measured-observation handoff to the unchanged combined RNJ+WZZT identifier.

No synthetic voltage is invented from an excitation score. AC simulation or
real meters must supply new P/Q/V/root observations in the existing contract.
"""
from __future__ import annotations

def fit_full_history(history_training, history_validation, future_training,
                     future_validation, output_dir, *, root_label=1,
                     search_seconds=25., lp_method=1, resume=False):
    """Append all supplied measured future episodes, forward original full history.

    No model scores or unknown topology are arguments. The existing identifier
    does exact observation/unit validation, pooling, acceptance, validation and
    final contraction. No topology code is copied here.
    """
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed
    ht, hv = list(history_training), list(history_validation)
    if not ht or not hv:
        raise ValueError("both complete historical splits are required")
    ft, fv = list(future_training), list(future_validation)
    return fit_observed(ht+ft, hv+fv, output_dir, root_label=root_label,
                        search_seconds=search_seconds, lp_method=lp_method, resume=resume)
