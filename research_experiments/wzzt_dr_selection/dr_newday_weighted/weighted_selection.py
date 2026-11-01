"""Frozen-model validation with changes confined to the new-day segment.

No fitting, candidate generation, parameter updates or truth are used. The
existing scorer computes each model's observation loss once. Its original
ungated 3% result is retained alongside the prespecified segment-weight rules.
"""
from __future__ import annotations

from copy import deepcopy
import math

from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import (
    VALIDATION_EPSILON,score_frozen,
)


HISTORY_FRACTION=.03
NEW_WEIGHTS=(1,2,4)
NEW_FRACTIONS=(.03,.02,.01)


def rule_id(weight,fraction):
    return f'w{int(weight)}_tau{round(100*fraction):d}pct'


def score(pool,new_scene,history_gate=True):
    """Return nine fixed-rule selections and the unchanged original baseline.

    For mean absolute losses vh/vd and row counts Nh/Nd, score S is
    Nh*vh+w*Nd*vd. The best-S reference j defines tolerance
    B=.03*Nh*vh_j+tau*w*Nd*vd_j. Candidate k is eligible if
    S_k<=S_j+B+epsilon*(Nh+w*Nd). Raw support count, S and frozen order break
    ties. All output maps are the previously frozen .16 maps.

    Absolute cell sums implement the same formula without a second loss
    pass. Epsilon remains in mean-loss units, not unnormalised sum units.
    Historical eligibility is always reported, and applied only when the
    explicit history_gate flag is true. Thus the original ungated baseline
    stays visible even if the historical gate removes a candidate.
    """
    if not isinstance(history_gate,bool):raise TypeError('history_gate must be bool')
    original=score_frozen(pool,new_scene)
    order=list(pool['selection_order'])
    by_id={m['candidate_id']:m for m in pool['models']}
    row_by_id={r['candidate_id']:r for r in original['rows']}
    historical_cells=int(pool['historical_validation_cells'])
    historical_rows=int(pool['validation_rows'])
    new_rows=int(original['new_validation_rows'])
    new_cells=int(original['rows'][0]['new_validation_cells'])
    n=len(pool['labels'])
    if min(historical_rows,new_rows,historical_cells,new_cells,n)<=0:
        raise ValueError('positive historical/new-day observation counts required')
    if historical_cells!=historical_rows*n or new_cells!=new_rows*n:
        raise ValueError('equal terminal-cell and row counts required')

    h_abs={k:float(row_by_id[k]['historical_validation_abs_sum']) for k in order}
    d_abs={k:float(row_by_id[k]['new_validation_abs_sum']) for k in order}
    if any(not math.isfinite(v) or v<0 for v in [*h_abs.values(),*d_abs.values()]):
        raise ValueError('finite nonnegative observed absolute errors required')
    h_loss={k:h_abs[k]/historical_cells for k in order}
    minimum_h=min(h_loss.values())
    h_eligible=[k for k in order if h_loss[k]<=1.03*minimum_h+VALIDATION_EPSILON]
    scope=h_eligible if history_gate else order
    raw_count={k:len(by_id[k]['point']['supports']) for k in order}
    index={k:i for i,k in enumerate(order)}
    rows=[]
    for k in order:
        rows.append(dict(candidate_id=k,historical_validation_mae=h_loss[k],
            new_validation_mae=d_abs[k]/new_cells,
            historical_validation_abs_sum=h_abs[k],new_validation_abs_sum=d_abs[k],
            raw_support_count=raw_count[k],
            frozen_output_support_count=len(by_id[k]['frozen_output_supports']),
            historical_eligible=k in h_eligible))

    rules=[]
    for weight in NEW_WEIGHTS:
        cells=historical_cells+weight*new_cells
        effective_rows=historical_rows+weight*new_rows
        means={k:(h_abs[k]+weight*d_abs[k])/cells for k in order}
        reference=min(scope,key=lambda k:(means[k],index[k]))
        for fraction in NEW_FRACTIONS:
            historical_tolerance=HISTORY_FRACTION*h_abs[reference]/cells
            new_tolerance=fraction*weight*d_abs[reference]/cells
            tolerance=historical_tolerance+new_tolerance
            # At common3%, this uses exactly the original selector's floating
            # expression; the other fractions implement the segment formula.
            threshold=(1.03*means[reference]+VALIDATION_EPSILON if fraction==.03
                else means[reference]+tolerance+VALIDATION_EPSILON)
            eligible=[k for k in scope if means[k]<=threshold]
            chosen=min(eligible,key=lambda k:(raw_count[k],means[k],index[k]))
            model=by_id[chosen]
            rule=dict(rule_id=rule_id(weight,fraction),new_day_weight=weight,
                history_fraction=HISTORY_FRACTION,new_day_fraction=fraction,
                history_gate_applied=history_gate,reference_candidate_id=reference,
                weighted_validation_mae=means,
                row_weighted_scores={k:means[k]*effective_rows for k in order},
                effective_validation_rows=effective_rows,
                historical_tolerance_mean=historical_tolerance,new_day_tolerance_mean=new_tolerance,
                total_tolerance_mean=tolerance,eligibility_threshold_mean=threshold,
                exit_margins={k:means[k]-threshold for k in order},
                historical_eligible_candidate_ids=h_eligible.copy(),
                eligible_candidate_ids=eligible,selected_candidate_id=chosen,
                selected_raw_support_count=raw_count[chosen],
                selected_output_support_count=len(model['frozen_output_supports']),
                selected_output_supports=deepcopy(model['frozen_output_supports']),
                selected_output_support_labels=deepcopy(model['frozen_output_support_labels']),
                selection_index=index[chosen],epsilon_mean_units=VALIDATION_EPSILON,
                epsilon_row_sum_units=VALIDATION_EPSILON*effective_rows,
                original_baseline_equivalence_checked=False)
            if weight==1 and fraction==.03 and scope==order:
                if chosen!=original['selected_candidate_id'] or eligible!=original['eligible_candidate_ids']:
                    raise AssertionError('original ungated3% baseline changed')
                if rule['selected_output_supports']!=original['selected_output_supports']:
                    raise AssertionError('frozen original output map changed')
                rule['original_baseline_equivalence_checked']=True
            rules.append(rule)

    return dict(status='complete',history_gate_applied=history_gate,
        history_fraction=HISTORY_FRACTION,historical_eligible_candidate_ids=h_eligible,
        history_gate_removed_candidate_ids=[k for k in order if k not in h_eligible],
        history_validation_rows=historical_rows,new_validation_rows=new_rows,
        actual_validation_rows=historical_rows+new_rows,
        loss_rows=rows,rules=rules,
        primary_rule=next(r for r in rules if r['new_day_weight']==2 and r['new_day_fraction']==.02),
        original_ungated_selection=deepcopy(original),
        new_prediction_pair_distance_l1=original['new_prediction_pair_distance_l1'],
        parameters_changed=False,calibration_changed=False,new_fit_calls=0,new_search_calls=0,
        probability_scope='deterministic validation-rule ablations; no confidence or recovery probability',
        rule_scope='history3% preserved; only the new-day sample weight and tolerance change')
