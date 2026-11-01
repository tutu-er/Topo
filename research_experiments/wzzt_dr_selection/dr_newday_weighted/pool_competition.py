"""Historical competition between distinct frozen final output families.

Only saved candidate scores, supports, weights and the original contraction
are read. The selected old point and every fitted parameter remain unchanged.
"""
from __future__ import annotations

import math
from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.prepare import _choose_competition
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract


EPS=1e-12


def _family(point,calibration):
    return {tuple(sorted(b)) for b in contract(point,calibration)['supports']}


def choose_final_competition(baseline):
    """Keep the actual old selection, seek a different eligible final family.

    Prefer the original next accepted same-path point only when its final
    output differs and its historical raw validation loss is within3% of the
    best raw point. Otherwise minimize final-family symmetric difference,
    raw support count, validation MAE, then original index.

    With no eligible distinct final family, retain the old pair as a labelled
    zero-purchase control. If even the old pair is unavailable, the selected
    point is repeated and explicitly labelled; no direction is invented.
    """
    points=baseline['selection_candidates']
    selected=baseline['models']['multistart_original_rule']['selected_uncontracted']
    index=selected.get('selection_index')
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(points):
        raise ValueError('saved selected raw index is unavailable')
    base=points[index]
    if base.get('preprocess','raw')!='raw':raise ValueError('selected historical point must be raw')
    calibration=baseline['calibrations']['raw']
    raw=[(i,p) for i,p in enumerate(points) if p.get('preprocess','raw')=='raw']
    if any(not math.isfinite(float(p['validation_mae'])) or p['validation_mae']<0 for _,p in raw):
        raise ValueError('finite nonnegative raw validation scores required')
    global_best=min(float(p['validation_mae']) for _,p in raw)
    cutoff=1.03*global_best+EPS
    eligible=[(i,p) for i,p in raw if p['validation_mae']<=cutoff]
    base_family=_family(base,calibration)
    distinct=[(i,p,_family(p,calibration)) for i,p in eligible if i!=index and _family(p,calibration)!=base_family]

    legacy=None;legacy_error=None
    try:
        old_base,old_other,old_rule=_choose_competition(baseline)
        if old_rule['selected_index']!=index or old_base!=base:raise AssertionError('old selected point changed')
        legacy=(old_rule['alternative_index'],old_other,old_rule)
    except ValueError as error:legacy_error=str(error)
    path=base.get('path_index')
    next_points=[(i,p) for i,p in raw if path is not None and p.get('origin')==base.get('origin')
        and p.get('path_index')==path+1 and len(p['supports'])>len(base['supports'])]
    next_point=min(next_points,key=lambda v:v[0]) if next_points else None
    next_valid=bool(next_point and any(i==next_point[0] for i,_,_ in distinct))
    if next_valid:
        other_index,other=next_point;reason='next accepted same-path point with distinct historically eligible final output'
        no_dispute=False
    elif distinct:
        other_index,other,_=min(distinct,key=lambda v:(len(base_family^v[2]),len(v[1]['supports']),v[1]['validation_mae'],v[0]))
        reason='closest distinct historically eligible final family; raw count, loss and original order break ties'
        no_dispute=False
    elif legacy is not None and legacy[1].get('preprocess','raw')=='raw':
        other_index,other,_=legacy;reason='no eligible distinct final output; retain old pair as zero-purchase control'
        no_dispute=True
    else:
        other_index,other=index,base;reason='no eligible distinct final output or old pair; duplicate selected control, no direction fabricated'
        no_dispute=True
    other_family=_family(other,calibration)
    rule=dict(selected_index=index,alternative_index=other_index,rule=reason,
        selected_origin=base.get('origin'),selected_path_index=path,
        alternative_origin=other.get('origin'),alternative_path_index=other.get('path_index'),
        selected_support_count=len(base['supports']),alternative_support_count=len(other['supports']),
        selected_final_support_count=len(base_family),alternative_final_support_count=len(other_family),
        selected_final_supports=[list(b) for b in sorted(base_family)],
        alternative_final_supports=[list(b) for b in sorted(other_family)],
        final_added_supports=[list(b) for b in sorted(other_family-base_family)],
        final_removed_supports=[list(b) for b in sorted(base_family-other_family)],
        final_symmetric_difference=len(base_family^other_family),final_output_different=base_family!=other_family,
        no_final_dispute=no_dispute,no_final_dispute_meaning='no different final family among all historical raw3%-eligible points',
        raw_global_best_validation_mae=global_best,historical_raw_cutoff=cutoff,
        historical_eligible_raw_indices=[i for i,_ in eligible],
        eligible_distinct_final_indices=[i for i,_,_ in distinct],
        legacy_alternative_index=legacy[0] if legacy else None,legacy_pair_error=legacy_error,
        original_next_same_path_index=next_point[0] if next_point else None,
        original_next_same_path_eligible_and_final_distinct=next_valid,
        duplicate_selected_control=other_index==index,
        planning_scope='fixed historical complete parameter points and original output maps; no test truth or future; no statistical coverage')
    return base,other,rule
