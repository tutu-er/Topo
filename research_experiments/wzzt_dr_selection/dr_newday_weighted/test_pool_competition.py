"""Pure saved-point mocks for final-output competition; no fit or case read."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pool_competition import choose_final_competition


def point(extra=None,weight=.1,loss=1.,origin='path',path=0):
    blocks=[[0],[1],[2],[0,1,2]]
    r=[.2,.2,.2,.2]
    if extra is not None:blocks=blocks+[extra];r=r+[weight]
    return dict(supports=blocks,r=r,x=[0.]*len(r),validation_mae=loss,
        preprocess='raw',origin=origin,path_index=path)


def baseline(points):
    return dict(selection_candidates=points,
        models={'multistart_original_rule':{'selected_uncontracted':dict(points[0],selection_index=0)}},
        calibrations={'raw':dict(a=1.,b=1.,median_root_depth=.5)})


def test_raw_difference_final_same_is_skipped_for_visible_eligible_family():
    data=baseline([point(),point([0,1],weight=.01,loss=.99,path=1),point([1,2],loss=.995,origin='bank',path=None)])
    base,other,rule=choose_final_competition(data)
    assert rule['selected_index']==0 and rule['alternative_index']==2
    assert rule['legacy_alternative_index']==1
    assert rule['final_symmetric_difference']==1
    assert rule['no_final_dispute'] is False
    assert base==data['selection_candidates'][0] and other==data['selection_candidates'][2]


def test_hist_outside_three_percent_is_rejected_and_old_pair_control_retained():
    data=baseline([point(),point([0,1],weight=.01,loss=.99,path=1),point([1,2],loss=1.1,origin='bank')])
    _,_,rule=choose_final_competition(data)
    assert rule['alternative_index']==1
    assert rule['no_final_dispute'] is True and rule['final_output_different'] is False
    assert rule['eligible_distinct_final_indices']==[]
    assert 2 not in rule['historical_eligible_raw_indices']


def test_valid_original_next_has_priority_over_better_loss_bank_point():
    data=baseline([point(),point([0,1],loss=1.01,path=1),point([1,2],loss=.99,origin='bank')])
    _,_,rule=choose_final_competition(data)
    assert rule['alternative_index']==1
    assert rule['original_next_same_path_eligible_and_final_distinct'] is True


def test_selector_never_changes_points_or_zero_weak_weights():
    data=baseline([point(),point([0,1],weight=1e-14,loss=.99,path=1),point([1,2],loss=.995,origin='bank')])
    data['selection_candidates'][1]['x'][-1]=0.
    before=json.dumps(data,sort_keys=True)
    choose_final_competition(data)
    assert json.dumps(data,sort_keys=True)==before
    assert data['selection_candidates'][1]['r'][-1]==1e-14


def test_no_old_pair_or_distinct_family_yields_explicit_duplicate_control():
    data=baseline([point()])
    _,_,rule=choose_final_competition(data)
    assert rule['no_final_dispute'] is True and rule['duplicate_selected_control'] is True
    assert rule['selected_index']==rule['alternative_index']==0
    assert rule['legacy_pair_error']


def test_nonraw_low_loss_is_not_a_comparable_raw_global_best():
    bad=point([1,2],loss=.001,origin='bank');bad['preprocess']='demean'
    data=baseline([point(),point([0,1],loss=1.01,path=1),bad])
    _,_,rule=choose_final_competition(data)
    assert rule['raw_global_best_validation_mae']==1.
    assert rule['alternative_index']==1


def test_fallback_ranking_uses_final_symmetric_difference_then_count_loss_index():
    # Legal leaf+{1,2} family: it removes the old common root and adds one
    # local block, so final difference2 despite fewer raw supports.
    bigger=point([1,2],loss=.99,origin='other')
    for key in ('supports','r','x'):bigger[key].pop(3)
    data=baseline([point(),point([0,1],weight=.01,loss=1.,path=1),bigger,
        point([1,2],loss=1.01,origin='bank'),point([0,1],loss=.995,origin='bank')])
    _,_,rule=choose_final_competition(data)
    assert rule['alternative_index']==4
    assert rule['final_symmetric_difference']==1
