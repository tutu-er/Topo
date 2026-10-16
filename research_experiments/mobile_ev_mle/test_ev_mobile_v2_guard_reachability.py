"""Small checks for the read-only reachability/cost explanation."""
from pathlib import Path
import pytest
from .ev_mobile_v2_guard_reachability import _same,extract,summarize


def test_saved_curve_matching_is_explicit_and_does_not_alias_different_routes():
    assert _same([[1.,0.]],[[1.+1e-9,0.]])
    assert not _same([[1.,0.]],[[0.,1.]])
    assert not _same([[1.]],[[1.,0.]])


def test_formal_path_is_rejected_before_reading_even_if_nonexistent():
    with pytest.raises(ValueError,match='formal paths'):
        extract(Path('outputs/mobile_v2_formal_cal_n3/case_000.json.gz'),{})


def test_complete_nonlinear_guard_can_allow_a_point_its_local_linear_floor_excludes():
    parent=0.;future=3.;floor=1.
    d1=lambda u:(u-1.)**2
    d2=lambda u:(u+1.)**2
    assert d1(parent)==d2(parent)==floor
    assert d1(future)>=floor and d2(future)>=floor
    assert (d1(future)+d2(future))/2>(d1(parent)+d2(parent))/2
    first_order_d1=d1(parent)+2*(parent-1)*(future-parent)
    assert first_order_d1<floor


def test_summary_distinguishes_scored_initial_presence_and_missing_parent():
    rows=[dict(seed=1,planning_success=False,actual_moves=0,selected_origin=None,
        counts=dict(no_gradient_steps=1,no_gradient_steps_parent_profile_unavailable=1),
        planning_seconds=30.,steps=[],old_action_audit=dict(old_was_moving=True,
            old_passed_capped_floor=True,old_action_present_in_scored_new_initials=False))]
    value=summarize(rows)
    assert value['counts']['no_gradient_steps_parent_profile_unavailable']==1
    assert value['old_feasible_moving_histories']==1
    assert value['old_feasible_moving_present_as_scored_initial']==0
