"""Input and trajectory guards for the initialization audit."""
import pandas as pd
from research_experiments.wzzt_dr_selection.baseline_alignment.initialization_audit import observation_only, new_supports


def test_observation_boundary_preserves_all_terminal_columns_and_exact_rows():
    frame=pd.DataFrame([[1.,2.,3.],[4.,5.,6.]],index=['t0','t1'],columns=[103,104,105])
    source={'name':'same_observations','P_terminal':frame,'Q_terminal':frame/3,
            'drop_target':frame/100,'truth_clades':[[103,104,105]],'network':'not permitted'}
    target=observation_only([source])[0]
    assert set(target)=={'name','P_terminal','Q_terminal','drop_target'}
    for key in ('P_terminal','Q_terminal','drop_target'):
        pd.testing.assert_frame_equal(target[key],source[key])
        assert target[key].shape==(2,3)
    assert target['P_terminal'].columns.tolist()==[103,104,105]


def test_trajectory_reports_replacement_without_treating_net_count_as_new_step():
    points=[{'support_labels':[[103,104,105],[113,114,115]]},
            {'support_labels':[[103,104,105],[113,114,115],[101,102,103,104,105]]},
            {'support_labels':[[103,104,105],[113,114,115],[101,102]]}]
    rows=new_supports(points)
    assert rows[1]['new_support_labels']==[[101,102,103,104,105]]
    assert rows[2]['new_support_labels']==[[101,102]]
    assert rows[2]['removed_support_labels']==[[101,102,103,104,105]]
    assert len(rows)==3
