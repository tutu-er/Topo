"""Pure saved-curve parsing and arithmetic; no scientific solver is run."""
from copy import deepcopy
import gzip
import json
import pytest
from . import ev_compact_plot as plot


def trial():
    power = [[2., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., -1.5, 0.], [0., 1.2, 0.], [0., 0., 0.]]
    connected = [[True, False, False], [False]*3, [False]*3] + [[False, True, False]]*3
    operation = dict(feasible=True, u_kw=deepcopy(power), connected=connected,
                     route=[dict(departure=t, arrival=t+1, kind=k) for t,k in enumerate(
                         ('connected', 'travel', 'setup', 'connected', 'connected', 'connected'))])
    row = dict(method='refined_v2g_mobile', status='complete', true_name='A', power_kw=power,
               fits=[dict(name='A', success=True, loss=2.), dict(name='B', success=True, loss=1.)], margin=-1.)
    return dict(config=dict(physical=dict(horizon=6, dt_hours=.25)), history=dict(p=[[0.,0.,0.]]),
                proposal=dict(bank=[dict(name='A'),dict(name='B')]), rows=[row],
                selection=dict(methods=dict(refined_v2g_mobile=dict(operation=operation, selected_origin='v2g_fixed'))))


def test_signed_curve_connected_zero_and_travel_setup_are_distinct():
    panel = plot.prepare_panel(trial(), 'refined_v2g_mobile')
    assert panel['actual_M'] == -1. and panel['selected_origin'] == 'v2g_fixed'
    assert [s['activity'] for s in panel['slots']] == ['connected', 'travel', 'setup', 'connected', 'connected', 'connected']
    assert panel['slots'][3]['net_power_kw'] == -1.5 and panel['slots'][3]['connected_site'] == 1
    assert panel['slots'][5]['net_power_kw'] == 0. and panel['slots'][5]['connected_site'] == 1
    assert panel['slots'][1]['net_power_kw'] == 0. and panel['slots'][1]['connected_site'] is None
    assert panel['slots'][-1]['stop_minutes'] == 90.


@pytest.mark.parametrize('change', [
    lambda t: t['rows'][0].update(margin=1.),
    lambda t: t['selection']['methods']['refined_v2g_mobile']['operation']['u_kw'][0].__setitem__(0, 3.),
    lambda t: t['selection']['methods']['refined_v2g_mobile']['operation']['connected'][0].__setitem__(0, False),
    lambda t: t['selection']['methods']['refined_v2g_mobile']['operation']['route'][0].update(kind='travel'),
    lambda t: t['selection']['methods']['refined_v2g_mobile']['operation']['route'][1].update(departure=0),
])
def test_inconsistent_saved_curve_or_annotation_is_rejected(change):
    saved = trial(); change(saved)
    with pytest.raises(ValueError):
        plot.prepare_panel(saved, 'refined_v2g_mobile')


def test_failure_and_missing_curve_are_retained_without_made_up_score():
    saved = trial(); saved['rows'][0] = dict(method='refined_v2g_mobile', status='operation_or_design_failure')
    panel = plot.prepare_panel(saved, 'refined_v2g_mobile')
    assert panel['actual_M'] is None and panel['slots'] == []
    assert panel['curve_unavailable_reason'] == 'no_saved_executed_power'


def test_actual_fit_failure_does_not_hide_saved_executable_curve():
    saved = trial(); saved['rows'][0]['status'] = 'joint_fit_failure'
    panel = plot.prepare_panel(saved, 'refined_v2g_mobile')
    assert panel['actual_M'] is None and len(panel['slots']) == 6


def test_no_connected_metadata_does_not_invent_zero_power_connection():
    saved = trial(); operation = saved['selection']['methods']['refined_v2g_mobile']['operation']
    operation.pop('connected'); operation.pop('route')
    panel = plot.prepare_panel(saved, 'refined_v2g_mobile')
    assert panel['slots'][3]['injection_site'] == 1 and panel['slots'][3]['connected_site'] is None
    assert panel['slots'][5]['plot_site'] is None and panel['slots'][5]['activity'] == 'unknown'


def test_duplicate_fit_is_not_a_complete_candidate_comparison():
    row = trial()['rows'][0]; row['fits'].append(deepcopy(row['fits'][0]))
    assert plot.actual_margin(row, ('A', 'B'))[0] is None


def test_milp_source_is_separate_from_whether_its_curve_is_new():
    saved = trial(); record = saved['selection']['methods']['refined_v2g_mobile']
    record['selected_origin'] = 'MILP'
    assert plot.prepare_panel(saved, 'refined_v2g_mobile')['is_new_MILP_curve'] is None
    record['initial_candidates'] = [dict(operation=dict(feasible=True, u_kw=deepcopy(saved['rows'][0]['power_kw'])))]
    assert plot.prepare_panel(saved, 'refined_v2g_mobile')['is_new_MILP_curve'] is False
    record['initial_candidates'][0]['operation']['u_kw'][0][0] += .5
    assert plot.prepare_panel(saved, 'refined_v2g_mobile')['is_new_MILP_curve'] is True


def saved_directory(tmp_path, kind):
    saved = trial(); saved['config']['physical']['device'] = dict(power_kw=6.)
    policy = dict(config=saved['config'], prediction='ensemble', master_form='shared_control_square',
                  domain='asset', source_sha256={'model.py': 'a' * 64})
    policy_id = plot.digest(json.dumps(policy, sort_keys=True).encode())
    freeze = dict(**policy, policy_id=policy_id, seeds=[1, 2, 3, 4], phase='development')
    (tmp_path / 'freeze.json').write_text(json.dumps(freeze))
    for seed in freeze['seeds']:
        value = dict(saved, **{key: item for key, item in policy.items() if key != 'config'},
                     seed=seed, phase='development', policy_id=policy_id)
        if kind == 'slim':
            value.update(artifact_schema='ev_compact_slim_trace_v1', delivery={'source_raw_sha256': 'b' * 64})
        name = f'trial_{seed}' + ('.slim' if kind == 'slim' else '') + '.json.gz'
        (tmp_path / name).write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))


@pytest.mark.parametrize('kind', ['raw', 'slim'])
def test_four_predeclared_raw_or_slim_inputs_bind_compressed_hash_and_policy(tmp_path, kind):
    saved_directory(tmp_path, kind)
    extracted = plot.figure_data(tmp_path, [1, 2, 3, 4], kind=kind)
    assert extracted['predeclared_seeds'] == [1, 2, 3, 4]
    assert len(extracted['histories']) == 4 and len(extracted['inputs']) == 4
    assert all(record['kind'] == kind and len(record['compressed_sha256']) == 64 for record in extracted['inputs'])
    assert all(history['panels'][1]['actual_M'] == -1. for history in extracted['histories'])
    assert extracted['new_fits'] == extracted['new_optimizations'] == extracted['new_actions_selected'] == 0


def test_calibration_and_ambiguous_or_unplanned_examples_are_refused(tmp_path):
    saved_directory(tmp_path, 'raw')
    with pytest.raises(ValueError, match='exactly four'):
        plot.figure_data(tmp_path, [1, 2, 3, 3])
    with pytest.raises(ValueError, match='seed absent'):
        plot.figure_data(tmp_path, [1, 2, 3, 5])
    (tmp_path / 'trial_1.slim.json.gz').write_bytes((tmp_path / 'trial_1.json.gz').read_bytes())
    with pytest.raises(ValueError, match='unambiguous'):
        plot.figure_data(tmp_path, [1, 2, 3, 4])
    freeze = json.loads((tmp_path / 'freeze.json').read_text()); freeze['phase'] = 'calibration'
    (tmp_path / 'freeze.json').write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='not calibration'):
        plot.figure_data(tmp_path, [1, 2, 3, 4], kind='raw')


def test_report_layout_is_explicit_and_readable_after_body_width_scaling():
    default = plot.layout_parameters(); report = plot.layout_parameters('report')
    assert (default['width_inches'], default['height_inches']) == (12., 8.6)
    assert (report['width_inches'], report['height_inches']) == (8., 8.5)
    assert report['annotation_fontsize_at_report_width_pt'] == pytest.approx(8.125)
    assert report['axis_label_fontsize_at_report_width_pt'] == pytest.approx(8.125)
    assert default['annotation_fontsize_pt'] == 8.  # The original preset is still selectable.
    with pytest.raises(ValueError, match='layout'):
        plot.layout_parameters('unrecorded')
