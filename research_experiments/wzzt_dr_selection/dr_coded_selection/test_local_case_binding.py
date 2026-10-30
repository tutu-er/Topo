"""Pure frozen-source and row-pair boundaries; never enter physical runtime."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_coded_selection import coded_physical as coded
from research_experiments.wzzt_dr_selection.dr_competition_bridge import physical_bridge as legacy


DEVICE = dict(baseline_kw=.2, baseline_kvar=.07, upper_kw=.4, start_hour=19, end_hour=21,
              initial_energy_kWh=0., final_energy_kWh=.4, efficiency=1., non_V2G=True)


class ReachedPhysicalRuntime(RuntimeError):
    pass


def source_binding(tmp_path):
    path = tmp_path / 'networks' / 'flynn16' / 'reference' / 'metadata.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(root_bus=1, base_kv=.4, base_mva=.2,
        source=dict(case_name='flynn_balanced_lv_16'))), encoding='utf-8')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case_name='flynn_balanced_lv_16')


def local_lock(tmp_path):
    return dict(status='frozen', case='flynn16', condition='reference', base_kw=200.,
        base_kv=.4, base_mva=.2, root_label=1, source_metadata=source_binding(tmp_path),
        labels=[201, 204, 209, 210], device_contract=dict(DEVICE),
        plans={'a': [[1, 1, 0, 0], [0, 0, 1, 1]]},
        directions={'a': [[1, -1, 0, 0], [0, 0, 1, -1]]}, templates={'a': 'opposed'},
        active_labels_by_row={'a': [[201, 204], [209, 210]]})


def write_lock(tmp_path, lock):
    path = tmp_path / 'lock.json'
    path.write_text(json.dumps(lock), encoding='utf-8')
    return path


def guard_runtime(monkeypatch):
    reached = []
    def forbidden(source):
        reached.append(str(source))
        raise ReachedPhysicalRuntime('pure boundary test reached runtime entry')
    copied = dict(vars(legacy))
    copied['_simulation_runtime'] = forbidden
    fake = SimpleNamespace(**copied)
    monkeypatch.setattr(coded, '_legacy', lambda repo: fake)
    return reached


def invoke(path, *, case='flynn16', expected_sha=None):
    return coded.generate_actual_events(path, coded.sha(path) if expected_sha is None else expected_sha,
        'a', path=0, case=case, voltage_sigma=.0002, repo_root=path.parent)


def test_distinct_local_pairs_are_returned_per_row(tmp_path):
    path = write_lock(tmp_path, local_lock(tmp_path))
    result = coded._lock(path, coded.sha(path), 'a', legacy)
    assert result[2] == 'opposed'
    assert np.array_equal(result[3], [[0, 1], [2, 3]])


def test_legacy_global_pair_repeats_without_changing_tuple_contract(tmp_path):
    lock = local_lock(tmp_path)
    lock.pop('active_labels_by_row')
    lock['active_labels'] = [201, 204]
    lock['plans']['a'] = [[1, 1, 0, 0], [1, 1, 0, 0]]
    lock['directions']['a'] = [[1, -1, 0, 0], [1, -1, 0, 0]]
    path = write_lock(tmp_path, lock)
    result = coded._lock(path, coded.sha(path), 'a', legacy)
    assert len(result) == 6 and result[2] == 'opposed'
    assert np.array_equal(result[3], [[0, 1], [0, 1]])


@pytest.mark.parametrize('pairs', [[[201, 204]], [[201, 201], [209, 210]], [[201, 999], [209, 210]]])
def test_malformed_or_unknown_row_pairs_fail_before_runtime(tmp_path, monkeypatch, pairs):
    lock = local_lock(tmp_path)
    lock['active_labels_by_row']['a'] = pairs
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ValueError):
        invoke(path)
    assert reached == []


def test_priced_node_outside_its_row_pair_fails_before_runtime(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock['active_labels_by_row']['a'] = [[201, 209], [209, 210]]
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ValueError):
        invoke(path)
    assert reached == []


@pytest.mark.parametrize('template,direction', [('same', [1, -1, 0, 0]),
                                              ('opposed', [1, 1, 0, 0]),
                                              ('alternating_opposed', [1, 1, 0, 0])])
def test_two_offered_nodes_must_match_declared_phase(tmp_path, monkeypatch, template, direction):
    lock = local_lock(tmp_path)
    lock['templates']['a'] = template
    lock['directions']['a'][0] = direction
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ValueError):
        invoke(path)
    assert reached == []


def test_single_and_zero_offer_do_not_require_a_fictitious_pair_response(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock['plans']['a'] = [[2, 0, 0, 0], [0, 0, 0, 0]]
    lock['directions']['a'] = [[1, 0, 0, 0], [0, 0, 0, 0]]
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ReachedPhysicalRuntime):
        invoke(path)
    assert len(reached) == 1


def test_wrong_lock_hash_fails_before_runtime(tmp_path, monkeypatch):
    path = write_lock(tmp_path, local_lock(tmp_path))
    reached = guard_runtime(monkeypatch)
    with pytest.raises(RuntimeError):
        invoke(path, expected_sha='0' * 64)
    assert reached == []


def test_frozen_case_mismatch_fails_before_runtime(tmp_path, monkeypatch):
    path = write_lock(tmp_path, local_lock(tmp_path))
    reached = guard_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        invoke(path, case='paper15')
    assert reached == []


def test_nonlegacy_missing_metadata_fails_before_runtime(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock.pop('source_metadata')
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        invoke(path)
    assert reached == []


def test_metadata_hash_mismatch_fails_before_runtime(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock['source_metadata']['sha256'] = '0' * 64
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(RuntimeError):
        invoke(path)
    assert reached == []


def test_power_base_mismatch_fails_before_runtime(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock['base_kw'] = 100.
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ValueError):
        invoke(path)
    assert reached == []


@pytest.mark.parametrize('orientations', [[1, -1], [1, 1, 0, 1]])
def test_invalid_event_orientations_fail_before_runtime(tmp_path, monkeypatch, orientations):
    lock = local_lock(tmp_path)
    lock['orientations_by_event'] = orientations
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ValueError):
        invoke(path)
    assert reached == []


def test_valid_frozen_event_signs_reach_runtime_boundary(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock['orientations_by_event'] = [1, -1, -1, 1]
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ReachedPhysicalRuntime):
        invoke(path)
    assert len(reached) == 1


def test_valid_nonlegacy_binding_reaches_only_runtime_boundary(tmp_path, monkeypatch):
    path = write_lock(tmp_path, local_lock(tmp_path))
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ReachedPhysicalRuntime):
        invoke(path)
    assert len(reached) == 1


def test_legacy_case_without_metadata_still_reaches_runtime_boundary(tmp_path, monkeypatch):
    lock = local_lock(tmp_path)
    lock['case'] = 'paper15'
    lock.pop('source_metadata')
    path = write_lock(tmp_path, lock)
    reached = guard_runtime(monkeypatch)
    with pytest.raises(ReachedPhysicalRuntime):
        invoke(path, case='paper15')
    assert len(reached) == 1


def test_legacy_global_offer_keeps_its_original_helper_pair_semantics(tmp_path):
    lock = local_lock(tmp_path)
    lock.pop('active_labels_by_row')
    lock['active_labels'] = [201, 204]
    lock['templates']['a'] = 'same'
    lock['plans']['a'] = [[1, 1, 1, 1], [0, 0, 0, 0]]
    lock['directions']['a'] = [[1, 1, 1, 1], [0, 0, 0, 0]]
    path = write_lock(tmp_path, lock)
    result = coded._lock(path, coded.sha(path), 'a', legacy)
    assert np.array_equal(result[0], lock['plans']['a'])
    assert np.array_equal(result[3], [[0, 1], [0, 1]])


@pytest.mark.parametrize('field,bad_value', [('root_bus', 2), ('base_kv', .5), ('base_mva', .3),
                                            ('terminals', [210, 209, 204, 201])])
def test_context_source_mismatch_stops_before_graph_check_or_profile(tmp_path, monkeypatch, field, bad_value):
    lock = local_lock(tmp_path)
    path = write_lock(tmp_path, lock)
    graph_checks = []
    def graph_forbidden(**kwargs):
        graph_checks.append(kwargs)
        raise AssertionError('metadata mismatch reached graph validation')
    network = SimpleNamespace(root_bus=1, base_kv=.4, base_mva=.2,
                              check_terminal_load_only=graph_forbidden)
    context = SimpleNamespace(net=network, terminals=lock['labels'])
    setattr(context if field == 'terminals' else network, field, bad_value)
    simulated = SimpleNamespace(_prepare_case_context=lambda *args, **kwargs: context)
    copied = dict(vars(legacy))
    copied['_simulation_runtime'] = lambda source: (simulated, {})
    copied['_physical_check_function'] = lambda repo: (None, Path(lock['source_metadata']['path']))
    fake = SimpleNamespace(**copied)
    monkeypatch.setattr(coded, '_legacy', lambda repo: fake)
    with pytest.raises(ValueError):
        invoke(path)
    assert graph_checks == []
