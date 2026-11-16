"""Worker integration with hand-built candidates, never native search or AC.

The real serialization, candidate append, frozen-bank checks, topology metrics
and cumulative SSE selector remain active. Only candidate search is replaced.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from ... import io
from ...identification.algorithm import _output
from ...identification.optimization import errors, response_matrices
from ...observations import fingerprint
from . import data, run


def _point(internal):
    supports = [[0], [1], [2]] + ([[0, 1]] if internal else [])
    return dict(supports=supports, r=[.04, .05, .06]+([.03] if internal else []),
                x=[.01, .02, .03]+([.025] if internal else []))


def _fixture(tmp_path, days=2):
    out = tmp_path/'run'; out.mkdir()
    assets = out/'assets'; assets.mkdir()
    t = np.arange(data.POINTS, dtype=float)
    p = np.column_stack((.02+t/100000, .03+.004*np.sin(t/13), .025+.003*np.cos(t/11)))
    q = np.column_stack((.004+.002*np.cos(t/7), -.001+.003*np.sin(t/17), .005+t/1000000))
    point = _point(True)
    r, x = response_matrices(point, 3); hstar = np.c_[r, x]
    root = 1.02+.0001*np.sin(t/19)
    clean = dict(P=p, Q=q, V=np.sqrt(root[:, None]**2-p@r.T-q@x.T),
                 root=root, labels=np.array([101, 102, 103]))
    noise = dict(P=np.full_like(p, .2), Q=np.full_like(q, -.3),
                 V=np.zeros_like(p), root=np.zeros(data.POINTS))
    # Deterministic voltage residual; the test still evaluates all real SSEs.
    noise['V'][:, 2] = .001*np.cos(t/23)
    np.savez_compressed(assets/'clean.npz', **clean)
    np.savez_compressed(assets/'noise_h00.npz', **noise)
    np.savez_compressed(assets/'Hstar.npz', Hstar=hstar,
                        support_json=np.asarray(json.dumps(point['supports'])))
    io.write(assets/'equipment.json', dict(devices=[
        dict(node_label=101, kind='pv'), dict(node_label=102, kind='ev')]))
    io.write(assets/'old_inputs.json', [])
    (assets/'archive.bin').write_bytes(b'fixture-only-not-a-source-archive')
    (assets/'source.py').write_text('# immutable fixture source\n', encoding='utf-8')
    world = dict(case='soumalas11', labels=[101, 102, 103],
        files={name:io.ref(assets/name) for name in ('clean.npz', 'noise_h00.npz', 'Hstar.npz')},
        old_entry=dict(equipment_spec=io.ref(assets/'equipment.json')), **data.indices())
    io.write(out/'worlds.json', [world])
    protocol = dict(sources=[io.ref(assets/'source.py')],
        old_inputs=io.ref(assets/'old_inputs.json'), archive=io.ref(assets/'archive.bin'),
        method_seconds=120., additional_E_seconds=60.,
        observed_error=dict(P=.005, Q=.005, V=.0005, root=.0005))
    io.write(out/'protocol.json', protocol)
    task = dict(id=f'soumalas11__d{days:02d}__h00', case='soumalas11',
                train_days=days, history_repeat=0)
    return out, world, task, clean, noise


def _search_mocks(monkeypatch, out, task, clean, noise):
    calls = []
    directory = out/'fits'/task['id']
    start = 576 if task['train_days'] == 2 else 0
    expected_p = clean['P']*(1+.005*noise['P'])
    expected_q = clean['Q']*(1+.005*noise['Q'])
    expected_y = (clean['root']*(1+.0005*noise['root']))[:, None]**2 - (
        clean['V']*(1+.0005*noise['V']))**2

    def fake_fit(training, validation, *, method, config):
        assert method == '101' and config.total_seconds == 120.
        assert (directory/'observations_sealed.json').exists()
        assert not (directory/'candidates_sealed.json').exists()
        assert training.points == 96*task['train_days'] and validation.points == 192
        for attr, values in [('p', expected_p), ('q', expected_q), ('y', expected_y)]:
            np.testing.assert_array_equal(getattr(training, attr), values[start:768])
            np.testing.assert_array_equal(getattr(validation, attr), values[768:960])
        assert set(vars(training)) == {'p', 'q', 'y', 'labels', 'voltage', 'root'}
        calls.append('fit')
        point = _point(False); point.update(mse=errors(training, point)['mse'])
        selected, mae, bank = _output([point], training, validation, 1e-6, config)
        return dict(method='101', selected=selected, selected_mse_view=deepcopy(selected),
            selected_mae_view=mae, frozen_candidates=bank, config=asdict(config),
            reference_mse=1e-6, events=[], QP=[], MIQP=[], common_seconds=0.,
            total_seconds=0., total_cap_exceeded=False, stop_reason='mock_search',
            handoff=dict(training_sha256=fingerprint(training),
                         validation_sha256=fingerprint(validation)))

    def fake_extra(original, training, *, seconds):
        assert calls == ['fit'] and seconds == 60.
        assert not (directory/'candidates_sealed.json').exists()
        np.testing.assert_array_equal(training.p, expected_p[start:768])
        calls.append('extra')
        point = _point(True); point['mse'] = errors(training, point)['mse']
        return dict(status='returned_point', closed_point=point, truth_used=False,
                    validation_used=False, new_day_used=False, total_seconds=0.)

    original_arrays = run.arrays

    def guarded_arrays(ref):
        if Path(ref['path']).name == 'Hstar.npz':
            assert calls == ['fit', 'extra']
            assert (directory/'candidates_sealed.json').exists()
            calls.append('truth_after_seal')
        return original_arrays(ref)

    monkeypatch.setattr(run, 'fit_observed', fake_fit)
    monkeypatch.setattr(run, 'additional_search', fake_extra)
    monkeypatch.setattr(run, 'arrays', guarded_arrays)
    return calls


@pytest.mark.parametrize('days', [2, 8])
def test_worker_preserves_observed_slices_seals_before_truth_and_scores_full_bank(tmp_path, monkeypatch, days):
    out, world, task, clean, noise = _fixture(tmp_path, days)
    calls = _search_mocks(monkeypatch, out, task, clean, noise)
    done = run.fit_task(out, task)
    result = io.read(done['result']['path'])
    assert done['status'] == 'complete', result
    assert calls == ['fit', 'extra', 'truth_after_seal']
    folder = out/'fits'/task['id']
    original = io.read(folder/'original_native.json')
    enriched = io.read(folder/'native.json')
    assert enriched['frozen_candidates'][:-1] == original['frozen_candidates']
    assert result['original_evaluation']['bank_size'] == 1
    assert result['original_evaluation']['history_topology']['internal_missing'] == 1
    evaluation = result['evaluation']
    assert evaluation['bank_size'] == 2
    assert evaluation['history_selected'] == evaluation['cumulative_selected'] == 1
    assert evaluation['bank_has_full_truth'] and evaluation['bank_has_internal_truth']
    assert evaluation['cumulative_selection']['total_cells'] == (192+96)*3
    assert not evaluation['cumulative_selection']['refitted']
    assert '192 historical validation + 96 new-day' in evaluation['scoring_rows']
    assert {r['role'] for r in result['matrix_rows']} == {
        'history_selected', 'cumulative_selected', 'best_true_internal'}
    for row in result['matrix_rows']:
        assert row['metrics']['points'] == 96*days
        assert row['design']['residual_cells'] == 96*days*3
        assert row['metrics']['R']['relative_error'] == 0.
        assert row['metrics']['X']['relative_error'] == 0.
    sealed = io.read(folder/'candidates_sealed.json')
    assert io.ref(sealed['native']['path']) == sealed['native']
    assert not sealed['truth_used'] and not sealed['new_day_used']
    assert run.freeze_history(enriched, run.data.slice_observations(
        data.observations(clean, noise), world, days)['validation']).parameter_sha256 == sealed['parameter_sha256']


def test_completed_worker_reuses_hashed_result_and_never_restarts_search(tmp_path, monkeypatch):
    out, _, task, clean, noise = _fixture(tmp_path)
    _search_mocks(monkeypatch, out, task, clean, noise)
    done = run.fit_task(out, task)
    assert done['status'] == 'complete'
    folder = out/'fits'/task['id']
    before = {p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    monkeypatch.setattr(run, 'fit_observed', lambda *a, **k:pytest.fail('Restarted completed fit'))
    monkeypatch.setattr(run, 'additional_search', lambda *a, **k:pytest.fail('Restarted extra search'))
    assert run.fit_task(out, task) == done
    assert before == {p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    Path(done['result']['path']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Sealed file changed'):
        run.fit_task(out, task)


def test_search_failure_is_retained_and_not_retried_as_a_fresh_budget(tmp_path, monkeypatch):
    out, _, task, _, _ = _fixture(tmp_path)
    count = []

    def fail(*args, **kwargs):
        count.append(1)
        raise RuntimeError('injected identifier failure')

    monkeypatch.setattr(run, 'fit_observed', fail)
    monkeypatch.setattr(run, 'additional_search', lambda *a, **k:pytest.fail('Search after failed fit'))
    done = run.fit_task(out, task); result = io.read(done['result']['path'])
    assert done['status'] == result['status'] == 'failed'
    assert result['stage'] == 'native101' and 'injected identifier failure' in result['error']
    assert result['task'] == task and 'RuntimeError' in result['traceback']
    assert (out/'fits'/task['id']/'observations_sealed.json').exists()
    assert not (out/'fits'/task['id']/'candidates_sealed.json').exists()
    assert run.fit_task(out, task) == done and len(count) == 1


def test_partial_worker_directory_refuses_budget_reset(tmp_path, monkeypatch):
    out, _, task, _, _ = _fixture(tmp_path)
    directory = out/'fits'/task['id']; directory.mkdir(parents=True)
    sentinel = directory/'started.json'; sentinel.write_text('partial-start', encoding='utf-8')
    monkeypatch.setattr(run, 'fit_observed', lambda *a, **k:pytest.fail('Partial fit was restarted'))
    with pytest.raises(RuntimeError, match='no automatic budget reset'):
        run.fit_task(out, task)
    assert sentinel.read_text(encoding='utf-8') == 'partial-start'
    assert not (directory/'complete.json').exists()


def test_prepare_keeps_failed_world_ledger_and_does_not_mark_one_world_ready(tmp_path, monkeypatch):
    old = tmp_path/'old.json'; io.write(old, [])
    archive = tmp_path/'archive.bin'; archive.write_bytes(b'fixture')
    monkeypatch.setattr(run, 'OLD_INPUTS', old)
    monkeypatch.setattr(run, 'ARCHIVE', archive)
    monkeypatch.setattr(run, 'source_paths', lambda:[])
    monkeypatch.setattr(run.validate, '_environment', lambda:{'scope':'mocked prepare test'})
    monkeypatch.setattr(run.validate, '_git_receipt', lambda paths:{})

    def incomplete(destination, *args):
        io.write(destination/'failures.json', [dict(case='pengwah18', error='injected preparation failure')])
        return [dict(case='soumalas11')]

    monkeypatch.setattr(run.data, 'prepare_worlds', incomplete)
    out = tmp_path/'prepared'
    with pytest.raises(RuntimeError, match='both locked physical worlds'):
        run.prepare(out)
    assert not (out/'prepared.json').exists()
    assert io.read(out/'worlds.json') == [dict(case='soumalas11')]
    assert io.read(out/'data'/'failures.json')[0]['case'] == 'pengwah18'
    assert len(io.read(out/'protocol.json')['tasks']) == 16
    with pytest.raises(FileExistsError, match='Preserve existing preparation'):
        run.prepare(out)
