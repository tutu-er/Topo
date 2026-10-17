"""Controller guards only: every process is mocked; no study worker executes."""
from hashlib import sha256
import gzip
import json

import pytest

from . import ev_ac_repair_control as tool


POLICY = 'controller-test-policy'


@pytest.fixture
def guarded(monkeypatch, tmp_path):
    version = dict(policy_id=POLICY, source_sha256={})
    monkeypatch.setattr(tool, 'freeze', lambda *_args, **_kwargs: version)
    monkeypatch.setattr(tool, 'hashes', lambda: {})
    monkeypatch.setattr(tool, '_pid_alive', lambda _pid: False)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('a guarded controller must not start a worker')
    monkeypatch.setattr(tool.subprocess, 'Popen', forbidden)
    root = tmp_path/'study'; root.mkdir()
    return root


def calibration_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    (root/'calibration').mkdir(exist_ok=True)
    specs = tool.study.specifications('calibration')
    hashes = {}
    for spec in specs:
        path = root/'calibration'/('history_%03d.json.gz'%spec['index'])
        tool.old.save_record(path, dict(spec=spec, status='process_failure'))
        hashes[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
    calibration = dict(histories=len(specs), G11_multiplier=2.,
        policy_id=POLICY, input_sha256=hashes)
    tool.atomic_json(root/'calibration.json', calibration)
    return calibration, specs


def test_freeze_snapshot_and_changed_source_are_immutable(monkeypatch, tmp_path):
    source_dir = tmp_path/'sources'; source_dir.mkdir()
    source = source_dir/'fixture.py'; source.write_bytes(b'alpha = 1\n')
    monkeypatch.setattr(tool, 'HERE', source_dir)
    monkeypatch.setattr(tool, 'hashes', lambda: {
        source.name: sha256(source.read_bytes()).hexdigest()})
    root = tmp_path/'version'
    frozen = tool.freeze(root, dict(draws=16))
    snapshot = (root/'source_snapshot.json.gz').read_bytes()
    assert 'fixture.py' in json.loads(gzip.decompress(snapshot))
    assert tool.freeze(root, dict(draws=16)) == frozen
    assert (root/'source_snapshot.json.gz').read_bytes() == snapshot
    source.write_bytes(b'alpha = 2\n')
    with pytest.raises(ValueError, match='immutable freeze'):
        tool.freeze(root, dict(draws=16))
    assert tool.analysis.load(root/'freeze.json') == frozen


@pytest.mark.parametrize('phase', ['calibration', 'evaluation'])
def test_partial_calibration_or_evaluation_is_forbidden(guarded, phase):
    with pytest.raises(ValueError, match='all 32 specifications'):
        tool.run_phase(guarded, phase, limit=1)


def test_evaluation_requires_calibration(guarded):
    with pytest.raises(ValueError, match='requires saved calibration'):
        tool.run_phase(guarded, 'evaluation')


def test_evaluation_rejects_partial_calibration_count(guarded):
    calibration, _ = calibration_fixture(guarded)
    calibration['histories'] = 1
    tool.atomic_json(guarded/'calibration.json', calibration)
    with pytest.raises(ValueError, match='all 32 calibration histories'):
        tool.run_phase(guarded, 'evaluation')


def test_evaluation_rejects_failed_or_foreign_policy(guarded):
    calibration, _ = calibration_fixture(guarded)
    calibration['G11_multiplier'] = None
    tool.atomic_json(guarded/'calibration.json', calibration)
    with pytest.raises(ValueError, match='no usable history'):
        tool.run_phase(guarded, 'evaluation')
    calibration.update(G11_multiplier=2., policy_id='foreign-policy')
    tool.atomic_json(guarded/'calibration.json', calibration)
    with pytest.raises(ValueError, match='policy does not match'):
        tool.run_phase(guarded, 'evaluation')


def test_evaluation_rejects_wrong_seed_and_changed_calibration_input(guarded):
    _, specs = calibration_fixture(guarded)
    path = guarded/'calibration'/'history_000.json.gz'
    wrong = dict(specs[0], seed=specs[0]['seed']+1)
    tool.old.save_record(path, dict(spec=wrong, status='process_failure'))
    with pytest.raises(ValueError, match='mismatched calibration specification'):
        tool.run_phase(guarded, 'evaluation')
    tool.old.save_record(path, dict(spec=specs[0], status='modified-record'))
    with pytest.raises(ValueError, match='inputs differ'):
        tool.run_phase(guarded, 'evaluation')


def test_evaluation_rejects_changed_bound_calibration(guarded):
    calibration_fixture(guarded)
    tool.atomic_json(guarded/'evaluation_calibration_binding.json',
        dict(policy_id=POLICY, calibration_sha256='different-file'))
    with pytest.raises(ValueError, match='changed after evaluation binding'):
        tool.run_phase(guarded, 'evaluation')


def test_live_controller_and_orphan_refuse_duplicate_launch(guarded, monkeypatch):
    guarded.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tool, '_pid_alive', lambda _pid: True)
    tool.atomic_json(guarded/'controller_lock.json', dict(pid=1234))
    with pytest.raises(RuntimeError, match='controller already active'):
        tool.run_phase(guarded, 'development')
    (guarded/'controller_lock.json').unlink()
    tool.atomic_json(guarded/'orphan_workers.json', dict(workers={'0': 2345}))
    with pytest.raises(RuntimeError, match='workers remain active'):
        tool.run_phase(guarded, 'development')


def test_hard_controller_crash_does_not_duplicate_saved_live_worker(guarded, monkeypatch):
    guarded.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tool, '_pid_alive', lambda pid: int(pid) == 2345)
    tool.atomic_json(guarded/'controller_lock.json', dict(pid=1234))
    tool.atomic_json(guarded/'controller.json', dict(pid=1234, status='running',
        workers={'0': dict(pid=2345, elapsed_seconds=20.)}))
    with pytest.raises(RuntimeError, match='worker'):
        tool.run_phase(guarded, 'development')


@pytest.mark.parametrize('status', ['launching', 'running'])
def test_unfinished_launch_claim_refuses_restart(guarded, monkeypatch, status):
    monkeypatch.setattr(tool, '_pid_alive', lambda pid: int(pid) == 2345)
    tool.atomic_json(guarded/'launch_claims'/'development_000.json',
        dict(status=status, pid=2345, controller_pid=1234))
    with pytest.raises(RuntimeError, match='launch claim|registered worker'):
        tool.run_phase(guarded, 'development')


def test_failed_workers_preserve_all_calibration_histories(guarded, monkeypatch):
    commands = []
    class ExitedWorker:
        returncode = 9
        def __init__(self, command, **_kwargs):
            index = int(command[command.index('--index')+1])
            self.claim = guarded/'launch_claims'/('calibration_%03d.json'%index)
            assert tool.analysis.load(self.claim)['status'] == 'launching'
            commands.append(command)
            self.pid = 40000+len(commands)
        def poll(self):
            assert tool.analysis.load(self.claim)['status'] == 'running'
            return self.returncode
    monkeypatch.setattr(tool.subprocess, 'Popen', ExitedWorker)
    monkeypatch.setattr(tool.time, 'sleep', lambda _seconds: None)
    tool.run_phase(guarded, 'calibration')
    expected = tool.study.specifications('calibration')
    assert len(commands) == len(expected) == 32
    assert {int(c[c.index('--index')+1]) for c in commands} == set(range(32))
    for spec in expected:
        row = tool.analysis.load(guarded/'calibration'/('history_%03d.json.gz'%spec['index']))
        assert row['spec'] == spec and row['status'] == 'process_failure'
        assert row['worker_exit_code'] == 9
    calibration = tool.analysis.load(guarded/'calibration.json')
    assert calibration['histories'] == 32
    assert calibration['available_histories'] == 0
    assert calibration['G11_multiplier'] is None
    assert len(calibration['input_sha256']) == 32
    summary = tool.analysis.load(guarded/'calibration_summary.json')
    assert summary['histories'] == 32
    assert len(summary['history_statuses']) == 32
    for group in summary['summaries'].values():
        counts = group['availability']
        assert counts['total_histories'] == counts['failed_histories'] == 32
        assert counts['histories'] == 0
        assert counts['failed_history_indices'] == list(range(32))
    assert not (guarded/'controller_lock.json').exists()
    with pytest.raises(ValueError, match='no usable history'):
        tool.run_phase(guarded, 'evaluation')


def test_cli_calibration_then_evaluation_keep_identical_parameters(monkeypatch):
    calls = []
    monkeypatch.setattr(tool, 'run_phase', lambda root, phase, **settings:
                        calls.append((root, phase, settings)))
    monkeypatch.setattr(tool.sys, 'argv', ['control', '--output', 'test-root',
        '--phase', 'calibration', '--workers', '1', '--draws', '64',
        '--replicates', '2', '--noise-repeats', '2', '--skip-clean-refit',
        '--then-evaluation'])
    tool.main()
    assert [call[1] for call in calls] == ['calibration', 'evaluation']
    for root, _, settings in calls:
        assert root == 'test-root'
        assert {key: settings[key] for key in
                ('workers', 'draws', 'replicates', 'noise_repeats', 'clean_refit')} == dict(
            workers=1, draws=64, replicates=2, noise_repeats=2, clean_refit=False)


@pytest.mark.parametrize('phase', ['development', 'evaluation'])
def test_cli_phase_chain_forbids_other_start_phases(monkeypatch, phase):
    calls = []
    monkeypatch.setattr(tool, 'run_phase', lambda *_args, **_settings: calls.append(True))
    monkeypatch.setattr(tool.sys, 'argv', ['control', '--output', 'test-root',
        '--phase', phase, '--then-evaluation'])
    with pytest.raises(SystemExit) as exc:
        tool.main()
    assert exc.value.code == 2
    assert calls == []
