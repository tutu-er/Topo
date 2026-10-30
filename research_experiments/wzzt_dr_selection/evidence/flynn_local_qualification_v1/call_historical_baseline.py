"""Direct one-call use of the existing shared cache and joint identifier."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import platform
import sys
import traceback

REPO = Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / 'baseline_protocol.json'
    packet = root / 'history_and_forecast.pkl'
    if args.mode == 'prepare':
        if protocol_path.exists():
            raise FileExistsError(protocol_path)
        qualification = root / 'qualification_result.json'
        q = json.loads(qualification.read_text(encoding='utf-8'))
        assert q['status'] == 'complete' and q['training_rows'] == q['validation_rows'] == 288
        assert q['terminal_labels'] == list(range(201, 217))
        protocol = dict(id='flynn16_b7_historical_joint_baseline_v1',
            created_utc=datetime.now(timezone.utc).isoformat(), command_source_sha256=sha(__file__),
            algorithm_source_sha256=source_hashes(), input_packet_sha256=sha(packet),
            qualification_sha256=sha(qualification),
            job_id=q['job_id'], root_label=q['root_label'], terminal_labels=q['terminal_labels'],
            training_rows=288, validation_rows=288, per_path_seconds=25., max_new_fits=1,
            max_extra_paths=2, fixed_family_LP_seconds=15.,
            full_algorithm='existing fit_observed via SharedFitter; original multistart3%/.16',
            measurement_or_response_scenarios_added=0, truth_read=False,
            status='frozen_before_identifier_call')
        write(protocol_path, protocol)
        print(json.dumps(dict(status='frozen', protocol_sha256=sha(protocol_path))))
        return
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert protocol['command_source_sha256'] == sha(__file__)
    assert protocol['algorithm_source_sha256'] == source_hashes()
    assert protocol['input_packet_sha256'] == sha(packet)
    if (root / 'baseline_ended.json').exists():
        raise FileExistsError('terminal baseline attempt retained; do not rerun')
    lock = root / 'baseline.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), protocol_sha256=sha(protocol_path)), stream)
    try:
        with packet.open('rb') as stream:
            history = pickle.load(stream)['history']
        fitter = SharedFitter(root / 'baseline_cache', sources=protocol['algorithm_source_sha256'], max_new_fits=1)
        result = fitter(history['training'], history['validation'], root / 'baseline_request',
            root_label=protocol['root_label'], search_seconds=protocol['per_path_seconds'])
        summary = dict(status=result['status'], protocol_sha256=sha(protocol_path),
            result_path=str(root / 'baseline_request/result.json'),
            result_sha256=sha(root / 'baseline_request/result.json'),
            training_rows=result['training_rows'], validation_rows=result['validation_rows'],
            LP_count=len(result.get('LP_solves', [])),
            search_stops=result.get('search_stop_statuses'), all_local_scans_complete=result.get('all_local_scans_complete'),
            model_names=list(result['models']), elapsed_seconds=result['elapsed_seconds'],
            truth_or_future_read=False, expected_new_fits=1,
            source='existing complete observed-only RNJ+WZZT call',
            environment=dict(python=sys.version, platform=platform.platform()))
        write(root / 'baseline_ended.json', summary)
        print(json.dumps(summary))
    except BaseException as exc:
        write(root / 'baseline_ended.json', dict(status='failed', exception_type=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc(), protocol_sha256=sha(protocol_path)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
