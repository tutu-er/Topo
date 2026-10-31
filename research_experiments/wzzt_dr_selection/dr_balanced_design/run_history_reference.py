"""Bounded JSON-only historical reference extraction and metadata regression.

The source is one already completed observed-only baseline; no scenarios,
household data, truth, new fits, LPs or AC runs are opened or executed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import unittest
import zipfile
from . import history_reference as ref
from . import test_history_reference as ref_tests
from . import test_coefficient_snapshot as snapshot_tests


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(protocol_path, output):
    protocol_path, output = Path(protocol_path), Path(output)
    p = json.loads(protocol_path.read_text(encoding='utf-8'))
    repo, source = Path(p['repository']), Path(p['historical_result'])
    for relative, expected in p['source_sha256'].items():
        if sha(repo/relative) != expected:
            raise ValueError('source changed after freeze: '+relative)
    if sha(source) != p['historical_result_sha256']:
        raise ValueError('historical result hash mismatch')
    if sha(p['historical_source_archive']) != p['historical_source_archive_sha256']:
        raise ValueError('historical source archive hash mismatch')
    with zipfile.ZipFile(p['historical_source_archive']) as archive:
        old_engine = archive.read(p['old_engine_archive_entry'])
    if hashlib.sha256(old_engine).hexdigest() != p['historical_engine_sha256']:
        raise ValueError('historical producer engine hash mismatch')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        raise FileExistsError('completed extraction is not silently rerun')
    lock = output/'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), protocol_sha256=sha(protocol_path)), stream)
    result = dict(status='running', protocol_sha256=sha(protocol_path),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform()),
                  real_identifier_calls=0, new_fixed_family_or_probability_LP=0, new_AC_or_future=0,
                  observed_power_voltage_or_household_files_opened=0)
    try:
        stream = io.StringIO()
        suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(ref_tests),
                                    unittest.defaultTestLoader.loadTestsFromModule(snapshot_tests)])
        test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (output/'unit_tests.txt').write_text(stream.getvalue(), encoding='utf-8')
        result['tests'] = dict(run=test_result.testsRun, failures=len(test_result.failures),
                               errors=len(test_result.errors), successful=test_result.wasSuccessful(),
                               solver_receipt_fixtures_not_real_fits=True)
        if not test_result.wasSuccessful() or test_result.testsRun != p['expected_unit_tests']:
            raise RuntimeError('reference metadata/regression checks did not pass')
        historical = json.loads(source.read_text(encoding='utf-8'))
        original_engine = [value for key, value in historical['source_sha256'].items()
                           if key.replace('\\', '/').endswith('/dr_pool_v2/engine.py')]
        if original_engine != [p['historical_engine_sha256']]:
            raise ValueError('result is not bound to archived historical engine')
        if historical['training_rows'] != 288 or historical['validation_rows'] != 288:
            raise ValueError('full historical row contract changed')
        packet = ref.extract_reference(historical, source_json_sha256=sha(source),
                                       scientific_reference_locked=False)
        audit_families = {ref.canonical(row['family']) for row in historical['LP_solves']}
        represented = sum(ref.canonical(row['complete_container']) in audit_families for row in packet['references'])
        if represented != len(packet['references']):
            raise ValueError('a reconstructed complete family lacks its old LP receipt')
        transitions = []
        for name, search in historical['searches'].items():
            for step in range(1, len(search['path'])):
                old = search['path'][step-1]['objective']
                new = search['path'][step]['objective']
                gain = old-new
                if not gain > 1e-10+.001*old:
                    raise ValueError('saved accepted path violates its numerical acceptance rule')
                transitions.append(dict(path=name, step=step, gain=gain,
                                         threshold=1e-10+.001*old))
        write(output/'reference_packet.json', packet)
        result.update(status='complete_history_metadata_audit_only', summary=packet['summary'],
                      training_rows=packet['training_rows'], validation_rows=packet['validation_rows'],
                      selection_rows_found_in_original_LP_receipts=represented,
                      historical_LP_receipts_read=len(historical['LP_solves']),
                      reconstructed_accepted_transitions=transitions,
                      reference_packet_sha256=sha(output/'reference_packet.json'),
                      risk_ready=packet['risk_ready'], scientific_reference_locked=False,
                      topology_posterior=False, full_parameter_or_tree_coverage=False,
                      old_coefficient_loss_recovered_exactly=False,
                      new_engine_change='additive full_fit metadata only; old objective/support/search/selection rules retained',
                      physical_DR_case_reopened=False)
    except Exception as exc:
        result.update(status='failed', exception_type=type(exc).__name__, exception=str(exc),
                      traceback=traceback.format_exc())
    finally:
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(output/'result.json', result)
        lock.unlink(missing_ok=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.output)
    print(json.dumps(dict(status=result['status'], tests=result.get('tests'),
                          summary=result.get('summary')), ensure_ascii=True))
    raise SystemExit(0 if result['status'].startswith('complete') else 1)
