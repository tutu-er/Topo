"""Integration admission checks; no optimization, fitting, or AC execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from . import run


class SmokeReadinessChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.index = self.root/'grid_index.json'
        self.source = self.root/'runtime.py'
        self.inputs = self.root/'inputs.json'
        self.protocol_path = self.root/'protocol.json'
        self.audit_path = self.root/'audit.json'
        self.write(self.index, {'inputs': ['unchanged']})
        self.source.write_text('frozen runtime\n', encoding='utf-8')
        self.write(self.inputs, [{'id': 'smoke'}])
        self.protocol = dict(smoke=True, forecasts=1, repeats=1, expected_rows=8,
            policies=list(run.panel.POLICIES), input_index=run.io.ref(self.index),
            inputs=run.io.ref(self.inputs), sources=[run.io.ref(self.source)])
        self.write(self.protocol_path, self.protocol)
        self.write(self.root/'complete.json', dict(status='complete', source_unchanged=True))
        self.report = dict(status='passed', expected_rows=8, audited_rows=8,
            audit_status_counts={'passed': 8}, task_errors=[], original_failed_tasks=[],
            original_execution_failed_rows=0, original_quote_exceptions=0,
            source_protocol=run.io.ref(self.protocol_path), source_inputs=run.io.ref(self.inputs))
        self.write(self.audit_path, self.report)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def change_protocol(self, **updates):
        self.protocol.update(updates)
        self.write(self.protocol_path, self.protocol)
        self.report['source_protocol'] = run.io.ref(self.protocol_path)
        self.write(self.audit_path, self.report)

    def test_current_audited_smoke_is_accepted_and_recorded(self):
        result = run.smoke_readiness(self.index, self.audit_path)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['audit'], run.io.ref(self.audit_path))
        self.assertEqual(result['verified_current_sources'], 1)

    def test_formal_cannot_skip_readiness_or_create_output_first(self):
        output = self.root/'formal'
        with self.assertRaisesRegex(ValueError, 'smoke-audit'):
            run.prepare(self.index, output)
        self.assertFalse(output.exists())

    def test_different_index_is_rejected(self):
        other = self.root/'other_index.json'
        self.write(other, {'inputs': ['different']})
        with self.assertRaisesRegex(ValueError, 'different grid input'):
            run.smoke_readiness(other, self.audit_path)

    def test_changed_runtime_is_rejected(self):
        self.source.write_text('changed runtime\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            run.smoke_readiness(self.index, self.audit_path)

    def test_failure_and_incomplete_audit_cannot_authorize(self):
        for update in ({'status': 'failed'}, {'audited_rows': 7},
                {'audit_status_counts': {'passed': 7, 'recorded_failure': 1}},
                {'original_execution_failed_rows': 1}, {'original_quote_exceptions': 1},
                {'original_failed_tasks': [{'id': 'failed'}]}, {'task_errors': ['failed']}):
            with self.subTest(update=update):
                report = deepcopy(self.report); report.update(update)
                self.write(self.audit_path, report)
                with self.assertRaises(ValueError):
                    run.smoke_readiness(self.index, self.audit_path)

    def test_non_smoke_or_wrong_scope_is_rejected(self):
        for update in ({'smoke': False}, {'repeats': 16}, {'forecasts': 2},
                {'expected_rows': 128}, {'sources': []}):
            with self.subTest(update=update):
                original = deepcopy(self.protocol)
                self.change_protocol(**update)
                with self.assertRaises(ValueError):
                    run.smoke_readiness(self.index, self.audit_path)
                self.protocol = original

    def test_source_completion_and_inputs_remain_bound(self):
        self.write(self.root/'complete.json', dict(status='complete', source_unchanged=False))
        with self.assertRaises(ValueError):
            run.smoke_readiness(self.index, self.audit_path)
        self.write(self.root/'complete.json', dict(status='complete', source_unchanged=True))
        self.inputs.write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            run.smoke_readiness(self.index, self.audit_path)


if __name__ == '__main__':
    unittest.main()
