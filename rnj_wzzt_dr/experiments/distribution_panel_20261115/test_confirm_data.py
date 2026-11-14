"""Confirmation readiness and unchanged preparation-body checks; no AC/fits."""
import ast
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import data,confirm_data


class ConfirmationDataTests(unittest.TestCase):
    def test_readiness_false_rejects_before_data_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'readiness.json';data.io.write(path,dict(confirmation_ready=False))
            with self.assertRaises(PermissionError):
                confirm_data.prepare(Path(folder)/'absent_data',Path(folder)/'new_output',path)
            self.assertFalse((Path(folder)/'new_output').exists())

    def test_readiness_checks_real_references_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);evidence=folder/'audit.json';data.io.write(evidence,dict(status='passed'))
            ready=folder/'readiness.json'
            data.io.write(ready,dict(confirmation_ready=True,checked_refs=[data.io.ref(evidence)]))
            frozen=data.io.ref(ready)
            self.assertEqual(confirm_data.check_readiness(frozen)['receipt'],frozen)
            data.io.write(evidence,dict(status='changed'))
            with self.assertRaises(ValueError):confirm_data.check_readiness(frozen)

    def test_missing_checked_refs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            ready=Path(folder)/'readiness.json';data.io.write(ready,dict(confirmation_ready=True))
            with self.assertRaises(ValueError):confirm_data.check_readiness(ready)

    def test_confirmation_case_body_preserves_development_algorithm(self):
        # These are the only authorized control-flow differences. All physical,
        # fitting, extra-pool, freezing and offline-evaluation calls must match.
        expected=inspect.getsource(data.generate_one)
        expected=expected.replace('def generate_one(task,out):','def prepare_one(task,out):')
        expected=expected.replace("if task['phase']!='development':raise PermissionError('Confirmation generation is not authorized by this data stage')",
            "if task['phase']!='confirmation':raise PermissionError('This entry is confirmation-only')\n    check_readiness(p['readiness'])")
        expected=expected.replace("directory=out/'development'/task['id']","directory=out/'confirmation'/task['id']")
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse(inspect.getsource(confirm_data.prepare_one))))

    def test_case_checks_readiness_before_profile_or_AC(self):
        with patch.object(confirm_data.io,'read',return_value=dict(readiness={})), \
             patch.object(confirm_data,'check_readiness',side_effect=PermissionError('not_ready')), \
             patch.object(confirm_data.profiles,'build_case') as profile:
            with self.assertRaises(PermissionError):confirm_data.prepare_one(dict(phase='confirmation'),'unused')
            profile.assert_not_called()


if __name__=='__main__':unittest.main()
