"""Persistence and task-boundary tests; no optimizer, AC or real run is opened."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import continuation as c


class ContinuationTests(unittest.TestCase):
    def test_exact_returned_point_and_solver_arguments_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            u=np.arange(12,dtype=float).reshape(3,4);args=(object(),object());env=object()
            kwargs=dict(mode='mgf_fixed',seconds=13.25,env=env,record_directory=Path(directory))
            receipt=dict(feasible=True,cost_yuan=1.25)
            with patch.object(c,'ORIGINAL_PLAN',return_value=(receipt,u)) as plan:
                returned,point=c.checkpoint_plan(*args,**kwargs)
            plan.assert_called_once_with(*args,**kwargs)
            self.assertIs(point,u);self.assertIs(returned,receipt)
            with np.load(c.original.old.checked(receipt['action_checkpoint'])) as z:
                np.testing.assert_array_equal(z['u'],u)
            self.assertEqual(receipt['action_checkpoint_array_sha256'],c.io.contract_hash(u))
            self.assertFalse((Path(directory)/'action_checkpoint.npz.tmp').exists())

    def test_no_point_has_explicit_marker_and_no_zero_point(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(c,'ORIGINAL_PLAN',return_value=(dict(feasible=False,status='unresolved'),None)):
                receipt,point=c.checkpoint_plan(record_directory=directory,seconds=4.)
            self.assertIsNone(point)
            self.assertFalse((Path(directory)/'action_checkpoint.npz').exists())
            self.assertEqual(c.io.read(receipt['action_checkpoint_none']['path'])['status'],'no_returned_point')

    def test_checkpoint_never_overwrites_existing_point(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(c,'ORIGINAL_PLAN',return_value=({},np.ones((2,2)))):
                c.checkpoint_plan(record_directory=directory)
                saved=c.io.ref(Path(directory)/'action_checkpoint.npz')
                with self.assertRaises(FileExistsError):c.checkpoint_plan(record_directory=directory)
            self.assertEqual(saved,c.io.ref(Path(directory)/'action_checkpoint.npz'))

    def test_initializer_keeps_original_callable_and_wraps_once(self):
        function=c.ORIGINAL_PLAN
        with patch.object(c.original.old,'initialize') as init,patch.object(c.original.solver,'plan'):
            c.initialize('somewhere');c.initialize('somewhere')
            self.assertEqual(init.call_count,2)
            self.assertIs(c.original.solver.plan,c.checkpoint_plan)
            self.assertIs(c.ORIGINAL_PLAN,function)

    def test_classification_only_absent_directories_are_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);e=[dict(id='a'),dict(id='b')]
            full=out/'fraction0.10/a/forecast00';full.mkdir(parents=True)
            for name in ('plans','results','seal'):c.io.write(full/(name+'.json'),{})
            c.io.write(full/'complete.json',dict(status='complete',plans=c.io.ref(full/'plans.json'),results=c.io.ref(full/'results.json')))
            (out/'fraction0.10/a/forecast01').mkdir()
            complete,partial,pending=c.classify(out,e,2)
            self.assertEqual(len(complete),1)
            self.assertEqual(partial,[dict(id='a',forecast_index=1)])
            self.assertEqual(pending,[dict(id='b',forecast_index=0),dict(id='b',forecast_index=1)])

    def test_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source.py';p.write_text('original')
            ref=c.io.ref(p);p.write_text('changed')
            with self.assertRaises(ValueError):
                c.verify_sources(dict(original_sources=[ref],source=ref,source_snapshot=ref,original_protocol=ref,
                    original_input=ref,planning_input_refs=[],completed_retained=[]))


if __name__=='__main__':unittest.main()
