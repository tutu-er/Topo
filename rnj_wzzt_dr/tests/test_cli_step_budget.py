"""Public CLI scheduling contract; no optimizer or real observations needed."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from .. import cli
from ..identification.algorithm import SearchConfig


class CLIStepBudgetTests(unittest.TestCase):
    def invoke(self, extra):
        train, validation = object(), object()
        fitted = dict(total_cap_exceeded=False, selected=dict(index=0))
        with TemporaryDirectory() as directory:
            output = Path(directory)/'new_output'
            args = ['fit', '--train', 'training.npz', '--validation', 'validation.npz',
                    '--output', str(output), '--seconds', '120', *extra]
            with patch.object(cli, 'fit_observed', return_value=fitted) as fit, \
                 patch.object(cli, 'load', side_effect=[train, validation]), \
                 patch.multiple(cli.io, ref=Mock(return_value={'path':'fixture','sha256':'fixture'}),
                                source_receipt=Mock(return_value={}), environment=Mock(return_value={}),
                                write=Mock()):
                # Explicit patch objects are accessible through their modules;
                # no filesystem output or observation loading is performed.
                result = cli.main(args)
                call = fit.call_args
                writes = list(cli.io.write.call_args_list)
        self.assertEqual(result, 0)
        self.assertEqual(call.args, (train, validation))
        request = next(c.args[1] for c in writes if c.args[0].name == 'request.json')
        self.assertIs(request['config'], call.kwargs['config'])
        self.assertTrue(any(c.args[0].name == 'complete.json' for c in writes))
        return call.kwargs, request

    def test_explicit_step_cap_reaches_fit_and_sealed_request(self):
        forwarded, request = self.invoke(['--method','100','--step-seconds','10'])
        self.assertEqual(forwarded['method'], '100')
        self.assertEqual(forwarded['config'].per_mip_seconds, 10.)
        self.assertEqual(forwarded['config'].total_seconds, 120.)
        self.assertEqual(forwarded['config'].output_reserve_seconds, 12.)
        self.assertEqual(request['method'], '100')

    def test_omitting_step_cap_preserves_algorithm_default(self):
        forwarded, _ = self.invoke([])
        self.assertEqual(forwarded['config'].per_mip_seconds, SearchConfig().per_mip_seconds)
        self.assertEqual(forwarded['method'], '111')
        self.assertEqual(forwarded['config'].internal_floor, SearchConfig().internal_floor)

    def test_nonpositive_or_nonfinite_step_cap_fails_before_loading_or_writing(self):
        for value in ('0','-10','nan','inf'):
            with self.subTest(value=value), TemporaryDirectory() as directory, \
                 patch.object(cli, 'fit_observed') as fit, patch.object(cli, 'load') as load, \
                 patch.object(cli.io, 'write') as write:
                with self.assertRaisesRegex(ValueError, 'Bounds and solver time allowances must be positive'):
                    cli.main(['fit','--train','training.npz','--validation','validation.npz',
                              '--output',str(Path(directory)/'new_output'),'--step-seconds',value])
                fit.assert_not_called(); load.assert_not_called(); write.assert_not_called()

    def test_help_identifies_W_cap_and_distinguishes_T_E_budget(self):
        text = StringIO()
        with redirect_stdout(text), self.assertRaises(SystemExit) as exit_status:
            cli.main(['fit','--help'])
        self.assertEqual(exit_status.exception.code, 0)
        help_text = ' '.join(text.getvalue().split())
        self.assertIn('--step-seconds', help_text)
        self.assertIn('W: cap each unknown-connection MIQP', help_text)
        self.assertIn('T/E use remaining stage time', help_text)


if __name__ == '__main__':
    unittest.main()
