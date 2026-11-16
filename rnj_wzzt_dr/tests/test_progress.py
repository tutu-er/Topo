"""Operational progress must retain scalar bounds and accepted-path checkpoints."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .. import cli, io
from ..progress import Progress


class ProgressTests(unittest.TestCase):
    def test_transient_windows_sharing_error_is_retried_without_duplicate_event(self):
        with TemporaryDirectory() as temp:
            original=io.write;calls=[]
            def sharing_error_once(path,value):
                calls.append(path)
                if len(calls)==1:raise PermissionError('Windows reader still holds file')
                return original(path,value)
            with patch.object(io,'write',side_effect=sharing_error_once), patch('rnj_wzzt_dr.progress.sleep') as pause:
                Progress(temp,method='101',loss='mse',mode='off')(dict(stage='initialized'))
            self.assertEqual(len(calls),2);pause.assert_called_once_with(.005)
            self.assertEqual(io.read(Path(temp)/'progress.json')['stage'],'initialized')
            self.assertEqual(len((Path(temp)/'events.jsonl').read_text().splitlines()),1)

    def test_json_trace_and_checkpoint_do_not_send_coefficients_to_stdout(self):
        with TemporaryDirectory() as temp:
            stream=StringIO();p=Progress(temp,method='101',loss='mae',mode='json',stream=stream)
            p(dict(stage='W_complete',point={'r':[42.]},path=[{'r':[42.]}]))
            p(dict(stage='native_progress',model_name='E',objective=float('inf'),bound=.5,nodes=3))
            current=io.read(Path(temp)/'progress.json');checkpoint=io.read(Path(temp)/'checkpoint.json')
            self.assertIsNone(current['objective']);self.assertEqual(current['bound'],.5)
            self.assertEqual(checkpoint['point']['r'],[42.]);self.assertFalse(checkpoint['budget_reset_allowed'])
            self.assertNotIn('42.',stream.getvalue());self.assertEqual(len((Path(temp)/'events.jsonl').read_text().splitlines()),2)

    def test_loss_and_long_preset_forward_without_changing_final_metric(self):
        with TemporaryDirectory() as temp, patch.object(cli,'load',return_value=object()), \
             patch.object(cli.io,'ref',return_value={}), patch.object(cli.io,'environment',return_value={}), \
             patch.object(cli.io,'source_receipt',return_value={}), \
             patch.object(cli,'fit_observed',return_value=dict(selected={'index':0},total_cap_exceeded=False)) as fit, \
             redirect_stdout(StringIO()):
            cli.main(['fit','--train','train.npz','--validation','val.npz','--method','101',
                '--loss','mae','--preset','long','--progress','off','--output',str(Path(temp)/'fit')])
            cfg=fit.call_args.kwargs['config']
            self.assertEqual((cfg.loss,cfg.total_seconds,cfg.per_mip_seconds,cfg.output_reserve_seconds),
                             ('mae',14400.,600.,600.))
            self.assertIsInstance(fit.call_args.kwargs['heartbeat'],Progress)

    def test_invalid_progress_interval_does_not_create_output(self):
        with TemporaryDirectory() as temp:
            out=Path(temp)/'fit'
            with self.assertRaises(ValueError):
                cli.main(['fit','--train','train.npz','--validation','val.npz',
                          '--progress-every','0','--output',str(out)])
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
