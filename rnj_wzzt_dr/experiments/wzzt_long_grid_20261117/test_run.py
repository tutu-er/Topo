"""Plan denominator, immutable reuse and controller failure bookkeeping."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ... import io
from . import data, run


class DriverTests(unittest.TestCase):
    def test_locked_grid_tasks_budgets_and_loss_pairs(self):
        entries=data.coordinates()+data.root_coordinates()+data.sampling_coordinates()
        tasks=run.task_specs(entries)
        self.assertEqual(Counter(t['panel'] for t in tasks),dict(long=96,grid=960,ablation=24,root_noise=96,sampling_control=96))
        self.assertEqual(sum(not t['method'].startswith('RNJ') for t in tasks),440)
        cells={}
        for task in tasks:
            key=(task['panel'],task['input_id'],task['method']);cells.setdefault(key,[]).append(task)
            if task['panel']=='long' and task['method']=='101':
                self.assertEqual((task['config']['total_seconds'],task['config']['per_mip_seconds']), (14400.,600.))
        self.assertEqual(len(cells),636)
        for panel in ('long','grid','root_noise','sampling_control'):
            for entry in entries:
                paired=[t for t in tasks if t['panel']==panel and t['input_id']==entry['id'] and t['loss']=='mae']
                if not paired:continue
                self.assertEqual(len(paired),3)
                self.assertEqual(len({(t['config']['total_seconds'],t['config']['output_reserve_seconds']) for t in paired}),1)
        for pair in cells.values():
            self.assertEqual({t['loss'] for t in pair},{'mse','mae'})
            a,b=(dict(t['config']) for t in pair);a.pop('loss');b.pop('loss');self.assertEqual(a,b)

    def test_changed_native_result_is_not_reused(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);io.write(root/'protocol.json',{})
            task=dict(task_id='x',result_dir='tasks/x')
            io.write(root/'tasks/x/native.json',{})
            io.write(root/'tasks/x/result.json',dict(task_id='x',status='complete',source_protocol=io.ref(root/'protocol.json'),
                native=io.ref(root/'tasks/x/native.json'),input_refs={}))
            self.assertEqual(run.reuse_receipt(root,task)['status'],'complete')
            io.write(root/'tasks/x/native.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'Changed sealed file'):run.reuse_receipt(root,task)


if __name__=='__main__':unittest.main()
