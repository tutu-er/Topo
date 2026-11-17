"""Denominator, parent barrier and real frozen subprocess verification."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ... import io
from ...identification.algorithm import SearchConfig
from . import run


def entries():
    rows=[]
    for case in ('paper15','flynn16','soumalas11','pengwah18'):
        for n in (48,96,192,384,768):
            for sd in (0.,.005,.01,.02):
                for repeat in (0,1):
                    rows.append(dict(id=f'{case}_{n}_{sd}_{repeat}',case=case,N=n,pq_sd=sd,repeat=repeat,
                        long_selected=n in (96,768) and sd in (.005,.02) and repeat==0))
    return rows


class DriverTests(unittest.TestCase):
    def test_complete_plan_has_matched_caps_and_rotated_order(self):
        tasks=run.task_specs(entries())
        self.assertEqual(Counter(t['panel'] for t in tasks),dict(long=48,grid=480))
        self.assertEqual(Counter(t['variant'] for t in tasks),dict(full=176,pool_only=176,skip=176))
        groups={}
        for task in tasks:groups.setdefault((task['panel'],task['input_id']),[]).append(task)
        for key,triple in groups.items():
            self.assertEqual({t['variant'] for t in triple},set(run.VARIANTS))
            configs=[{k:v for k,v in t['config'].items() if k!='w_mode'} for t in triple]
            self.assertEqual(configs[0],configs[1]);self.assertEqual(configs[1],configs[2])
            self.assertEqual(configs[0]['e_seconds'],6900. if key[0]=='long' else 270.)
        first=[t['variant'] for t in tasks if t['panel']=='long']
        self.assertEqual(first[:9],['full','pool_only','skip','pool_only','skip','full','skip','full','pool_only'])

    def test_parent_requires_terminal_receipt_and_no_lock(self):
        with TemporaryDirectory() as temp:
            parent=Path(temp)
            self.assertFalse(run.parent_ready(parent)[0])
            io.write(parent/'execution.json',{'status':'all_terminal'})
            self.assertTrue(run.parent_ready(parent)[0])
            (parent/'running.lock').write_text('123')
            self.assertEqual(run.parent_ready(parent),(False,'parent_controller_active'))
            (parent/'running.lock').unlink()
            io.write(parent/'execution.json',{'status':'stopped_with_pending'})
            self.assertFalse(run.parent_ready(parent)[0])

    def test_real_private_worker_processes_all_three_modes_without_future_data(self):
        with TemporaryDirectory() as temp:
            out=Path(temp);package,sources=run.snapshot(out)
            driver=package/'experiments'/run.STUDY.name/'run.py'
            rng=np.random.default_rng(172023);p=rng.uniform(.05,.2,(24,3));q=rng.uniform(.01,.08,(24,3))
            y=p@np.array([[.3,.1,0.],[.1,.4,0.],[0.,0.,.2]])+q*.1+rng.normal(0.,.001,p.shape)
            for name in ('train','validation'):
                np.savez(out/(name+'.npz'),P=p,Q=q,Y=y,labels=[101,102,103])
            entry=dict(id='fixture',labels=[101,102,103],files={k+'.npz':io.ref(out/(k+'.npz')) for k in ('train','validation')})
            tasks=[]
            for mode in run.VARIANTS:
                cfg=SearchConfig(w_mode=mode,e_seconds=4.,total_seconds=15.,per_mip_seconds=2.,
                    output_reserve_seconds=2.,internal_floor=0.)
                tasks.append(dict(task_id=mode,input_id='fixture',variant=mode,panel='long',method='101',loss='mse',
                    config=asdict(cfg),result_dir='tasks/'+mode))
            io.write(out/'inputs.json',[entry]);io.write(out/'tasks.json',tasks)
            io.write(out/'protocol.json',dict(source_root=str(package),private_driver=str(driver),sources=sources,
                inputs_ref=io.ref(out/'inputs.json'),tasks_ref=io.ref(out/'tasks.json')))
            for mode in run.VARIANTS:
                process=subprocess.run([sys.executable,'-B',str(driver),'worker','--out',str(out),'--task-id',mode],
                    cwd=temp,text=True,capture_output=True,timeout=35)
                self.assertEqual(process.returncode,0,process.stdout+process.stderr)
                result=io.read(out/'tasks'/mode/'result.json')
                self.assertEqual(result['status'],'complete',result.get('traceback'))
                self.assertFalse(result['truth_used']);self.assertFalse(result['new_day_loaded'])
                native=io.read(out/'tasks'/mode/'native.json')
                self.assertEqual(native['config']['w_mode'],mode)
                if mode=='skip':self.assertFalse(any(e['stage']=='W' for e in native['events']))
                if mode=='pool_only':self.assertFalse(any(e['stage']=='W' and e['solver'].get('solve_started') for e in native['events']))
                self.assertTrue((out/'tasks'/mode/'stages').exists())


if __name__=='__main__':unittest.main()
