"""Pairing, fresh-score boundaries and fail-closed quotation checkpoints."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ... import io
from . import run


class TinyCase:
    R=4
    resources=[dict(physical_device_id='pv_4'),dict(physical_device_id='battery_4'),
               dict(physical_device_id='pv_4'),dict(physical_device_id='battery_4')]


def entry(level=.005,repeat=0,case='pengwah18'):
    return dict(id=f'{case}_{repeat}_{level}',case=case,index=18,original_input_index=18,
        history_repeat=repeat,noise_repeat=repeat,pq_relative_sd=level,
        meter=dict(additional_PQ_relative_sd=level,additional_V_relative_sd=.0005),
        training_rows=192,validation_rows=96,new_rows=96)


class RunTest(unittest.TestCase):
    def test_noise_levels_share_eta_but_repeats_do_not(self):
        first,_=run.device_multipliers(TinyCase(),entry(.0),3)
        other,_=run.device_multipliers(TinyCase(),entry(.02),3)
        np.testing.assert_array_equal(first,other)
        self.assertEqual(first[0],first[2]);self.assertEqual(first[1],first[3])
        second,_=run.device_multipliers(TinyCase(),entry(.0,1),3)
        self.assertFalse(np.array_equal(first,second))
        reordered=TinyCase();reordered.resources=list(reversed(TinyCase.resources))
        permuted,_=run.device_multipliers(reordered,entry(),3)
        np.testing.assert_array_equal(permuted,first[::-1])

    def test_evaluation_path0_and_other_paths_remain_distinct(self):
        values={name:np.arange(np.prod(shape)).reshape(shape).astype(float)
                for name,shape in [('P',(16,96,2)),('Q',(16,96,2)),('V',(16,96,2)),('root',(16,96))]}
        first=run.evaluation_noises(values,0,2);second=run.evaluation_noises(values,1,2)
        np.testing.assert_array_equal(first[0],values['P'][0])
        self.assertFalse(np.array_equal(first[0],second[0]))
        bad=deepcopy(values);bad['root']=bad['root'][:1]
        with self.assertRaises(ValueError): run.evaluation_noises(bad,0,2)

    def test_fixed_smoke_never_chooses_by_candidate_results(self):
        entries=[entry(q,h,c) for c in ('pengwah18','soumalas11') for h in range(4) for q in run.PQ_LEVELS]
        chosen=run.selected_entries(entries,True)
        self.assertEqual(len(chosen),2)
        self.assertTrue(all(run.history_repeat(e)==0 and run.pq_sd(e)==.005 for e in chosen))
        with self.assertRaises(ValueError): run.selected_entries([e for e in entries if e['case']!='soumalas11'],True)
        bad=entry();bad['validation_rows']=48
        with self.assertRaises(ValueError): run.selected_entries([bad])
        bad=entry();bad['meter']['additional_V_relative_sd']=.001
        with self.assertRaises(ValueError): run.selected_entries([bad])

    def test_completed_quote_requires_exact_saved_action_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=dict(path='protocol',sha256='abc');name='single_normal_fixed'
            io.write(d/(name+'_plan.json'),dict(feasible=True,cost_yuan=0.))
            np.savez_compressed(d/(name+'_action.npz'),action=np.zeros((24,2)))
            io.write(d/(name+'_completed.json'),dict(protocol=p,plan=io.ref(d/(name+'_plan.json')),
                action=io.ref(d/(name+'_action.npz'))))
            q,u=run.load_quote_checkpoint(d,name,p)
            self.assertTrue(q['feasible']);self.assertEqual(u.shape,(24,2))
            with self.assertRaises(ValueError):run.load_quote_checkpoint(d,name,dict(p,sha256='changed'))
            np.savez_compressed(d/(name+'_action.npz'),action=np.ones((24,2)))
            with self.assertRaises(ValueError):run.load_quote_checkpoint(d,name,p)

    def test_interrupted_search_never_resets_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);name='single_mgf_fixed'
            io.write(d/(name+'_started.json'),dict(allowed_seconds=1200))
            with self.assertRaisesRegex(RuntimeError,'remaining budget'):
                run.load_quote_checkpoint(d,name,dict(path='p',sha256='x'))

    def test_no_actual_data_before_planning_seal(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run.old,'context') as actual:
            with self.assertRaises(FileNotFoundError):run.execute({}, {},Path(tmp),{},None,None,None,{}, {})
            actual.assert_not_called()

    def test_formal_cannot_bypass_independent_preflight(self):
        with self.assertRaisesRegex(ValueError,'smoke audit'):run.validate_preflight(None,Path('index.json'))

    def test_missing_fit_keeps_the_forty_cell_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);io.write(d/'protocol.json',dict(seed=run.SEED))
            tasks=[entry(q,h,c) for c in ('pengwah18','soumalas11') for h in range(4) for q in run.PQ_LEVELS]
            records=[dict(task=t,status='failed' if i==0 else 'complete') for i,t in enumerate(tasks)]
            manifest=dict(inputs=tasks[1:],attempts=records,planned_inputs=40,source_unchanged=True,
                protocol=io.ref(d/'protocol.json'))
            io.write(d/'index.json',manifest)
            _,successful=run.load_index(d/'index.json')
            self.assertEqual(len(successful),39)
            manifest['attempts']=records[1:];io.write(d/'index.json',manifest)
            with self.assertRaisesRegex(ValueError,'forty'):
                run.load_index(d/'index.json')


if __name__=='__main__':unittest.main()
