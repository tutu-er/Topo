"""Controller-only tests: no history NPZ, actual future, AC or LP is loaded."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('coded_controller', Path(__file__).with_name('controller.py'))
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)

class Frame:
    def __init__(self, values): self.values=list(values)
    def __len__(self): return len(self.values)
    @property
    def iloc(self): return self
    def __getitem__(self, key): return Frame(self.values[key])
    def copy(self): return Frame(self.values)

def scene(start=0):
    return {k:Frame(range(start,start+96)) for k in controller.OBS_FIELDS}

class ControllerTests(unittest.TestCase):
    def test_checkpoint_retains_history_and_excludes_future(self):
        history={'training':[scene(i*96) for i in range(3)],'validation':[scene(1000+i*96) for i in range(3)]}
        packet={'training':[scene(2000),scene(3000)],'validation':[scene(4000),scene(5000)]}
        packet['metadata']={'training_event_ids':[0,1],'validation_event_ids':[2,3]}
        train,val=controller.checkpoint_observations(history,packet,'early')
        self.assertEqual(sum(map(lambda s:len(s['P_terminal']),train)),372)
        self.assertEqual(sum(map(lambda s:len(s['P_terminal']),val)),288)
        self.assertIs(train[0],history['training'][0])
        self.assertIs(val[-1],history['validation'][-1])
        self.assertEqual(train[-1]['P_terminal'].values,list(range(2000,2084)))
        packet['training'][1]=scene(-9999)
        train2,val2=controller.checkpoint_observations(history,packet,'early')
        self.assertEqual(train2[-1]['P_terminal'].values,train[-1]['P_terminal'].values)
        final_train,final_val=controller.checkpoint_observations(history,packet,'final')
        self.assertEqual(sum(len(s['P_terminal']) for s in final_train),480)
        self.assertEqual(sum(len(s['P_terminal']) for s in final_val),480)
        packet['metadata']['training_event_ids']=[1]
        packet['training']=packet['training'][1:]
        with self.assertRaises(ValueError): controller.checkpoint_observations(history,packet,'early')

    def test_checkpoint_costs_are_not_added_or_future_known(self):
        rows=[[1,0],[1,0]]
        receipt={'event_records':[dict(event=e,status='complete',actual_payment=.2,
            committed_payment=.2,checks={'energy_return_pass':True}) for e in range(4)]}
        early=controller.payment_account(receipt,'early',rows)
        final=controller.payment_account(receipt,'final',rows)
        self.assertEqual(early['paid_through_checkpoint'],.2)
        self.assertAlmostEqual(early['unstarted_event_worst_payment'],.6)
        self.assertAlmostEqual(early['future_realized_payment_evaluator_only'],.6)
        self.assertAlmostEqual(final['full_contract_realized_payment'],.8)
        self.assertEqual(early['pending_accepted_rebound_kwh'],0)
        self.assertEqual(early['accepted_committed_payment_through_checkpoint'],.2)
        self.assertFalse(early['future_event_cancellation'])
        with self.assertRaises(ValueError): controller.payment_account(receipt,'early',[[2,2],[2,2]])

    def test_later_physical_failure_keeps_early_payment_and_unknowns(self):
        receipt={'event_records':[dict(event=0,status='complete',actual_payment=.2,checks={'energy_return_pass':True})]}
        early=controller.payment_account(receipt,'early',[[1],[1]])
        final=controller.payment_account(receipt,'final',[[1],[1]])
        self.assertEqual(early['paid_through_checkpoint'],.2)
        self.assertEqual(early['pending_accepted_rebound_kwh'],0)
        self.assertIsNone(early['full_contract_realized_payment'])
        self.assertIsNone(final['paid_through_checkpoint'])
        self.assertIsNone(final['pending_accepted_rebound_kwh'])

    def test_summary_keeps_failures_and_unique_fit_denominator(self):
        evaluation={name:dict(full_recovery=True,full_block_errors=0) for name in ('rnj_bank','multistart_original_rule')}
        records=[dict(policy='zero',checkpoint=point,status='complete',observed_fingerprint=point,
            fit_path='fit_'+point,evaluation=evaluation,all_local_scans_complete=True,actual_payment=0)
            for point in ('early','final')]
        records.append(dict(policy='same',checkpoint='early',status='timeout',actual_payment=.2))
        result=controller.summarize(records)
        self.assertEqual(result['records'],3)
        self.assertEqual(result['statuses']['timeout'],1)
        self.assertEqual(result['unique_complete_fits'],2)
        self.assertEqual(result['status'],'partial')
        failed=next(g for g in result['groups'] if g['policy']=='same' and g['checkpoint']=='early')
        self.assertEqual(failed['structure_denominator'],0)
        self.assertEqual(failed['known_paid_sum'],.2)

    def test_prepare_requires_finished_prices_and_detects_lock_drift(self):
        fake=types.ModuleType('research_experiments.wzzt_dr_selection.dr_pool_v2.adapter')
        fake.source_hashes=lambda:{}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); summary=base/'case_plans.json'; actual=base/'actual'
            controller.write(summary,{'status':'unresolved'})
            with self.assertRaises(ValueError):
                controller.prepare(actual,summary,base/'worker.py',base/'physical.py')
            self.assertFalse(actual.exists())
            worker=base/'worker.py'; physical=base/'physical.py'
            worker.write_text('# fake worker; never executed\n',encoding='utf-8')
            physical.write_text('# fake physical; never executed\n',encoding='utf-8')
            row={'job_id':'pilot_b8','bundle':8,'voltage_sigma':.001}
            for role in ('train','validation'):
                p=base/(role+'.fixture');p.write_bytes(b'past-only')
                row[role]={'path':str(p),'catalog_sha256':controller.sha(p)}
            locks={}
            for law in ('local','common'):
                p=base/(law+'.json')
                controller.write(p,{'status':'frozen','plans':{f'policy{i}':[[0],[0]] for i in range(8)}})
                locks[law]={'path':str(p),'sha256':controller.sha(p)}
            controller.write(summary,{'status':'complete','cases':[{'job_id':'pilot_b8','voltage_sigma':.001,'history_row':row,'locks':locks}]})
            with patch.dict(controller.sys.modules,{fake.__name__:fake}):
                cfg=controller.prepare(actual,summary,worker,physical)
            self.assertEqual(cfg['planned_records'],32)
            self.assertFalse(cfg['actual_future_loaded_at_prepare'])
            self.assertEqual(cfg['early']['validation_rows'],288)
            self.assertEqual(cfg['price_planning_endpoint'].split(';')[0],'final')
            self.assertEqual(controller.verify(actual)['status'],'frozen_before_actual_future')
            Path(locks['local']['path']).write_text('{}',encoding='utf-8')
            with self.assertRaises(RuntimeError): controller.verify(actual)

if __name__=='__main__': unittest.main()
