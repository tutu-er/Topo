"""Controller-only tests: no history NPZ, actual future, AC or LP is loaded."""
from __future__ import annotations
import importlib.util
import pickle
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
    result={k:Frame(range(start,start+96)) for k in controller.OBS_FIELDS}
    result.update(name='fixture',units={k:'pu' for k in controller.OBS_FIELDS},
        drop_target=Frame(range(start,start+96)))
    result['units']['drop_target']='pu2'
    return result

class ControllerTests(unittest.TestCase):
    def test_real_pandas_early_contract_passes_real_normalize(self):
        # Real observation validation only: no build_bank, LP, QP, AC, producer
        # or actual future is called. All arrays below are fixed test fixtures.
        import numpy as np
        import pandas as pd
        controller.sys.path[:0]=[str(controller.REPO),str(controller.REPO/'rnj_wzzt_core')]
        from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize
        def full(day):
            idx=pd.date_range(pd.Timestamp('2000-01-01')+pd.Timedelta(days=day),periods=96,freq='15min')
            labels=[101,102]
            p=np.full((96,2),.02)+np.arange(96)[:,None]*1e-5
            q=.35*p; root=np.full(96,1.02)
            v=np.full((96,2),1.01)-np.arange(96)[:,None]*1e-6
            frames=[pd.DataFrame(a,index=idx,columns=labels) for a in (p,q,v)]
            return dict(name=f'fixture_{day}',P_terminal=frames[0],Q_terminal=frames[1],
                V_terminal=frames[2],V_root_observed=pd.Series(root,index=idx,name='V_root_measured'),
                drop_target=pd.DataFrame(root[:,None]**2-v**2,index=idx,columns=labels),
                units={'P_terminal':'pu','Q_terminal':'pu','V_terminal':'pu','V_root_observed':'pu','drop_target':'pu2'})
        history={'training':[full(i) for i in range(3)],'validation':[full(i) for i in range(3,6)]}
        packet={'training':[full(6),full(7)],'validation':[full(8),full(9)],
            'metadata':{'training_event_ids':[0,1],'validation_event_ids':[2,3]}}
        train,valid=controller.checkpoint_observations(history,packet,'early')
        obs,days,counts=normalize(train,'training')
        vobs,_,_=normalize(valid,'validation')
        self.assertEqual(len(obs[0]),372)
        self.assertEqual(len(vobs[0]),288)
        self.assertEqual(counts['2000-01-07 00:00:00'],84)
        self.assertEqual(obs[0].index[-1],pd.Timestamp('2000-01-07 20:45:00'))
        self.assertEqual(train[-1]['name'],packet['training'][0]['name'])
        self.assertEqual(train[-1]['units'],packet['training'][0]['units'])
        pd.testing.assert_frame_equal(train[-1]['drop_target'],packet['training'][0]['drop_target'].iloc[:84])
        # Parent writes a genuinely prefix-only packet. The worker receives no
        # event1 or future validation values even at the serialization boundary.
        packet['training'][1]['P_terminal'].iloc[:,:]=98765.
        packet['validation'][0]['P_terminal'].iloc[:,:]=54321.
        worker_packet=pickle.loads(pickle.dumps(controller.identifier_packet(packet,'early'),protocol=4))
        self.assertEqual(len(worker_packet['training']),1)
        self.assertEqual(worker_packet['validation'],[])
        self.assertEqual(worker_packet['metadata']['training_event_ids'],[0])
        self.assertEqual(worker_packet['metadata']['validation_event_ids'],[])
        self.assertTrue(worker_packet['metadata']['precut'])
        self.assertEqual(len(worker_packet['training'][0]['P_terminal']),84)
        for k in controller.OBS_FIELDS:
            self.assertEqual(len(worker_packet['training'][0][k]),84)
            self.assertLess(float(np.max(worker_packet['training'][0][k].to_numpy())),10)
        worker_train,worker_val=controller.checkpoint_observations(history,worker_packet,'early')
        worker_obs,_,_=normalize(worker_train,'training')
        pd.testing.assert_frame_equal(worker_obs[0],obs[0])
        self.assertEqual(len(normalize(worker_val,'validation')[0][0]),288)
        worker_packet['validation']=[packet['validation'][0]]
        with self.assertRaises(ValueError): controller.checkpoint_observations(history,worker_packet,'early')
        # Demonstrate that the former four-field-only truncation violates the
        # real interface; this regression is about the boundary, not fitting.
        old={k:packet['training'][0][k].iloc[:84] for k in controller.OBS_FIELDS}
        with self.assertRaises(ValueError): normalize(history['training']+[old],'training')

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
            fit_path='fit_'+point,evaluation=evaluation,all_local_scans_complete=True,actual_payment=0,
            operation_checks_pass=True)
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
            self.assertIn(str((controller.BASE/'end_to_end/physical_case.py').resolve()),cfg['source_sha256'])
            self.assertEqual(cfg['price_planning_endpoint'].split(';')[0],'final')
            self.assertEqual(controller.verify(actual)['status'],'frozen_before_actual_future')
            Path(locks['local']['path']).write_text('{}',encoding='utf-8')
            with self.assertRaises(RuntimeError): controller.verify(actual)

    def test_operation_failure_keeps_structure_but_excludes_feasible_denominator(self):
        checks={k:True for k in controller.OPERATION_CHECKS}
        checks['capacity_pass']=False
        receipt={'event_records':[dict(event=0,status='complete',checks=checks)],'all_operation_checks_pass':False}
        operation=controller.operation_account(receipt,'early')
        self.assertFalse(operation['operation_checks_pass'])
        self.assertIn({'event':0,'check':'capacity_pass'},operation['operation_failed_checks'])
        evaluation={name:dict(full_recovery=True,full_block_errors=0) for name in ('rnj_bank','multistart_original_rule')}
        row=dict(policy='same',checkpoint='early',status='complete',observed_fingerprint='x',fit_path='fit',
            evaluation=evaluation,all_local_scans_complete=True,actual_payment=.2,**operation)
        result=controller.summarize([row])
        group=next(g for g in result['groups'] if g['checkpoint']=='early' and g['policy']=='same')
        self.assertEqual(group['all_structural_denominator'],1)
        self.assertEqual(group['structure_denominator'],0)
        self.assertEqual(group['operation_failed_complete'],1)
        self.assertEqual(group['wzzt_full'],0)
        self.assertEqual(group['wzzt_full_all_structural'],1)
        self.assertEqual(group['known_paid_sum'],.2)
        # A later failure does not retroactively invalidate event0 operation.
        receipt['event_records'][0]['checks']['capacity_pass']=True
        receipt['event_records'].append(dict(event=1,status='physical_failed',checks={}))
        self.assertTrue(controller.operation_account(receipt,'early')['operation_checks_pass'])
        self.assertFalse(controller.operation_account(receipt,'final')['operation_checks_pass'])

    def test_controls_only_exact_seven_declared_denominator_and_unresolved_method(self):
        fake=types.ModuleType('research_experiments.wzzt_dr_selection.dr_pool_v2.adapter')
        fake.source_hashes=lambda:{}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);summary=base/'case_plans.json';actual=base/'actual'
            worker=base/'worker.py';physical=base/'physical.py'
            worker.write_text('# never executed\n',encoding='utf-8');physical.write_text('# never executed\n',encoding='utf-8')
            row={'job_id':'pilot_b8','bundle':8,'voltage_sigma':.001}
            for role in ('train','validation'):
                p=base/(role+'.fixture');p.write_bytes(b'past-only')
                row[role]={'path':str(p),'catalog_sha256':controller.sha(p)}
            locks={}
            for law in ('local','common'):
                p=base/(law+'.json');controller.write(p,{'status':'frozen','plans':{name:[[0],[0]] for name in controller.CONTROLS}})
                locks[law]={'path':str(p),'sha256':controller.sha(p)}
            payload={'status':'complete','controls_only':True,
                'cases':[{'job_id':'pilot_b8','voltage_sigma':.001,'history_row':row,'locks':locks}]}
            controller.write(summary,payload)
            with self.assertRaises(ValueError): controller.prepare(actual,summary,worker,physical)
            self.assertFalse(actual.exists())
            unresolved={}
            for role,status in (('protocol','frozen'),('result','unresolved')):
                p=base/('original_'+role+'.json');controller.write(p,{'status':status,'original_wall_limit_seconds':900})
                unresolved[role]={'path':str(p),'sha256':controller.sha(p)}
            payload['planning_unresolved']=unresolved;controller.write(summary,payload)
            with patch.dict(controller.sys.modules,{fake.__name__:fake}):
                cfg=controller.prepare(actual,summary,worker,physical)
            self.assertTrue(cfg['controls_only'])
            self.assertEqual(cfg['planned_records'],28)
            self.assertEqual(len(cfg['policies']),7)
            self.assertEqual({r['status'] for r in cfg['missing_method_records']},{'price_unresolved'})
            self.assertEqual(len(cfg['missing_method_records']),4)
            self.assertTrue(all(r['policy']=='wzzt_best' for r in cfg['missing_method_records']))
            for ref in unresolved.values(): self.assertIn(ref['path'],cfg['input_sha256'])
            empty=controller.summarize([],cfg)
            self.assertEqual(empty['planned'],28)
            self.assertEqual(empty['records'],0)
            self.assertEqual(empty['price_unresolved_record_count'],4)
            self.assertEqual(len(empty['groups']),14)
            self.assertEqual({g['policy'] for g in empty['groups']},set(controller.CONTROLS))
            self.assertTrue(all(g['planned']==2 and g['complete']==0 for g in empty['groups']))

if __name__=='__main__': unittest.main()
