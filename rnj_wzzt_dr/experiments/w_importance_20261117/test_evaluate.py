"""Pure arithmetic and file-receipt checks; never fit or solve a model."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from rnj_wzzt_dr.experiments.w_importance_20261117 import evaluate as e


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return e.base.file_ref(path)


def simple_topology(exact=True,f1=1.):
    return dict(internal=dict(exact=exact,f1=f1,fp=0 if exact else 1,fn=0 if exact else 1),
                full=dict(exact=exact))


def paired_row(ident,*,status='complete',correct=True,seconds=2.,f1=1.):
    return dict(input_id=ident,status=status,seconds=seconds,
        bank=dict(reference_mse=1.,selections={s:dict(mse=dict(topology=simple_topology(correct,f1))) for s in e.SPLITS}))


def fixture(temp):
    root=Path(temp);n=3;labels=[301,302,303]
    branches=[dict(from_bus=0,to_bus=301,r_ohm=.1,x_ohm=.02),
              dict(from_bus=0,to_bus=302,r_ohm=.2,x_ohm=.04),
              dict(from_bus=0,to_bus=303,r_ohm=.3,x_ohm=.06)]
    import csv
    with (root/'branches.csv').open('w',encoding='utf8',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(branches[0]));writer.writeheader();writer.writerows(branches)
    metadata=write(root/'metadata.json',dict(root_bus=0,base_kv=1.,base_mva=1.))
    evaluation=write(root/'evaluation.json',dict(evaluation_case_refs={
        'metadata.json':metadata,'branches.csv':e.base.file_ref(root/'branches.csv')}))
    point=dict(supports=[[0],[1],[2]],r=[.2,.4,.6],x=[.04,.08,.12])
    R,X=e.base.matrices(point,n)
    files={};arrays={}
    for name,count in (('train',4),('validation',3),('new_day',2)):
        p=(np.arange(count*n).reshape(count,n)+1)/100.;q=np.flip(p,axis=1)*.13
        value=dict(P=p,Q=q,Y=p@R.T+q@X.T+.001,labels=np.asarray(labels))
        np.savez(root/(name+'.npz'),**value);arrays[name]=value
        files[name+'.npz']=e.base.file_ref(root/(name+'.npz'))
    entry=dict(id='x',case='toy',N=4,pq_sd=.005,repeat=0,status='complete',labels=labels,
               files=files,evaluation_ref=evaluation,zbase_ohm=1.)
    cfg=dict(w_mode='full',loss='mse',alpha=.003,zero_tol=1e-9,validation_tie=1e-12,
             gain_tol=1e-10,total_seconds=600.,e_seconds=270.)
    native=dict(method='101',config=cfg,reference_mse=1.,reference_loss=1.,events=[],MIQP=[],
                path=[point],uncontracted_output_path=[point],frozen_candidates=[dict(index=0,point=point)],
                selected=dict(index=0),truth_used=False,test_used=False,total_seconds=2.)
    return root,entry,cfg,native,arrays


def setup_run(temp,*,variants=('full','pool_only','skip'),completed=('full',),failed=()):
    root,entry,cfg,native,arrays=fixture(temp)
    inputs=write(root/'inputs.json',[entry]);tasks=[]
    for variant in variants:
        config={**cfg,'w_mode':variant};ident='grid__x__'+variant
        tasks.append(dict(task_id=ident,input_id='x',panel='grid',variant=variant,method='101',
                          loss='mse',result_dir='tasks/'+ident,config=config))
    task_ref=write(root/'tasks.json',tasks)
    protocol=write(root/'protocol.json',dict(inputs_ref=inputs,tasks_ref=task_ref,sources=[]))
    for task in tasks:
        folder=root/task['result_dir'];variant=task['variant']
        if variant in completed:
            model=deepcopy(native);model['config']['w_mode']=variant
            ref=write(folder/'native.json',model)
            write(folder/'result.json',dict(task_id=task['task_id'],source_protocol=protocol,status='complete',
                variant=variant,truth_used=False,new_day_loaded=False,native=ref,input_refs={k:entry['files'][k] for k in ('train.npz','validation.npz')},
                input_fingerprints={k:e.base.observation_fingerprint(arrays[k]) for k in ('train','validation')},
                reference_mse=1.,training_reference_loss=1.,seconds=2.))
        elif variant in failed:
            write(folder/'result.json',dict(task_id=task['task_id'],source_protocol=protocol,status='failed',error='numerical check failed',seconds=1.))
            (folder/'events.jsonl').write_text(json.dumps(dict(stage='native_progress',callback_event='FINAL',status=9,objective=1.,bound=.5,gap=.5))+'\n',encoding='utf8')
    return root,tasks


class EvaluatorTests(unittest.TestCase):
    def test_paired_losses_and_failures_are_both_retained(self):
        left=[paired_row('a'),paired_row('b'),paired_row('c')]
        right=[paired_row('a',correct=False,f1=.5,seconds=1.),paired_row('b',status='failed'),
               paired_row('c',seconds=1.)]
        result=e.paired(left,right,'validation')
        self.assertEqual((result['planned_inputs'],result['completed_pairs'],result['unpaired_or_incomplete']),(3,2,1))
        self.assertEqual(result['full']['loss_ids'],['a']);self.assertEqual(result['right_status_counts']['failed'],1)
        self.assertEqual(result['internal_F1_delta_mean'],-.25)
        self.assertEqual(result['right_over_left_time_ratio_median'],.5)

    def test_pool_only_native_solve_is_audit_failure(self):
        native=dict(method='101',config=dict(w_mode='pool_only',loss='mse'),events=[dict(stage='W',solver={'solve_started':True})])
        with self.assertRaisesRegex(ValueError,'performed a native W solve'):
            e.validate_treatment(dict(variant='pool_only',method='101',loss='mse'),native)

    def test_paired_calibration_and_RNJ_pool_are_checked(self):
        x=paired_row('a');y=paired_row('a');y['bank']['reference_mse']=2.
        with self.assertRaisesRegex(ValueError,'different MSE calibration'):e.paired([x],[y],'validation')
        y['bank']['reference_mse']=1.
        x['attribution']=dict(pool=dict(available=True,rnj_pool_supports=[[0]]))
        y['attribution']=dict(pool=dict(available=True,rnj_pool_supports=[[1]]))
        with self.assertRaisesRegex(ValueError,'different RNJ support pools'):e.paired([x],[y],'validation')

    def test_pairs_separate_closed_E_from_time_limits_and_unstarted(self):
        left=[paired_row(c) for c in ('a','b','c')];right=deepcopy(left)
        for row in left:row['solver']=dict(events=[dict(stage='E',solve_started=True,status=2,native_subproblem_certificate=True)])
        right[0]['solver']=deepcopy(left[0]['solver'])
        right[1]['solver']=dict(events=[dict(stage='E',solve_started=True,status=9,timed_out=True,native_subproblem_certificate=False)])
        right[2]['solver']=dict(events=[])
        result=e.paired(left,right,'validation')['E_quality_strata']
        self.assertEqual(result['both_closed']['input_ids'],['a'])
        self.assertEqual(result['at_least_one_time_limit']['input_ids'],['b'])
        self.assertEqual(result['other_unclosed_or_not_started']['input_ids'],['c'])

    def test_skip_rejects_any_W_event(self):
        native=dict(method='101',config=dict(w_mode='skip',loss='mse'),events=[dict(stage='W',solver={'solve_started':False})])
        with self.assertRaisesRegex(ValueError,'skip mode contains'):
            e.validate_treatment(dict(variant='skip',method='101',loss='mse'),native)

    def test_equal_declared_E_cap_is_checked_against_native_allowance(self):
        native=dict(method='101',config=dict(w_mode='skip',loss='mse',e_seconds=270.),
            events=[dict(stage='E',stage_seconds_cap=270.,solver={'requested_solve_seconds':271.})])
        with self.assertRaisesRegex(ValueError,'more than its assigned allowance'):
            e.validate_treatment(dict(variant='skip',method='101',loss='mse'),native)

    def test_missing_RNJ_pool_is_explicitly_unavailable(self):
        with TemporaryDirectory() as temp:
            root,entry,_,native,_=fixture(temp);reader=e.base.Reader();data=e.base.load_input(reader,entry)
            bank=e.base.evaluate_bank(native,data['train'],data['validation'],data['new_day'],data['geometry'])
            result=e.attribution(native,bank,data['geometry'])
            self.assertFalse(result['pool']['available'])
            self.assertEqual(result['selected']['validation']['origin'],'basis')
            self.assertTrue(result['W_prefix']['validation']['topology']['full']['exact'])

    def test_failure_partial_calls_never_become_completed_models(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            events=[dict(stage='native_progress',callback_event='FINAL',status=s,wall_seconds=i+1.) for i,s in enumerate((2,9,9))]
            (root/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in events)+'\n',encoding='utf8')
            write(root/'checkpoint.json',dict(path=[{},{}]))
            result=e.partial_failure_evidence(e.base.Reader(),root)
            self.assertEqual((result['native_final_records'],result['native_time_limits'],result['accepted_path_points']),(3,2,2))

    def test_failed_stage_receipts_recover_missing_callback_without_double_count(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            events=[dict(stage='W_started',iteration=0),
                dict(stage='native_progress',callback_event='FINAL',status=9),
                dict(stage='W_complete'),dict(stage='W_started',iteration=1)]
            (root/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in events)+'\n',encoding='utf8')
            for ordinal,status in enumerate((9,2)):
                write(root/'stages'/f'{ordinal:04d}_W.json',dict(event=dict(stage='W',solver=dict(solve_started=True,status=status))))
            result=e.partial_failure_evidence(e.base.Reader(),root)
            self.assertEqual((result['scalar_native_FINAL_records'],result['complete_stage_native_records']),(1,2))
            self.assertEqual((result['native_final_records'],result['native_time_limits']),(2,1))

    def test_full_file_interface_pending_denominator_and_frozen_score(self):
        with TemporaryDirectory() as temp:
            root,_=setup_run(temp)
            result=e.evaluate_run(root,root/'analysis')
            self.assertEqual(result['status'],'incomplete');self.assertEqual(result['planned'],3)
            self.assertEqual(result['completed'],1);self.assertFalse(result['all_tasks_terminal'])
            self.assertEqual(len(result['errors']),0)
            full=next(r for r in result['panels'] if r['variant']=='full')
            self.assertEqual(full['metrics']['validation']['full_correct'],1)
            skip=next(r for r in result['panels'] if r['variant']=='skip')
            self.assertIsNone(skip['metrics']['validation']['full_accuracy_completed'])
            self.assertIsNone(skip['metrics']['validation']['full_accuracy_planned'])

    def test_completed_and_partial_solver_counts_stay_separate(self):
        with TemporaryDirectory() as temp:
            root,_=setup_run(temp,completed=('full','skip'),failed=('pool_only',))
            result=e.evaluate_run(root,root/'analysis')
            self.assertEqual(result['status'],'passed');self.assertTrue(result['all_tasks_terminal'])
            group=next(r for r in result['panels'] if r['variant']=='pool_only')
            self.assertEqual(group['status_counts'],{'failed':1})
            self.assertEqual(group['solver']['partial_native_time_limits'],1)
            self.assertEqual(group['solver']['completed_native_time_limits'],0)
            self.assertEqual(group['metrics']['validation']['full_accuracy_planned'],0.)

    def test_changed_native_or_input_is_not_scored(self):
        with TemporaryDirectory() as temp:
            root,tasks=setup_run(temp)
            write(root/tasks[0]['result_dir']/'native.json',{'changed':True})
            result=e.evaluate_run(root,root/'analysis')
            self.assertEqual(result['status'],'failed');self.assertEqual(result['completed'],0)
            self.assertEqual(result['status_counts']['audit_failed'],1)

    def test_unmatched_budgets_are_rejected_before_reporting(self):
        with TemporaryDirectory() as temp:
            root,tasks=setup_run(temp)
            tasks[1]['config']['e_seconds']=271.
            task_ref=write(root/'tasks.json',tasks)
            protocol=json.loads((root/'protocol.json').read_text());protocol['tasks_ref']=task_ref
            write(root/'protocol.json',protocol)
            with self.assertRaisesRegex(ValueError,'budgets or model settings differ'):
                e.evaluate_run(root,root/'analysis')

    def test_final_report_refuses_pending_while_retaining_evidence(self):
        with TemporaryDirectory() as temp:
            root,_=setup_run(temp)
            with self.assertRaisesRegex(RuntimeError,'pending tasks'):
                e.study(root,allow_partial=False)
            self.assertEqual(len(list((root/'evaluations').glob('*/report.json'))),1)


if __name__=='__main__':unittest.main()
