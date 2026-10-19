from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from . import ev_mobile_v3_summary as summary


def spec(i=0,phase='development',seed=None):
    return dict(index=i,phase=phase,bank='n3',H=8,T=12,seed=100+i if seed is None else seed,family_index=0)


def record(s=None):
    s=spec() if s is None else s
    op=dict(audit=dict(feasible=True,moves=0,driver_cost=0.,cost_units='worker_minutes',travel_energy_kwh=0.,
        grid_energy_kwh=3.,throughput_kwh=5.,conversion_loss_kwh=.2),energy_kwh=[24.,26.85])
    method=dict(success=True,planning_success=True,power_kw=[[1.,0.],[-1.,0.]],operation=op,
        evaluation=dict(complete=True,losses=dict(a=10.,b=15.),M=5.,seconds=.4))
    r=dict(spec=s,status='complete',true_name='a',methods={name:deepcopy(method) for name in summary.METHODS},
        training_result=dict(fits=[dict(candidate='a',r=[.01],x=[.02]),dict(candidate='b',r=[.01,.02],x=[.02,.03])]),planning={})
    for name in summary.METHODS[2:]:
        r['planning'][name]=dict(seconds=1.,standalone_seconds=2.,raw=dict(success=True,selected_origin='square_parent',selected_score=1.,steps=[]))
    return r


class SummaryTests(unittest.TestCase):
    def test_missing_record_preserves_all_method_denominators(self):
        r=record();x=summary.summarize_records([spec(),spec(1)],{0:r},candidate_count=2)
        for m in x['overall']['methods'].values():
            self.assertEqual(m['denominator'],2);self.assertEqual(m['usable'],1);self.assertEqual(m['missing_or_failed'],1)
        p=x['overall']['mobility']['projection']
        self.assertEqual((p['ties'],p['missing_pairs']),(1,1))

    def test_failed_plan_fallback_separate_from_execution(self):
        r=record();r['methods']['projection_mobile']['planning_success']=False
        x=summary.summarize_records([spec()],{0:r},candidate_count=2)['overall']['methods']['projection_mobile']
        self.assertEqual(x['planning_status'],dict(failed=1));self.assertEqual(x['usable'],1)
        self.assertEqual(x['execution_status'],dict(success=1));self.assertEqual(x['fit_status'],dict(success=1))

    def test_mobile_predicted_gain_actual_loss_direction(self):
        r=record();r['planning']['projection_mobile']['raw']['selected_score']=2.
        r['methods']['projection_mobile']['evaluation'].update(losses=dict(a=10.,b=13.),M=3.)
        p=summary.summarize_records([spec()],{0:r},candidate_count=2)['overall']['mobility']['projection']
        self.assertEqual((p['a_wins'],p['ties'],p['a_losses'],p['missing_pairs']),(0,0,1,0))
        self.assertEqual(p['prediction_actual']['predicted_gain_actual_loss_count'],1)

    def test_pair_missing_member_preserves_history(self):
        r=record();del r['methods']['projection_mobile']
        p=summary.summarize_records([spec()],{0:r},candidate_count=2)['overall']['mobility']['projection']
        self.assertEqual((p['complete_pairs'],p['missing_pairs']),(0,1))

    def test_dimensions_from_saved_edges_not_name(self):
        r=record();r['training_result']['fits'][0].update(r=[.01,.02],x=[.02,.03])
        r['training_result']['fits'][1].update(r=[.01],x=[.02])
        r['methods']['constant']['evaluation'].update(losses=dict(a=10.,b=11.),M=1.)
        a=summary.project_method(r,'constant',candidate_count=2)
        self.assertEqual(a['raw_argmin'],['a']);self.assertEqual(a['AIC_argmin'],['b'])

    def test_actual_argmin_tie_not_unique(self):
        r=record();r['methods']['constant']['evaluation'].update(losses=dict(a=10.,b=10.),M=0.)
        a=summary.project_method(r,'constant',candidate_count=2)
        self.assertFalse(a['raw_unique_correct']);self.assertTrue(a['raw_contains_truth'])

    def test_secondary_AIC_margin_preserves_primary_M(self):
        r=record();r['training_result']['fits'][0].update(r=[.01,.02],x=[.02,.03])
        r['training_result']['fits'][1].update(r=[.01],x=[.02])
        r['methods']['constant']['evaluation'].update(losses=dict(a=10.,b=11.),M=1.)
        a=summary.project_method(r,'constant',candidate_count=2)
        self.assertEqual(a['M'],1.);self.assertEqual(a['M_AIC'],-3.)
        g=summary.summarize_records([spec()],{0:r},candidate_count=2)['overall']['methods']['constant']
        self.assertEqual(g['M_AIC']['mean'],-3.);self.assertEqual(g['M_AIC_negative'],1)

    def test_bad_saved_M_retained_as_invalid(self):
        r=record();r['methods']['constant']['evaluation']['M']=999.
        a=summary.project_method(r,'constant',candidate_count=2)
        self.assertIsNone(a['M']);self.assertIn('saved M mismatch',a['issues'])

    def test_missing_AIC_dimensions_does_not_drop_M_denominator(self):
        r=record();r['training_result']['fits']=[]
        x=summary.summarize_records([spec()],{0:r},candidate_count=2)['overall']['methods']['constant']
        self.assertEqual(x['usable'],1);self.assertEqual(x['decisions']['AIC']['known'],0)
        self.assertEqual(x['decisions']['raw']['known'],1)

    def test_M_negative_is_exact_and_pair_tie_tolerance_separate(self):
        r=record();r['methods']['constant']['evaluation'].update(losses=dict(a=10.,b=10.-1e-10),M=-1e-10)
        x=summary.summarize_records([spec()],{0:r},candidate_count=2)['overall']['methods']['constant']
        self.assertEqual(x['M_negative'],1);self.assertEqual(x['M_zero_with_tolerance'],1)

    def test_charge_discharge_grid_side_energy(self):
        a=summary.project_method(record(),'constant',candidate_count=2)
        self.assertEqual(a['operation']['charging_energy_kwh'],4.)
        self.assertEqual(a['operation']['discharging_energy_kwh'],1.)

    def test_true_candidate_training_and_joint_line_errors(self):
        r=record();r['truth']=dict(r=[1.,2.],x=[2.,4.])
        r['training_result']['fits'][0].update(r=[1.1,2.2],x=[1.8,3.6],success=True)
        r['methods']['constant']['evaluation']['comparisons']=dict(a=dict(joint=dict(post=dict(success=True,state=dict(r=[1.2,2.4],x=[2.1,4.2])))))
        errors=summary.project_method(r,'constant',candidate_count=2)['parameter_errors']['errors']
        for key,value in dict(training_R=.1,training_X=.1,joint_R=.2,joint_X=.05).items():
            self.assertTrue(errors[key]['valid']);self.assertAlmostEqual(errors[key]['value'],value)
        grouped=summary.summarize_records([spec()],{0:r},candidate_count=2)['by_bank_H']['n3/H8']['methods']['constant']['parameter_errors']
        self.assertEqual(grouped['joint_R']['valid'],1);self.assertAlmostEqual(grouped['joint_R']['mean'],.2)

    def test_missing_truth_or_final_state_retains_error_denominator(self):
        r=record();r['truth']=dict(r=[1.],x=[2.])
        x=summary.summarize_records([spec(),spec(1)],{0:r},candidate_count=2)['overall']['methods']['constant']['parameter_errors']
        self.assertEqual(x['training_R']['denominator'],2);self.assertEqual(x['training_R']['valid'],1)
        self.assertEqual(x['joint_R']['denominator'],2);self.assertEqual(x['joint_R']['valid'],0)
        self.assertEqual(x['joint_R']['missing'],2);self.assertIsNone(x['joint_R']['mean'])
        r.pop('truth');e=summary.project_method(r,'constant',candidate_count=2)['parameter_errors']['errors']['training_R']
        self.assertFalse(e['valid']);self.assertIsNone(e['value'])

    def test_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):summary.summarize_records([spec(),spec(1,seed=100)],{},candidate_count=2)

    def test_fatal_failure_not_dropped(self):
        s=spec();x=summary.summarize_records([s],{0:dict(spec=s,status='fatal_failure',failure='broken')},candidate_count=2)
        self.assertEqual(x['overall']['history_denominator'],1)
        self.assertEqual(x['overall']['methods']['constant']['missing_or_failed'],1)

    def test_file_phases_separate_and_policy_checked(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);dev=spec();evaluation=spec(phase='evaluation',seed=999)
            freeze=dict(methods=list(summary.METHODS),policy_id='p',source_sha256={'dummy':'abc'},config=dict(candidate_count=2),
                specifications=dict(development=[dev],evaluation=[evaluation]))
            (root/'freeze.json').write_text(json.dumps(freeze),encoding='utf8')
            for phase,s in [('development',dev),('evaluation',evaluation)]:
                (root/phase).mkdir();r=record(s);r['policy_id']='p'
                (root/phase/'000.json.gz').write_bytes(gzip.compress(json.dumps(r).encode()))
            x=summary.run(root,root/'summary.json',require_complete=True)
            self.assertEqual(x['phases']['development']['overall']['history_denominator'],1)
            self.assertEqual(x['phases']['evaluation']['overall']['history_denominator'],1)
            bad=record(evaluation);bad['policy_id']='bad'
            (root/'evaluation/000.json.gz').write_bytes(gzip.compress(json.dumps(bad).encode()))
            with self.assertRaises(ValueError):summary.run(root,root/'other.json',require_complete=True)


if __name__=='__main__':unittest.main()
