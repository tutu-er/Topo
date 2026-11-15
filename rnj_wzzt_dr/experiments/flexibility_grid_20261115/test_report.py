"""Synthetic aggregation checks; no fitting, procurement solve, or AC run."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from ... import io
from . import report


def task(world='paper15__2024-01-04', stride=1, multiplier=1., order=0):
    return dict(id=world+f'__s{stride}__m{str(multiplier).replace(".","p")}',
        original_id=world, case=world.split('__')[0],start_date=world.split('__')[1],
        stride=stride,meter_multiplier=multiplier,grid_order=order,
        training_rows=192//stride,validation_rows=96//stride,new_points=96,forecast_relative_sd=.05)


def row(t, policy, repeat=0, *, correct=True, weight=.95, evidence_gain=0.):
    cost = dict(total_yuan=3.,activation_yuan=1.,availability_yuan=2.,active_activation_yuan=.2,
        reactive_activation_yuan=.8,reserved_P_kw=1.,reserved_Q_kvar=3.)
    e = dict(h=.1,J=1.,bias=-.1,meter=.2,clean_logodds=1.+evidence_gain,
             observed_logodds=1.2+evidence_gain)
    diag = dict(K_nominal_forecast_mu=2.+2*evidence_gain,K_actual_input_mu=1.8,
        actual_power_minus_nominal_forecast=-.2,clean_minus_actual_K=-.8,
        observed_minus_clean=.2,clean_minus_nominal_K=-1.)
    return dict(id=t['id'],case=t['case'],start_date=t['start_date'],date_cluster=t['start_date'],season='winter',
        forecast=0,repeat=repeat,strategy=policy,audit_status='passed',original_status='completed',
        evaluation=dict(native_correct=correct,native_full_support_exact=correct,p_true=weight,
            true_high=weight>=.9,wrong_high=not correct and weight<=.1,
            pair=[4,2],evidence=e,historical_mean_diagnostic=diag),
        voltage_screen_passed=True,recovered_and_voltage_screen_passed=correct,
        planned_cost=deepcopy(cost),settled_cost=deepcopy(cost),
        requested=[dict(unit='kW',absolute_energy=2.),dict(unit='kvar',absolute_energy=8.)],
        delivered=[dict(unit='kW',absolute_energy=1.8),dict(unit='kvar',absolute_energy=7.2)],
        electrical=dict(active_loss_kwh=1.,reactive_loss_kvarh=.1,root_signed_kwh=100.,
            root_import_kwh=105.,root_export_kwh=5.,maximum_branch_current_A=10.,
            all_bus_voltage=dict(minimum=.99,maximum=1.03)))


def quote(t, mode='normal_fixed', *, status=9, feasible=True, zero=False):
    return dict(id=t['id'],forecast=0,policy=mode,new_gate_feasible=feasible,
        zero_action_sufficient=zero,nonzero_executed=not zero,optimizer_invoked=not zero,
        anchor_fallback=not feasible,zero_fallback=False,native_status=status,status='time_limit' if status==9 else 'zero_sufficient',
        relative_gap=.5 if feasible and not zero else 0. if zero else None,
        cost_yuan=1. if feasible and not zero else 0. if zero else None,
        lower_bound_yuan=.5 if feasible and not zero else 0. if zero else None,
        charged_total_seconds=5.,quote_wall_seconds=4.,source='initial' if not zero else 'zero',
        initial=dict(feasible=True,name='common_lp',cost_yuan=1.))


def fixture(directory, *, failed_last=True):
    worlds=['paper15__2024-01-04','soumalas11__2024-01-04','pengwah18__2024-01-04','flynn16__2024-12-19']
    tasks=[task(w,s,m,i) for i,(w,s,m) in enumerate((w,s,m) for w in worlds
           for s in report.STRIDES for m in report.MULTIPLIERS)]
    ledger=[dict(task=t,status='failed' if failed_last and i==35 else 'complete') for i,t in enumerate(tasks)]
    entries=[t for t,a in zip(tasks,ledger) if a['status']=='complete']
    records=[row(t,p,r,correct=(p!='no_dfs' or r==1),weight=.95 if p!='no_dfs' or r==1 else .05)
             for t in entries for r in range(2) for p in report.POLICIES]
    quotes=[quote(t,mode) for t in entries for mode in report.MODES]
    coverage=[dict(id=t['id'],truth_in_native_bank=True,truth_in_target_set=True,
                   history_correct=False,full_support_truth_in_native_bank=True) for t in entries]
    for name,value in [('inputs',entries),('ledger',ledger),('failures',[])]:
        io.write(directory/(name+'.json'),value)
    for name,value in [('records',records),('quotes',quotes)]:
        (directory/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in value),encoding='utf8')
    protocol=dict(phase='grid_development',smoke=False,forecasts=1,repeats=2,
        policies=list(report.POLICIES),modes=list(report.MODES),inputs=io.ref(directory/'inputs.json'),
        qualification_ledger=io.ref(directory/'ledger.json'),preparation_failures=io.ref(directory/'failures.json'))
    io.write(directory/'protocol.json',protocol)
    audit=dict(status='passed',source_protocol=io.ref(directory/'protocol.json'),source_inputs=protocol['inputs'],
        records=io.ref(directory/'records.jsonl'),quotes=io.ref(directory/'quotes.jsonl'),
        input_coverage=coverage,rotation_unmatched_fees=[])
    io.write(directory/'audit.json',audit)
    return tasks,ledger,entries,audit


class ReportChecks(unittest.TestCase):
    def test_formal_ledger_is_complete_and_dimensions_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks,ledger,entries,_=fixture(Path(directory))
            dims=report.dimensions(ledger,entries)
            self.assertEqual(len(dims),36)
            self.assertEqual(dims[tasks[-1]['id']]['new_day_scoring_fraction'],.8)
            with self.assertRaisesRegex(ValueError,'36-cell'):
                report.dimensions(ledger[:-1],entries)
            changed=deepcopy(entries);changed[0]['stride']=2
            with self.assertRaisesRegex(ValueError,'dimension changed'):
                report.dimensions(ledger,changed)

    def test_paired_gains_losses_and_failed_costs(self):
        t=task(); rows=[]
        for r in range(2):
            rows += [row(t,'no_dfs',r,correct=r==1,weight=.95 if r==1 else .05),
                     row(t,'normal_fixed',r,correct=r==0,weight=.95 if r==0 else .05)]
        comparison=report.paired(rows,'normal_fixed','no_dfs')
        self.assertEqual((comparison['gained'],comparison['lost']),(1,1))
        self.assertNotIn('dates',comparison)
        failed=dict(rows[-1],audit_status='recorded_failure',original_status='execution_failed',evaluation={})
        summary=report.summarize([rows[1],failed])
        self.assertEqual((summary['n'],summary['recovered'],summary['cost_denominator']),(2,1,1))
        self.assertEqual(summary['mean_p_true_completed'],.95)
        self.assertEqual(summary['settled_cost_mean_completed']['total_yuan'],3.)

    def test_evidence_pair_and_unavailable_truth(self):
        t=task(); base=row(t,'no_dfs'); action=row(t,'normal_fixed',evidence_gain=-2.)
        answer=report.evidence_changes([base,action],'normal_fixed')
        self.assertEqual(answer['mean_forecast_K_logodds_increment'],-4.)
        self.assertAlmostEqual(answer['mean_actual_clean_logodds_increment'],-2.)
        self.assertAlmostEqual(answer['mean_actual_observed_logodds_increment'],-2.)
        self.assertEqual(answer['negative_observed_increment_count'],1)
        self.assertEqual(answer['negative_threshold'],-1e-8)
        changed=deepcopy(action);changed['evaluation']['pair']=[2,4]
        with self.assertRaisesRegex(ValueError,'pair changed'):
            report.evidence_changes([base,changed],'normal_fixed')
        for value in (base,action):value['evaluation'].pop('pair')
        answer=report.evidence_changes([base,action],'normal_fixed')
        self.assertIsNone(answer['mean_actual_observed_logodds_increment'])
        self.assertEqual(answer['unavailable_fixed_pair'],1)

    def test_timeout_fallback_and_nonzero_gap_separated_from_zero(self):
        t=task(); quotes=[quote(t),quote(t,status=None,zero=True),quote(t,feasible=False)]
        answer=report.solver_summary(quotes,'normal_fixed',4)
        self.assertEqual(answer['missing_quotes'],1)
        self.assertEqual(answer['native_time_limit_quotes'],2)
        self.assertEqual(answer['anchor_fallbacks'],1)
        self.assertEqual(answer['zero_cost_certificates']['n'],1)
        self.assertEqual(answer['nonzero_feasible_quotes']['gap']['mean'],.5)
        self.assertEqual(answer['gap_mean_feasible'],.25)
        self.assertEqual(answer['final_equal_to_initial_cost'],1)

    def test_full_report_preserves_failed_cell_and_all_csv_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';source.mkdir();fixture(source)
            out=Path(temporary)/'report'
            result=report.run(source/'audit.json',out)
            self.assertEqual((len(result['cells']),len(result['by_stride_meter']),result['physical_worlds']),(36,9,4))
            self.assertEqual(result['full_pipeline_strategy_rows'],36*2*8)
            self.assertEqual(len(result['failed_qualification']),1)
            self.assertEqual(len(result['failed_execution_rows']),16)
            no=result['overall']['groups']['no_dfs'];new=result['overall']['groups']['normal_fixed']
            self.assertEqual((no['n'],no['recovered'],new['recovered']),(72,35,70))
            self.assertEqual(result['overall']['paired_vs_no_dfs']['normal_fixed']['gained'],35)
            self.assertEqual(result['overall']['solver']['normal_fixed']['missing_quotes'],1)
            covered=result['by_candidate_coverage']['truth_present']['groups']['no_dfs']
            unknown=result['by_candidate_coverage']['coverage_unknown']['groups']['no_dfs']
            self.assertEqual((covered['n'],unknown['n'],unknown['recorded_failures']),(70,2,2))
            self.assertEqual(covered['n']+unknown['n'],no['n'])
            self.assertAlmostEqual(no['mean_p_true_completed'],.5)
            self.assertIsNone(result['cells'][-1]['groups']['normal_fixed']['mean_p_true_completed'])
            for name,n in [('grid_cells.csv',288),('stride_meter.csv',72)]:
                with (out/name).open(encoding='utf-8-sig',newline='') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))),n)
            self.assertEqual(io.read(out/'complete.json')['status'],'complete')
            self.assertNotIn('comparisons',result)
            with self.assertRaises(FileExistsError):report.run(source/'audit.json',out)

    def test_rejects_failed_audit_or_changed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary);_,_,_,audit=fixture(source)
            audit['status']='failed';io.write(source/'audit.json',audit)
            with self.assertRaisesRegex(ValueError,'passed independent audit'):
                report.run(source/'audit.json',source/'out')
            self.assertFalse((source/'out').exists())
            audit['status']='passed';io.write(source/'audit.json',audit)
            (source/'records.jsonl').write_text('',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Changed audit input'):
                report.run(source/'audit.json',source/'out')
            self.assertFalse((source/'out').exists())


if __name__ == '__main__':
    unittest.main()
