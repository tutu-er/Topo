"""Data-boundary and append-only checks without historical fits or new AC."""
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from . import data
from ...observations import Observation,fingerprint
from ...identification.optimization import errors,predict


class DataTests(unittest.TestCase):
    def test_calendar_is_fixed_and_disjoint(self):
        tasks=data.calendar()
        self.assertEqual(len(tasks),112)
        self.assertEqual(sum(t['phase']=='development' for t in tasks),16)
        self.assertEqual(sum(t['phase']=='confirmation' for t in tasks),96)
        self.assertEqual(len({t['id'] for t in tasks}),112)
        self.assertEqual([t['index'] for t in tasks],list(range(112)))
        self.assertEqual(len({t['start_date'] for t in tasks}),28)

    def test_preflight_checks_each_network_and_future_flags(self):
        task=data.calendar()[0]
        times=pd.date_range(pd.Timestamp(task['local_start']).tz_convert('UTC'),periods=384,freq='15min')
        finite=np.ones((384,18),bool);count=np.full((384,18),15,np.uint8)
        count[0,17]=11;finite[320,4]=False
        rows=data.preflight_rows([task,dict(task,case='pengwah18',terminal_count=18)],times,finite,count,np.arange(1,19))
        self.assertEqual([r['incomplete_node_time_bins'] for r in rows],[1,2])
        self.assertEqual(rows[0]['incomplete_bins_by_split'],dict(train=0,validation=0,new_day=1))
        self.assertFalse(rows[0]['future_day_complete'])
        self.assertIsNone(rows[0]['replacement_date'])
        with self.assertRaises(TypeError):
            data.preflight_rows([task],times,finite.astype(float),count,np.arange(1,19))

    def test_missing_timestamp_is_not_interpolated(self):
        task=data.calendar()[0]
        times=pd.date_range(pd.Timestamp(task['local_start']).tz_convert('UTC'),periods=383,freq='15min')
        row=data.preflight_rows([task],times,np.ones((383,18),bool),np.full((383,18),15),np.arange(1,19))[0]
        self.assertEqual(row['missing_timestamp_rows'],1)
        self.assertEqual(row['incomplete_node_time_bins'],15)

    def test_confirmation_has_no_execution_entry(self):
        with patch.object(data.io,'read',return_value={}),patch.object(data.physical,'generate') as generator:
            with self.assertRaises(PermissionError):data.generate_one(dict(phase='confirmation'),'unused')
            generator.assert_not_called()

    @staticmethod
    def toy():
        rng=np.random.default_rng(5);p=rng.normal(.1,.03,(12,3));q=rng.normal(.03,.02,(12,3))
        empty=Observation(p,q,np.zeros_like(p),(101,102,103))
        one=dict(supports=[[0],[1],[2]],r=[.02,.03,.01],x=[.01,.01,.01])
        train=Observation(p,q,predict(empty,one)+rng.normal(0,.0001,p.shape),empty.labels)
        validation=Observation(p*.9,q*1.1,train.y*.95,train.labels)
        one['mse']=errors(train,one)['mse']
        two=dict(supports=[[0],[1],[2],[0,1]],r=[.018,.028,.01,.002],x=[.009,.009,.01,.001])
        two['mse']=errors(train,two)['mse']
        config=data.validate._config(30.)
        selected,mae,bank=data._output([one],train,validation,one['mse'],config)
        original=dict(method='101',events=[],config=asdict(config),reference_mse=one['mse'],
            frozen_candidates=bank,selected=selected,selected_mse_view=selected,selected_mae_view=mae,
            handoff=dict(validation_sha256=fingerprint(validation)))
        return original,two,train,validation

    def test_append_preserves_original_bank_and_selection_rule(self):
        original,point,train,validation=self.toy();copy=deepcopy(original)
        enriched=data.append_candidate(original,point,train,validation)
        self.assertEqual(original,copy)
        self.assertEqual(enriched['frozen_candidates'][:1],original['frozen_candidates'])
        self.assertEqual(len(enriched['frozen_candidates']),2)
        data.freeze_history(enriched,validation)
        self.assertEqual(data.append_candidate(original,None,train,validation),original)

    def test_search_uses_only_recorded_pool_and_training(self):
        original,point,train,_=self.toy()
        absent=data.additional_search(original,train)
        self.assertEqual(absent['status'],'not_run_no_recorded_E_pool')
        original['events']=[dict(stage='E',pool=point['supports'],keep=[[0],[1],[2]],point=original['selected']['point'])]
        receipt=dict(closed_point=point,total_seconds=.1)
        with patch.object(data.pool_diagnostic,'run_search',return_value=receipt) as search:
            result=data.additional_search(original,train)
            self.assertIs(search.call_args.args[0],train)
            self.assertEqual(search.call_args.args[-1],60.)
            self.assertEqual(result['status'],'returned_point')
            self.assertFalse(result['validation_used'])
            self.assertFalse(result['new_day_used'])


if __name__=='__main__':unittest.main()
