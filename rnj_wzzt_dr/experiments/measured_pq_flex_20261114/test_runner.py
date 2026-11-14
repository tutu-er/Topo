"""Small algebra and incremental-domain checks; no identification or AC batch."""
import unittest
import numpy as np

from ...observations import Observation
from .domain import IncrementCase
from .runner import pair_components,pair_delta,evidence_summary,forecast_power


def fixture():
    context=dict(node_mapping=[dict(label=i+101,feeder_id=i+1,scale=.5) for i in range(3)],
        resource_metadata_at_history_start=[dict(feeder=i+1,date='2024-01-01',PV_systems_kW=20+i,
            EV_chargers_kW=11+i,hot_water_tanks_kW=5+i) for i in range(3)])
    prediction=np.zeros((96,6));prediction[:,:3]=-.02
    return context,prediction


class DomainTests(unittest.TestCase):
    def test_windows_units_and_response(self):
        context,forecast=fixture();case=IncrementCase(context,(101,102,103),forecast,200)
        self.assertTrue(case.audit(np.zeros(case.shape))['passed'])
        for j,r in enumerate(case.resources):
            allowed=(case.clock>=10)&(case.clock<14) if r['kind']=='pv' else ((case.clock>=18)|(case.clock<7) if r['kind']=='ev' else np.ones(24,bool))
            self.assertTrue(np.all(case.upper[~allowed,j]==0))
        direction=np.sin(np.arange(case.dimension)).reshape(case.shape)
        u,_=case.linear_optimum(direction)
        pp,qq=case.aggregate(case.delivery(u))
        self.assertEqual(pp.shape,(96,3));self.assertTrue(np.all(qq==0))
        self.assertAlmostEqual(pp.sum()*.25,u.sum())
        actual,receipt=case.execute(u,np.zeros((96,3)),np.full(case.R,.95))
        self.assertTrue(receipt['resource_audit']['passed'])
        for j,r in enumerate(case.resources):
            if r['kind']=='pv':self.assertTrue(np.all(actual[:,j]==0))
            elif r['kind']=='ev':self.assertAlmostEqual(actual[:,j].sum(),0.,places=8)
            else:
                state=0.
                for value in actual[:,j]:
                    state=.98*state+value
                    self.assertLessEqual(abs(state),r['contract_kw']+1e-7)
                self.assertAlmostEqual(state,0.,places=8)

    def test_PV_uses_minimum_quarter_and_no_baseline(self):
        context,forecast=fixture();forecast[8,0]=.001
        case=IncrementCase(context,(101,102,103),forecast,200)
        j=next(i for i,r in enumerate(case.resources) if r['kind']=='pv' and r['node_label']==101)
        self.assertEqual(case.upper[2,j],0.)
        with self.assertRaises(RuntimeError):case.device_baseline_pq_kw()


class EvidenceTests(unittest.TestCase):
    def test_right_target_and_common_cross_terms(self):
        random=np.random.default_rng(404);H=random.normal(size=(3,2,4));C=np.zeros_like(H)
        C[:,:,:2]=np.array([.2,-.1,.4])[:,None,None]
        P=random.normal(size=(7,4));P2=P+random.normal(scale=.2,size=P.shape)
        true=random.normal(size=(2,4));history=np.array([2.,1.,3.]);hnc=np.array([1.,4.,2.]);sigma=.7
        def component(power):
            y=power@true.T+.03
            obs=Observation(power[:,:2]+.01,power[:,2:]-.02,y+.04,(1,2))
            return pair_components(H,C,history,sigma,power,y,obs,hnc)
        before,after=component(P),component(P2)
        fb=pair_components(H,C,history,sigma,P);fa=pair_components(H,C,history,sigma,P2)
        ctx=dict(score=dict(native_index=2),history_runner_native_index=0)
        change,summary=evidence_summary(ctx,before,after,fb,fa,before,after)
        for row in summary['target_pairs']:
            self.assertAlmostEqual(row['actual_logodds_change'],row['actual_physical_J_gain']+row['actual_bias_change']+row['actual_meter_change'],places=9)
        self.assertEqual(summary['nearest_competitor']['competitor'],0)
        np.testing.assert_allclose(before['J'],before['J_without_common']+before['J_common']+before['J_cross'])
        self.assertGreater(float(np.max(abs(before['J_cross']))),1e-6)
        _,missing=evidence_summary(ctx,before,after,fb,fa,None,None)
        self.assertFalse(missing['exact_requested_diagnostic_available'])
        self.assertIsNone(missing['nearest_competitor']['exact_physical_J_gain'])
        self.assertIn('observed_logodds_without_common',change)
        self.assertIn('clean_logodds_without_common',before)

    def test_forecast_stream_is_repeatable_and_input_specific(self):
        P=np.ones((96,6))
        np.testing.assert_array_equal(forecast_power(P,20261114,7,2),forecast_power(P,20261114,7,2))
        self.assertFalse(np.array_equal(forecast_power(P,20261114,7,2),forecast_power(P,20261114,0,2)))


if __name__=='__main__':unittest.main()
