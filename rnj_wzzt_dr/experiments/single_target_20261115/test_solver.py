"""Factory default compatibility and adapter argument/metadata tests."""
import unittest
from unittest.mock import patch

import numpy as np

from ..distribution_gate_20261114 import solver as engine
from ..distribution_gate_20261114.test_solver import ToyCase, ToyEvidence
from . import solver
from .test_evidence import measured_toy


class ZeroEvidence(ToyEvidence):
    def verify_expansion(self,u):
        return dict(margins=np.ones(1),guard_gains=[],pair_failure=np.zeros(1),feasible=True)


class FactoryTests(unittest.TestCase):
    def test_omitted_none_and_explicit_original_factory_equal(self):
        case=ToyCase()
        results=[]
        with patch.object(engine,'DistributionEvidence',ZeroEvidence):
            for kwargs in ({},{'evidence_factory':None},{'evidence_factory':ZeroEvidence}):
                receipt,point=engine.plan(case,{},None,np.zeros((1,1)),1.,
                    mode='normal_fixed',seconds=2.,env=None,**kwargs)
                results.append((receipt,point))
        for receipt,point in results:
            self.assertEqual(receipt['status'],'zero_sufficient')
            self.assertFalse(receipt['optimizer_invoked'])
            self.assertEqual(receipt['cost_yuan'],0.)
            np.testing.assert_array_equal(point,np.zeros(case.shape))
            np.testing.assert_array_equal(receipt['baseline']['margins'],results[0][0]['baseline']['margins'])

    def test_wrapper_passes_original_arguments_and_attaches_selection(self):
        case,score,val,forecast,_=measured_toy()
        seeds={'known':np.zeros(case.shape)};captured={}
        def run(c,s,v,f,b,**kwargs):
            captured.update(kwargs)
            self.assertIs(c,case);self.assertIs(s,score);self.assertIs(v,val)
            self.assertIs(f,forecast);self.assertEqual(b,200.)
            evidence=kwargs['evidence_factory'](c,s,v,f,b,mode=kwargs['mode'],target_mass=kwargs['target_mass'],
                meter_relative_sd=kwargs['meter_relative_sd'],forecast_relative_sd=kwargs['forecast_relative_sd'])
            self.assertEqual(evidence.targets,[0])
            return dict(evidence_source={'path':'original-evidence'},feasible=False),None
        with patch.object(solver.engine,'plan',side_effect=run):
            receipt,point=solver.plan(case,score,val,forecast,200.,mode='mgf_fixed',
                seconds=600.,env='sentinel',seed_points=seeds,meter_relative_sd=.001,
                forecast_relative_sd=.05,record_directory='private-log')
        self.assertIsNone(point)
        self.assertIs(captured['seed_points'],seeds)
        self.assertEqual(captured['seconds'],600.)
        self.assertEqual(captured['env'],'sentinel')
        self.assertEqual(captured['record_directory'],'private-log')
        self.assertEqual(receipt['historical_target_selection']['selected_native'],0)
        self.assertTrue(receipt['single_target'])
        self.assertFalse(receipt['original_reverse_scalar_guards_retained'])
        self.assertEqual(receipt['base_evidence_source']['path'],'original-evidence')


if __name__=='__main__':unittest.main()
