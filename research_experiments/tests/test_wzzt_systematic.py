"""Data amount, leakage and paired physical counterfactual checks."""
import unittest

import numpy as np
import pandas as pd

from research_experiments.wzzt_systematic.common import indices, scenes_from_observed


class SamplingProtocolTests(unittest.TestCase):
    def test_nested_physical_sampling_preserves_daily_extent(self):
        # More samples must reveal the same physical days rather than change
        # the dynamics, meter RNG draw or train/test trajectory seed.
        for total in (24,48,96,288):
            selection=indices(total)
            self.assertEqual(3*len(selection),total)
            self.assertEqual(selection[0],0)
            self.assertGreaterEqual(selection[-1],84)
        for a,b in ((24,48),(48,96),(96,288)):
            self.assertTrue(set(indices(a))<set(indices(b)))

    def test_root_measurement_shared_and_day_boundaries_retained(self):
        p=pd.DataFrame(np.arange(24*2).reshape(24,2),columns=[7,9])
        q=p*.1
        v=pd.DataFrame(np.ones((24,2)),columns=p.columns)
        root=pd.Series(np.linspace(1.,1.04,24))
        scenes=scenes_from_observed((p,q,v,root))
        self.assertEqual([len(s["P_terminal"]) for s in scenes],[8,8,8])
        for scene in scenes:
            rows=scene["P_terminal"].index
            np.testing.assert_allclose(scene["drop_target"],np.repeat((root.loc[rows].to_numpy()**2-1)[:,None],2,axis=1))


if __name__=="__main__":
    unittest.main()
