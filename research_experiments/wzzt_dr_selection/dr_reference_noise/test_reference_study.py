"""Noise isolation and unchanged-load/DR contract tests, no solver."""
import unittest,numpy as np
import reference_study as s

class ReferenceStudyTests(unittest.TestCase):
    def test_three_regimes_and_original_settings(self):
        c=s.configuration()
        self.assertEqual([(x['pq_relative'],x['voltage_relative']) for x in c['noise_regimes']],[(0.,.0002),(.005,0.),(.005,.0002)])
        self.assertEqual(c['budget'],1.6);self.assertEqual(c['fit_settings']['coefficient_bound'],2.)
        self.assertEqual(c['seeds'],dict(natural=884001,measurement=884002,acceptance_and_independent_code=884003))
        self.assertEqual(c['c_levels'],[0.,.02,.6]);self.assertEqual(c['impedance_scale'],1.)

    def test_channel_isolation_shared_draws_and_matched_truth(self):
        c=s.configuration();R,X,_=s.original.truth(.4,.6,[0,1],c);p,q,_=s.original.natural_day(0,c)
        exact=s.original.observed_scene(p,q,R,X,0,dict(pq_relative=0.,voltage_relative=0.),c)
        v,pw,both=[s.original.observed_scene(p,q,R,X,0,r,c) for r in c['noise_regimes']]
        for key in ('P_terminal','Q_terminal'):
            np.testing.assert_array_equal(v[key],exact[key]);np.testing.assert_array_equal(pw[key],both[key])
        for key in ('V_terminal','V_root_observed'):
            np.testing.assert_array_equal(pw[key],exact[key]);np.testing.assert_array_equal(v[key],both[key])
        for r in c['noise_regimes']:
            first=None
            for pair in c['true_pairs']:
                rr,xx,_=s.original.truth(.4,.6,pair,c);h=s.history_packet(rr,xx,r,c)
                digest=__import__('hashlib').sha256(__import__('pickle').dumps(h,protocol=4)).hexdigest()
                if first is None:first=digest
                self.assertEqual(digest,first)

    def test_original_loads_and_controls_used_without_modification(self):
        c=s.configuration()
        for policy in c['main_policies']:
            draw=s.original.control(policy,c)
            self.assertLessEqual(draw['worst_payment'],1.6)
            np.testing.assert_array_equal(draw['orientation'],[1,-1,1,1])
            if policy=='pair_random':
                np.testing.assert_array_equal(draw['accepted'],[[1,0,0],[0,0,1],[1,0,1],[1,0,1]])
                self.assertAlmostEqual(float(sum(draw['committed_payment'])),1.2)

if __name__=='__main__':unittest.main()
