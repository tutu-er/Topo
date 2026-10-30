"""Pure controls/observed-array tests; no LP, AC or actual future is loaded."""
import unittest
import numpy as np
import random_code as code


def fixture(c=.05):
    """Conditional two-event analytic fixture, not a sampled success rate."""
    n=3;events=2;t=16;h=24;rng=np.random.default_rng(3124)
    p0=rng.normal(.02,.002,size=(t,n));q0=rng.normal(.006,.001,size=(t,n))
    hp=rng.normal(.02,.002,size=(h,n));hq=rng.normal(.006,.001,size=(h,n))
    dp=np.zeros((t,n));wave=code.WAVE_KW/200.
    dp[:8,0]=wave;dp[:8,2]=-wave;dp[8:,0]=-wave;dp[8:,2]=wave
    k=.1*np.eye(n);k[:2,:2]+=c
    p=p0+dp;q=q0+.35*dp
    observed=dict(history_train=dict(P=hp,Q=hq,Y=hp@k.T),future_train=dict(P=p,Q=q,Y=p@k.T))
    controls=dict(future_train=dict(delta_P_known_kw=dp*200,delta_Q_known_kvar=.35*dp*200,
        code_weight_known_kw=dp[:,0]*200,
        event_ids=np.repeat([0,1],8),pair_by_event={0:[0,2],1:[0,2]},
        orientation_by_event={0:1,1:-1},accepted_by_event={0:[1,0,1],1:[1,0,1]},
        price_by_event={0:[1,0,1],1:[1,0,1]},base_kw=200.,acceptance_before_code=True))
    config=dict(pq_observation_mode='trusted_exact',executed_control_mode='verified_exact',
        y_error_bound=0.,band_fixed_before_codes=True,numeric_tolerance=1e-10)
    return observed,controls,config


class RandomCodeTests(unittest.TestCase):
    def test_contract_randomization_and_partial_cost(self):
        d=code.sample_controls([[1,1,0],[1,1,0]],[[0,1],[0,1]],seed=8310)
        self.assertEqual(d['worst_payment'],1.6)
        self.assertTrue(np.isin(d['orientation'],[-1,1]).all())
        np.testing.assert_allclose(d['committed_payment'],.2*np.sum(d['prices']*d['accepted'],axis=1))
        self.assertIsNone(d['actual_payment'])
        np.testing.assert_allclose(d['delta_p_kw'].sum(axis=1),0,atol=1e-14)
        np.testing.assert_allclose(np.maximum(-d['delta_p_kw'],0).sum(axis=1)*.25,.2*d['accepted'])
        self.assertLessEqual(abs(d['delta_p_kw']).max(),.2)
        zero=code.sample_controls([[0,0,0],[0,0,0]],[[0,1],[0,1]],seed=8310)
        np.testing.assert_array_equal(zero['orientation'],d['orientation'])
        self.assertFalse(zero['accepted'].any())
        self.assertTrue(d['acceptance_before_code'])

    def test_budget_and_event_identity(self):
        with self.assertRaises(ValueError):code.sample_controls([[2,2],[2,2]],[[0,1],[0,1]],seed=1)
        d=code.sample_controls([[1,1,0],[1,1,0]],[[0,1],[0,1]],seed=8310)
        s=code.controls_as_samples(d,[0,1])
        self.assertEqual(s['delta_P_known_kw'].shape,(192,3))
        self.assertTrue((s['event_ids'][:96]==0).all())
        self.assertTrue((s['delta_P_known_kw'][:76]==0).all())
        with self.assertRaises(ValueError):code.controls_as_samples(d,[0,0])

    def test_observed_moments_are_same_raw_data_left_multiplication(self):
        observed,controls,_=fixture()
        result=code.observation_moments(observed,controls)
        w=controls['future_train']['code_weight_known_kw']/200.
        w=w/(w@w)
        for key in ('P','Q','Y'):
            np.testing.assert_allclose(result[key][0],w@observed['future_train'][key],atol=1e-14)
        self.assertEqual(result['whole_event_randomization_units'],2)
        self.assertTrue(result['no_new_observations'])
        self.assertFalse(result['statistical_coverage'])

    def test_no_response_moments_do_not_become_paid_information(self):
        d=code.sample_controls([[0,0,0],[0,0,0]],[[0,2],[0,2]],seed=8310)
        control=code.controls_as_samples(d,[0,1],samples_per_event=8,window_start=0)
        rng=np.random.default_rng(1)
        obs=dict(future_train={k:rng.normal(size=(16,3)) for k in ('P','Q','Y')})
        result=code.observation_moments(obs,dict(future_train=control))
        self.assertTrue(result['rows'][0]['known_control_first_stage_zero'])
        self.assertTrue(result['rows'][0]['includes_partial_and_nonparticipation'])
        np.testing.assert_array_equal(d['committed_payment'],0)

    def test_conditional_exact_positive_block(self):
        observed,controls,config=fixture()
        result=code.discover_candidates(observed,controls,config)
        self.assertIn([0,1],result['candidate_supports'])
        self.assertEqual(result['direct_family'],[[0],[1],[2],[0,1]])
        self.assertFalse(result['certificate'])
        self.assertLess(result['diagnostics'][0]['numeric_input_closure_error'],1e-10)
        np.testing.assert_allclose(result['diagnostics'][0]['residualized_contrast_Y'],[.15,.05,-.1],atol=1e-11)

    def test_zero_effect_is_unresolved_not_fake_negative_certificate(self):
        result=code.discover_candidates(*fixture(0.))
        self.assertIsNone(result['direct_family'])
        self.assertFalse(result['certificate'])
        self.assertEqual(result['diagnostics'][0]['status'],'unresolved_signed_membership')

    def test_noise_can_make_perfect_fake_code_correlation(self):
        observed,controls,config=fixture(0.)
        dp=controls['future_train']['delta_P_known_kw']/200.
        # A particular independently drawn nuisance realization can equal the
        # assigned code. The sign itself is not a topology certificate.
        observed['future_train']['Y'][:,:2]+=.04*dp[:,0,None]
        no_band=dict(config);no_band.pop('y_error_bound')
        no_band['band_fixed_before_codes']=False
        result=code.discover_candidates(observed,controls,no_band)
        self.assertEqual(result['candidate_supports'],[])
        self.assertGreater(result['diagnostics'][0]['raw_code_contrast_Y'][1],0)
        covered=dict(config,y_error_bound=.00004)
        result=code.discover_candidates(observed,controls,covered)
        self.assertIsNone(result['direct_family'])
        self.assertNotIn(1,result['diagnostics'][0]['signed_band']['sides'][0]['forced'])
        self.assertFalse(result['statistical_coverage'])

    def test_unknown_PQ_error_does_not_get_unbounded_gain_certificate(self):
        observed,controls,config=fixture()
        config['pq_observation_mode']='noisy_unknown_gain'
        result=code.discover_candidates(observed,controls,config)
        self.assertEqual(result['candidate_supports'],[])
        self.assertEqual(result['diagnostics'][0]['status'],'unresolved_observation_model')
        config['allow_unclosed_heuristic']=True
        result=code.discover_candidates(observed,controls,config)
        self.assertIn([0,1],result['candidate_supports'])
        self.assertFalse(result['certificate'])
        self.assertIsNone(result['direct_family'])

    def test_known_control_can_be_absorbed_by_remaining_observed_input(self):
        observed,controls,config=fixture()
        signal=controls['future_train']['delta_P_known_kw'][:,0]/200.
        observed['future_train']['P'][:,1]=signal
        observed['history_train']['P'][:,1]=0.
        result=code.discover_candidates(observed,controls,config)
        self.assertEqual(result['candidate_supports'],[])
        self.assertEqual(result['diagnostics'][0]['status'],'unresolved_control_absorbed_by_observed_nuisance')

    def test_validation_and_truth_are_not_inputs(self):
        observed,controls,config=fixture()
        class Forbidden:
            def __array__(self,*args,**kwargs):raise AssertionError('validation was read')
        observed['future_validation']=Forbidden();observed['history_validation']=Forbidden()
        result=code.discover_candidates(observed,controls,config)
        self.assertFalse(result['uses_validation'])
        observed['true_K']=np.eye(3)
        with self.assertRaises(ValueError):code.discover_candidates(observed,controls,config)


if __name__=='__main__':unittest.main()
