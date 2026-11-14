"""Declared-noise algebra and backward-compatible solver wiring checks."""
import unittest
from unittest.mock import patch

import numpy as np

from ..distribution_gate_20261114.evidence import DistributionEvidence
from ..distribution_gate_20261114 import solver
from ..distribution_gate_20261114 import probability
from ..distribution_panel_20261115.audit import noise_parameters
from ..distribution_gate_20261114.test_evidence import toy, joint_log_mgf
from ..distribution_gate_20261114.test_solver import ToyCase as ScalarCase


def consistent_history(stride=1):
    case,score,val,forecast,rng=toy()
    score['labels']=list(val.labels)
    validation={k:getattr(val,v)[::stride].copy() for k,v in
        (('P','p'),('Q','q'),('Y','y'),('V','voltage'),('root','root'))}
    power=np.c_[validation['P'],validation['Q']]
    for row in score['native_bank']:
        row['history_sse']=float(np.square(validation['Y']-power@np.asarray(row['H']).T).sum())
        row['internal']=1
    for row in score['parameters']:
        row['history_sse']=score['native_bank'][row['index']]['history_sse']
    score['cells']=validation['Y'].size
    return case,score,val,validation,forecast,rng


class NoiseParameters(unittest.TestCase):
    def test_explicit_original_defaults_are_bitwise_identical(self):
        case, score, val, forecast, _ = toy()
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            implicit=DistributionEvidence(case,score,val,forecast,200.,mode=mode,target_mass=1.)
            explicit=DistributionEvidence(case,score,val,forecast,200.,mode=mode,target_mass=1.,
                meter_relative_sd=.0005,forecast_relative_sd=.05)
            for field in ('sigma_e','power_sd','means','mu_q','mu_a','mu_c',
                          'variance_q','variance_a','variance_c','gate_q','gate_a','gate_c'):
                np.testing.assert_array_equal(getattr(implicit,field),getattr(explicit,field))
            self.assertEqual(implicit.metadata['meter_relative_sd'],.0005)
            self.assertEqual(implicit.metadata['forecast_relative_sd'],.05)

    def test_meter_variance_scales_quadratically_without_changing_the_mean(self):
        case, score, val, forecast, rng = toy()
        u=rng.uniform(case.lower,case.upper)
        low=DistributionEvidence(case,score,val,forecast,200.,mode='normal_fixed',target_mass=1.,meter_relative_sd=.0005)
        high=DistributionEvidence(case,score,val,forecast,200.,mode='normal_fixed',target_mass=1.,meter_relative_sd=.001)
        a,b=low.direct(u),high.direct(u)
        np.testing.assert_array_equal(a['mu'],b['mu'])
        np.testing.assert_allclose(b['variance'],4*a['variance'],rtol=1e-14,atol=0)
        np.testing.assert_array_equal(high.sigma_e,4*low.sigma_e)
        power=forecast+case.shift_pq_pu(u,200.)
        for k,row in enumerate(high.rows):
            d=power@(high.H[row['target_rep']]-high.native_H[row['competitor_native']]).T
            expected=((2*.001*np.median(val.root)**2)**2*np.square(d.sum(1)).sum()
                      +(2*.001*np.median(val.voltage)**2)**2*np.square(d).sum())/high.sigma2**2
            self.assertAlmostEqual(float(b['variance'][k]),expected,places=12)

    def test_forecast_scale_only_changes_the_random_power_law(self):
        case,score,val,forecast,_=toy()
        a=DistributionEvidence(case,score,val,forecast,200.,forecast_relative_sd=.05)
        b=DistributionEvidence(case,score,val,forecast,200.,forecast_relative_sd=.10)
        np.testing.assert_array_equal(b.power_sd,2*a.power_sd)
        np.testing.assert_array_equal(b.sigma_p_diag,4*a.sigma_p_diag)
        for mode in ('normal_fixed','mgf_fixed'):
            a=DistributionEvidence(case,score,val,forecast,200.,mode=mode,forecast_relative_sd=.05)
            b=DistributionEvidence(case,score,val,forecast,200.,mode=mode,forecast_relative_sd=.20)
            for field in ('mu_q','variance_q','gate_q','gate_a','gate_c','power_sd'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))

    def test_nondefault_noise_matches_independent_joint_gaussian_integral(self):
        case,score,val,forecast,rng=toy()
        e=DistributionEvidence(case,score,val,forecast,200.,mode='mgf_forecast',target_mass=1.,
            meter_relative_sd=.0012,forecast_relative_sd=.12)
        u=rng.uniform(case.lower,case.upper);power=forecast+case.shift_pq_pu(u,200.);info=e.direct(u)
        for k,row in enumerate(e.rows):
            i,j,lam=row['target_rep'],row['competitor_native'],row['lambda_selected']
            expected=lam*(row['tie']-row['h'])
            for t in range(len(power)):
                expected+=joint_log_mgf(e.mean_matrices[k],e.H[i]-e.native_H[j],e.sigma2,
                    e.sigma_e,e.power_sd[t],power[t],lam)
            self.assertAlmostEqual(info['log_pair_failure_bound'][k],expected,places=11)

    def test_zero_variance_limits_and_invalid_values(self):
        case,score,val,forecast,_=toy()
        fixed=DistributionEvidence(case,score,val,forecast,200.,mode='mgf_fixed')
        degenerate=DistributionEvidence(case,score,val,forecast,200.,forecast_relative_sd=0.)
        for field in ('gate_q','gate_a','gate_c'):
            np.testing.assert_array_equal(getattr(fixed,field),getattr(degenerate,field))
        deterministic=DistributionEvidence(case,score,val,forecast,200.,mode='normal_fixed',meter_relative_sd=0.)
        info=deterministic.direct(np.zeros(case.shape))
        np.testing.assert_array_equal(info['variance'],0.)
        for name in ('meter_relative_sd','forecast_relative_sd'):
            for value in (-.01,np.inf,np.nan):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    DistributionEvidence(case,score,val,forecast,200.,**{name:value})

    def test_solver_forwards_values_and_records_them_without_optimization(self):
        received=[]
        class ZeroEvidence:
            targets=[0];rows=[];guards=[]
            def __init__(self,*args,**kwargs):received.append(kwargs)
            def verify_expansion(self,u):
                return dict(margins=np.array([]),pair_failure=np.array([]),guard_gains=np.array([]),feasible=True)
        with patch.object(solver,'DistributionEvidence',ZeroEvidence):
            quote,action=solver.plan(ScalarCase(),{},None,np.zeros((1,1)),1.,mode='mgf_forecast',
                seconds=1.,env=None,meter_relative_sd=.001,forecast_relative_sd=.1)
        self.assertEqual(received[0]['meter_relative_sd'],.001)
        self.assertEqual(received[0]['forecast_relative_sd'],.1)
        self.assertEqual(quote['meter_relative_sd'],.001)
        self.assertEqual(quote['forecast_relative_sd'],.1)
        self.assertEqual(quote['status'],'zero_sufficient')
        self.assertFalse(quote['optimizer_invoked'])
        np.testing.assert_array_equal(action,0.)

    def test_independent_model_defaults_and_history_lengths(self):
        for stride in (1,2,4):
            _,score,_,val,forecast,_=consistent_history(stride)
            a=probability.build_model(score,val)
            b=probability.build_model(score,val,meter_relative_sd=.0005,forecast_relative_sd=.05)
            self.assertEqual(a['validation_cells'],val['Y'].size)
            for field in ('H','means','amplitude','history','covariance'):
                np.testing.assert_array_equal(a[field],b[field])
            for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
                self.assertEqual(probability.exact_gates(a,forecast,forecast,mode),
                                 probability.exact_gates(b,forecast,forecast,mode))

    def test_independent_nondefault_gates_match_joint_integral_and_production(self):
        case,score,val,validation,forecast,rng=consistent_history()
        settings=dict(meter_relative_sd=.0012,forecast_relative_sd=.12)
        model=probability.build_model(score,validation,**settings)
        baseline=probability.build_model(score,validation)
        np.testing.assert_allclose(model['covariance'],baseline['covariance']*(.0012/.0005)**2,
                                   rtol=1e-15,atol=0)
        u=rng.uniform(case.lower,case.upper);power=forecast+case.shift_pq_pu(u,200.)
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            independently=probability.exact_gates(model,forecast,power,mode)
            production=DistributionEvidence(case,score,val,forecast,200.,mode=mode,**settings)
            direct=production.direct(u)
            self.assertEqual(len(independently['rows']),len(production.rows))
            for index,row in enumerate(independently['rows']):
                self.assertEqual(row['target_native'],production.rows[index]['target_native'])
                self.assertEqual(row['competitor_native'],production.rows[index]['competitor_native'])
                self.assertAlmostEqual(row['margin'],direct['margins'][index],places=10)
                self.assertAlmostEqual(row['fixed_input_variance'],direct['variance'][index],places=10)
                i,j=row['target_native'],row['competitor_native'];D=model['H'][i]-model['H'][j]
                E=model['means'][row['target']]-model['H'][i]
                B=(.5*D.T@D+.5*(E.T@D+D.T@E))/model['sigma2']
                lam=row['lambda_selected'];expected=lam*(independently['tie']-row['h'])
                sd=.12*abs(forecast) if mode=='mgf_forecast' else np.zeros_like(forecast)
                for t in range(len(power)):
                    expected+=joint_log_mgf(B,D,model['sigma2'],model['covariance'],sd[t],power[t],lam)
                self.assertAlmostEqual(row['log_failure_bound'],expected,places=11)

    def test_independent_zero_noise_and_invalid_values(self):
        _,score,_,validation,forecast,_=consistent_history()
        model=probability.build_model(score,validation,meter_relative_sd=0.,forecast_relative_sd=0.)
        np.testing.assert_array_equal(model['covariance'],0.)
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            result=probability.exact_gates(model,forecast,forecast,mode)
            self.assertTrue(all(row['variance']==0. for row in result['rows']))
        for name in ('meter_relative_sd','forecast_relative_sd'):
            for value in (-1.,np.inf,np.nan):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    probability.build_model(score,validation,**{name:value})

    def test_audit_noise_precedence_and_validation(self):
        self.assertEqual(noise_parameters({},{}),
            dict(pq_relative_sd=.005,meter_relative_sd=.0005,forecast_relative_sd=.05))
        p=dict(meter=dict(additional_PQ_relative_sd=.0025,additional_V_relative_sd=.00025),forecast_relative_sd=.03)
        self.assertEqual(noise_parameters({},p),
            dict(pq_relative_sd=.0025,meter_relative_sd=.00025,forecast_relative_sd=.03))
        entry=dict(meter=dict(additional_PQ_relative_sd=.01,additional_V_relative_sd=.001),forecast_relative_sd=.12)
        self.assertEqual(noise_parameters(entry,p),
            dict(pq_relative_sd=.01,meter_relative_sd=.001,forecast_relative_sd=.12))
        for entry in ({'forecast_relative_sd':np.nan},{'meter':{'additional_PQ_relative_sd':-1}}):
            with self.assertRaises(ValueError):noise_parameters(entry,p)


if __name__=='__main__':unittest.main()
