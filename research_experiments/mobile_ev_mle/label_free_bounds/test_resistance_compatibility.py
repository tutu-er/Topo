"""Independent chain-rule, domain and feasible-point tests for direct R floors."""
import unittest
import json
from pathlib import Path
import tempfile
from dataclasses import asdict

import numpy as np

from .bounds import Measurements, PhysicalDomain, TreeSpec
from .resistance_compatibility import (
    ac_voltage_resistance, fit_resistance_compatible_point, project_qp_start,
    resistance_limits, resistance_to_polar,
    run_pilot,
)
from .compatibility import physical_checks


class ResistanceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tree = TreeSpec(root=0, edges=((0, 1), (1, 2), (1, 3)), observed_nodes=(1, 2, 3))
        self.p = np.array([[.05, .08, .06], [.07, .03, .10], [.08, .06, .04], [.04, .09, .07]])
        self.q = self.p*np.array([.4, .3, .5])
        self.root = np.array([1., 1.005, .995, 1.])
        self.domain = PhysicalDomain(z_min=.01, z_max=.03, r_min=.006,
            v_min=.8, v_max=1.1, power_caps=.3, ratio_min=.25, ratio_max=4.)
        # The first edge is z_min-dominated; the last is r_min-dominated.
        self.x = np.r_[[.25, .4, .7], [.5, 1., 2.]]

    def observations(self, voltage):
        return Measurements(p=self.p, q=self.q, v=voltage, root_v=self.root,
                            sigma_p=.001, sigma_q=.001, sigma_v=.0005)

    def test_chain_against_independent_finite_differences_both_max_branches(self):
        polar, chain = resistance_to_polar(self.x, self.domain, derivatives=True)
        finite = np.empty_like(chain)
        for j in range(len(self.x)):
            step = np.zeros_like(self.x); step[j] = 1e-6
            finite[:, j] = (resistance_to_polar(self.x+step, self.domain)
                            -resistance_to_polar(self.x-step, self.domain))/(2e-6)
        np.testing.assert_allclose(chain, finite, rtol=2e-7, atol=2e-9)
        self.assertTrue(np.isfinite(polar).all())

    def test_joining_surface_has_declared_one_sided_derivative(self):
        # At k*=sqrt((z_min/r_min)^2-1), the r_min branch is active for k>k*.
        kink = np.sqrt((.01/.006)**2-1)
        x = self.x.copy(); x[3:] = kink
        polar, chain = resistance_to_polar(x, self.domain, derivatives=True)
        h = 1e-7
        for j in range(3, 6):
            step = np.zeros_like(x); step[j] = h
            forward = (resistance_to_polar(x+step, self.domain)-polar)/h
            np.testing.assert_allclose(chain[:, j], forward, rtol=1e-6, atol=1e-7)
        # Both sides away from the kink also agree with ordinary central differences.
        for offset in [-1e-3, 1e-3]:
            y=x.copy(); y[3:]+=offset
            _, derivative=resistance_to_polar(y,self.domain,derivatives=True)
            for j in range(3, 6):
                step=np.zeros_like(y);step[j]=1e-7
                central=(resistance_to_polar(y+step,self.domain)-resistance_to_polar(y-step,self.domain))/(2e-7)
                np.testing.assert_allclose(derivative[:,j],central,rtol=2e-6,atol=2e-8)

    def test_exact_AC_chain_jacobian(self):
        point = ac_voltage_resistance(self.tree, self.p, self.q, self.root,
                                      self.x, self.domain, derivatives=True)
        finite = np.empty_like(point["magnitude_jacobian"])
        for j in range(len(self.x)):
            step = np.zeros_like(self.x); step[j] = 2e-6
            plus = ac_voltage_resistance(self.tree,self.p,self.q,self.root,self.x+step,self.domain)
            minus = ac_voltage_resistance(self.tree,self.p,self.q,self.root,self.x-step,self.domain)
            finite[:,:,j]=(plus["magnitude"]-minus["magnitude"])/(4e-6)
        np.testing.assert_allclose(point["magnitude_jacobian"],finite,rtol=8e-5,atol=8e-11)

    def test_full_parameter_box_stays_in_physical_domain_and_projection(self):
        lower,upper=resistance_limits(self.tree,self.domain)
        rng=np.random.default_rng(20261018)
        for fraction in [np.zeros(6),np.ones(6),*rng.uniform(size=(40,6))]:
            x=lower+fraction*(upper-lower)
            polar=resistance_to_polar(x,self.domain)
            z=np.exp(polar[:3]+1j*polar[3:])
            self.assertTrue(np.all(z.real>=self.domain.r_min-1e-12))
            self.assertTrue(np.all(abs(z)>=self.domain.z_min-1e-12))
            self.assertTrue(np.all(abs(z)<=self.domain.z_max+1e-12))
            self.assertTrue(np.all(z.imag/z.real>=self.domain.ratio_min-1e-12))
            self.assertTrue(np.all(z.imag/z.real<=self.domain.ratio_max+1e-12))
        projected=project_qp_start(self.tree,self.domain,np.r_[[-1.,.001,.3],[.3,-1.,.001]])
        self.assertTrue(np.all(projected>=lower))
        self.assertTrue(np.all(projected<=upper))

    def test_ratio_cap_from_resistance_floor_and_empty_domain(self):
        domain=PhysicalDomain(z_min=.01,z_max=.03,r_min=.02,v_min=.8,v_max=1.1,
                              power_caps=.3,ratio_min=.25,ratio_max=4.)
        _,upper=resistance_limits(self.tree,domain)
        np.testing.assert_allclose(upper[3:],np.sqrt((.03/.02)**2-1))
        bad=PhysicalDomain(z_min=.01,z_max=.03,r_min=.029,v_min=.8,v_max=1.1,
                           power_caps=.3,ratio_min=1.,ratio_max=4.)
        with self.assertRaisesRegex(ValueError,"empty or zero-width"):
            resistance_limits(self.tree,bad)

    def test_legal_compatible_upper_and_large_upper_never_reject(self):
        point=ac_voltage_resistance(self.tree,self.p,self.q,self.root,self.x,self.domain)
        theta=np.r_[point["impedance"].real,point["impedance"].imag]
        result=fit_resistance_compatible_point(self.tree,self.observations(point["magnitude"]),
                                              self.domain,theta,max_nfev=2)
        self.assertTrue(result["ac_noise_ball_feasible"])
        self.assertTrue(all(result["best_point"]["checks"].values()))
        self.assertEqual(len(result["starts"]),2)
        self.assertFalse(result["reject"])
        failed=fit_resistance_compatible_point(self.tree,self.observations(np.full_like(self.p,.90)),
                                              self.domain,theta,max_nfev=1,early_stop=False)
        self.assertTrue(failed["available"])
        self.assertFalse(failed["ac_noise_ball_feasible"])
        self.assertFalse(failed["reject"])
        self.assertTrue(all(r["evaluation_limit_reached"] for r in failed["starts"]))

    def test_shared_physical_checker_also_enforces_new_resistance_floor(self):
        point=ac_voltage_resistance(self.tree,self.p,self.q,self.root,self.x,self.domain)
        stronger=PhysicalDomain(z_min=.01,z_max=.03,r_min=.025,v_min=.8,v_max=1.1,
            power_caps=.3,ratio_min=.25,ratio_max=4.)
        checked=physical_checks(self.tree,self.observations(point['magnitude']),stronger,point)
        self.assertFalse(checked['resistance_floor'])

    def test_unmetered_source_is_explicitly_unsupported(self):
        tree=TreeSpec(root=0,edges=self.tree.edges,observed_nodes=self.tree.observed_nodes,source_node=1)
        result=fit_resistance_compatible_point(tree,self.observations(np.ones_like(self.p)),
                                              self.domain,np.ones(6))
        self.assertEqual(result["status"],"unsupported_source")
        self.assertFalse(result["available"])
        self.assertFalse(result["reject"])

    def test_pilot_accepts_alpha_stored_in_candidate_certificate(self):
        point=ac_voltage_resistance(self.tree,self.p,self.q,self.root,self.x,self.domain)
        theta=np.r_[point["impedance"].real,point["impedance"].imag].tolist()
        domain=asdict(self.domain)
        domain['power_caps']={str(i):.3 for i in self.tree.observed_nodes}
        meters=dict(id='unit',source_mode='none',source_node=1,
            observed_nodes=list(self.tree.observed_nodes),p=self.p.tolist(),q=self.q.tolist(),
            v=point['magnitude'].tolist(),root_v=self.root.tolist(),
            sigma_p=.001,sigma_q=.001,sigma_v=.0005,sigma_root=None)
        assets=dict(domain=domain,observed_nodes=list(self.tree.observed_nodes),
            candidates=[dict(id='c00',root=0,source_node=1,
                edges=self.tree.edges,observed_nodes=self.tree.observed_nodes)])
        qp=dict(history_id='unit',candidate_results=[dict(candidate_id='c00',success=True,
            result=dict(alpha=.05,theta=theta,ac_sse_lower=0.))])
        scratch=Path(__file__).resolve().parents[3]/'outputs'/'label_free_test_tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='topo_rmin_test_',dir=scratch) as directory:
            root=Path(directory).resolve()
            self.assertTrue(root.is_relative_to(scratch.resolve()))
            for name,record in [('meters',meters),('assets',assets),('qp',qp)]:
                (root/(name+'.json')).write_text(json.dumps(record),encoding='utf-8')
            result=run_pilot(root/'meters.json',root/'assets.json',root/'qp.json',root/'out',max_nfev=2)
            self.assertEqual(result['compatible_candidates'],['c00'])


if __name__ == "__main__":
    unittest.main()
