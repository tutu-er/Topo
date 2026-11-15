"""Meaningful physics checks, including state recursion independent of rows."""
from copy import deepcopy
import unittest

import numpy as np

from .domain import EquipmentCase
from .resources import build_spec, content_hash
from .test_resources import context


def make_case():
    labels = [101, 102, 103, 104]
    spec = build_spec(context(), labels, '2024-01-04')
    return EquipmentCase(spec, labels, np.full((96, 8), .01), 200.)


class DomainTest(unittest.TestCase):
    def test_zero_and_no_baseline_double_count(self):
        case = make_case(); zero = np.zeros(case.shape)
        self.assertTrue(case.audit(zero)['passed'])
        np.testing.assert_array_equal(case.shift_pq_pu(zero, 200.), np.zeros((96, 8)))
        with self.assertRaises(RuntimeError): case.device_baseline_pq_kw()
        decomposition = case.baseline_decomposition()
        p0 = np.asarray(decomposition['device_p_kw'])
        background = np.asarray(decomposition['residual_background_p_kw'])
        np.testing.assert_allclose(p0+background, case.forecast[:, :4]*200., rtol=0, atol=1e-15)

    def test_prelocked_battery_states_and_shared_response(self):
        case = make_case(); u = np.zeros(case.shape)
        device = next(d for d in case.devices if d['kind']=='battery')
        j, q = case.p_index[device['id']], case.q_index[device['id']]
        charge_h = int(np.flatnonzero(case.upper[:, j]>0)[0])
        discharge_h = int(np.flatnonzero(case.lower[:, j]<0)[0])
        amount = .1*device['power_kw']
        u[charge_h, j] = amount
        u[discharge_h, j] = -amount*device['charging_efficiency']*device['discharging_efficiency']
        u[charge_h, q] = .2*device['inverter_kva']
        self.assertTrue(case.audit(u)['passed'])
        e = device['initial_soc']*device['energy_kwh']; states=[e/device['energy_kwh']]
        for p in u[:, j]:
            e += device['charging_efficiency']*max(p,0)-max(-p,0)/device['discharging_efficiency']
            states.append(e/device['energy_kwh'])
        np.testing.assert_allclose(case.device_states(u)[device['id']]['soc'], states, atol=1e-15)
        delivered, receipt = case.execute(u, np.zeros((96, case.N)), np.full(case.P_R, .93))
        np.testing.assert_allclose(delivered, .93*u)
        self.assertTrue(receipt['resource_audit']['passed'])
        eta = np.ones(case.R); eta[q] = .9
        with self.assertRaises(ValueError): case.execute(u, np.zeros((96, case.N)), eta)
        wrong = u.copy(); wrong[charge_h,j] = -amount
        self.assertFalse(case.audit(wrong)['passed'])

    def test_inverter_circle_cannot_deliver_full_P_and_Q(self):
        case = make_case(); u = np.zeros(case.shape)
        d = next(d for d in case.devices if d['kind']=='battery')
        p, q = case.p_index[d['id']], case.q_index[d['id']]
        h = int(np.flatnonzero(case.upper[:,p]>0)[0])
        u[h,p] = d['power_kw']; u[h,q] = d['reactive_cap_kvar']
        result = case.audit(u)
        self.assertGreater(result['max_inverter_relative_violation'], .15)
        self.assertFalse(result['passed'])

    def test_PV_all_four_points_and_EV_no_export(self):
        case = make_case(); zero = np.zeros(case.shape)
        d = next(d for d in case.devices if d['kind']=='pv'); p=case.p_index[d['id']]
        gross=np.asarray(d['available_generation_kw']).reshape(24,4)
        np.testing.assert_array_equal(case.upper[:,p], gross.min(1))
        h=int(np.argmax(gross.max(1)-gross.min(1)))
        u=zero.copy();u[h,p]=gross[h].mean()
        self.assertFalse(case.audit(u)['passed'])
        ev=next(d for d in case.devices if d['kind']=='ev');j=case.p_index[ev['id']]
        self.assertTrue(np.all(np.asarray(ev['baseline_p_kw']).reshape(24,4)+case.lower[:,j,None]>=0))
        u=zero.copy();u[0,j]=-.01
        self.assertFalse(case.audit(u)['passed'])

    def test_heat_recursion_and_mapping_units(self):
        case=make_case();u=np.zeros(case.shape)
        d=next(d for d in case.devices if d['kind']=='thermal');j=case.p_index[d['id']]
        rho=d['hourly_temperature_retention'];a=.03*d['power_kw']
        u[1,j]=a;u[2,j]=-rho*a
        self.assertTrue(case.audit(u)['passed'])
        z=0.;states=[d['initial_temperature_C']]
        for shift in u[:,j]:
            z=rho*z+shift
            states.append(d['initial_temperature_C']+z/d['heat_capacity_kwh_per_C'])
        np.testing.assert_allclose(case.device_states(u)[d['id']]['temperature_C'],states,atol=1e-14)
        p,q=case.aggregate(case.delivery(u));node=case.labels.index(d['node_label'])
        np.testing.assert_allclose(p[:,node],np.repeat(u[:,j],4));self.assertTrue(np.all(q==0))
        self.assertAlmostEqual(p.sum()/4.,u[:,j].sum())

    def test_spec_tampering_rejected(self):
        c=make_case();bad=deepcopy(c.spec);bad['devices'][0]['power_kw']*=2
        with self.assertRaises(ValueError): EquipmentCase(bad,c.labels,c.forecast,200.)

    def test_attach_convex_qcp_matches_fixed_feasible_point(self):
        import gurobipy as gp
        case=make_case();u=np.zeros(case.shape)
        d=next(d for d in case.devices if d['kind']=='thermal');j=case.p_index[d['id']]
        u[1,j]=.01*d['power_kw'];u[2,j]=-d['hourly_temperature_retention']*u[1,j]
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model(env=env) as model:
                model.Params.TimeLimit=1.;model.Params.Threads=1
                action, fee=case.attach(model)
                model.addConstr(action==u);model.setObjective(fee)
                model.optimize()
                self.assertGreater(model.SolCount,0)
                self.assertAlmostEqual(model.ObjVal,case.fee(u),places=7)
                self.assertTrue(case.audit(np.asarray(action.X))['passed'])


if __name__=='__main__':unittest.main()
