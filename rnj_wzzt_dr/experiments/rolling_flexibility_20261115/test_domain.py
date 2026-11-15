"""Physical recurrence, task inheritance and robust-first-step boundaries."""
import unittest
import numpy as np

from .domain import Fleet, DT
from ..equipment_flex_20261115.test_domain import make_case


def make_fleet(band=False):
    old=make_case()
    arrays=None if not band else dict(labels=np.array(old.labels),kappa=np.ones(old.N),
        band_kvar=np.zeros((24,old.N)),decision_hours=(8+np.arange(24))%24)
    return Fleet(old.spec,old.labels,old.base_kw,arrays)


class DomainTests(unittest.TestCase):
    def test_quarter_hour_baseline_and_units(self):
        f=make_fleet();self.assertEqual(f.shape,(96,f.R))
        self.assertTrue(f.audit_prefix(np.zeros(f.shape))['passed'])
        pv=next(d for d in f.devices if d['kind']=='pv');j=f.p_index[pv['id']]
        np.testing.assert_array_equal(f.upper[:,j],pv['available_generation_kw'])
        self.assertGreater(np.max(np.ptp(f.upper[:,j].reshape(24,4),axis=1)),0)
        u=np.zeros((1,f.R));u[0,j]=.2
        self.assertAlmostEqual(f.activation_fee(u,10),DT*.2*f.fee_plus[10,j])

    def test_thermal_four_steps_exactly_match_hour_model(self):
        f=make_fleet();device=next(d for d in f.devices if d['kind']=='thermal')
        j=f.p_index[device['id']];amount=.01*device['power_kw']
        prefix=np.zeros((4,f.R));prefix[:,j]=amount
        state=f.audit_prefix(prefix)['states'][device['id']]['current']
        expected=device['initial_temperature_C']+amount/device['heat_capacity_kwh_per_C']
        self.assertAlmostEqual(state,expected,places=11)
        self.assertAlmostEqual(f.states[device['id']]['rho_15min']**4,device['hourly_temperature_retention'],places=14)

    def test_mapping_gain_and_no_duplicate_baseline(self):
        f=make_fleet();case=f.remaining(np.zeros((0,f.R)),np.zeros(f.shape),np.full(f.P_R,.94))
        u=np.zeros(case.shape);u[2,0]=.1
        np.testing.assert_allclose(case.shift_pq_pu(u,f.base_kw),u@case.response_effect(np.eye(2*f.N),f.base_kw).T)
        np.testing.assert_array_equal(case.shift_pq_pu(np.zeros(case.shape),f.base_kw),np.zeros((96,2*f.N)))
        np.testing.assert_allclose(case.total_delta(u),.94*u)
        np.testing.assert_allclose(case.request_delta(u),u)

    def test_caution_limits_only_the_next_executed_step(self):
        f=make_fleet();prefix=np.zeros((8,f.R));service=np.zeros((88,f.R))
        original=f.remaining(prefix,service,np.ones(f.P_R))
        cautious=f.remaining(prefix,service,np.ones(f.P_R),caution_fraction=.25)
        np.testing.assert_allclose(cautious.lower[0],.25*original.lower[0])
        np.testing.assert_allclose(cautious.upper[0],.25*original.upper[0])
        np.testing.assert_array_equal(cautious.lower[1:],original.lower[1:])
        np.testing.assert_array_equal(cautious.upper[1:],original.upper[1:])

    def test_delayed_ev_first_step_gain_endpoint_rejects_overdelivery(self):
        f=make_fleet();d=next(d for d in f.devices if d['kind']=='ev');j=f.p_index[d['id']]
        prefix=np.zeros((90,f.R));prefix[56,j]=-.5*d['power_kw']
        service=np.zeros((6,f.R));service[1,j]=.5*d['power_kw']
        case=f.remaining(prefix,service,np.full(f.P_R,.9))
        u=np.zeros(case.shape);u[0,j]=.5*d['power_kw']/.9;u[1,j]=-u[0,j]
        self.assertTrue(f.audit_prefix(np.vstack((prefix,case.total_delta(u))))['passed'])
        checked=case.audit(u)
        self.assertFalse(checked['first_step_robust_reachable'])
        self.assertFalse(checked['passed'])
        self.assertFalse(checked['first_step_endpoints'][1]['remaining_reachable'])
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model(env=env) as model:
                model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.TimeLimit=2.
                action,fee=case.attach(model);model.addConstr(action==u);model.setObjective(fee)
                model.optimize()
                self.assertEqual(model.Status,gp.GRB.INFEASIBLE)
                self.assertEqual(model.SolCount,0)

    def test_unknown_eta_circle_is_checked_at_both_endpoints(self):
        f=make_fleet();d=next(d for d in f.devices if d['kind']=='pv');p=f.p_index[d['id']];q=f.q_index[d['id']]
        k=int(np.argmax(d['available_generation_kw']))
        case=f.remaining(np.zeros((k,f.R)),np.zeros((96-k,f.R)),np.full(f.P_R,.9))
        head=np.sqrt(d['inverter_kva']**2-f.original_baseline[k,p]**2)
        u=np.zeros(case.shape);u[0,q]=.99*head/.9
        self.assertLess(u[0,q],d['reactive_cap_kvar'])
        self.assertTrue(f.audit_prefix(np.vstack((case.tail['prefix'],case.total_delta(u))))['passed'])
        self.assertFalse(case.audit(u)['first_step_endpoints'][1]['physical_passed'])

    def test_original_terminal_not_reset_after_battery_charge(self):
        f=make_fleet();d=next(d for d in f.devices if d['kind']=='battery');j=f.p_index[d['id']]
        prefix=np.zeros((9,f.R));prefix[8,j]=.1*d['power_kw']
        state=f.audit_prefix(prefix)['states'][d['id']]
        self.assertGreater(state['current'],state['target'])
        zero=np.zeros((87,f.R))
        self.assertFalse(f.audit_prefix(np.vstack((prefix,zero)))['passed'])
        service=zero.copy();service[40,j]=-.1*d['power_kw']*d['charging_efficiency']*d['discharging_efficiency']
        self.assertTrue(f.audit_prefix(np.vstack((prefix,service)))['passed'])
        case=f.remaining(prefix,service,np.ones(f.P_R))
        self.assertTrue(case.audit(np.zeros(case.shape))['passed'])

    def test_service_qcp_repairs_state_and_reports_band_conflict(self):
        import gurobipy as gp
        f=make_fleet(band=True);d=next(d for d in f.devices if d['kind']=='battery');j=f.p_index[d['id']]
        prefix=np.zeros((9,f.R));prefix[8,j]=.1*d['power_kw']
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            point,receipt=f.service_plan(prefix,env=env,seconds=3.)
        self.assertIsNotNone(point);self.assertTrue(receipt['physical_audit']['passed'])
        self.assertFalse(receipt['band']['passed'])
        self.assertGreater(receipt['cost_yuan'],0)

    def test_unreachable_service_does_not_return_zero(self):
        f=make_fleet();d=next(d for d in f.devices if d['kind']=='ev');j=f.p_index[d['id']]
        prefix=np.zeros((92,f.R));prefix[56,j]=-.1*d['power_kw']
        point,receipt=f.service_plan(prefix,env=None,seconds=1.)
        self.assertIsNone(point);self.assertFalse(receipt['feasible'])
        self.assertEqual(receipt['status'],'prefix_infeasible_or_unreachable')

    def test_attach_matches_nominal_and_robust_fixed_point(self):
        import gurobipy as gp
        f=make_fleet();k=94;prefix=np.zeros((k,f.R));case=f.remaining(prefix,np.zeros((2,f.R)),np.ones(f.P_R))
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model(env=env) as model:
                model.Params.OutputFlag=0;model.Params.TimeLimit=2.;model.Params.Threads=1
                u,fee=case.attach(model);model.addConstr(u==np.zeros(case.shape));model.setObjective(fee);model.optimize()
                self.assertGreater(model.SolCount,0)
                self.assertTrue(case.audit(np.asarray(u.X))['passed'])
                self.assertAlmostEqual(model.ObjVal,case.fee(np.asarray(u.X)),places=7)


if __name__=='__main__':unittest.main()
