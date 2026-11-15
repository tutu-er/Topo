import numpy as np
import pytest
from .solve import plan,expression


class Case:
    H=R=1;shape=(1,1)
    lower=np.zeros(shape);upper=np.ones(shape);service_delta=np.zeros(shape)
    def array(self,u):return np.asarray(u,float).reshape(self.shape)
    def request_delta(self,u):return self.array(u)
    def fee(self,u):return float(self.array(u).sum())
    def audit(self,u):
        u=self.array(u)
        return {'passed':bool(np.all(u>=-1e-8) and np.all(u<=1+1e-8))}
    def attach(self,m):
        u=m.addMVar(self.shape,lb=0.,ub=1.)
        return u,u.sum()


class Evidence:
    mode='mgf_fixed';rows=[dict(alpha=.1,z=1.)]
    gate_q=np.ones((1,1,1,1));gate_a=np.zeros((1,1,1));gate_c=np.array([-.25])
    guards=[(0,1,0.,np.array([[.1]]),np.zeros((1,1)))]
    def verify_expansion(self,u):
        v=float(np.asarray(u)[0,0]);margin=v*v+float(self.gate_c[0])
        return dict(feasible=margin>=-1e-7,margins=[margin],guard_gains=[.1*v*v])


@pytest.fixture
def env():
    import gurobipy as gp
    with gp.Env(empty=True) as e:
        e.setParam('OutputFlag',0);e.start();yield e


def test_native_finds_known_nonconvex_optimum_with_nonempty_constant_guard(env):
    point,receipt=plan(Case(),Evidence(),env=env,seconds=3.,paid_capacity=np.zeros(1),capacity_prices=np.zeros(1))
    assert receipt['feasible'] and receipt['source']=='native_found'
    assert np.isclose(point[0,0],.5,atol=1e-5)
    assert receipt['initial'][0]['feasible'] is False


def test_solver_actually_improves_a_feasible_warm_start(env):
    point,receipt=plan(Case(),Evidence(),env=env,seconds=3.,paid_capacity=np.zeros(1),capacity_prices=np.zeros(1),warm=np.array([[.8]]))
    assert receipt['source']=='native_improved_initial'
    assert point[0,0]<.6


def test_original_zero_proof_does_not_claim_encoded_buffer(env):
    e=Evidence();e.gate_c=np.array([5e-9])
    p,r=plan(Case(),e,env=env,seconds=1.,paid_capacity=np.zeros(1),capacity_prices=np.zeros(1))
    assert r['global_optimality_verified'] and not r['audit']['encoded_passed']
    assert 'original direct domain' in r['bound_scope']


def test_constant_quadratic_guard_sums_over_time_not_resources():
    u=np.array([[1.,2.],[3.,4.],[5.,6.]])
    q=np.eye(2);a=np.zeros_like(u)
    assert expression(u,q,a,0.)==float(np.square(u).sum())
