"""Fault-injection tests: preserve physical service and pre-dispatch ordering."""
from types import SimpleNamespace
import numpy as np
import pytest
from ... import io
from ...observations import Observation
from .state import FrozenBank
from .config import Config
from . import run


def setup(monkeypatch,tmp_path):
    fleet=SimpleNamespace(N=1,R=1,P_R=1,labels=(1,),base_kw=1.,B=np.array([[1.],[0.]]),
        device_ids=['d'],resources=[dict(physical_device_id='d',contract_kw=1.)],
        spec={'pricing':{'grid_buy_yuan_per_kwh':{'day':.5,'night':.3,'evening':.8}}})
    fleet.service_plan=lambda prefix,**kwargs:(np.zeros((96-len(prefix),1)),dict(feasible=True,status='firm_service'))
    fleet.audit_prefix=lambda prefix:dict(passed=True)
    fleet.activation_fee=lambda delta,start=0:float(.25*np.abs(delta).sum())
    def remaining(prefix,service,gains,**kwargs):
        return SimpleNamespace(shape=service.shape,service_delta=service,
            request_delta=lambda u:service+u,total_delta=lambda u:service+gains*u)
    fleet.remaining=remaining
    bank=FrozenBank(H=np.zeros((1,1,2)),means=np.zeros((1,1,2)),initial_sse=np.array([.1]),
        topology_ids=('g',),internal=np.array([0]),sigma2=.001,labels=(1,),reference_mse=.1,tie=1e-12,
        initial_cells=96,root_median=1.,voltage_median=1.,fingerprint='fault_test')
    monkeypatch.setattr(run,'load_planning',lambda entry:({},np.zeros((96,2)),fleet,bank))
    def broken(*a,**kw):raise ArithmeticError('injected numerical quote failure')
    monkeypatch.setattr(run,'evidence',broken)
    class Evaluation:
        def __init__(self,entry,fleet,repeat):self.k=0
        def save_randomness(self,path):
            np.savez(path,eta=np.array([.93]));return {'randomness':io.ref(path)}
        def step(self,k,service,extra):
            # Future data is not returned before this same-step request exists.
            seals=list(tmp_path.rglob(f'step{k:03d}/seal.json'))
            assert len(seals)==1
            seal=io.read(seals[0]);assert seal['information_cutoff_step']==k-1
            assert seal['snapshot']['step']==k
            assert k==self.k;self.k+=1
            obs=Observation(np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1)),(1,),np.ones((1,1)),np.ones(1))
            return dict(observation=obs,total_actual_delta=service+.93*extra,extra_actual=.93*extra,
                clean_P=obs.p,clean_Q=obs.q,clean_V=obs.voltage,clean_root=obs.root,
                base_P=obs.p,base_Q=obs.q,ac={'status':'reused_sealed_zero_row'},
                ac_state={'voltage_magnitude_pu':np.ones((1,1))},root_kw=0.,minimum_voltage=1.,maximum_voltage=1.)
        def final_evaluation(self,obs):
            assert self.k==96 and len(obs.p)==96
            return dict(native_selected_index=0,native_map_correct=True,native_full_support_exact=True,
                truth_family_id='g',historical_target_correct=True)
    monkeypatch.setattr(run,'Evaluator',Evaluation)
    return dict(id='fault_test',case='soumalas11',history_repeat=0,pq_relative_sd=.005,
        meter={'additional_PQ_relative_sd':.005,'additional_V_relative_sd':.0005}),fleet


@pytest.mark.parametrize('policy',['no_dfs','rolling_mgf_fixed'])
def test_all_96_physical_steps_survive_quote_failure_and_are_sealed_first(monkeypatch,tmp_path,policy):
    entry,fleet=setup(monkeypatch,tmp_path)
    result=run.run_trajectory(entry,policy,0,tmp_path,Config(),None)
    assert result['status']=='complete' and result['completed_steps']==96
    assert result['quote_exceptions']==(96 if policy.startswith('rolling') else 0)
    assert result['nonzero_extra_rounds']==0 and result['service_yuan']==0.
    assert result['evaluation']['profile_true_weight']==1.
    assert result['evaluation']['native_selected_index']==0
