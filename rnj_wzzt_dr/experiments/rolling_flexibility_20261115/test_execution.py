from types import SimpleNamespace
import numpy as np
from .execution import DeliveryEstimator, CostLedger


def fleet():
    return SimpleNamespace(R=2,P_R=1,device_ids=['d'],resources=[
        dict(physical_device_id='d',contract_kw=2.),dict(physical_device_id='d',capacity_kvar=.8)],
        spec={'pricing':{'grid_buy_yuan_per_kwh':{'day':.5,'night':.3,'evening':.8}}},
        activation_fee=lambda x,start=0: .25*float(np.abs(x).sum()))


def test_unrequested_device_keeps_prior_then_uses_only_measured_delivery():
    x=DeliveryEstimator(fleet())
    assert np.array_equal(x.update([0.,0.],[0.,0.]),[.95,.95])
    assert np.allclose(x.update([0.,.4],[0.,.368]),[.92,.92])


def test_daily_capacity_peak_not_recharged_and_recovery_is_not_free():
    x=CostLedger(fleet())
    a=x.settle(0,np.array([[2.,.4]]),np.array([[1.,.2]]),1.)
    b=x.settle(1,np.array([[1.,.2]]),np.array([[-1.,0.]]),-1.)
    assert np.isclose(a['availability_increment_yuan'],1.2)
    assert b['availability_increment_yuan']==0.
    assert np.isclose(b['activation_yuan'],.25)
    assert b['root_energy_yuan']<0


def test_capacity_reserves_firm_service_even_when_extra_cancels_net_request():
    x=CostLedger(fleet())
    r=x.settle(0,np.zeros((1,2)),np.array([[.1,.02]]),0.,service_tail=np.array([[1.,.2]]))
    assert np.allclose(r['paid_capacity'],[1.,.2])
    assert np.isclose(r['availability_increment_yuan'],.6)
    assert r['activation_yuan']>0.
