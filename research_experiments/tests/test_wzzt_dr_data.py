import numpy as np
import pandas as pd
import pytest
from research_experiments.wzzt_dr_selection.end_to_end.data_inputs import (
    MENUS,_daily_matrices,_ratio_trajectory,split_response_banks,
)

def test_control_adjustment_uses_full_hour_shape_and_no_recentring():
    historical=np.ones((5,2,24))*2
    actual=np.ones((2,24))*2
    actual[:,17]=4
    control=np.ones((2,24))*4
    factors=_ratio_trajectory(actual,historical,control,historical,{'before':1,'event':1,'after':1})
    assert factors[17]==1
    assert np.all(factors[np.arange(24)!=17]==.5)
    assert factors.mean()!=1

def test_future_response_bank_never_splits_a_source_date():
    records=[]
    for menu in MENUS:
        for day in ['2021-02-09','2021-02-11']:
            for group in ['A','B']:
                records.append({'menu':menu,'date':day,'group':group,'factors':np.arange(24)/100+1})
    bank,meta=split_response_banks(records)
    for menu in MENUS:
        assert bank['action'][menu].shape==(2,24,1)
        assert {x['date'] for x in meta['action'][menu]}.isdisjoint({x['date'] for x in meta['validation'][menu]})
        np.testing.assert_array_equal(bank['action'][menu][0,:,0],np.arange(24)/100+1)

def test_missing_future_menu_date_fails_not_filled_or_resampled():
    rows=[{'menu':m,'date':'2021-02-09','group':'A','factors':np.ones(24)} for m in MENUS]
    with pytest.raises(ValueError,match='>=2'):split_response_banks(rows)

def test_hourly_energy_keeps_scale_and_missing_meter_not_silently_dropped():
    frame=pd.DataFrame([{'ID':i,'Date':'2021-02-09','Hour':h,'Demand_kWh':v} for i,v in [('a',2),('b',3)] for h in range(1,25)])
    dates,cube=_daily_matrices(frame,['b','a'])
    assert cube.shape==(1,2,24)
    np.testing.assert_array_equal(cube.sum(axis=1),np.full((1,24),5))
    with pytest.raises(ValueError,match='incomplete'):_daily_matrices(frame.iloc[1:],['a','b'])
