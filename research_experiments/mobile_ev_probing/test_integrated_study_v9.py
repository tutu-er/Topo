"""Checks the data/controller contract, independent of topology success counts."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import integrated_study_v9 as study


@pytest.mark.parametrize('n',[6,12])
@pytest.mark.parametrize('mode',['gain_dominated','jitter_dominated','precision_control'])
def test_paired_meter_truth_and_ev_constraints(n,mode):
    cfg=(n,False,'partial',mode,9811)
    base,truth=study.simulate(*cfg)
    ev,other=study.simulate(*cfg,'mobile',(0,n-1))
    h=truth['history_count']
    for key in base:
        np.testing.assert_array_equal(base[key][:h],ev[key][:h])
    for key in ['source_p','meter_gains','meter_jitter','Q']:
        np.testing.assert_array_equal(truth[key],other[key])
    increments,_=study.action(n,'mobile',(0,n-1))
    np.testing.assert_allclose(other['P'][h:]-truth['P'][h:],increments,atol=1e-16)
    for observed,actual in [(base,truth),(ev,other)]:
        assert np.all(abs(observed['P']-actual['P'])<=observed['P_error'])
        assert np.all(abs(observed['Q']-actual['Q'])<=observed['Q_error'])
        assert np.all(abs(observed['Y_drop']-actual['Y'])<=observed['drop_error_bound'])
        bound=actual['meter_settings']['total_bound']
        assert abs(actual['voltage_measured']-actual['voltage_terminal']).max()<=bound
        assert abs(actual['root_voltage_measured']-1).max()<=bound
    assert other['device']['energy_kwh']==27
    assert other['device']['travel_kwh']==2


def test_refresh_never_drops_old_unresolved(monkeypatch):
    obs,_=study.simulate(6,False,'partial','gain_dominated',9812)
    old_rows=[dict(id='old',supports=[[0,2]],rank=1,linear_contrast_mae=0.)]
    old=dict(n_terminal=6,candidates=old_rows)
    new=dict(n_terminal=6,candidates=[dict(id='new',supports=[[0,1]],rank=1,linear_contrast_mae=1.)])
    monkeypatch.setattr(study,'propose',lambda *args,**kwargs:new)
    result=study.append_and_refit(obs,old,dict(retained=['old'],records={'old':dict(status='unresolved')}),1,1.)
    assert {r['id'] for r in result['candidates']}=={'old','new'}
    assert result['retained_count']==2  # one new proposal cap does not drop old evidence


def test_refresh_respects_valid_same_domain_exclusion(monkeypatch):
    obs,_=study.simulate(6,False,'partial','gain_dominated',9813)
    old=dict(n_terminal=6,candidates=[dict(id='excluded',supports=[[0,2]],rank=1,linear_contrast_mae=0.)])
    new=dict(n_terminal=6,candidates=[dict(id='excluded',supports=[[0,2]],rank=1,linear_contrast_mae=0.),
                                    dict(id='new',supports=[[0,1]],rank=2,linear_contrast_mae=1.)])
    monkeypatch.setattr(study,'propose',lambda *args,**kwargs:new)
    result=study.append_and_refit(obs,old,dict(retained=[],records={'excluded':dict(status='excluded')}),1,1.)
    assert {r['id'] for r in result['candidates']}=={'new'}
