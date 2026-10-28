"""Independent numerical and observation-boundary checks for the DR pool bridge."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario
from research_experiments.wzzt_dr_selection.dr_pool_v2 import adapter
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays, normalize, matrices
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import select


def scenario(day, rows=8, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range(day, periods=rows, freq="h")
    labels = [101,102,103]
    p = pd.DataFrame(rng.uniform(.01,.08,(rows,3)),index=index,columns=labels)
    q = pd.DataFrame(rng.uniform(.005,.03,(rows,3)),index=index,columns=labels)
    r,x = matrices(3,[(0,),(1,),(2,),(0,1)], [.08,.07,.09,.05], [.025,.03,.035,.02])
    y = p.to_numpy()@r+q.to_numpy()@x
    root = pd.Series(np.ones(rows),index=index)
    v = pd.DataFrame(np.sqrt(1-y),index=index,columns=labels)
    return make_observed_scenario(p,q,v,root,str(day))


def test_four_real_days_with_unequal_samples_are_centered_separately():
    scenes=[scenario(f"2021-02-{day:02}",rows,day) for day,rows in zip((3,4,5,9),(4,6,8,10))]
    obs,days,counts=normalize(scenes,"training")
    assert list(counts.values())==[4,6,8,10]
    centered=arrays(obs,"demean",day_ids=days)
    for day in dict.fromkeys(days):
        rows=np.asarray(days)==day
        for array in centered:
            np.testing.assert_allclose(array[rows].mean(axis=0),0.,atol=1e-16)
    for a,b in zip(arrays(obs),[np.concatenate([s['P_terminal'].to_numpy() for s in scenes]),
                               np.concatenate([s['Q_terminal'].to_numpy() for s in scenes]),
                               np.concatenate([s['drop_target'].to_numpy() for s in scenes])]):
        np.testing.assert_array_equal(a,b)


def test_columns_and_time_alignment_are_rejected_before_output(tmp_path):
    train,valid=scenario("2021-02-03"),scenario("2021-02-10")
    valid['Q_terminal']=valid['Q_terminal'][[103,102,101]]
    with pytest.raises(ValueError,match="columns and time order"):
        adapter.fit_observed([train],[valid],tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    valid=scenario("2021-02-10")
    valid['V_root_observed'].index=valid['V_root_observed'].index+pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match="exactly the P time index"):
        adapter.fit_observed([train],[valid],tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_overlapping_training_validation_and_single_point_days_rejected(tmp_path):
    s=scenario("2021-02-03")
    with pytest.raises(ValueError,match="disjoint"):
        adapter.fit_observed([s],[s],tmp_path/'bad')
    with pytest.raises(ValueError,match="at least two"):
        normalize([scenario("2021-02-03",1)],"training")


def test_physical_day_ids_must_not_override_timestamp_dates():
    s=scenario("2021-02-03",4)
    s['day_ids']=['a','a','b','b']
    with pytest.raises(ValueError,match="disagree"):
        normalize([s],"training")


def test_fixed_family_lp_matches_independent_l1_median():
    index=pd.date_range('2021-02-03',periods=3,freq='h')
    p=pd.DataFrame(np.ones((3,1)),index=index,columns=[101]);q=p*0
    target=np.array([.1,.4,.9]);root=pd.Series(np.ones(3),index=index)
    v=pd.DataFrame(np.sqrt(1-target)[:,None],index=index,columns=[101])
    lp=PoolLP((p,q,v,root),[(0,)],bound=2.)
    try:
        point=lp.fit([(0,)])
        assert point['r'][0]==pytest.approx(.4,abs=1e-9)
        assert point['objective']==pytest.approx((.3+.5)/3,abs=1e-9)
        assert lp.solves[0]['status']==2 and lp.solves[0]['accepted']
        assert lp.solves[0]['max_constraint_violation']<1e-7
    finally:
        lp.close()


def test_selection_replays_original_saved_benchmark_without_refitting():
    fixture=json.loads((Path(adapter.__file__).parent/'selection_fixture.json').read_text(encoding='utf-8'))
    result=select(fixture['points'],fixture['calibration'])
    assert result['supports']==fixture['expected']['supports']
    np.testing.assert_allclose(result['r'],fixture['expected']['r'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(result['x'],fixture['expected']['x'],atol=1e-12,rtol=0)
    assert result['selection_validation_mae']==result['selected_uncontracted']['validation_mae']


def test_small_standalone_adapter_returns_three_models_and_resumes_without_refit(tmp_path,monkeypatch):
    train=[scenario('2021-02-03',8,1),scenario('2021-02-04',8,2)]
    valid=[scenario('2021-02-10',8,3)]
    result=adapter.fit_observed(train,valid,tmp_path,search_seconds=.5)
    assert result['status']=='complete'
    assert set(result['models'])=={'rnj_bank','baseline25','multistart_original_rule'}
    assert result['generated_tree_count']==60
    assert result['training_rows']==16 and list(result['training_day_counts'].values())==[8,8]
    assert len(result['pool_support_labels'])==result['pool_size']
    assert all(s['status']==2 for s in result['LP_solves'])
    for point in result['models'].values():
        assert len(point['r_matrix'])==3
        assert point['train_mae']>=0
        assert 'selected_uncontracted' in point
    assert not (tmp_path/'fit.lock').exists()
    monkeypatch.setattr(adapter,'build_bank',lambda *a,**kw:pytest.fail('resume called a new fit'))
    repeated=adapter.fit_observed(train,valid,tmp_path,search_seconds=.5,resume=True)
    assert repeated['fingerprint']==result['fingerprint']
