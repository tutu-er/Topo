"""Synthetic arithmetic and hash tests; no actual data, solver or true graph."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

HERE=Path(__file__).resolve().parent
# Existing selector imports the repository core package. This is only import
# setup for synthetic unit tests; it neither reads research data nor fits.
CORE=Path.cwd()/'rnj_wzzt_core'
if CORE.is_dir() and str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
SPEC=importlib.util.spec_from_file_location('frozen_pool_under_test',HERE/'frozen_pool.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def scene(p,y,labels):
    p=np.asarray(p,float);y=np.asarray(y,float)
    index=pd.date_range('2000-01-01',periods=len(p),freq='15min')
    return dict(P_terminal=pd.DataFrame(p,index=index,columns=labels),
        Q_terminal=pd.DataFrame(np.zeros_like(p),index=index,columns=labels),
        V_terminal=pd.DataFrame(np.sqrt(1-y),index=index,columns=labels),
        V_root_observed=pd.Series(np.ones(len(p)),index=index),
        drop_target=pd.DataFrame(y,index=index,columns=labels))


def fixture_files(tmp_path):
    labels=[201,202,213];n=3
    p=np.column_stack([np.linspace(.01,.03,96),np.linspace(.02,.04,96),np.linspace(.03,.05,96)])
    coarse=dict(supports=[[0],[1],[2],[0,1,2]],r=[.1,.1,.1,.2],x=[.05,.05,.05,.1],
        preprocess='raw',origin='tree_initialized_correction',path_index=0)
    refined=dict(supports=coarse['supports']+[[0,1]],r=coarse['r']+[.04],x=coarse['x']+[.03],
        preprocess='raw',origin='tree_initialized_correction',path_index=1)
    R,X=MODULE._matrices(coarse,n);y=p@R.T
    day=scene(p,y,labels)
    history=dict(labels=labels,base_kw=200.,root_label=1,training=[deepcopy(day) for _ in range(3)],validation=[deepcopy(day) for _ in range(3)])
    points=[coarse,refined]
    for point in points:
        r,x=MODULE._matrices(point,n);prediction=p@r.T
        point.update(objective=float(np.abs(prediction-y).mean()),validation_mae=float(np.abs(prediction-y).mean()))
    selected=dict(coarse,selection_index=0)
    baseline=dict(selection_candidates=points,models={'multistart_original_rule':{'selected_uncontracted':selected}},
        calibrations={'raw':{'a':1.,'b':1.,'median_root_depth':.5}})
    hp=tmp_path/'history.pkl'
    with hp.open('wb') as stream:pickle.dump({'history':history},stream)
    fp=tmp_path/'historical_fit.json';fp.write_text(json.dumps(baseline),encoding='utf-8')
    cp=tmp_path/'case.json';cp.write_text(json.dumps(dict(labels=labels,base_kw=200.,root_label=1,
        history_packet=ref(hp),historical_baseline_result=ref(fp))),encoding='utf-8')
    return cp,hp,fp,day,baseline


def test_freeze_uses_existing_rule_replays_losses_and_freezes_output(tmp_path):
    cp,hp,fp,day,old=fixture_files(tmp_path)
    out=tmp_path/'frozen.json';pool=MODULE.freeze_pool(cp,out)
    assert pool['status']=='frozen' and pool['train_rows']==pool['validation_rows']==288
    assert pool['historical_validation_cells']==864
    assert pool['selection_order']==['coarse','refined']
    assert [m['historical_index'] for m in pool['models']]==[0,1]
    assert pool['dispute_added_supportlabels']==[[201,202]]
    assert pool['models'][0]['point']==old['selection_candidates'][0]
    assert pool['models'][1]['point']==old['selection_candidates'][1]
    # Positive weak added block.04+.03 remains in prediction but lies below
    # unchanged.16*.5 output threshold.08. Nothing tightens its fitted weight.
    fine=pool['models'][1]
    assert [0,1] in fine['point']['supports']
    assert [0,1] not in fine['frozen_output_supports']
    assert fine['point']['r'][-1]==.04 and fine['point']['x'][-1]==.03
    assert pool['new_fit_calls']==pool['new_search_calls']==0
    assert not pool['new_future_or_truth_read']
    assert json.loads(out.read_text(encoding='utf-8'))['pool_sha256']==pool['pool_sha256']
    with pytest.raises(FileExistsError):MODULE.freeze_pool(cp,out)


def test_frozen_parameters_do_not_change_when_new_target_changes(tmp_path):
    cp,_,_,day,_=fixture_files(tmp_path);pool=MODULE.freeze_pool(cp,tmp_path/'frozen.json')
    before=deepcopy(pool);first=MODULE.score_frozen(pool,day)
    changed=deepcopy(day)
    target=np.full((96,3),.05)
    changed['V_terminal'].iloc[:,:]=np.sqrt(1-target)
    changed['drop_target'].iloc[:,:]=target
    second=MODULE.score_frozen(pool,changed)
    assert pool==before
    assert first['cumulative_validation_rows']==second['cumulative_validation_rows']==384
    assert first['rows']!=second['rows']
    assert first['new_fit_calls']==second['new_fit_calls']==0
    np.testing.assert_array_equal(MODULE.predict(pool['models'][0],day),MODULE.predict(pool['models'][0],changed))


def test_source_hash_and_stored_validation_loss_must_match(tmp_path):
    cp,hp,fp,_,_=fixture_files(tmp_path)
    hp.write_bytes(hp.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash mismatch'):MODULE.freeze_pool(cp,tmp_path/'frozen.json')
    # Repair the hash only to test an independently wrong saved validation
    # value, not to conceal an altered research input.
    case=json.loads(cp.read_text(encoding='utf-8'));case['history_packet']=ref(hp)
    baseline=json.loads(fp.read_text(encoding='utf-8'));baseline['selection_candidates'][1]['validation_mae']+=.1
    fp.write_text(json.dumps(baseline),encoding='utf-8');case['historical_baseline_result']=ref(fp)
    cp.write_text(json.dumps(case),encoding='utf-8')
    with pytest.raises(ValueError,match='validation loss fails replay'):MODULE.freeze_pool(cp,tmp_path/'frozen2.json')


def test_full_history_selection_and_day_only_are_separate():
    labels=[201,202]
    p0=dict(supports=[[0],[1]],r=[1.,1.],x=[0.,0.])
    p1=dict(supports=[[0],[1],[0,1]],r=[1.,1.,.05],x=[0.,0.,0.])
    models=[]
    for identity,point,historical_mae in [('coarse',p0,.1),('refined',p1,.11)]:
        r,x=MODULE._matrices(point,2)
        models.append(dict(candidate_id=identity,point=point,labels=labels,R=r.tolist(),X=x.tolist(),
            historical_validation_abs_sum=historical_mae*288*2,
            frozen_output_supports=point['supports'],
            frozen_output_support_labels=[[labels[i] for i in s] for s in point['supports']]))
    pool=dict(status='frozen',models=models,selection_order=['coarse','refined'],labels=labels,
        historical_validation_cells=576,validation_rows=288,
        calibration=dict(a=1.,b=1.,median_root_depth=.25))
    new=scene(np.full((96,2),.1),np.full((96,2),.11),labels)
    score=MODULE.score_frozen(pool,new)
    assert score['selected_candidate_id']=='coarse'
    assert score['day_only_selection']['selected_candidate_id']=='refined'
    assert score['cumulative_validation_rows']==384
    assert score['cumulative_current_coarse_exit_margin']<0
    assert score['day_only_current_coarse_exit_margin']>0
    assert score['rows'][0]['cumulative_validation_mae']==pytest.approx(.0775)
    assert score['rows'][1]['cumulative_validation_mae']==pytest.approx(.0825)


def test_column_order_and_pool_mutation_are_rejected(tmp_path):
    cp,_,_,day,_=fixture_files(tmp_path);pool=MODULE.freeze_pool(cp,tmp_path/'frozen.json')
    wrong=deepcopy(day)
    wrong['P_terminal']=wrong['P_terminal'][[202,201,213]]
    with pytest.raises(ValueError,match='terminal order'):MODULE.predict(pool['models'][0],wrong)
    changed=deepcopy(pool);changed['models'][0]['R'][0][0]+=.01
    with pytest.raises(ValueError,match='pool content changed'):MODULE.score_frozen(changed,day)


def test_zero_weights_are_preserved_in_frozen_parameters():
    point=dict(supports=[[0],[1],[0,1]],r=[0.,.2,0.],x=[0.,.1,0.])
    r,x=MODULE._matrices(point,2)
    output=MODULE._output(point,dict(a=1.,b=1.,median_root_depth=1.))
    assert point['r']==[0.,.2,0.]
    assert output==[[0],[1]]
    assert r[0,0]==x[0,0]==0.
    assert point['supports']==[[0],[1],[0,1]]


def test_reuses_existing_selector_and_contraction(tmp_path,monkeypatch):
    from research_experiments.wzzt_dr_selection.dr_pool_v2 import selection
    original_select=selection.select;original_contract=selection.contract
    calls=dict(select=0,contract=0)
    def select(points,calibration):
        calls['select']+=1
        return original_select(points,calibration)
    def contract(point,calibration):
        calls['contract']+=1
        return original_contract(point,calibration)
    monkeypatch.setattr(selection,'select',select)
    monkeypatch.setattr(selection,'contract',contract)
    cp,_,_,day,_=fixture_files(tmp_path)
    pool=MODULE.freeze_pool(cp,tmp_path/'frozen.json')
    MODULE.score_frozen(pool,day)
    assert calls['select']==2  # cumulative primary plus named day-only diagnostic
    assert calls['contract']==4  # two freeze maps plus two original selections
