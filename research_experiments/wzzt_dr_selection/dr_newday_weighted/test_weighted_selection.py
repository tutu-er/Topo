"""Pure frozen-predictor checks; no saved case, solver, future or truth read."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import weighted_selection as ws
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract


def fixture(history=(1.02,1.),day=(1.04,1.),nh=3,nd=1,shared=.01):
    labels=[201,202];cal=dict(a=1.,b=1.,median_root_depth=.01)
    coarse=dict(supports=[[0],[1]],r=[day[0],day[0]],x=[0.,0.])
    fine=dict(supports=[[0],[1],[0,1]],r=[day[1]-2*shared,day[1]-2*shared,shared],x=[0.,0.,0.])
    assert min(fine['r'])>=0
    models=[]
    for identity,point,hist in zip(('coarse','refined'),(coarse,fine),history):
        r=np.zeros((2,2));x=np.zeros((2,2))
        for block,rv,xv in zip(point['supports'],point['r'],point['x']):
            r[np.ix_(block,block)]+=rv;x[np.ix_(block,block)]+=xv
        output=contract(point,cal)['supports']
        models.append(dict(candidate_id=identity,point=point,labels=labels,R=r.tolist(),X=x.tolist(),
            historical_validation_abs_sum=hist*nh*2,
            frozen_output_supports=output,
            frozen_output_support_labels=[[labels[i] for i in b] for b in output]))
    pool=dict(status='frozen',models=models,selection_order=['coarse','refined'],labels=labels,
        historical_validation_cells=nh*2,validation_rows=nh,calibration=cal)
    scene=dict(P_terminal=np.ones((nd,2)),Q_terminal=np.zeros((nd,2)),drop_target=np.zeros((nd,2)))
    return pool,scene


def rules(result):return {(r['new_day_weight'],r['new_day_fraction']):r for r in result['rules']}


def test_original_ungated_baseline_exact_and_single_loss_pass(monkeypatch):
    pool,scene=fixture();calls=[];old=ws.score_frozen
    def counted(*args,**kwargs):
        calls.append(1);return old(*args,**kwargs)
    monkeypatch.setattr(ws,'score_frozen',counted)
    result=ws.score(pool,scene,history_gate=False);rule=rules(result)[1,.03]
    assert len(calls)==1
    assert rule['original_baseline_equivalence_checked']
    assert rule['selected_candidate_id']==result['original_ungated_selection']['selected_candidate_id']
    assert rule['eligible_candidate_ids']==result['original_ungated_selection']['eligible_candidate_ids']
    assert rule['selected_output_supports']==result['original_ungated_selection']['selected_output_supports']


def test_only_new_day_gets_extra_sample_weight():
    pool,scene=fixture();result=ws.score(pool,scene,history_gate=False);r=rules(result)[2,.02]
    assert r['effective_validation_rows']==5
    assert r['weighted_validation_mae']['coarse']==pytest.approx((3*1.02+2*1.04)/5)
    assert r['historical_tolerance_mean']==pytest.approx(.03*3/5)
    assert r['new_day_tolerance_mean']==pytest.approx(.02*2/5)
    assert r['epsilon_row_sum_units']==pytest.approx(5e-12)
    assert result['actual_validation_rows']==4


def test_manual_synergy_requires_both_new_day_changes():
    pool,scene=fixture();rs=rules(ws.score(pool,scene,history_gate=True))
    assert rs[1,.03]['selected_candidate_id']=='coarse'
    assert rs[1,.02]['selected_candidate_id']=='coarse'
    assert rs[2,.03]['selected_candidate_id']=='coarse'
    assert rs[2,.02]['selected_candidate_id']=='refined'
    assert rs[2,.02]['exit_margins']['coarse']==pytest.approx(.002)


def test_history_gate_not_hidden_in_original_baseline():
    pool,scene=fixture(history=(1.04,1.),day=(.1,.5))
    result=ws.score(pool,scene,history_gate=True)
    assert result['historical_eligible_candidate_ids']==['refined']
    assert result['history_gate_removed_candidate_ids']==['coarse']
    assert result['original_ungated_selection']['selected_candidate_id']=='coarse'
    assert all(r['selected_candidate_id']=='refined' for r in result['rules'])
    assert not rules(result)[1,.03]['original_baseline_equivalence_checked']


def test_epsilon_is_mean_loss_not_absolute_sum():
    pool,scene=fixture(history=(0.,0.),day=(1.5e-12,0.),shared=0.)
    rule=rules(ws.score(pool,scene,history_gate=False))[2,.02]
    assert rule['weighted_validation_mae']['coarse']==pytest.approx(.6e-12)
    assert rule['eligibility_threshold_mean']==1e-12
    assert 'coarse' in rule['eligible_candidate_ids']
    assert rule['selected_candidate_id']=='coarse'


def test_reference_is_recomputed_after_new_day_weight_change():
    pool,scene=fixture(history=(1.,1.02),day=(1.05,1.))
    rs=rules(ws.score(pool,scene,history_gate=True))
    assert rs[1,.02]['reference_candidate_id']=='coarse'
    assert rs[2,.02]['reference_candidate_id']=='refined'


def test_zero_and_weak_weights_and_frozen_maps_are_unchanged():
    pool,scene=fixture(history=(0.,0.),day=(0.,0.),shared=0.)
    point=pool['models'][1]['point'];point['x'][2]=1e-14
    pool['models'][1]['X']=(np.ones((2,2))*1e-14).tolist()
    before=json.dumps(pool,sort_keys=True)
    result=ws.score(pool,scene)
    assert json.dumps(pool,sort_keys=True)==before
    assert point['r']==[0.,0.,0.] and point['x'][2]==1e-14
    assert all(r['selected_candidate_id']=='coarse' for r in result['rules'])
    assert result['parameters_changed'] is False
    assert result['new_fit_calls']==result['new_search_calls']==0


def test_tie_break_uses_raw_count_then_score_then_frozen_order():
    pool,scene=fixture(history=(0.,0.),day=(0.,0.),shared=0.)
    result=ws.score(pool,scene,history_gate=False)
    # The contracted outputs both have2 blocks, but original raw counts2/3
    # still favour coarse. Selection must not count contracted blocks.
    assert all(r['selected_raw_support_count']==2 for r in result['rules'])
    assert all(r['selected_candidate_id']=='coarse' for r in result['rules'])


def test_history_gate_flag_must_be_explicit_bool():
    pool,scene=fixture()
    with pytest.raises(TypeError):ws.score(pool,scene,history_gate=1)
