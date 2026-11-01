import importlib.util
from pathlib import Path
import sys
from copy import deepcopy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import design_adapter as module


def test_inventory_and_disputes_selected_without_mutating_full_pool(monkeypatch):
    pool=dict(dispute_added_supportlabels=[[1,2],[3,4]],dispute_removed_supportlabels=[[4,5]],models=['unchanged'])
    original=deepcopy(pool)
    def fake(local,history,units):
        if units>4:raise ValueError('past load cannot contain that inventory')
        b=local['dispute_added_supportlabels'][0]
        return dict(qualification=dict(selected_labels=b,nodes=[dict(node_label=x,unit_response_column_micro=x) for x in b]))
    monkeypatch.setattr(module,'single_design',fake)
    d=module.choose_design(pool,None)
    assert d['qualification']['selected_labels']==[3,4]
    assert d['qualification']['all_added_disputes']==[[1,2],[3,4]]
    assert d['qualification']['attempts'][-1]['units']==4
    assert pool==original


def test_no_capacity_is_not_silently_dropped(monkeypatch):
    def bad(*a,**k):raise ValueError('no component')
    monkeypatch.setattr(module,'single_design',bad)
    with pytest.raises(ValueError,match='inventory1..5'):
        module.choose_design(dict(dispute_added_supportlabels=[[1,2]],dispute_removed_supportlabels=[]),None)


def test_final_dispute_focus_overrides_empty_raw_family_difference(monkeypatch):
    pool=dict(labels=[301,302,303],dispute_added_supportlabels=[],dispute_removed_supportlabels=[],
        competition_rule=dict(final_added_supports=[[0,2]],final_removed_supports=[]),models=['fullmodels'])
    original=deepcopy(pool)
    def fake(local,history,units):
        assert local['models']==['fullmodels']
        assert local['dispute_added_supportlabels']==[[301,303]]
        return dict(qualification=dict(selected_labels=[301,302],nodes=[dict(node_label=i,unit_response_column_micro=1) for i in [301,302]]))
    monkeypatch.setattr(module,'single_design',fake)
    result=module.choose_design(pool,None)
    assert result['qualification']['raw_added_disputes']==[]
    assert result['qualification']['all_added_disputes']==[[301,303]]
    assert pool==original
