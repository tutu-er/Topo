"""Small score/denominator tests; no AC, fit or MILP calls."""
import copy

import pytest

from .ev_mobile_v3_proxy_replay import render, saved_source_score, signs, summarize


def record():
    p=[[1.,0.],[2.,0.]]
    profile=dict(success=True,power_kw=p,pairs=[dict(a='a',b='b',success=True),
                                              dict(a='b',b='a',success=True)])
    return dict(candidates=[dict(name='a'),dict(name='b')],
        methods=[dict(name='scalar_fixed',power_kw=p)],selection=dict(methods=dict(
        scalar_fixed=dict(raw=dict(selected_profile=profile,selected_score=3.,selected_aggregation=None)))))


def test_saved_score_requires_exact_power_and_complete_pairs():
    raw=record();assert saved_source_score(raw,'scalar_fixed')['score']==3.
    raw['selection']['methods']['scalar_fixed']['raw']['selected_profile']['pairs'].pop()
    with pytest.raises(ValueError,match='No complete'):
        saved_source_score(raw,'scalar_fixed')


def test_nearby_unscored_action_never_inherits_a_score():
    raw=record()
    raw['selection']['methods']['scalar_fixed']['raw']['selected_profile']=copy.deepcopy(
        raw['selection']['methods']['scalar_fixed']['raw']['selected_profile'])
    raw['selection']['methods']['scalar_fixed']['raw']['selected_profile']['power_kw'][0][0]+=1e-11
    with pytest.raises(ValueError,match='exact selected'):
        saved_source_score(raw,'scalar_fixed')


def test_complete_initial_can_supply_the_saved_score_without_new_fit():
    raw=record();row=raw['selection']['methods']['scalar_fixed']['raw']
    profile=row.pop('selected_profile');row['initial_candidates']=[dict(profile=profile,score=3.,aggregation={})]
    assert saved_source_score(raw,'scalar_fixed')['path']=='initial_candidates[0].profile'


@pytest.mark.parametrize('p,a,category',[(2,-1,'reversed'),(-2,1,'reversed'),(2,1,'same_order'),
                                      (0,1,'predicted_tie'),(2,0,'actual_tie'),(0,0,'both_tied')])
def test_both_ranking_directions_and_ties(p,a,category):
    assert signs(p,a)['category']==category


def test_summary_keeps_failed_and_postselected_events_separate():
    rows=[dict(success=True,role='predeclared_representative',old_ranking=signs(1,-1),new_ranking=signs(-1,-1)),
          dict(success=False,role='predeclared_representative'),
          dict(success=True,role='observed_decision_change_explanation',old_ranking=signs(1,1),new_ranking=signs(-1,1))]
    s=summarize(rows)
    assert s['denominator']==3 and s['complete']==2 and s['failed']==1
    assert s['representatives']['denominator']==2
    assert s['events']['denominator']==1
    assert s['old_rise_actual_fall']==1 and s['new_rise_actual_fall']==0
    assert s['corrected_old_reversals']==1 and s['new_reversals']==1


def test_report_preserves_math_commands(tmp_path):
    path=tmp_path/'report.md'
    render(dict(summary=summarize([]),histories=[]),path)
    text=path.read_text(encoding='utf-8')
    assert r'\tfrac12' in text and r'\frac1K' in text and r'\phi' in text
    assert '\t' not in text and '\f' not in text
