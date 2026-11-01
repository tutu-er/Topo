import numpy as np
import pytest
from research_experiments.wzzt_dr_selection.dr_validation_excitation.coarse_exit_reprice import projection
from research_experiments.wzzt_dr_selection.dr_validation_excitation.affine_solver import evaluate_affine


def table(values):
    return [dict(scenario=0,validation_size=1,acceptance_margin=1.,families=[
        dict(family_id=j,has_target=(j==1),constant_loss=v,
             residual_base=np.array([]),residual_price=np.empty((0,8)))
        for j,v in enumerate(values)])]


def test_current_coarse_best_includes_itself_in_global_minimum():
    data=projection(table([1.,2.]),0)
    r=evaluate_affine(data,np.zeros(8))
    assert r['scenario_margins'][0] == pytest.approx(1.-1.03-1e-12)
    assert not r['sample_requirement_met']


def test_complex_no_C_can_remain_while_current_coarse_exits():
    original=table([1.1,1.,1.01])
    old=evaluate_affine(original,np.zeros(8))
    new=evaluate_affine(projection(original,0),np.zeros(8))
    assert old['scenario_margins'][0]<0<new['scenario_margins'][0]
    assert new['scenario_margins'][0] == pytest.approx(1.1-1.03-1e-12)
    assert not original[0]['families'][2]['has_target']
    assert len(projection(original,0)[0]['families'])==4  # 3 families + one expression alias
    assert sum(f['comparison_alias'] for f in projection(original,0)[0]['families'])==1


def test_projection_preserves_true_C_presence_as_metadata():
    original=table([1.1,1.,1.01]);new=projection(original,0)
    assert [f['original_has_target'] for f in new[0]['families']]==[False,True,False,False]
    assert new[0]['actual_distinct_family_count']==3
    assert new[0]['comparison_kind']=='current_coarse_vs_best_over_all_families'
