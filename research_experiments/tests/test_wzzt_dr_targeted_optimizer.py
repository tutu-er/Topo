"""Fraction/enumeration checks for the precomputed-state pricing MILP.

These artificial loss tables test optimization arithmetic, not WZZT recovery.
No observations, simulation truth, identifier or AC model is accessed.
"""
from fractions import Fraction
import itertools
import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_targeted_pricing.optimizer import (
    PAIR_STATES,explicit_loss_intervals,marginalized_gamma,optimize_price_plan,
    pair_probabilities,
)


def fraction_enumeration(losses, module_ids):
    probabilities=[Fraction(0),Fraction(1,2),Fraction(3,4)]
    rows=[]
    for m,ci,cj in itertools.product(range(len(module_ids)),range(3),range(3)):
        if ci+cj>2:continue
        pi,pj=probabilities[ci],probabilities[cj]
        pair=[(pi if i else 1-pi)*(pj if j else 1-pj) for i,j in PAIR_STATES]
        value=sum((pair[t]*pair[v]*Fraction(int(losses[m,t,v]))
                   for t in range(4) for v in range(4)),Fraction(0))
        fee=2*Fraction(2,5)*(ci*pi+cj*pj)
        rows.append((value,fee,module_ids[m],(ci,cj)))
    risk=min(row[0] for row in rows);fee=min(row[1] for row in rows if row[0]==risk)
    return risk,fee,{(row[2],row[3]) for row in rows if row[:2]==(risk,fee)}


def assert_solver_receipts(output):
    assert len(output['solver_receipts'])==2
    for receipt in output['solver_receipts']:
        assert receipt['status']==0 and receipt['solver_reported_optimal']
        assert receipt['independent_primal_pass']
        assert receipt['mip_dual_bound'] is not None
        assert receipt['mip_gap']==pytest.approx(0.)
    assert output['independent_upper_optimality_check']['pass_check']
    assert output['independent_lex_fee_check']['pass_check']
    assert output['AC_calls']==output['identifier_calls']==0


def test_complementarity_exact_sixteen_states_and_hard_budget():
    losses=np.ones((1,4,4));losses[0,3,3]=0
    result=optimize_price_plan(losses,module_ids=['joint_witness'],seed=11)
    assert result['status']=='optimal'
    assert result['selected']['prices']==[1.,1.]
    assert result['selected']['risk_lower']==pytest.approx(float(Fraction(15,16)))
    assert result['selected']['expected_fee']==pytest.approx(float(Fraction(4,5)))
    assert result['selected']['worst_fee']==pytest.approx(float(Fraction(8,5)))
    assert len(result['feasible_columns'])==6
    assert all(sum(v['prices'])<=2 for v in result['feasible_columns'])
    assert_solver_receipts(result)


def test_one_resource_suffices_then_concentrated_price_is_selected():
    losses=np.ones((1,4,4));losses[0,[1,3],:]=0
    result=optimize_price_plan(losses,module_ids=['single_is_enough'],seed=11)
    assert result['status']=='optimal'
    assert result['selected']['prices']==[2.,0.]
    assert result['selected']['risk_upper']==pytest.approx(float(Fraction(1,4)))
    assert result['selected']['expected_fee']==pytest.approx(float(Fraction(6,5)))
    assert_solver_receipts(result)


def test_zero_benefit_selects_no_reward_and_reports_all_module_ties():
    losses=np.zeros((2,4,4))
    a=optimize_price_plan(losses,module_ids=['first','second'],seed=920)
    b=optimize_price_plan(losses,module_ids=['first','second'],seed=920)
    assert a['status']=='optimal'
    assert a['selected']['prices']==[0.,0.] and a['selected']['expected_fee']==0
    assert a['selected']==b['selected']
    assert {(v['module_id'],tuple(v['prices'])) for v in a['all_exact_lex_ties']}=={
        ('first',(0.,0.)),('second',(0.,0.))}
    assert_solver_receipts(a)


def test_train_validation_are_not_the_same_participation_draw():
    losses=np.ones((4,4));losses[3,3]=0
    pi=pair_probabilities(.5,.5)
    diagonal=float(sum(pi[s]*losses[s,s] for s in range(4)))
    independent=float(np.sum(np.outer(pi,pi)*losses))
    assert diagonal==pytest.approx(.75)
    assert independent==pytest.approx(float(Fraction(15,16)))
    gamma=marginalized_gamma(losses,losses,pi)
    assert gamma['training_loss_lower']==pytest.approx([1,1,1,.75])
    assert gamma['value']==pytest.approx(-.25)
    assert gamma['value']!=losses[3,3]-losses[1,1]-losses[2,2]+losses[0,0]


def test_fraction_full_enumeration_agrees_with_milp_and_all_lex_ties():
    losses=np.random.default_rng(10817).integers(0,4,size=(3,4,4))
    ids=['alpha','beta','gamma']
    expected_risk,expected_fee,expected_ties=fraction_enumeration(losses,ids)
    result=optimize_price_plan(losses,module_ids=ids,seed=9)
    assert result['status']=='optimal'
    assert result['selected']['risk_upper']==pytest.approx(float(expected_risk))
    assert result['selected']['expected_fee']==pytest.approx(float(expected_fee))
    got={(r['module_id'],tuple(int(v) for v in r['prices'])) for r in result['all_exact_lex_ties']}
    assert got==expected_ties
    assert_solver_receipts(result)


def test_unknown_failures_remain_intervals_and_do_not_select_unverified_price():
    values=np.full((1,4,4),np.nan);values[0,0,0]=1
    with pytest.raises(ValueError,match='explicit loss intervals'):
        optimize_price_plan(values)
    bounds=explicit_loss_intervals(values,missing_lower=0,missing_upper=4)
    assert len(bounds['missing_cells'])==15
    result=optimize_price_plan(bounds['lower'],bounds['upper'])
    assert result['status']=='interval_unresolved'
    assert result['selected'] is None
    assert result['upper_candidate'] is not None
    assert len(result['uncertain_cells'])==15
    assert result['finite_domain_gap_upper']>0
    assert result['all_exact_lex_ties'] is None
    assert_solver_receipts(result)


def test_interval_dominance_can_prove_no_reward_without_filling_unknown_losses():
    values=np.full((1,4,4),np.nan);values[0,0,0]=0
    bounds=explicit_loss_intervals(values,missing_lower=2,missing_upper=3)
    result=optimize_price_plan(bounds['lower'],bounds['upper'])
    assert result['status']=='interval_dominance_certified'
    assert result['selected']['prices']==[0.,0.]
    assert result['finite_domain_gap_upper']==0
    assert len(result['uncertain_cells'])==15
    assert not result['loss_table_exact']
    assert_solver_receipts(result)


def test_infeasible_positive_only_menu_is_retained_without_selection():
    result=optimize_price_plan(np.ones((1,4,4)),prices=(1,2),probabilities=(.5,.75),budget=.1)
    assert result['status']=='optimizer_unresolved'
    assert result['selected'] is None
    assert result['feasible_columns']==[]
    assert result['solver_receipts'][0]['status']==2
    assert not result['solver_receipts'][0]['solver_reported_optimal']


def test_invalid_shapes_intervals_and_probabilities_are_rejected():
    with pytest.raises(ValueError):optimize_price_plan(np.zeros((1,4)))
    with pytest.raises(ValueError):optimize_price_plan(np.ones((1,4,4)),np.zeros((1,4,4)))
    with pytest.raises(ValueError):optimize_price_plan(np.zeros((1,4,4)),probabilities=(0,.5,1.1))
    with pytest.raises(ValueError):optimize_price_plan(np.zeros((1,4,4)),budget=-1)
    with pytest.raises(ValueError):explicit_loss_intervals(np.full((1,4,4),np.nan),missing_lower=3,missing_upper=2)
