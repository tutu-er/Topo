"""Pure numerical tests; synthetic loss tables are not WZZT effect evidence."""
from fractions import Fraction
import itertools
import numpy as np
import pytest
from research_experiments.wzzt_dr_selection.dr_targeted_pricing.event_optimizer import optimize_event_prices
from research_experiments.wzzt_dr_selection.dr_targeted_pricing.optimizer import (
    optimize_price_plan, explicit_loss_intervals, PAIR_STATES,
)


def exact_columns(losses,ids):
    p=[Fraction(0),Fraction(1,2),Fraction(3,4)];rows=[]
    for m in range(len(ids)):
        for q in itertools.product(range(3),repeat=4):
            if sum(q)>4:continue
            a=[(p[q[0]] if i else 1-p[q[0]])*(p[q[1]] if j else 1-p[q[1]]) for i,j in PAIR_STATES]
            b=[(p[q[2]] if i else 1-p[q[2]])*(p[q[3]] if j else 1-p[q[3]]) for i,j in PAIR_STATES]
            risk=sum((a[t]*b[v]*Fraction(int(losses[m,t,v])) for t in range(4) for v in range(4)),Fraction(0))
            fee=Fraction(2,5)*sum((q[k]*p[q[k]] for k in range(4)),Fraction(0))
            rows.append((risk,fee,ids[m],q))
    return rows


def receipts(result):
    assert len(result['solver_receipts'])==2
    for row in result['solver_receipts']:
        assert row['status']==0 and row['solver_reported_optimal']
        assert row['independent_primal_pass'] and row['mip_dual_bound'] is not None
        assert row['mip_gap']==pytest.approx(0.)
    assert result['independent_upper_optimality_check']['pass_check']
    assert result['independent_lex_fee_check']['pass_check']
    assert result['AC_calls']==result['identifier_calls']==0


def test_validation_joint_evidence_can_receive_all_incentives():
    loss=np.ones((1,4,4));loss[0,:,3]=0
    free=optimize_event_prices(loss,module_ids=['validation_joint'],seed=31)
    restricted=optimize_price_plan(loss,module_ids=['validation_joint'],seed=31)
    assert free['status']=='optimal' and restricted['status']=='optimal'
    assert free['selected']['prices']==[[0,0],[2,2]]
    assert all(isinstance(x,int) for q in free['selected']['prices'] for x in q)
    assert free['selected']['risk_upper']==pytest.approx(float(Fraction(7,16)))
    assert free['selected']['expected_fee']==pytest.approx(float(Fraction(6,5)))
    assert free['selected']['worst_fee']==pytest.approx(float(Fraction(8,5)))
    assert restricted['selected']['risk_upper']==pytest.approx(float(Fraction(3,4)))
    assert free['selected']['training_Gamma']['value']==0
    assert free['selected']['validation_Gamma']['value']==-1
    receipts(free)


def test_one_validation_resource_avoids_unnecessary_training_payment():
    loss=np.ones((1,4,4));loss[0,:,[1,3]]=0
    free=optimize_event_prices(loss,module_ids=['validation_single'])
    restricted=optimize_price_plan(loss,module_ids=['validation_single'])
    assert free['selected']['prices']==[[0,0],[2,0]]
    assert free['selected']['risk_upper']==restricted['selected']['risk_upper']==pytest.approx(.25)
    assert free['selected']['expected_fee']==pytest.approx(.6)
    assert restricted['selected']['expected_fee']==pytest.approx(1.2)
    assert free['selected']['worst_fee']==pytest.approx(.8)
    receipts(free)


def test_four_way_interaction_requires_independent_train_validation_law():
    loss=np.ones((1,4,4));loss[0,3,3]=0
    result=optimize_event_prices(loss,module_ids=['all_four'])
    assert result['selected']['prices']==[[1,1],[1,1]]
    assert result['selected']['risk_upper']==pytest.approx(float(Fraction(15,16)))
    assert result['selected']['risk_upper']!=pytest.approx(.75)
    assert result['selected']['training_Gamma']['value']==pytest.approx(-.25)
    assert result['selected']['validation_Gamma']['value']==pytest.approx(-.25)
    receipts(result)


def test_fraction_enumeration_and_repeated_quote_subdomain_agree():
    loss=np.random.default_rng(10823).integers(0,5,size=(2,4,4));ids=['left','right']
    exact=exact_columns(loss,ids);best=min(x[0] for x in exact)
    fee=min(x[1] for x in exact if x[0]==best)
    ties={(x[2],x[3]) for x in exact if x[:2]==(best,fee)}
    result=optimize_event_prices(loss,module_ids=ids,seed=13)
    assert result['status']=='optimal'
    assert result['feasible_price_vectors_per_plan']==50
    assert len(result['feasible_columns'])==100
    assert result['selected']['risk_upper']==pytest.approx(float(best))
    assert result['selected']['expected_fee']==pytest.approx(float(fee))
    got={(v['module_id'],tuple(v['prices'][0]+v['prices'][1])) for v in result['all_exact_lex_ties']}
    assert got==ties
    repeated=[x for x in exact if x[3][:2]==x[3][2:]]
    assert len(repeated)==2*6
    assert best<=min(x[0] for x in repeated)
    assert all(sum(v['prices'][0]+v['prices'][1])<=4 for v in result['feasible_columns'])
    receipts(result)


def test_zero_gain_preserves_no_reward_and_all_seeded_module_ties():
    loss=np.zeros((2,4,4))
    a=optimize_event_prices(loss,module_ids=['a','b'],seed=103)
    b=optimize_event_prices(loss,module_ids=['a','b'],seed=103)
    assert a['selected']==b['selected']
    assert a['selected']['prices']==[[0,0],[0,0]] and a['selected']['expected_fee']==0
    assert len(a['all_exact_lex_ties'])==2
    assert {v['module_id'] for v in a['all_exact_lex_ties']}=={'a','b'}
    receipts(a)


def test_missing_loss_intervals_do_not_create_a_verified_price():
    values=np.full((1,4,4),np.nan);values[0,0,0]=1
    with pytest.raises(ValueError,match='explicit loss intervals'):optimize_event_prices(values)
    bounds=explicit_loss_intervals(values,missing_lower=0,missing_upper=4)
    result=optimize_event_prices(bounds['lower'],bounds['upper'])
    assert result['status']=='interval_unresolved' and result['selected'] is None
    assert result['upper_candidate'] is not None
    assert result['finite_domain_gap_upper']>0 and len(result['uncertain_cells'])==15
    assert result['all_exact_lex_ties'] is None
    receipts(result)


def test_interval_dominance_can_certify_no_reward_with_unknown_paid_outcomes():
    values=np.full((1,4,4),np.nan);values[0,0,0]=0
    bounds=explicit_loss_intervals(values,missing_lower=2,missing_upper=3)
    result=optimize_event_prices(bounds['lower'],bounds['upper'])
    assert result['status']=='interval_dominance_certified'
    assert result['selected']['prices']==[[0,0],[0,0]]
    assert result['finite_domain_gap_upper']==0
    assert len(result['uncertain_cells'])==15
    receipts(result)


def test_empty_price_domain_is_retained_without_fake_solver_success():
    result=optimize_event_prices(np.ones((1,4,4)),prices=(1,2),probabilities=(.5,.75),budget=.1)
    assert result['status']=='infeasible_finite_domain' and result['selected'] is None
    assert result['feasible_columns']==[] and result['solver_receipts']==[]
    assert result['finite_domain_lower_bound'] is None


def test_invalid_price_or_loss_domain_is_rejected():
    with pytest.raises(ValueError):optimize_event_prices(np.zeros((1,4)))
    with pytest.raises(ValueError):optimize_event_prices(np.ones((1,4,4)),np.zeros((1,4,4)))
    with pytest.raises(ValueError):optimize_event_prices(np.zeros((1,4,4)),prices=(0,.5,1))
    with pytest.raises(ValueError):optimize_event_prices(np.zeros((1,4,4)),probabilities=(0,.5,1.2))
    with pytest.raises(ValueError):optimize_event_prices(np.zeros((1,4,4)),budget=-1)
