"""Arithmetic and missing-state tests for the separate exploratory case."""
from itertools import product

import pytest

from . import ev_mobile_v4_gap_case as g


def _meta():
    FC,MC=g.METHODS
    return dict(nominal={FC:dict(Phi=1.,true_ref_m=1.),MC:dict(Phi=5.,true_ref_m=7.)},
        observed_halfM={'shift':{FC:0.,MC:26.}},selection_basis='observed development reversal')


def test_exploratory_factor_and_score_gap_decomposition():
    FC,MC=g.METHODS;rows={}
    for bits in g.core.STATES:
        L,B,S=bits;delta=4+2*L-3*B+5*S+6*L*B+9*L*B*S;key=''.join(map(str,bits))
        for method,value in [(FC,10.),(MC,10.+delta)]:rows[method,key]=dict(status='complete',halfM=value)
    result=g.event(_meta(),rows)
    assert result['complete'] and not result['independent_confirmation'] and not result['used_to_change_policy']
    assert result['delta_000']==4. and result['delta_111']==23.
    assert result['Shapley_delta_halfM']['contributions']==pytest.approx(dict(lines=8.,background=3.,source=8.))
    assert result['metric_aggregation_discrepancy']==2.
    assert result['nominal_vs_same_inference_rule_discrepancy']==-2.
    assert result['observed_minus_111_residual']==3.
    assert result['full_gap_decomposition_error']==0.


def test_incomplete_exploration_is_not_given_shapley_values_or_zero_fills():
    rows={(m,''.join(map(str,bits))):dict(status='complete',halfM=1.)
          for bits,m in product(g.core.STATES,g.METHODS)}
    rows[g.METHODS[1],'111']=dict(status='fit_failure',halfM=None)
    result=g.event(_meta(),rows)
    assert not result['complete'] and result['states']['111']['delta_halfM'] is None
    assert 'Shapley_delta_halfM' not in result


def test_empty_output_keeps_only_declared_64_candidate_slots(tmp_path):
    summary=g.summarize(tmp_path)
    assert summary['target_slot_denominator']==16 and summary['candidate_slot_denominator']==64
    assert summary['candidate_slots']==dict(expected=64,attempted=0,success=0,failed=0,not_run=64)
    assert summary['unique_evaluation_calls']==0
    assert summary['selected_after_observing_development_reversal']
    assert not summary['independent_confirmation'] and not summary['used_to_change_policy']


def test_exploratory_methods_do_not_change_predeclared_attribution_constants():
    assert g.METHODS==('guarded__fixed_charge','guarded__mobile_charge')
    assert g.core.METHODS==('guarded__fixed_bidir','guarded__mobile_bidir')
    assert g.core.INDICES==(0,6) and g.core.TARGET_SLOTS==64 and g.core.CANDIDATE_SLOTS==256
    assert g.INDEX==1 and g.REGIME=='shift'
