"""Pure independent-audit arithmetic and no-early-array-read gates."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from audit_multicase import audit,plan_settlement


def plan():
    shape=np.zeros(96);shape[24:28]=-1.;shape[28:32]=1.
    resources=[dict(resource_id=name,node_label=node,baseline_kw=.2*n,power_upper_kw=.4*n,
        prefix_delta_lower_kwh=-.4*n,prefix_delta_upper_kwh=.4*n,available=True,reactive_ratio=.35)
        for name,node,n in [('focus',202,4),('partner',213,5)]]
    modules=[dict(module_id=r['resource_id'],profiles={r['resource_id']:shape.tolist()},
        amplitude_upper_kw=r['baseline_kw']) for r in resources]
    return dict(resources=resources,modules=modules,prices=[.5,.5],amplitudes=[.2,.25],
        elasticities=[.4,.5],response_factors=[[.15,.05],[.15,-.05]])


def check(name,condition):
    if not condition:raise AssertionError(name)


def test_mixed_n4_n5_settlement_is_correct_and_generic():
    p=plan();before=deepcopy(p)
    result=plan_settlement(p,[.2,-.3],[202,213],check,'toy')
    assert result['inventory_units']=={'focus':4,'partner':5}
    assert result['actual']==pytest.approx([.203,.26125])
    assert result['fee']==pytest.approx(.232125)
    assert result['expected_fee']==pytest.approx(.225)
    assert result['hard_fee']==pytest.approx(.27)
    assert p==before


def test_zero_contract_remains_zero_without_positivity_floor():
    p=plan();p['prices']=[0.,0.];p['amplitudes']=[0.,0.]
    result=plan_settlement(p,[-1.,1.],[202,213],check,'zero')
    assert result['fee']==result['expected_fee']==result['hard_fee']==0


def test_wrong_scope_is_rejected():
    with pytest.raises(AssertionError,match='declared_nodes'):
        plan_settlement(plan(),[0.,0.],[201,213],check,'toy')


def test_fee_over_budget_is_rejected():
    p=plan();p['prices']=[1.5,1.5];p['amplitudes']=[.6,.75]
    with pytest.raises(AssertionError,match='hard_budget'):
        plan_settlement(p,[0.,0.],[202,213],check,'toy')


def test_audit_waits_before_opening_any_protocol_or_private_arrays(tmp_path):
    # There is deliberately no protocol to parse. Missing predictive/scored
    # gates must return before any attempt to consume the sentinel file.
    a=tmp_path/'actual';a.mkdir()
    (a/'observations_frozen.json').write_text('not yet valid JSON')
    result=audit(tmp_path)
    assert result['status']=='waiting_for_frozen_predictions'
    assert result['private_arrays_opened'] is False
    assert result['new_AC']==result['new_solver_calls']==result['new_fits']==0
