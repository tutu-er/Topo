"""Compact evidence must preserve final states, failures and saved analysis."""
from copy import deepcopy

import pytest

from . import ev_mobile_v4_evidence as evidence
from . import gradient_price_study as io
from .test_ev_mobile_v4_analysis import _record
from .test_ev_mobile_v4_audit import _fixture,predictive


def with_predictive_intermediates(record):
    value=deepcopy(record)
    reference=predictive([[0.]])
    for row in reference['references']:
        for fit in row['fits'].values():
            fit.update(state=dict(r=[.01],x=[.003],source_p=[0.]*100,source_q=[0.]*100),
                blocks=dict(history=dict(residual=[[.01]*3]*100)),
                optimizer=dict(success=True,nfev=20,diagnostics=dict(active_bound_count=1),
                               repeated_large_array=[[float(i) for i in range(50)]]*30),
                optimizer_recomputed_score=dict(loss=fit['loss'],blocks=dict(residual=[[.2]*3]*100)),
                feasible_witness=dict(loss=fit['loss'],blocks=dict(residual=[[.3]*3]*100)))
    for family in ('aligned','guarded'):
        value['families'][family]['cache']={'one':deepcopy(reference)}
    value['reporting_cache']={'one':deepcopy(reference)}
    return value


def test_compact_preserves_complete_final_data_and_analysis_with_explicit_omissions():
    record=with_predictive_intermediates(_record())
    compact=evidence.compact_record(record,pool_min_bytes=128)
    restored=evidence.restore(compact)
    assert not compact['complete_raw_reconstruction']
    assert len(compact['omissions'])==3*4*4
    for region in ('actual','methods','training_result','forecast'):
        assert restored[region]==record[region]
    fit=restored['reporting_cache']['one']['references'][0]['fits']['a']
    assert 'blocks' not in fit
    assert 'repeated_large_array' not in fit['optimizer']
    assert fit['optimizer']['diagnostics']=={'active_bound_count':1}
    assert fit['state']==record['reporting_cache']['one']['references'][0]['fits']['a']['state']
    validation=evidence.validate_record(record,compact,physical_replay=False)
    assert validation['success'] and validation['analysis_equal']
    assert all(validation['exact_final_regions'].values())


def test_failed_optimizer_inside_omitted_predictive_region_is_retained_in_full():
    record=with_predictive_intermediates(_record())
    failed=dict(success=False,status='iteration_limit',nfev=250,
                parameter_vector=list(range(80)),failure='optimizer did not converge')
    record['reporting_cache']['one']['references'][0]['fits']['a']['optimizer']=failed
    compact=evidence.compact_record(record,pool_min_bytes=128)
    restored=evidence.restore(compact)
    failures=restored['_evidence_provenance']['failure_evidence']
    assert any(row['complete_failed_node']==failed for row in failures)
    assert evidence.validate_record(record,compact,physical_replay=False)['success']


def test_lossless_pooling_keeps_repeated_operations_without_omitting_solver_bounds():
    record=with_predictive_intermediates(_record())
    repeated=dict(best_bound=3.,incumbent=2.,gap=1.,z=[.1]*500,
                  selected_route=dict(arcs=list(range(200))))
    record['solver_extra']={'first':deepcopy(repeated),'second':deepcopy(repeated)}
    compact=evidence.compact_record(record,pool_min_bytes=128)
    restored=evidence.restore(compact)
    assert restored['solver_extra']==record['solver_extra']
    assert restored['solver_extra']['first'] is not restored['solver_extra']['second']
    assert all(not row['path'].startswith('/solver_extra') for row in compact['omissions'])


def test_tampered_pool_or_wrong_source_is_rejected():
    record=with_predictive_intermediates(_record())
    compact=evidence.compact_record(record,pool_min_bytes=128)
    tampered=deepcopy(compact)
    key=next(iter(tampered['objects']))
    tampered['objects'][key]=dict(tamper=True)
    with pytest.raises(ValueError,match='payload hash'):
        evidence.restore(tampered)
    # Rehashing the outer envelope cannot hide an object-content substitution.
    tampered['payload_sha256']=evidence._hash({k:v for k,v in tampered.items() if k!='payload_sha256'})
    with pytest.raises(ValueError,match='content-addressed'):
        evidence.restore(tampered)
    changed=deepcopy(record);changed['true_name']='b'
    with pytest.raises(ValueError,match='source object'):
        evidence.validate_record(changed,compact,physical_replay=False)


def test_restore_matches_complete_independent_physical_audit(monkeypatch):
    record,calls=_fixture(monkeypatch)
    record['spec']['seed']=17
    # Existing auditor fixture has legal operation/state coverage and mock
    # branch-current replay, so the compact test never runs production AC.
    record=io.serial(record)
    for family in ('aligned','guarded'):
        for profile in record['families'][family]['cache'].values():
            for row in profile['references']:
                for fit in row['fits'].values():
                    fit['optimizer']=dict(success=True,nfev=4,discarded_vector=[1.]*100)
    compact=evidence.compact_record(record,pool_min_bytes=128)
    check=evidence.validate_record(record,compact,physical_replay=True)
    assert check['success']
    assert check['original_physical_success'] and check['restored_physical_success']
    assert check['physical_audit_equal']
    assert len(calls)==12  # Two complete saved-state audits, no optimizer.
