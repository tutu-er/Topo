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


def _compact_directory(tmp_path):
    directory=tmp_path/'compact';(directory/'evaluation').mkdir(parents=True)
    record=with_predictive_intermediates(_record());record['policy_id']='policy'
    frozen=dict(policy_id='policy',source_sha256={},
        specifications=dict(development=[],evaluation=[record['spec']]))
    evidence._write(directory/'freeze.json',frozen)
    evidence._write(directory/'source_snapshot.json.gz',dict(source_sha256={},source_base64={}))
    evidence._write(directory/'evaluation'/'000.json.gz',evidence.compact_record(record,pool_min_bytes=128))
    return directory,record


def test_restore_cli_yields_existing_analysis_and_audit_reader_schema(tmp_path,monkeypatch):
    import sys
    from . import ev_mobile_v4_analysis as analysis
    from . import ev_mobile_v4_audit as audit
    directory,original=_compact_directory(tmp_path);output=tmp_path/'restored'
    monkeypatch.setattr(sys,'argv',['ev_mobile_v4_evidence','restore',
        '--directory',str(directory),'--output',str(output)])
    evidence.main()
    restored,_=analysis._read(output/'evaluation'/'000.json.gz')
    assert audit.old.read(output/'evaluation'/'000.json.gz')==restored
    assert restored['policy_id']=='policy'
    assert restored['actual']==original['actual']
    assert restored['_evidence_provenance']['complete_raw_reconstruction'] is False
    assert len(restored['_evidence_provenance']['omissions'])==48
    assert analysis.summarize_records([original['spec']],{0:original})==analysis.summarize_records([restored['spec']],{0:restored})
    manifest=evidence._read(output/'restore_manifest.json')
    assert manifest['success'] and manifest['current_scientific_sources_match']
    assert manifest['record_count']==1 and manifest['phases']==['evaluation']
    assert manifest['optimizer_calls']==manifest['new_AC_calls']==manifest['new_simulation_calls']==0
    with pytest.raises(FileExistsError):
        evidence.restore_directory(directory,output)


def test_restore_requires_frozen_phase_coverage_and_valid_source_snapshot(tmp_path):
    import base64
    directory,record=_compact_directory(tmp_path)
    frozen=evidence._read(directory/'freeze.json')
    frozen['specifications']['evaluation'].append({**record['spec'],'index':1})
    # This is only a temporary test input, not an existing research record.
    (directory/'freeze.json').write_bytes(evidence._bytes(frozen))
    with pytest.raises(ValueError,match='coverage'):
        evidence.restore_directory(directory,tmp_path/'missing-record')
    assert not (tmp_path/'missing-record').exists()
    frozen['specifications']['evaluation'].pop()
    frozen['source_sha256']={'not_a_real_scientific_file.py':'wrong_hash'}
    (directory/'freeze.json').write_bytes(evidence._bytes(frozen))
    snapshot=dict(source_sha256=frozen['source_sha256'],source_base64={
        'not_a_real_scientific_file.py':base64.b64encode(b'placeholder').decode()})
    (directory/'source_snapshot.json.gz').write_bytes(__import__('gzip').compress(evidence._bytes(snapshot)))
    with pytest.raises(ValueError,match='snapshot hash'):
        evidence.restore_directory(directory,tmp_path/'wrong-source')
    assert not (tmp_path/'wrong-source').exists()
