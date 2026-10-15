"""Lossless inference payload, unchanged statistics, immutable delivery hashes."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from . import ev_systematic_delivery as delivery
from . import ev_systematic_analyze as analysis
from .test_ev_systematic_analyze import record as statistical_record


def raw_record(seed,phase='development'):
    value=statistical_record(seed,phase)
    state=dict(r=[.004,.003],x=[.001,.0008],source_p=[.1,.11],source_q=[.03,.04])
    block=dict(success=True,state=state,residual=[.01,-.02],block_losses=dict(history=3.,future=2.),
        normalizer=dict(v_sigma=.0001,master_p_sigma=.001),optimizer_failure=None,
        feasible_fixed_witness=dict(state=deepcopy(state),loss=5.))
    value.update(config=dict(fixed_training=True),seeds=dict(background=12),truth=dict(r=[.005],x=[.001]),
        training=dict(p=[[.1]],q=[[.03]],v=[[.999]]),training_meter=dict(v_sigma=.0001),
        validation=dict(p=[[.12]],q=[[.04]],v=[[.998]]),validation_meter=dict(v_sigma=.0001),
        future_meter=dict(v_sigma=.00005),training_result=dict(fits=[deepcopy(block)]),
        history_prefits=dict(a=dict(fixed=deepcopy(block),joint=deepcopy(block))),
        future_records=dict(action=dict(observations=dict(p=[[.1]],q=[[.03]],v=[[1.]]),
            comparisons=dict(a=dict(fixed=deepcopy(block),joint=deepcopy(block))),failures=dict(b='retained source failure'))))
    many=[dict(c=[i/100.,0.,.1],B=[[.01,.02]],source_QP=dict(success=True,residual=[.1,.2])) for i in range(1000)]
    value['common_cost']=dict(training_seconds=1.7,shared_fit_seconds=.8)
    value['selection']=dict(seconds=1.2,setup_seconds=.11,context_seconds=.12,uses_truth=False,frozen_pairs=[dict(name='a_to_b',c=[1.,2.],B=[[1.],[2.]],A0=[[1.,0.],[0.,1.]])],
        methods=dict(scalar_fixed=dict(planning_actual_seconds=.3,planning_seconds=.4,
            dependencies=['square_fixed','context'],power_kw=[[0.],[1.]],success=True,
            raw=dict(success=True,selected_origin='scalar_MILP_3',selected_score=2.,table_QP_calls=600,
                action_QP_calls=12,grids=[dict(grid_points=3,accepted=True,raw_AC_score=2.,
                    table=dict(success=True,QP_calls=600,seconds=.1,tables=many),
                    raw=dict(success=True,frozen_objective_lower=1.9,frozen_objective_upper=2.1,
                        frozen_absolute_gap=.2,solver_proxy_incumbent=2.,solver_proxy_best_bound=2.1,
                        solver_mip_gap=.05,added_binary_count=0,added_continuous_count=4))],
                iterations=[dict(accepted=False,before_score=1.,candidate_score=.9,
                    rejection_reason='full AC score not improved',MILP=dict(status=1,linear_best_bound=2.))],
                failed_extra=dict(success=False,message='failed witness retained',residual=[3.,4.])))))
    return value


def write_raw(directory,records):
    directory.mkdir(parents=True)
    (directory/'freeze.json').write_bytes(b'{"source_id":"source-v1","config_id":"config-v1"}\n')
    (directory/'source_snapshot.json.gz').write_bytes(gzip.compress(b'{"source":"original bytes"}',mtime=17))
    for row in records:
        path=directory/f"case_{row['seed']}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(row,sort_keys=True).encode(),mtime=23))
    return directory


def strip_file_metadata(value):
    omitted={'source_files','calibration_files','threshold_file','threshold_file_sha256','directory'}
    if isinstance(value,dict):return {k:strip_file_metadata(v) for k,v in value.items() if k not in omitted}
    if isinstance(value,list):return [strip_file_metadata(v) for v in value]
    return value


def test_compact_preserves_all_nonselection_physics_states_and_original_metadata_bytes(tmp_path):
    original=raw_record(1);raw=write_raw(tmp_path/'raw',[original]);out=tmp_path/'compact'
    manifest=delivery.compact(raw,out)
    saved=delivery._read(out/'case_1.json.gz')
    assert delivery._retained(saved)==delivery._retained(original)
    for key in ('training_result','history_prefits','future_records','methods'):
        assert saved[key]==original[key]
    assert (out/'freeze.json').read_bytes()==(raw/'freeze.json').read_bytes()
    assert (out/'source_snapshot.json.gz').read_bytes()==(raw/'source_snapshot.json.gz').read_bytes()
    grid=saved['selection']['methods']['scalar_fixed']['raw']['grids'][0]
    assert grid['raw']['frozen_objective_upper']==2.1
    assert grid['table']['QP_calls']==600 and 'tables' not in grid['table']
    reasons=json.dumps(saved['selection']['delivery_failure_evidence'])
    assert 'failed witness retained' in reasons and 'full AC score not improved' in reasons
    assert not saved['delivery_provenance']['full_raw_byte_reconstruction_supported']
    assert saved['common_cost']==original['common_cost']
    for key in ('setup_seconds','context_seconds'):
        assert saved['selection'][key]==original['selection'][key]
    for key in ('planning_actual_seconds','dependencies','planning_seconds','power_kw'):
        assert saved['selection']['methods']['scalar_fixed'][key]==original['selection']['methods']['scalar_fixed'][key]
    assert manifest['case_byte_ratio']<.65
    assert delivery.verify(out,raw_directory=raw)['exact_retained_payload_cases']==1


@pytest.mark.parametrize('mode',['compact','analysis'])
def test_analyzer_descriptive_thresholds_and_decisions_match_exactly_except_file_metadata(tmp_path,mode):
    raw_cal=write_raw(tmp_path/'raw_cal',[raw_record(i,'calibration') for i in range(19)])
    raw_eval=write_raw(tmp_path/'raw_eval',[raw_record(100+i,'evaluation') for i in range(4)])
    compact_cal=tmp_path/'compact_cal';compact_eval=tmp_path/'compact_eval'
    produce=getattr(delivery,mode)
    produce(raw_cal,compact_cal);produce(raw_eval,compact_eval)
    rc=analysis.read_phase(raw_cal);cc=analysis.read_phase(compact_cal)
    re=analysis.read_phase(raw_eval);ce=analysis.read_phase(compact_eval)
    assert analysis.summarize(rc['records'])==analysis.summarize(cc['records'])
    assert strip_file_metadata(analysis.descriptive_analysis(re))==strip_file_metadata(analysis.descriptive_analysis(ce))
    raw_out=tmp_path/'stats_raw';compact_out=tmp_path/'stats_compact'
    rt=analysis.calibrate(rc,raw_out);ct=analysis.calibrate(cc,compact_out)
    assert strip_file_metadata(rt)==strip_file_metadata(ct)
    rv=analysis.evaluate(re,rt,raw_out,calibration_records=rc['records'])
    cv=analysis.evaluate(ce,ct,compact_out,calibration_records=cc['records'])
    assert strip_file_metadata(rv)==strip_file_metadata(cv)


def test_analysis_lite_keeps_complete_statistical_projection_and_explicitly_omits_physics(tmp_path):
    original=raw_record(1);original.update(source_id='source-v1',source_unchanged=True,
        actual_physics_calls=7,seconds=3.4,failure='retained pipeline warning')
    raw=write_raw(tmp_path/'raw',[original]);out=tmp_path/'analysis'
    manifest=delivery.analysis(raw,out);saved=delivery._read(out/'case_1.json.gz')
    assert delivery._retained(saved)==analysis.project_analysis_record(original)
    assert saved['methods']==original['methods'] and saved['common_cost']==original['common_cost']
    assert not any(k in saved for k in ('training','validation','future_records','training_result','history_prefits','selection'))
    provenance=saved['delivery_provenance']
    assert provenance['analysis_only'] and not provenance['full_meter_and_fit_evidence_included']
    assert provenance['phase']==original['phase'] and provenance['source_id']=='source-v1'
    assert provenance['raw_sha256']==hashlib.sha256((raw/'case_1.json.gz').read_bytes()).hexdigest()
    assert manifest['schema']==delivery.ANALYSIS_SCHEMA and manifest['analysis_only']
    for name in ('freeze.json','source_snapshot.json.gz'):
        assert (out/name).read_bytes()==(raw/name).read_bytes()
    assert delivery.verify(out,raw_directory=raw)['exact_retained_payload_cases']==1
    with pytest.raises(FileExistsError):delivery.analysis(raw,out)


def test_analysis_scope_tampering_and_reprojecting_delivery_are_rejected(tmp_path):
    raw=write_raw(tmp_path/'raw',[raw_record(1)]);out=tmp_path/'analysis';delivery.analysis(raw,out)
    with pytest.raises(ValueError,match='already a compact or analysis-lite'):
        delivery.analysis(out,tmp_path/'again')
    case=out/'case_1.json.gz';saved=delivery._read(case)
    saved['delivery_provenance']['full_meter_and_fit_evidence_included']=True
    data=gzip.compress(delivery._encoded(saved),mtime=0);case.write_bytes(data)
    path=out/'delivery_manifest.json';manifest=delivery._read(path)
    manifest['entries'][0].update(compact_bytes=len(data),compact_sha256=hashlib.sha256(data).hexdigest())
    manifest['manifest_id']=delivery._object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})
    path.write_bytes(delivery._encoded(manifest))
    with pytest.raises(ValueError,match='scope'):delivery.verify(out)


def test_analysis_lite_bundle_is_identified_separately_from_complete_compact(tmp_path):
    raw=write_raw(tmp_path/'raw',[raw_record(1)]);out=tmp_path/'analysis';delivery.analysis(raw,out)
    manifest=delivery.bundle(out,tmp_path/'bundle')
    assert manifest['analysis_only'] and manifest['schema']=='systematic_analysis_lite_bundle_v1'
    assert 'meters and all fitted states are absent' in manifest['restore_scope']


@pytest.mark.parametrize('tamper',['manifest','compact','raw_mapping'])
def test_tampered_mapping_or_file_hash_is_rejected(tmp_path,tamper):
    raw=write_raw(tmp_path/'raw',[raw_record(1)]);out=tmp_path/'compact';delivery.compact(raw,out)
    path=out/'delivery_manifest.json';manifest=delivery._read(path)
    if tamper=='compact':manifest['entries'][0]['compact_sha256']='0'*64
    else:manifest['entries'][0]['raw_sha256']='0'*64
    if tamper!='manifest':manifest['manifest_id']=delivery._object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})
    path.write_bytes(delivery._encoded(manifest))
    with pytest.raises(ValueError,match='hash mismatch|mapping mismatch'):
        delivery.verify(out)


def test_existing_outputs_and_recompacting_compact_records_are_rejected(tmp_path):
    raw=write_raw(tmp_path/'raw',[raw_record(1)]);out=tmp_path/'compact';delivery.compact(raw,out)
    before=(out/'case_1.json.gz').read_bytes()
    with pytest.raises(FileExistsError):delivery.compact(raw,out)
    assert (out/'case_1.json.gz').read_bytes()==before
    with pytest.raises(ValueError,match='already a compact'):delivery.compact(out,tmp_path/'again')


def test_none_selection_and_pipeline_failure_keep_inference_failure_records(tmp_path):
    record=raw_record(1);record['selection']=None;record['status']='pipeline_failure';record['failure']='AttributeError fixture'
    raw=write_raw(tmp_path/'raw',[record]);out=tmp_path/'compact';delivery.compact(raw,out)
    saved=delivery._read(out/'case_1.json.gz')
    assert saved['failure']=='AttributeError fixture' and saved['future_records']==record['future_records']
    assert saved['selection']['source_selection_available'] is False


def test_segmented_archives_stay_under_limit_and_reproduce_compact_files_only(tmp_path):
    raw=write_raw(tmp_path/'raw',[raw_record(i) for i in range(3)]);compact=tmp_path/'compact'
    delivery.compact(raw,compact);pack=tmp_path/'packs';limit=12000
    manifest=delivery.bundle(compact,pack,max_bytes=limit)
    restored=tmp_path/'restored';restored.mkdir()
    for entry in manifest['archives']:
        data=(pack/entry['file']).read_bytes()
        assert len(data)==entry['bytes']<=limit
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
        with tarfile.open(pack/entry['file'],'r:xz') as tar:
            for member in tar.getmembers():
                assert Path(member.name).name==member.name
                (restored/member.name).write_bytes(tar.extractfile(member).read())
    assert delivery.verify(restored,raw_directory=raw)['success']
    for path in compact.iterdir():assert path.read_bytes()==(restored/path.name).read_bytes()
    with pytest.raises(FileExistsError):delivery.bundle(compact,pack,max_bytes=limit)


def test_mapping_cannot_read_outside_delivery_directory(tmp_path):
    raw=write_raw(tmp_path/'raw',[raw_record(1)]);out=tmp_path/'compact';delivery.compact(raw,out)
    path=out/'delivery_manifest.json';manifest=delivery._read(path);manifest['metadata'][0]['name']='../raw/freeze.json'
    manifest['manifest_id']=delivery._object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})
    path.write_bytes(delivery._encoded(manifest))
    with pytest.raises(ValueError,match='basename'):delivery.verify(out)
