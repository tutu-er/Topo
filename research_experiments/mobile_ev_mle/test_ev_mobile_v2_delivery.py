"""Eleven-method scope, immutable hashes, replay and calibration preservation."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from . import ev_mobile_v2_delivery as delivery
from . import ev_reliability_v2 as rel
from .test_ev_systematic_delivery import raw_record as old_record


def record(seed,phase='development'):
    value=old_record(seed,phase);template=value['methods'][0]
    value['methods']=[]
    for j,name in enumerate(delivery.METHODS):
        row=deepcopy(template);row['name']=name;row['planning_actual_seconds']=j/100.
        row['planning_seconds']=j/10.;value['methods'].append(row)
    value.update(schema='mobile_reliability_EV_v2',physical_bank='n3',
        candidates=[dict(name='a',supports=[[0,1,2],[0],[1],[2]]),
            dict(name='b',supports=[[0,1,2],[0,1],[0],[1],[2]]),
            dict(name='c',supports=[[0,1,2],[1,2],[0],[1],[2]])])
    value['config'].update(training_points=8)
    return value


def write_raw(directory,records):
    directory.mkdir(parents=True)
    (directory/'freeze.json').write_text(json.dumps(dict(source_id='source-v1',config_id='config-v1',
        config=dict(methods=list(delivery.METHODS)))),encoding='utf8')
    (directory/'source_snapshot.json.gz').write_bytes(gzip.compress(b'{"source":"unchanged snapshot bytes"}',mtime=17))
    for row in records:
        (directory/f"case_{row['seed']}.json.gz").write_bytes(gzip.compress(json.dumps(row,sort_keys=True).encode(),mtime=23))
    return directory


@pytest.mark.parametrize('mode',['compact','analysis'])
def test_eleven_methods_snapshots_and_retained_payload_are_preserved(tmp_path,mode):
    original=record(1);raw=write_raw(tmp_path/'raw',[original]);out=tmp_path/mode
    manifest=getattr(delivery,mode)(raw,out);saved=delivery.base._read(out/'case_1.json.gz')
    expected=delivery.project_analysis_record(original) if mode=='analysis' else delivery.base._retained(original)
    assert delivery.base._retained(saved)==expected
    assert saved['methods']==original['methods'] and len(saved['methods'])==11
    assert manifest['declared_methods']==list(delivery.METHODS) and manifest['physical_banks']==['n3']
    assert saved['delivery_provenance']['physical_bank']=='n3'
    for name in ('freeze.json','source_snapshot.json.gz'):
        assert (raw/name).read_bytes()==(out/name).read_bytes()
    if mode=='analysis':
        assert not any(k in saved for k in ('training','future_records','history_prefits','selection'))
        assert saved['delivery_provenance']['analysis_only']
    else:
        assert saved['future_records']==original['future_records']
        assert 'tables' not in saved['selection']['methods']['scalar_fixed']['raw']['grids'][0]['table']
    checked=delivery.verify(out,raw_directory=raw)
    assert checked['success'] and checked['exact_retained_payload_cases']==1 and checked['method_count']==11


@pytest.mark.parametrize('mode',['compact','analysis'])
def test_immutable_outputs_and_repackaged_records_are_rejected(tmp_path,mode):
    raw=write_raw(tmp_path/'raw',[record(1)]);out=tmp_path/mode
    getattr(delivery,mode)(raw,out);before=(out/'case_1.json.gz').read_bytes()
    with pytest.raises(FileExistsError):getattr(delivery,mode)(raw,out)
    assert (out/'case_1.json.gz').read_bytes()==before
    with pytest.raises(ValueError,match='already a compact'):getattr(delivery,mode)(out,tmp_path/'again')


@pytest.mark.parametrize('tamper',['methods','mode','margin','drift'])
def test_record_validation_rejects_missing_methods_modes_bad_numbers_and_drift(tamper):
    value=record(1)
    if tamper=='methods':value['methods'].pop()
    elif tamper=='mode':value['methods'][0]['inference'].pop('joint_total')
    elif tamper=='margin':value['methods'][0]['inference']['joint_total']['M']=float('inf')
    else:value['source_unchanged']=False
    with pytest.raises(ValueError):delivery.project_analysis_record(value)


def test_declaration_and_physical_bank_tampering_rejected_even_with_rehashed_bytes(tmp_path):
    raw=write_raw(tmp_path/'raw',[record(1)]);out=tmp_path/'analysis';delivery.analysis(raw,out)
    case=out/'case_1.json.gz';saved=delivery.base._read(case)
    saved['delivery_provenance']['physical_bank']='n4'
    data=gzip.compress(delivery.base._encoded(saved),mtime=0);case.write_bytes(data)
    path=out/'delivery_manifest.json';manifest=delivery.base._read(path)
    manifest['entries'][0].update(compact_bytes=len(data),compact_sha256=hashlib.sha256(data).hexdigest())
    manifest['manifest_id']=delivery.base._object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})
    path.write_bytes(delivery.base._encoded(manifest))
    with pytest.raises(ValueError,match='physical-bank'):delivery.verify(out,raw_directory=raw)
    manifest['declared_methods'].pop();manifest['manifest_id']=delivery.base._object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})
    path.write_bytes(delivery.base._encoded(manifest))
    with pytest.raises(ValueError,match='declaration'):delivery.verify(out)


def test_failed_inference_slots_and_failure_reasons_survive_analysis_projection(tmp_path):
    value=record(1);value.update(status='pipeline_failure',failure='retained optimizer exception')
    value['methods'][0]['inference']['joint_total'].update(complete=False,losses={},M=None)
    raw=write_raw(tmp_path/'raw',[value]);out=tmp_path/'analysis';delivery.analysis(raw,out)
    saved=delivery.base._read(out/'case_1.json.gz')
    assert saved['failure']==value['failure'] and saved['methods']==value['methods']
    assert len(saved['methods'])==11 and not saved['methods'][0]['inference']['joint_total']['complete']


@pytest.mark.parametrize('mode',['compact','analysis'])
def test_all_five_reliability_rules_thresholds_and_evaluation_survive_delivery(tmp_path,mode):
    raw_cal=write_raw(tmp_path/'raw_cal',[record(i,'calibration') for i in range(12)])
    raw_eval=write_raw(tmp_path/'raw_eval',[record(100+i,'evaluation') for i in range(3)])
    out_cal=tmp_path/'pack_cal';out_eval=tmp_path/'pack_eval'
    getattr(delivery,mode)(raw_cal,out_cal);getattr(delivery,mode)(raw_eval,out_eval)
    rc,cc=rel.read_phase(raw_cal),rel.read_phase(out_cal)
    re,ce=rel.read_phase(raw_eval),rel.read_phase(out_eval)
    rt=rel.calibrate(rc,tmp_path/'raw_stats',expected_count=12)
    ct=rel.calibrate(cc,tmp_path/'pack_stats',expected_count=12)
    assert rt['rules']==ct['rules'] and rt['candidate_supports']==ct['candidate_supports']
    rr=rel.evaluate(re,rt,tmp_path/'raw_stats',expected_count=3)
    cr=rel.evaluate(ce,ct,tmp_path/'pack_stats',expected_count=3)
    for field in ('overall','by_history','by_case','by_H','by_truth'):
        assert rr[field]==cr[field]


def test_segmented_archives_restore_valid_eleven_method_files(tmp_path):
    raw=write_raw(tmp_path/'raw',[record(i) for i in range(3)])
    out=tmp_path/'analysis';delivery.analysis(raw,out);archive=tmp_path/'bundles'
    manifest=delivery.bundle(out,archive,max_bytes=16000);restored=tmp_path/'restored';restored.mkdir()
    for row in manifest['archives']:
        assert row['bytes']<=16000
        with tarfile.open(archive/row['file'],'r:xz') as tar:
            for member in tar.getmembers():
                assert Path(member.name).name==member.name
                (restored/member.name).write_bytes(tar.extractfile(member).read())
    assert delivery.verify(restored,raw_directory=raw)['success']
