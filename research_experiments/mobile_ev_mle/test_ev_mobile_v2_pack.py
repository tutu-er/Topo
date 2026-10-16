"""Tiny synthetic packaging data; no real calibration/evaluation is accessed."""
import base64
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random

import pytest

from . import ev_mobile_v2_pack as pack


def _fixture(tmp_path,monkeypatch):
    root=tmp_path/'raw_outputs';root.mkdir();hashes={'fixture.py':sha256(b'fixture only').hexdigest()}
    frozen={}
    for version in ('orig30','guard30','guard60'):
        config=dict(methods=list(pack.delivery.METHODS),version=version)
        value=dict(config=config,source_sha256=hashes,environment=dict(python='fixture'))
        value['policy_id']=pack.merging._object_hash(dict(config=config,source_sha256=hashes));frozen[version]=value
    monkeypatch.setattr(pack,'POLICY',frozen['guard60']['policy_id'])
    snapshot=dict(source_sha256=hashes,source_base64={'fixture.py':base64.b64encode(b'fixture only').decode()})
    for number,(name,bank,phase,indices) in enumerate(pack.PLAN):
        source=root/name;source.mkdir();version='orig30' if number<5 else 'guard30' if number<8 else 'guard60';freeze=frozen[version]
        (source/'freeze.json').write_bytes(pack.delivery.base._encoded(freeze))
        (source/'source_snapshot.json.gz').write_bytes(gzip.compress(pack.delivery.base._encoded(snapshot),mtime=number))
        for index in indices:
            first=(101210001 if bank=='n3' else 101310001) if phase=='calibration' else (101220001 if bank=='n3' else 101320001) if phase=='evaluation' else (101100001 if bank=='n3' else 101100101) if phase=='pilot' else (101110001 if bank=='n3' else 101120001)
            seed=first+index
            modes={mode:dict(losses={'star':1.,'clade':2.},complete=True,M=1.) for mode in pack.delivery.MODES}
            record=dict(schema='mobile_reliability_EV_v2',physical_bank=bank,phase=phase,index=index,seed=seed,
                family_index=0,case_id='fixture',true_name='star',candidate_miss=False,status='complete',
                policy_id=freeze['policy_id'],source_sha256=hashes,source_unchanged=True,config=dict(training_points=96),
                methods=[dict(name=method,inference=modes,operation=dict(success=True),planning_failure=None) for method in pack.delivery.METHODS],
                training=dict(p=[[.1]]),future_records=dict(actual_state=dict(r=[.003],x=[.0008])),
                selection=dict(B=list(range(1000)) if phase=='evaluation' and index==0 else [1.]))
            (source/f'case_{index:03d}_{seed}.json.gz').write_bytes(gzip.compress(pack.delivery.base._encoded(record),mtime=17))
    return root


def test_declared_plan_has_352_slots_and_does_not_access_data():
    value=pack.declared_plan()
    assert value['logical_directories']==14 and value['experiment_slots']==352
    assert value['compact_representatives']==dict(orig30=4,guard60=10)
    assert value['complete_raw_records']==2


def test_full_small_mock_flow_uses_existing_tools_preserves_raw_and_preselected_phases(tmp_path,monkeypatch):
    root=_fixture(tmp_path,monkeypatch);before={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in root.rglob('case_*.json.gz')}
    evidence=root/'evidence';result=pack.pack(root,evidence)
    assert result['experiment_slots']==352 and result['logical_directories']==14
    assert [g['case_count'] for g in result['compact_groups']]==[4,10]
    guard=result['compact_groups'][1]['records']
    assert {(r['physical_bank'],r['phase'],r['index']) for r in guard}=={
        (bank,phase,index) for bank in ('n3','n4') for phase,indices in (('calibration',(0,)),('evaluation',(0,12,24,36))) for index in indices}
    assert len(result['complete_raw']['records'])==2
    assert all(r['phase']=='evaluation' and r['index']==0 for r in result['complete_raw']['records'])
    for row in result['packages']:
        path=evidence/row['file'];assert path.stat().st_size==row['bytes']<=20*1024*1024
        assert sha256(path.read_bytes()).hexdigest()==row['sha256']
    for relative,digest in before.items():assert sha256((root/relative).read_bytes()).hexdigest()==digest
    assert not list(evidence.rglob('case_*.json.gz'))  # no unbundled raw/compact duplication in delivered tree
    raw=evidence/'bundles'/'selected_complete_raw';manifest=pack.delivery.base._read(raw/'raw_bundle_manifest.json')
    for row in manifest['files']:
        if not row['original_file'].startswith('case_'):continue
        content=b''.join((raw/p['file']).read_bytes() for p in row['parts'])
        restored=json.loads(gzip.decompress(content));assert restored['selection']['B']==list(range(1000))
    assert pack.verify_raw_bundle(raw)['success']
    with pytest.raises(FileExistsError):pack.pack(root,evidence)


@pytest.mark.parametrize('tamper',['bytes','offset','whole_hash'])
def test_exact_raw_chunks_keep_large_files_without_dropping_fields_and_reject_tampering(tmp_path,tamper):
    source=tmp_path/'source';source.mkdir();content=random.Random(17).randbytes(10001)
    (source/'case_000_fixture.json.gz').write_bytes(content);out=tmp_path/'raw_bundle'
    result=pack._raw_bundle(source,out,max_bytes=2048)
    assert len(result['files'][0]['parts'])==5
    restored=b''.join((out/p['file']).read_bytes() for p in result['files'][0]['parts'])
    assert restored==content
    if tamper=='bytes':
        path=out/result['files'][0]['parts'][0]['file'];path.write_bytes(path.read_bytes()+b'changed')
    else:
        if tamper=='offset':result['files'][0]['parts'][1]['offset']=0
        else:result['files'][0]['sha256']='0'*64
        result['manifest_id']=pack.delivery.base._object_hash({k:v for k,v in result.items() if k!='manifest_id'})
        (out/'raw_bundle_manifest.json').write_bytes(pack.delivery.base._encoded(result))
    with pytest.raises(ValueError):pack.verify_raw_bundle(out)


def test_missing_declared_records_and_outside_outputs_target_are_rejected_before_output(tmp_path,monkeypatch):
    root=_fixture(tmp_path,monkeypatch);path=next((root/pack.PLAN[0][0]).glob('case_*.json.gz'));path.unlink()
    output=root/'evidence'
    with pytest.raises(ValueError,match='count'):pack.pack(root,output)
    assert not output.exists()
    with pytest.raises(ValueError,match='boundary'):pack.pack(root,tmp_path/'outside')
