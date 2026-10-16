"""Small synthetic phase fixtures; no real formal batch is opened."""
import base64
from copy import deepcopy
import errno
import gzip
from hashlib import sha256
import json

import pytest

from . import ev_mobile_v2_merge as merge
from . import ev_mobile_v2_delivery as delivery
from . import ev_reliability_v2 as rel


def _fixture(tmp_path,*,bank='n3',phase='calibration',count=4):
    payload={'fake_source_a.py':b'print(1)\n','fake_source_b.md':b'fixture only\n'}
    hashes={k:sha256(v).hexdigest() for k,v in payload.items()}
    config=dict(methods=list(delivery.METHODS),modes=list(merge.MODES),
        calibration_histories_per_bank=count,evaluation_histories_per_bank=count,
        n3_formal_cases=['ordinary96'],n4_formal_cases=['twinblock12'])
    frozen=dict(config=config,source_sha256=hashes,environment=dict(python='fixture'))
    frozen['policy_id']=merge._object_hash(dict(config=config,source_sha256=hashes))
    snapshot=dict(source_sha256=hashes,source_base64={k:base64.b64encode(v).decode() for k,v in payload.items()})
    parts=[tmp_path/'part_a',tmp_path/'part_b'];records=[]
    n=3 if bank=='n3' else 4;root=list(range(n))
    for index in range(count):
        seed=merge.SEED_START[bank,phase]+index
        methods=[dict(name=name,planning_failure=None,operation=dict(success=True),
            inference={mode:dict(complete=True,losses={'star':1.,'clade':2.},M=1.) for mode in merge.MODES}) for name in delivery.METHODS]
        records.append(dict(schema=merge.SCHEMA,physical_bank=bank,phase=phase,index=index,seed=seed,
            case_id='ordinary96' if bank=='n3' else 'twinblock12',family_index=index%4,
            policy_id=frozen['policy_id'],source_sha256=hashes,source_unchanged=True,status='complete',
            true_name='star',candidate_miss=False,methods=methods,config=dict(training_points=96),
            candidates=[dict(name='star',supports=[root,*[[i] for i in range(n)]]),
                dict(name='clade',supports=[root,[0,1],*[[i] for i in range(n)]])]))
    for j,directory in enumerate(parts):
        directory.mkdir();(directory/'freeze.json').write_bytes(merge._encoded(frozen))
        (directory/'source_snapshot.json.gz').write_bytes(gzip.compress(merge._encoded(snapshot),mtime=123+j))
        subset=records[:count//2] if j==0 else records[count//2:]
        (directory/'checkpoint.json').write_bytes(merge._encoded(dict(bank=bank,phase=phase,planned=len(subset),completed=len(subset))))
        for row in subset:
            (directory/f"case_{row['index']:03d}_{row['seed']}.json.gz").write_bytes(gzip.compress(merge._encoded(row),mtime=17))
    return parts,frozen,records


def _case(directory,index):return next(directory.glob(f'case_{index:03d}_*.json.gz'))
def _rewrite(path,value):path.write_bytes(gzip.compress(merge._encoded(value),mtime=17))


def test_hardlink_merge_preserves_original_bytes_all_metadata_and_decoded_snapshot_equality(tmp_path):
    parts,frozen,_=_fixture(tmp_path);out=tmp_path/'merged'
    assert (parts[0]/'source_snapshot.json.gz').read_bytes()!=(parts[1]/'source_snapshot.json.gz').read_bytes()
    result=merge.merge(parts,out,bank='n3',phase='calibration',expected_count=4,expected_policy_id=frozen['policy_id'])
    assert result['case_count']==result['unique_seed_count']==4
    # The sandbox/filesystem may disallow creating hard links; copied bytes
    # are the declared fallback and their exact hash is checked below.
    assert sum(result['materialization_counts'].values())==4
    assert set(result['materialization_counts'])<={'hardlink','copy'}
    for row in result['entries']:
        assert (out/row['output_file']).read_bytes()==(parts[row['source_chunk']]/row['source_file']).read_bytes()
    assert (out/'freeze.json').read_bytes()==(parts[0]/'freeze.json').read_bytes()
    assert (out/'source_snapshot.json.gz').read_bytes()==(parts[0]/'source_snapshot.json.gz').read_bytes()
    assert merge.verify(out,bank='n3',phase='calibration',expected_count=4,expected_policy_id=frozen['policy_id'])['success']
    phase=rel.read_phase(out,expected_phase='calibration')
    assert len(phase['records'])==4 and len(phase['methods'])==11


def test_pipeline_and_planning_failures_remain_in_the_denominator(tmp_path):
    parts,_,_=_fixture(tmp_path);path=_case(parts[0],0);row,_=merge._read(path)
    row.update(status='pipeline_failure',failure='kept failure')
    for method in row['methods']:
        method['planning_failure']='kept planning failure';method['operation']['success']=False
        for value in method['inference'].values():value.update(complete=False,losses={},M=None)
    _rewrite(path,row);result=merge.merge(parts,tmp_path/'merged',bank='n3',phase='calibration')
    assert result['case_count']==4 and result['failed_record_slots']==1
    assert result['failure_counts']==dict(planning_failures=11,operation_failures=11,incomplete_inference_slots=33)
    assert result['entries'][0]['failure']=='kept failure'


@pytest.mark.parametrize('tamper',['schema','bank','phase','index','seed','policy','sources','source_drift','methods','mode','case','family'])
def test_wrong_record_identity_schedule_or_schema_is_rejected_before_output(tmp_path,tamper):
    parts,_,_=_fixture(tmp_path);path=_case(parts[1],2);row,_=merge._read(path)
    if tamper=='schema':row['schema']='wrong'
    elif tamper=='bank':row['physical_bank']='n4'
    elif tamper=='phase':row['phase']='evaluation'
    elif tamper=='index':row['index']=4
    elif tamper=='seed':row['seed']+=10
    elif tamper=='policy':row['policy_id']='0'*64
    elif tamper=='sources':row['source_sha256']={}
    elif tamper=='source_drift':row['source_unchanged']=False
    elif tamper=='methods':row['methods'].pop()
    elif tamper=='mode':row['methods'][0]['inference'].pop('joint_total')
    elif tamper=='case':row['case_id']='not_declared'
    else:row['family_index']=4
    _rewrite(path,row);out=tmp_path/'merged'
    with pytest.raises(ValueError):merge.merge(parts,out,bank='n3',phase='calibration')
    assert not out.exists()


@pytest.mark.parametrize('tamper',['missing','duplicate','extra_file','freeze','snapshot'])
def test_missing_extra_duplicate_or_changed_chunk_evidence_rejected(tmp_path,tamper):
    parts,_,_=_fixture(tmp_path)
    if tamper=='missing':_case(parts[1],2).unlink()
    elif tamper=='duplicate':
        row,_=merge._read(_case(parts[0],0));_rewrite(_case(parts[1],2),row)
    elif tamper=='extra_file':(parts[1]/'case_extra.tmp').write_bytes(b'extra')
    elif tamper=='freeze':
        path=parts[1]/'freeze.json';row,_=merge._read(path);row['environment']['python']='different';path.write_bytes(merge._encoded(row))
    else:
        path=parts[1]/'source_snapshot.json.gz';row,_=merge._read(path)
        row['source_base64']['fake_source_a.py']=base64.b64encode(b'changed').decode();_rewrite(path,row)
    with pytest.raises(ValueError):merge.merge(parts,tmp_path/'merged',bank='n3',phase='calibration')


def test_copy_fallback_and_readonly_idempotent_verification(tmp_path,monkeypatch):
    parts,frozen,_=_fixture(tmp_path);out=tmp_path/'merged'
    def unavailable(*args,**kwargs):raise OSError(errno.EXDEV,'fixture cross-device')
    monkeypatch.setattr(merge.os,'link',unavailable)
    result=merge.merge(parts,out,bank='n3',phase='calibration')
    assert result['materialization_counts']=={'copy':4}
    before={p.name:p.read_bytes() for p in out.iterdir()}
    assert merge.verify(out,expected_policy_id=frozen['policy_id'])['success']
    assert merge.verify(out,expected_policy_id=frozen['policy_id'])['success']
    with pytest.raises(FileExistsError):merge.merge(parts,out,bank='n3',phase='calibration')
    assert {p.name:p.read_bytes() for p in out.iterdir()}==before


@pytest.mark.parametrize('tamper',['case','metadata','manifest','extra'])
def test_restart_verification_rejects_changed_or_extra_output(tmp_path,tamper):
    parts,_,_=_fixture(tmp_path);out=tmp_path/'merged';merge.merge(parts,out,bank='n3',phase='calibration')
    if tamper=='case':
        # A source hard link is intentionally shared; only fixture bytes change.
        p=_case(out,0);p.write_bytes(p.read_bytes()+b'changed')
    elif tamper=='metadata':(out/'source_000_checkpoint.json').write_bytes(b'changed')
    elif tamper=='extra':(out/'case_extra.json').write_bytes(b'{}')
    else:
        p=out/'merge_manifest.json';r,_=merge._read(p);r['case_count']=5;p.write_bytes(merge._encoded(r))
    with pytest.raises(ValueError):merge.verify(out)


def test_analysis_lite_blocks_are_supported_and_raw_mix_is_rejected(tmp_path):
    parts,_,_=_fixture(tmp_path);lite=[tmp_path/'lite_a',tmp_path/'lite_b']
    for source,target in zip(parts,lite):delivery.analysis(source,target)
    result=merge.merge(lite,tmp_path/'lite_merged',bank='n3',phase='calibration')
    assert result['evidence_kind']=='analysis_lite' and merge.verify(tmp_path/'lite_merged')['success']
    assert len(rel.read_phase(tmp_path/'lite_merged')['records'])==4
    with pytest.raises(ValueError,match='mix'):merge.merge([parts[0],lite[1]],tmp_path/'mixed',bank='n3',phase='calibration')
