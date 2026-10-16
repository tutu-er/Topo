"""Pure-IO restore tests; no restored scientific source is ever executed."""
from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import json
from pathlib import Path
import struct

import pytest

from . import ev_bayes_response_restore as restore
from . import ev_bayes_small_evidence as pack


def digest(payload):return sha256(payload).hexdigest()


def write_snapshot(folder,sources,*,policy='tiny-policy'):
    hashes={name:digest(payload) for name,payload in sources.items()}
    frozen=dict(policy_id=policy,source_sha256=hashes)
    snapshot=dict(source_sha256=hashes,source_base64={name:base64.b64encode(payload).decode('ascii') for name,payload in sources.items()})
    (folder/'preflight_freeze.json').write_bytes(pack.canonical_json(frozen))
    (folder/'preflight_source_snapshot.json.gz').write_bytes(gzip.compress(pack.canonical_json(snapshot),mtime=0))
    return frozen,snapshot


@pytest.fixture
def evidence(tmp_path):
    folder=tmp_path/'evidence';folder.mkdir()
    sources={'tiny.py':b'raise RuntimeError("restored source must never execute")\n',
        '../mobile_ev_probing/helper.py':b'# sibling\n',
        '../../terminal_case33/data/toy.py':b'# parent relative\n',
        'rnj_wzzt_core/rnj_wzzt/toy.py':b'# core\n'}
    write_snapshot(folder,sources)
    records=[];rows=[]
    for index,name in enumerate(restore.ARCHIVES):
        record=dict(policy_id='tiny-policy',spec=dict(phase='development',index=index),
            values=[-0.0,1e-300,1e300]+[float(i) for i in range(300)],
            huge=2**100,typed=[1,1.0,-0.0,True,None],nested={'zero':-0.0})
        original=folder/('original_%d.json.gz'%index)
        original.write_bytes(gzip.compress(pack.canonical_json(record),compresslevel=9,mtime=123))
        packed=pack.pack_record(original,folder/name)
        rows.append(dict(packed=packed,checked=pack.verify_record(folder/name,original)))
        records.append(record)
    (folder/'archive_audit.json').write_bytes(pack.canonical_json(dict(records=rows)))
    # Decoys must never be selected instead of the two preflight sidecars.
    (folder/'frozen_batch.json').write_text('{"policy_id":"wrong-batch"}',encoding='utf-8')
    (folder/'frozen_source_snapshot.json.gz').write_bytes(gzip.compress(b'{}'))
    return folder,sources,records


def test_restore_source_bytes_sidecars_and_exact_typed_records(evidence,tmp_path):
    folder,sources,records=evidence;target=tmp_path/'fresh'
    result=restore.restore_preflight(folder,target)
    assert result['passed'] and result['source_count']==4
    assert result['restored_sources_executed'] is False
    assert json.loads((target/'restore_audit.json').read_bytes())==result
    for name,payload in sources.items():
        assert restore._safe_source_path(target,name).read_bytes()==payload
    for source,destination in restore.SIDECARS.items():
        assert (folder/source).read_bytes()==(target/'outputs/preflight'/destination).read_bytes()
    for index,original in enumerate(records):
        restored,_=pack.read_record(target/'outputs/preflight/development'/('history_%03d.json.gz'%index))
        assert pack.exact_equal(original,restored)==(True,None)
        assert type(restored['huge']) is int and restored['huge']==2**100
        assert type(restored['typed'][0]) is int and type(restored['typed'][1]) is float
        assert struct.pack('>d',restored['values'][0])==struct.pack('>d',-0.0)
        assert result['records'][index]['verify_record']['passed']
        assert result['records'][index]['original_compressed_sha256']!=result['records'][index]['restored_compressed_sha256']


def test_refuses_even_empty_existing_target(evidence,tmp_path):
    folder,_,_=evidence;target=tmp_path/'fresh';target.mkdir()
    with pytest.raises(FileExistsError):restore.restore_preflight(folder,target)
    assert list(target.iterdir())==[]


@pytest.mark.parametrize('name',['/outside.py','C:/outside.py','C:relative.py',
    '\\server\\share\\outside.py','../../../outside.py','rnj_wzzt_core/../../outside.py',
    'x.py:stream','CON.py','folder/trailing.','folder/trailing ','a//b.py','a?b.py'])
def test_rejects_unsafe_snapshot_names_before_writes(evidence,tmp_path,name):
    folder,_,_=evidence;write_snapshot(folder,{name:b'not executed'})
    target=tmp_path/'fresh'
    with pytest.raises(ValueError):restore.restore_preflight(folder,target)
    assert not target.exists()


@pytest.mark.parametrize('sources',[
    {'x.py':b'a','./x.py':b'b'},
    {'X.py':b'a','x.py':b'b'},
    {'a':b'a','a/b.py':b'b'},
    {'../../outputs/preflight/freeze.json':b'collision'},
    {'../../restore_audit.json':b'collision'},
])
def test_rejects_aliases_and_file_collisions(evidence,tmp_path,sources):
    folder,_,_=evidence;write_snapshot(folder,sources);target=tmp_path/'fresh'
    with pytest.raises(ValueError):restore.restore_preflight(folder,target)
    assert not target.exists()


@pytest.mark.parametrize('kind',['hash','base64','manifest'])
def test_rejects_bad_snapshot_before_writes(evidence,tmp_path,kind):
    folder,sources,_=evidence;frozen,snapshot=write_snapshot(folder,sources)
    first=next(iter(snapshot['source_base64']))
    if kind=='hash':snapshot['source_base64'][first]=base64.b64encode(b'corrupted').decode()
    elif kind=='base64':snapshot['source_base64'][first]='not:base64!'
    else:snapshot['source_sha256'].pop(first)
    (folder/'preflight_source_snapshot.json.gz').write_bytes(gzip.compress(pack.canonical_json(snapshot)))
    target=tmp_path/'fresh'
    with pytest.raises((ValueError,TypeError)):restore.restore_preflight(folder,target)
    assert not target.exists()


@pytest.mark.parametrize('kind',['archive_sha','checked','policy','index'])
def test_rejects_invalid_archive_or_identity_before_writes(evidence,tmp_path,kind):
    folder,_,_=evidence;audit=json.loads((folder/'archive_audit.json').read_bytes())
    row=audit['records'][0]
    if kind=='archive_sha':row['packed']['archive_sha256']='0'*64
    elif kind=='checked':row['checked']['passed']=False
    else:
        record,_=pack.unpack_record(folder/restore.ARCHIVES[0])
        if kind=='policy':record['policy_id']='wrong-policy'
        else:record['spec']['index']=1
        original=folder/'replacement.json';original.write_bytes(pack.canonical_json(record))
        # Existing evidence is changed only in this isolated test fixture.
        row['packed']=pack.pack_record(original,folder/restore.ARCHIVES[0],overwrite=True)
        row['checked']=pack.verify_record(folder/restore.ARCHIVES[0],original)
    (folder/'archive_audit.json').write_bytes(pack.canonical_json(audit));target=tmp_path/'fresh'
    with pytest.raises(ValueError):restore.restore_preflight(folder,target)
    assert not target.exists()


def test_output_root_normalized_before_path_checks(evidence,tmp_path):
    folder,_,_=evidence;parent=tmp_path/'parent';parent.mkdir()
    result=restore.restore_preflight(folder,parent/'..'/'fresh')
    assert Path(result['output_root'])==(tmp_path/'fresh').resolve()


@pytest.mark.parametrize('position',['output','input','archive'])
def test_rejects_symlink_before_writes(evidence,tmp_path,monkeypatch,position):
    folder,_,_=evidence;target=tmp_path/'fresh'
    bad={'output':target,'input':folder,'archive':folder/restore.ARCHIVES[0]}[position].absolute()
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda self: self.absolute()==bad or original(self))
    with pytest.raises(ValueError,match='symbolic links/junctions'):restore.restore_preflight(folder,target)
    assert not target.exists()


def test_parent_directory_link_rejected(evidence,tmp_path,monkeypatch):
    folder,_,_=evidence;parent=tmp_path/'linked';target=parent/'fresh'
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda self: self.absolute()==parent.absolute() or original(self))
    with pytest.raises(ValueError,match='symbolic links/junctions'):restore.restore_preflight(folder,target)
    assert not target.exists()


def test_cli_uses_expected_audit_filename(evidence,tmp_path,monkeypatch,capsys):
    folder,_,_=evidence;target=tmp_path/'fresh'
    monkeypatch.setattr('sys.argv',['restore','--input',str(folder),'--output-root',str(target)])
    restore.main();result=json.loads(capsys.readouterr().out)
    assert result['passed'] and result['record_count']==2
    assert Path(result['manifest'])==target/'restore_audit.json'
    assert result['restored_sources_executed'] is False
