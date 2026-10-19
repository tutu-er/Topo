"""Source-only restoration controls; no scientific solver is called."""
import base64
from copy import deepcopy
import gzip
import json
import pytest
from . import ev_profile_source_restore as restore


def fixture(tmp_path):
    source=tmp_path/'research_experiments/mobile_ev_mle';source.mkdir(parents=True)
    names=['../mobile_ev_probing/shared.py']+[f'file_{i}.py' for i in range(24)]
    raw={name:b'first\nsecond\r\nthird\n' for name in names}
    hashes={name:restore.digest(data) for name,data in raw.items()}
    frozen={'hashes':hashes}
    saved={'schema':'source_bytes_base64','freeze_sha256':'a'*64,'sources':{name:{'sha256':hashes[name],'bytes_base64':base64.b64encode(data).decode()} for name,data in raw.items()}}
    for name,data in raw.items():
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    freeze=tmp_path/'freeze.json';snapshot=tmp_path/'source.json.gz'
    def write():
        freeze.write_text(json.dumps(frozen),encoding='utf-8')
        snapshot.write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    write()
    return source,freeze,snapshot,raw,frozen,saved,write


def test_exact25_default_audit_is_read_only(tmp_path):
    source,freeze,snapshot,raw,*_=fixture(tmp_path)
    report=restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source)
    assert report['success'] and report['checked_count']==report['exact_after_count']==25
    assert report['exact_before_count']==25 and report['restored_count']==0
    assert not report['restore_requested'] and report['scientific_fit_calls']==0
    assert all((source/name).read_bytes()==data for name,data in raw.items())


def test_mixed_newlines_require_explicit_restore_and_keep_frozen_original_bytes(tmp_path):
    source,freeze,snapshot,raw,*_=fixture(tmp_path)
    name=next(iter(raw));target=source/name;uniform=raw[name].replace(b'\r\n',b'\n').replace(b'\n',b'\r\n');target.write_bytes(uniform)
    report=restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source)
    assert not report['success'] and report['restoration_safe'] and report['line_ending_only_count']==1
    assert target.read_bytes()==uniform and report['restored_count']==0
    repaired=restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source,restore_line_endings=True)
    assert repaired['success'] and repaired['restored_count']==1 and repaired['exact_after_count']==25
    assert target.read_bytes()==raw[name]
    assert all(row['after_sha256']==row['expected_sha256'] for row in repaired['files'])


def test_genuine_content_change_blocks_whole_batch_even_with_safe_newline_file(tmp_path):
    source,freeze,snapshot,raw,*_=fixture(tmp_path);names=list(raw)
    first=source/names[0];first.write_bytes(raw[names[0]].replace(b'\r\n',b'\n'));before=first.read_bytes()
    (source/names[-1]).write_bytes(b'genuine edit\n')
    report=restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source,restore_line_endings=True)
    assert not report['restoration_safe'] and report['content_changed_count']==1 and report['restored_count']==0
    assert first.read_bytes()==before and (source/names[-1]).read_bytes()==b'genuine edit\n'


def test_missing_file_blocks_whole_batch_and_is_not_created(tmp_path):
    source,freeze,snapshot,raw,*_=fixture(tmp_path);names=list(raw)
    first=source/names[0];first.write_bytes(raw[names[0]].replace(b'\r\n',b'\n'));before=first.read_bytes()
    missing=source/names[-1];missing.unlink()
    report=restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source,restore_line_endings=True)
    assert not report['restoration_safe'] and report['missing_count']==1 and report['restored_count']==0
    assert first.read_bytes()==before and not missing.exists()


def test_snapshot_policy_keys_hashes_and_decoded_bytes_must_match_external_freeze(tmp_path):
    source,freeze,snapshot,raw,frozen,saved,write=fixture(tmp_path);original=deepcopy(saved);name=next(iter(raw))
    for damage in ('provenance_format','keys','hash','decoded_bytes','base64','external_hash'):
        saved.clear();saved.update(deepcopy(original))
        if damage=='provenance_format':saved['freeze_sha256']='invalid'
        elif damage=='keys':saved['sources'].pop(name)
        elif damage=='hash':saved['sources'][name]['sha256']='b'*64
        elif damage=='decoded_bytes':saved['sources'][name]['bytes_base64']=base64.b64encode(b'wrong content').decode()
        elif damage=='base64':saved['sources'][name]['bytes_base64']='invalid!base64'
        else:frozen['hashes'][name]='b'*64
        write()
        with pytest.raises(ValueError):restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source,restore_line_endings=True)
        assert all((source/n).read_bytes()==data for n,data in raw.items())


def test_absolute_escape_duplicate_targets_and_wrong_root_are_rejected(tmp_path):
    source,freeze,snapshot,raw,frozen,saved,write=fixture(tmp_path);name=next(iter(raw));original_frozen=deepcopy(frozen);original_saved=deepcopy(saved)
    for bad in ('../../outside.py','Z:/outside.py','/outside.py','file_0.py','./file_1.py','dir//file.py','dir\\file.py','NUL.py','file_1.py.'):
        frozen.clear();frozen.update(deepcopy(original_frozen));saved.clear();saved.update(deepcopy(original_saved))
        # Replace the sibling source with another spelling; file_0 is a duplicate target.
        replacement='unused/../file_0.py' if bad=='file_0.py' else bad
        frozen['hashes'][replacement]=frozen['hashes'].pop(name)
        saved['sources'][replacement]=saved['sources'].pop(name);write()
        with pytest.raises(ValueError):restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source,restore_line_endings=True)
        assert all((source/n).read_bytes()==data for n,data in raw.items())
    with pytest.raises(ValueError,match='direct child'):
        restore.audit_source_bytes(snapshot,freeze=freeze,source_root=tmp_path/'not_research/source')


def test_freeze_newline_hash_is_provenance_and_not_a_source_map_override(tmp_path):
    source,freeze,snapshot,raw,frozen,saved,write=fixture(tmp_path)
    saved['freeze_sha256']=restore.digest(freeze.read_bytes());write()
    before=freeze.read_bytes();freeze.write_bytes(before+b'\r\n')
    result=restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source)
    assert result['success'] and not result['freeze_provenance_bytes_match']
    frozen['hashes'][next(iter(raw))]='b'*64;write()
    with pytest.raises(ValueError):restore.audit_source_bytes(snapshot,freeze=freeze,source_root=source,restore_line_endings=True)
    assert all((source/name).read_bytes()==data for name,data in raw.items())
