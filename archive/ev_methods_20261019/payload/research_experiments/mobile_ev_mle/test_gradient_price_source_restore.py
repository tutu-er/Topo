"""Exact-byte restore tests; altered code and escaped paths are never written."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from .gradient_price_source_restore import audit_source_bytes


def snapshot_fixture(tmp_path,entries):
    source=tmp_path/'research_experiments'/'mobile_ev_mle';source.mkdir(parents=True)
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    snap=tmp_path/'snapshot.json.gz';freeze=tmp_path/'freeze.json'
    saved=dict(schema='gradient_price_source_bytes_v1',source_sha256=hashes,
               files={name:base64.b64encode(data).decode('ascii') for name,data in entries.items()})
    snap.write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    freeze.write_text(json.dumps(dict(source_sha256=hashes)))
    return source,snap,freeze,saved


def test_exact_audit_and_restore_are_no_ops(tmp_path):
    expected=b"x = 1\r\nprint(x)\r\n"
    source,snap,freeze,_=snapshot_fixture(tmp_path,{'a.py':expected})
    file=source/'a.py';file.write_bytes(expected);before=file.stat().st_mtime_ns
    for restore in (False,True):
        result=audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=restore)
        assert result['success'] and result['checked_count']==1 and result['restored_count']==0
        assert file.read_bytes()==expected and file.stat().st_mtime_ns==before


def test_read_only_reports_then_restores_only_line_endings(tmp_path):
    expected=b"x = 1\r\nprint(x)\r\n"
    source,snap,freeze,_=snapshot_fixture(tmp_path,{'a.py':expected})
    file=source/'a.py';current=expected.replace(b'\r\n',b'\n');file.write_bytes(current)
    audit=audit_source_bytes(snap,source_root=source,freeze=freeze)
    assert not audit['success'] and audit['restoration_safe'] and audit['line_ending_only_count']==1
    assert file.read_bytes()==current
    result=audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert result['success'] and result['restored_count']==1 and result['exact_after_count']==1
    assert file.read_bytes()==expected


def test_genuine_code_change_blocks_every_restore_write(tmp_path):
    expected=b'x = 1\r\n'
    source,snap,freeze,_=snapshot_fixture(tmp_path,{'a.py':expected,'b.py':expected})
    (source/'a.py').write_bytes(b'x = 1\n');(source/'b.py').write_bytes(b'x = 2\n')
    result=audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert not result['success'] and not result['restoration_safe'] and result['restored_count']==0
    assert result['content_changed_count']==1
    assert (source/'a.py').read_bytes()==b'x = 1\n'
    assert (source/'b.py').read_bytes()==b'x = 2\n'


@pytest.mark.parametrize('name',['../../outside.py','C:/outside.py','C:outside.py','/outside.py'])
def test_escaped_and_absolute_paths_rejected_without_writes(tmp_path,name):
    source,snap,freeze,_=snapshot_fixture(tmp_path,{name:b'print(1)\n'})
    outside=tmp_path/'outside.py';outside.write_bytes(b'private sentinel')
    with pytest.raises(ValueError,match='path|absolute'):
        audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert outside.read_bytes()==b'private sentinel'


def test_mobile_ev_probing_neighbor_is_inside_allowed_root(tmp_path):
    expected=b'print(1)\r\n'
    source,snap,freeze,_=snapshot_fixture(tmp_path,{'../mobile_ev_probing/a.py':expected})
    neighbor=source.parent/'mobile_ev_probing';neighbor.mkdir();file=neighbor/'a.py';file.write_bytes(b'print(1)\n')
    result=audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert result['success'] and file.read_bytes()==expected


def test_snapshot_byte_hash_tamper_is_rejected_before_writes(tmp_path):
    source,snap,freeze,saved=snapshot_fixture(tmp_path,{'a.py':b'x = 1\r\n'})
    file=source/'a.py';file.write_bytes(b'x = 1\n')
    saved['files']['a.py']=base64.b64encode(b'x = 2\r\n').decode()
    snap.write_bytes(gzip.compress(json.dumps(saved).encode()))
    with pytest.raises(ValueError,match='SHA-256'):
        audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert file.read_bytes()==b'x = 1\n'


def test_external_freeze_mismatch_and_missing_file_refuse_restore(tmp_path):
    source,snap,freeze,_=snapshot_fixture(tmp_path,{'a.py':b'x = 1\r\n'})
    result=audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert not result['success'] and result['missing_count']==1 and result['restored_count']==0
    freeze.write_text(json.dumps(dict(source_sha256={'a.py':'0'*64})))
    with pytest.raises(ValueError,match='external freeze'):
        audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert not (source/'a.py').exists()


def test_bare_carriage_return_change_is_not_allowed_crlf_lf_restore(tmp_path):
    source,snap,freeze,_=snapshot_fixture(tmp_path,{'a.py':b'x = 1\nprint(x)\n'})
    file=source/'a.py';file.write_bytes(b'x = 1\rprint(x)\r')
    result=audit_source_bytes(snap,source_root=source,freeze=freeze,restore_line_endings=True)
    assert not result['success'] and result['content_changed_count']==1
    assert file.read_bytes()==b'x = 1\rprint(x)\r'
