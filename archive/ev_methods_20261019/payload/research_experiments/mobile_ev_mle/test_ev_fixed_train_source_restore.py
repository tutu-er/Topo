"""New-snapshot adapter; all writes are limited to tiny D-root fixtures."""
import base64
import gzip
import json
import sys

import pytest

from . import ev_fixed_train_source_restore as restore


def _fixture(tmp_path):
    repo=tmp_path/"repository";source=repo/"research_experiments/mobile_ev_mle";source.mkdir(parents=True)
    directory=source/"evidence/fixture";directory.mkdir(parents=True)
    data=b"first\r\nsecond\nthird\r\n"
    names=("local.py","../mobile_ev_probing/neighbor.py","rnj_wzzt_core/package/core.py")
    hashes={};encoded={}
    for name in names:
        path=repo/restore.safe.source_repository_path(name);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data);hashes[name]=restore.safe.digest(data);encoded[name]=base64.b64encode(data).decode()
    snapshot=directory/"source_snapshot.json.gz";freeze=directory/"freeze.json"
    snapshot.write_bytes(gzip.compress(json.dumps(dict(source_sha256=hashes,source_base64=encoded)).encode(),mtime=0))
    freeze.write_text(json.dumps(dict(source_sha256=hashes,source_count=3)),encoding="utf-8")
    options=dict(source_root=source,repository_root=repo,expected_count=3)
    return directory,source,repo,data,options


def test_default_readonly_then_explicit_newline_only_restore_and_unchanged_evidence(tmp_path):
    directory,source,repo,data,options=_fixture(tmp_path)
    evidence={p.name:p.read_bytes() for p in directory.iterdir()}
    target=source/"local.py";target.write_bytes(data.replace(b"\r\n",b"\n"));old=target.read_bytes()
    audit=restore.audit_source_bytes(directory,**options)
    assert not audit["success"] and audit["restoration_safe"] and audit["source_writes"]==0
    assert target.read_bytes()==old and audit["line_ending_only_count"]==1
    restored=restore.audit_source_bytes(directory,restore_line_endings=True,**options)
    assert restored["success"] and restored["restored_count"]==1 and restored["exact_after_count"]==3
    assert target.read_bytes()==data
    assert all((directory/name).read_bytes()==value for name,value in evidence.items())
    assert not restored["temporary_adapter_retained"]


def test_genuine_change_or_missing_source_blocks_whole_batch_before_newline_writes(tmp_path):
    directory,source,repo,data,options=_fixture(tmp_path)
    local=source/"local.py";local.write_bytes(data.replace(b"\r\n",b"\n"));before=local.read_bytes()
    changed=repo/"research_experiments/mobile_ev_probing/neighbor.py";changed.write_bytes(b"changed content\n")
    report=restore.audit_source_bytes(directory,restore_line_endings=True,**options)
    assert not report["success"] and report["content_changed_count"]==1 and report["source_writes"]==0
    assert local.read_bytes()==before and changed.read_bytes()==b"changed content\n"
    changed.unlink()
    report=restore.audit_source_bytes(directory,restore_line_endings=True,**options)
    assert not report["success"] and report["missing_count"]==1 and report["source_writes"]==0
    assert local.read_bytes()==before and not changed.exists()


def test_bad_snapshot_bytes_or_manifest_never_overwrite_source(tmp_path):
    directory,source,repo,data,options=_fixture(tmp_path)
    path=directory/"source_snapshot.json.gz"
    saved=restore.safe.load(path);saved["source_base64"]["local.py"]=base64.b64encode(b"wrong").decode()
    path.write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    with pytest.raises(ValueError,match="bytes/hash mismatch"):
        restore.audit_source_bytes(directory,restore_line_endings=True,**options)
    assert (source/"local.py").read_bytes()==data
    saved["source_sha256"]["local.py"]="0"*64
    path.write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    with pytest.raises(ValueError,match="manifests differ"):
        restore.audit_source_bytes(directory,restore_line_endings=True,**options)


def test_adapter_reuses_path_safety_and_requires_bounded_cleanup_area(tmp_path,monkeypatch,capsys):
    directory,source,repo,data,options=_fixture(tmp_path)
    with pytest.raises(ValueError,match="inside the intended repository"):
        restore.audit_source_bytes(directory,temporary_root=tmp_path/"outside",**options)
    for name in ("../../escape.py","../other_parent/protocol.md","a.py:alternate","RNJ_WZZT_CORE/a.py"):
        with pytest.raises(ValueError):restore.safe.source_repository_path(name)
    no_count={k:v for k,v in options.items() if k!="expected_count"}
    with pytest.raises(ValueError,match="78 or 79"):
        restore.audit_source_bytes(directory,**no_count)
    # A restore-path exception may follow a write in a race; report its write
    # count as unknown. The default read-only path can safely report zero.
    cli=["source_restore","--directory",str(directory),"--source-root",str(source),"--repository-root",str(repo)]
    for requested in (False,True):
        monkeypatch.setattr(sys,"argv",cli+(["--restore-line-endings"] if requested else []))
        with pytest.raises(SystemExit) as stopped:restore.main()
        assert stopped.value.code==2
        result=json.loads(capsys.readouterr().out)
        assert result["source_writes"] is None if requested else result["source_writes"]==0
        assert (source/"local.py").read_bytes()==data
