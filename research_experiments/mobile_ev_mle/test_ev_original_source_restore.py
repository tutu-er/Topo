"""Whole-batch newline-only recovery and canonical-target safety tests."""
import base64,gzip,hashlib,json,os,sys
from pathlib import Path
import pytest
from . import ev_original_source_restore as restore


def fixture(tmp_path,names=('a.py','../mobile_ev_probing/b.py','rnj_wzzt_core/package/c.py')):
    repo=tmp_path/'repo';source=repo/'research_experiments/mobile_ev_mle';source.mkdir(parents=True)
    data=b'first\r\nsecond\nthird\r\n';saved={};hashes={}
    for name in names:
        path=repo/restore.source_repository_path(name);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data);hashes[name]=restore.digest(data)
        saved[name]=dict(sha256=hashes[name],bytes_base64=base64.b64encode(data).decode())
    snap=tmp_path/'source_snapshot.json.gz';snap.write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(dict(source_sha256=hashes)),encoding='utf-8')
    args=dict(snapshot=snap,freeze=freeze,source_root=source,repository_root=repo,expected_count=len(names))
    return repo,source,data,saved,args


def test_default_read_only_and_exact_mixed_newline_restoration(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path)
    path=source/'a.py';path.write_bytes(data.replace(b'\r\n',b'\n'));before=path.read_bytes()
    checked=restore.audit_source_bytes(**args)
    assert not checked['success'] and checked['restoration_safe'] and checked['restored_count']==0
    assert checked['line_ending_only_count']==1 and path.read_bytes()==before
    written=restore.audit_source_bytes(**args,restore_line_endings=True)
    assert written['success'] and written['restored_count']==1 and written['exact_after_count']==3
    assert path.read_bytes()==data
    assert restore.audit_source_bytes(**args)['all_bytes_match']


def test_real_content_change_refuses_whole_batch_without_writing_newlines(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path)
    (source/'a.py').write_bytes(data.replace(b'\r\n',b'\n'))
    other=repo/restore.source_repository_path('../mobile_ev_probing/b.py');other.write_bytes(b'genuine edit\n')
    before={name:(repo/restore.source_repository_path(name)).read_bytes() for name in saved}
    report=restore.audit_source_bytes(**args,restore_line_endings=True)
    assert not report['success'] and report['content_changed_count']==1 and report['restored_count']==0
    assert all((repo/restore.source_repository_path(name)).read_bytes()==value for name,value in before.items())


def test_missing_source_is_never_created_and_blocks_batch(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path)
    (source/'a.py').write_bytes(data.replace(b'\r\n',b'\n'));before=(source/'a.py').read_bytes()
    missing=repo/restore.source_repository_path('../mobile_ev_probing/b.py');missing.unlink()
    r=restore.audit_source_bytes(**args,restore_line_endings=True)
    assert r['missing_count']==1 and not r['success'] and r['restored_count']==0
    assert not missing.exists() and (source/'a.py').read_bytes()==before


@pytest.mark.parametrize('name',['../../escape.py','../elsewhere/a.py','./a.py','a//b.py','a/../b.py',
    'C:/escape.py','\\\\server\\file','a\\b.py','a.py:stream','a.py.','NUL.py','Research_Experiments/a.py','RNJ_WZZT_CORE/a.py'])
def test_paths_reject_escape_and_aliases(name):
    with pytest.raises(ValueError):restore.source_repository_path(name)


def test_snapshot_bytes_and_external_hashes_both_checked(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path)
    saved['a.py']['bytes_base64']=base64.b64encode(b'changed').decode()
    args['snapshot'].write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    with pytest.raises(ValueError,match='bytes/hash mismatch'):restore.audit_source_bytes(**args,restore_line_endings=True)
    assert (source/'a.py').read_bytes()==data


def test_duplicate_json_keys_rejected(tmp_path):
    bad=tmp_path/'duplicate.json';bad.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate JSON'):restore.load(bad)


def test_case_alias_target_rejected(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path,names=('a.py',))
    (source/'a.py').rename(source/'A.py')
    with pytest.raises(ValueError,match='case alias'):restore.audit_source_bytes(**args,restore_line_endings=True)


def test_symlink_outside_rejected(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path,names=('a.py',))
    outside=tmp_path/'outside.py';outside.write_bytes(data)
    target=source/'a.py';target.unlink()
    try:target.symlink_to(outside)
    except OSError as e:pytest.skip('symlink permission unavailable: '+str(e))
    with pytest.raises(ValueError,match='symlink/junction'):restore.audit_source_bytes(**args,restore_line_endings=True)
    assert outside.read_bytes()==data


def test_hardlink_duplicate_target_rejected(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path,names=('a.py','b.py'))
    (source/'b.py').unlink()
    try:os.link(source/'a.py',source/'b.py')
    except OSError as e:pytest.skip('hardlink unavailable: '+str(e))
    with pytest.raises(ValueError,match='hardlink/duplicate'):restore.audit_source_bytes(**args,restore_line_endings=True)


def test_lone_CR_change_is_content_change(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path,names=('a.py',))
    (source/'a.py').write_bytes(data.replace(b'\r\n',b'\r'))
    report=restore.audit_source_bytes(**args,restore_line_endings=True)
    assert report['content_changed_count']==1 and report['restored_count']==0


def test_default_production_count_requires_68(tmp_path):
    repo,source,data,saved,args=fixture(tmp_path)
    del args['expected_count']
    with pytest.raises(ValueError,match='68-source'):restore.audit_source_bytes(**args)


def test_linklike_component_rejected_without_OS_link_privilege(tmp_path,monkeypatch):
    repo,source,data,saved,args=fixture(tmp_path,names=('a.py',))
    target=source/'a.py';actual=restore._linklike
    monkeypatch.setattr(restore,'_linklike',lambda p:p==target or actual(p))
    with pytest.raises(ValueError,match='symlink/junction'):restore.audit_source_bytes(**args,restore_line_endings=True)
    assert target.read_bytes()==data


def test_resolved_path_escape_rejected_without_OS_link_privilege(tmp_path,monkeypatch):
    repo,source,data,saved,args=fixture(tmp_path,names=('a.py',))
    target=source/'a.py';outside=tmp_path/'outside.py';outside.write_bytes(data);actual=Path.resolve
    def escaped(self,*a,**kw):return outside if self==target else actual(self,*a,**kw)
    monkeypatch.setattr(Path,'resolve',escaped)
    with pytest.raises(ValueError,match='escapes repository'):restore.audit_source_bytes(**args,restore_line_endings=True)
    assert outside.read_bytes()==data


def _cli_args(args,count=None):
    value=['source_restore','--snapshot',str(args['snapshot']),'--freeze',str(args['freeze']),
        '--source-root',str(args['source_root']),'--repository-root',str(args['repository_root'])]
    if count is not None:value.extend(['--expected-source-count',str(count)])
    return value


@pytest.mark.parametrize('count',[68,70])
def test_explicit_68_and_70_CLI_are_readonly(tmp_path,monkeypatch,capsys,count):
    names=tuple('source_'+str(i)+'.py' for i in range(count-2))+('../mobile_ev_probing/peer.py','rnj_wzzt_core/unit/root.py')
    repo,source,data,saved,args=fixture(tmp_path,names=names)
    monkeypatch.setattr(sys,'argv',_cli_args(args,count));restore.main()
    result=json.loads(capsys.readouterr().out)
    assert result['checked_count']==count and result['success'] and result['restored_count']==0
    assert result['restore_requested'] is False
    assert all((repo/restore.source_repository_path(n)).read_bytes()==data for n in names)


def test_70_CLI_newline_restore_writes_only_fixture_D_tree(tmp_path,monkeypatch,capsys):
    names=tuple('source_'+str(i)+'.py' for i in range(70))
    repo,source,data,saved,args=fixture(tmp_path,names=names)
    (source/names[0]).write_bytes(data.replace(b'\r\n',b'\n'))
    monkeypatch.setattr(sys,'argv',_cli_args(args,70)+['--restore-line-endings'])
    restore.main();result=json.loads(capsys.readouterr().out)
    assert result['success'] and result['checked_count']==70 and result['restored_count']==1
    assert all((source/n).read_bytes()==data for n in names)


def test_CLI_default68_rejects70_and_invalid_count_is_not_allowed(tmp_path,monkeypatch,capsys):
    names=tuple('source_'+str(i)+'.py' for i in range(70))
    repo,source,data,saved,args=fixture(tmp_path,names=names)
    monkeypatch.setattr(sys,'argv',_cli_args(args))
    with pytest.raises(SystemExit) as e:restore.main()
    assert e.value.code==2 and '68-source' in capsys.readouterr().out
    monkeypatch.setattr(sys,'argv',_cli_args(args,69))
    with pytest.raises(SystemExit) as e:restore.main()
    assert e.value.code==2
    assert all((source/n).read_bytes()==data for n in names)
