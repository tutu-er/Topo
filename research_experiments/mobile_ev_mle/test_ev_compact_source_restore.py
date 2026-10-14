"""Compact wrapper: source identity and newline-only recovery boundaries."""
import base64,gzip,hashlib,json
import pytest
from . import ev_compact_source_restore as compact


def setup(tmp_path,n=72):
    repo=tmp_path/'repo';source=repo/'research_experiments/mobile_ev_mle'
    source.mkdir(parents=True);directory=tmp_path/'evidence';directory.mkdir()
    data=b'first\r\nsecond\n';saved={};hashes={}
    for i in range(n):
        name=f'source_{i}.py';(source/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
        saved[name]=dict(sha256=hashes[name],bytes_base64=base64.b64encode(data).decode())
    payload=dict(config={'methods':['primary']},prediction='ensemble',
        master_form='shared_control_square',domain='asset',source_sha256=hashes)
    policy_id=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    freeze=dict(**payload,policy_config=payload,policy_id=policy_id)
    (directory/'freeze.json').write_text(json.dumps(freeze),encoding='utf-8')
    (directory/'source_snapshot.json.gz').write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
    return repo,source,directory,data,freeze


def test_exact_and_readonly_then_explicit_newline_restore(tmp_path):
    repo,source,directory,data,freeze=setup(tmp_path)
    args=dict(source_root=source,repository_root=repo)
    assert compact.audit_directory(directory,**args)['exact_after_count']==72
    changed=(source/'source_0.py');changed.write_bytes(data.replace(b'\r\n',b'\n'))
    result=compact.audit_directory(directory,**args)
    assert not result['success'] and result['restored_count']==0
    assert changed.read_bytes()!=data
    result=compact.audit_directory(directory,**args,restore_line_endings=True)
    assert result['success'] and result['restored_count']==1 and changed.read_bytes()==data
    assert result['policy_identity']['policy_id']==freeze['policy_id']


@pytest.mark.parametrize('count',[68,70,71,73])
def test_requires_exact_72_frozen_sources(tmp_path,count):
    repo,source,directory,data,freeze=setup(tmp_path,count)
    with pytest.raises(ValueError,match='72-source'):
        compact.audit_directory(directory,source_root=source,repository_root=repo)


def test_changed_policy_refuses_before_any_source_write(tmp_path):
    repo,source,directory,data,freeze=setup(tmp_path)
    target=source/'source_0.py';target.write_bytes(data.replace(b'\r\n',b'\n'));before=target.read_bytes()
    freeze['policy_id']='0'*64
    (directory/'freeze.json').write_text(json.dumps(freeze),encoding='utf-8')
    with pytest.raises(ValueError,match='policy_id'):
        compact.audit_directory(directory,source_root=source,repository_root=repo,restore_line_endings=True)
    assert target.read_bytes()==before


def test_changed_content_refuses_whole_batch(tmp_path):
    repo,source,directory,data,freeze=setup(tmp_path)
    target=source/'source_0.py';target.write_bytes(data.replace(b'\r\n',b'\n'));before=target.read_bytes()
    (source/'source_1.py').write_bytes(b'changed code\n')
    result=compact.audit_directory(directory,source_root=source,repository_root=repo,restore_line_endings=True)
    assert not result['success'] and result['content_changed_count']==1 and result['restored_count']==0
    assert target.read_bytes()==before
