"""Byte restoration, payload integrity and hostile-path archive tests."""
import gzip,io,json,tarfile
from pathlib import Path
import pytest
from . import ev_original_evidence_archive as archive


def dataset(root):
    root.mkdir();payload=b'{  "unicode": "\\u4e2d", "values": [1,2,3] }\n'*200
    (root/'freeze.json').write_bytes(b'{ "seed" : 1 }\r\n')
    with gzip.GzipFile(filename='exact-source.json',mode='wb',fileobj=(root/'source_snapshot.json.gz').open('wb'),mtime=1234,compresslevel=9) as f:f.write(payload)
    (root/'trial_1.json.gz').write_bytes(gzip.compress(payload,compresslevel=6,mtime=2376))
    (root/'review_v1.json').write_bytes(b'{"ok":true}\n')
    (root/'analysis.csv').write_bytes(b'name,value\r\na,1\r\n')
    (root/'unrelated.bin').write_bytes(b'not selected')
    return sorted(p.name for p in root.iterdir() if p.name!='unrelated.bin')


def test_exact_roundtrip_retains_headers_sourcebytes_and_plain_bytes(tmp_path):
    source=tmp_path/'raw';names=dataset(source);before={n:(source/n).read_bytes() for n in names}
    packed=archive.pack_directory(source,tmp_path/'compact.tar.xz')
    restored=archive.restore_archive(tmp_path/'compact.tar.xz',tmp_path/'restored')
    assert packed['files']==len(names) and packed['raw_inputs_unchanged'] and not packed['raw_inputs_removed']
    assert restored['all_payloads_match'] and restored['all_original_bytes_match']
    assert set(p.name for p in (tmp_path/'restored').iterdir())==set(names)
    assert archive.compare_restored(source,tmp_path/'restored',names)['all_byte_equal']
    assert all((source/n).read_bytes()==before[n] for n in names)
    with tarfile.open(tmp_path/'compact.tar.xz','r:xz') as tar:
        manifest=json.load(tar.extractfile(archive.MANIFEST))
        rec=next(r for r in manifest['files'] if r['name']=='source_snapshot.json.gz')
        assert rec['gzip']['members'][0]['mtime']==1234
        assert rec['gzip']['members'][0]['flags']&8
        assert tar.extractfile(rec['member']).read()==gzip.decompress(before[rec['name']])
        assert manifest['json_reformatted'] is False


def test_multimember_gzip_and_zero_padding_roundtrip(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    value=gzip.compress(b'{"part":1}\n',mtime=1)+gzip.compress(b'{"part":2}\n',compresslevel=1,mtime=2)+b'\0\0'
    (raw/'trial_two.json.gz').write_bytes(value)
    archive.pack_directory(raw,tmp_path/'two.tar.xz')
    result=archive.restore_archive(tmp_path/'two.tar.xz',tmp_path/'restored')
    assert result['all_original_bytes_match']
    assert (tmp_path/'restored'/'trial_two.json.gz').read_bytes()==value


def test_cross_runtime_gzip_difference_preserves_payload_and_discloses_hash(tmp_path,monkeypatch):
    source=tmp_path/'raw';dataset(source)
    archive.pack_directory(source,tmp_path/'compact.tar.xz',filenames=['source_snapshot.json.gz'])
    real_compress=gzip.compress
    def alternate(data,compresslevel=9,*,mtime=None):return real_compress(data,compresslevel=1,mtime=mtime)
    monkeypatch.setattr(archive.gzip,'compress',alternate)
    result=archive.restore_archive(tmp_path/'compact.tar.xz',tmp_path/'restored')
    row=result['files'][0]
    assert result['all_payloads_match'] and not result['all_original_bytes_match']
    assert row['restored_sha256']!=row['original_sha256'] and 'not claimed' in row['note']
    assert gzip.decompress((tmp_path/'restored'/row['name']).read_bytes())==gzip.decompress((source/row['name']).read_bytes())


def rewrite(original,target,mutator):
    with tarfile.open(original,'r:xz') as tar:
        entries=[(m.name,tar.extractfile(m).read()) for m in tar.getmembers()]
    entries=mutator(entries)
    with tarfile.open(target,'w:xz') as tar:
        for name,value in entries:
            if isinstance(value,tarfile.TarInfo):tar.addfile(value)
            else:archive._add(tar,name,value)


@pytest.mark.parametrize('bad_name',['../escape','payload/../../escape','C:/escape','payload\\escape'])
def test_unexpected_unsafe_members_rejected_before_creating_output(tmp_path,bad_name):
    source=tmp_path/'raw';dataset(source);original=tmp_path/'original.tar.xz';bad=tmp_path/'bad.tar.xz'
    archive.pack_directory(source,original,filenames=['freeze.json'])
    rewrite(original,bad,lambda entries:entries+[(bad_name,b'bad')])
    with pytest.raises(ValueError):archive.restore_archive(bad,tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_symlink_member_and_duplicate_member_rejected(tmp_path):
    source=tmp_path/'raw';dataset(source);original=tmp_path/'original.tar.xz';archive.pack_directory(source,original,filenames=['freeze.json'])
    link=tarfile.TarInfo('payload/link');link.type=tarfile.SYMTYPE;link.linkname='../outside'
    rewrite(original,tmp_path/'link.tar.xz',lambda entries:entries+[('payload/link',link)])
    rewrite(original,tmp_path/'duplicate.tar.xz',lambda entries:entries+[entries[-1]])
    for name in ('link','duplicate'):
        with pytest.raises(ValueError):archive.restore_archive(tmp_path/(name+'.tar.xz'),tmp_path/name)


def test_payload_corruption_is_rejected(tmp_path):
    source=tmp_path/'raw';dataset(source);original=tmp_path/'original.tar.xz';archive.pack_directory(source,original,filenames=['freeze.json'])
    def damage(entries):return [(n,b'X'*len(v) if n.startswith('payload/') else v) for n,v in entries]
    rewrite(original,tmp_path/'bad.tar.xz',damage)
    with pytest.raises(ValueError,match='hash mismatch'):archive.restore_archive(tmp_path/'bad.tar.xz',tmp_path/'restored')


@pytest.mark.parametrize('name',['../bad','C:bad','a/b','a\\b','CON','NUL.json','trailing.','trailing ','a\nname','a|b'])
def test_source_names_reject_traversal_and_windows_ambiguity(tmp_path,name):
    root=tmp_path/'raw';root.mkdir()
    with pytest.raises(ValueError):archive.pack_directory(root,tmp_path/'a.tar.xz',filenames=[name])


def test_restore_new_directory_and_exclusive_archive_output(tmp_path):
    source=tmp_path/'raw';dataset(source);path=tmp_path/'archive.tar.xz';archive.pack_directory(source,path)
    with pytest.raises(FileExistsError):archive.pack_directory(source,path)
    (tmp_path/'existing').mkdir()
    with pytest.raises(FileExistsError):archive.restore_archive(path,tmp_path/'existing')


def test_explicit_selection_and_size_limits(tmp_path):
    source=tmp_path/'raw';dataset(source)
    archive.pack_directory(source,tmp_path/'one.tar.xz',filenames=['freeze.json'])
    restored=archive.restore_archive(tmp_path/'one.tar.xz',tmp_path/'restored')
    assert len(restored['files'])==1 and restored['files'][0]['name']=='freeze.json'
    with pytest.raises(ValueError):archive.pack_directory(source,tmp_path/'small.tar.xz',max_file_bytes=10)
    with pytest.raises(ValueError):archive.select_files(source,filenames=['freeze.json','freeze.json'])
