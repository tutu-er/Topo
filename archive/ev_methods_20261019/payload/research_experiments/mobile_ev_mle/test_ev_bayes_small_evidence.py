"""Pure IO evidence tests; no AC, inference, optimizer, or experiment imports."""
import gzip
from hashlib import sha256
import json
import struct
import zipfile

import numpy as np
import pytest

from . import ev_bayes_small_evidence as e


def write(path,record):
    path.write_bytes(gzip.compress(e.canonical_json(record),mtime=0));return path


def rewrite(source,target,change):
    with zipfile.ZipFile(source) as z:entries={n:z.read(n) for n in z.namelist()}
    change(entries)
    with zipfile.ZipFile(target,'w') as z:
        for name,value in entries.items():z.writestr(name,value)


def sample():
    special=[0.,-0.,float(np.nextafter(0.,1.)),float(np.finfo(float).max),float.fromhex('0x1.0000000000001p+0'),-1e-300,1.25,2.]
    floating=special*64
    return dict(a=floating,same=floating.copy(),matrix=[floating[:256],floating[256:]],
        integer=list(range(-128,128)),unsigned=[2**63+i for i in range(256)],
        huge_integer=[2**100+i for i in range(256)],boolean=[True,False]*128,
        mixed=[1,1.,True,None]*128,scalar=dict(p=-0.,i=2**130,b=True,n=None),
        collision={e.REFERENCE_KEY:'arrays/user_dictionary.npy'},ragged=[[1.],[2.,3.]])


def test_bit_exact_types_large_integer_negative_zero_and_dedup_roundtrip(tmp_path):
    original=sample();source=write(tmp_path/'000.json.gz',original);before=source.read_bytes()
    archive=tmp_path/'record.zip';report=e.pack_record(source,archive)
    restored,manifest=e.unpack_record(archive)
    assert original==restored and e.exact_equal(original,restored)[0]
    assert struct.pack('>d',restored['a'][1])==struct.pack('>d',-0.)
    assert type(restored['mixed'][0]) is int and type(restored['mixed'][1]) is float
    assert type(restored['mixed'][2]) is bool and restored['mixed'][3] is None
    assert restored['huge_integer'][255]==2**100+255
    assert manifest['source']['original_compressed_sha256']==sha256(before).hexdigest()
    assert manifest['source']['canonical_json_sha256']==sha256(e.canonical_json(original)).hexdigest()
    assert report['statistics']['deduplicated_references']>=1
    assert source.read_bytes()==before
    assert e.verify_record(archive,source)['passed']


@pytest.mark.parametrize('compression',['deflate','lzma'])
def test_fixed_zip_metadata_and_same_input_produce_identical_archive(tmp_path,compression):
    source=write(tmp_path/'000.json.gz',sample())
    a,b=tmp_path/'a.zip',tmp_path/'b.zip';e.pack_record(source,a,compression=compression);e.pack_record(source,b,compression=compression)
    assert a.read_bytes()==b.read_bytes()
    with zipfile.ZipFile(a) as z:
        assert all(i.date_time==e.ZIP_DATE for i in z.infolist())
        assert len(z.namelist())==len(set(z.namelist()))
    assert e.verify_record(a,source)['passed']


def test_default_no_overwrite_and_fresh_data_restore(tmp_path):
    source=write(tmp_path/'000.json.gz',sample());archive=tmp_path/'record.zip'
    e.pack_record(source,archive);before=archive.read_bytes()
    with pytest.raises(FileExistsError):e.pack_record(source,archive)
    with pytest.raises(ValueError,match='differ'):e.pack_record(source,source)
    assert archive.read_bytes()==before
    target=tmp_path/'fresh';result=e.restore_directory(archive,target)
    assert e.exact_equal(e.read_record(source)[0],e.read_record(result['restored_record'])[0])[0]
    assert sorted(p.name for p in target.iterdir())==['000.json.gz','restore_manifest.json']
    with pytest.raises(FileExistsError):e.restore_directory(archive,target)
    assert result['scientific_sources_extracted'] is False


def test_path_traversal_and_scientific_source_filename_are_rejected(tmp_path):
    source=write(tmp_path/'000.json.gz',sample());archive=tmp_path/'record.zip';e.pack_record(source,archive)
    bad=tmp_path/'bad.zip'
    def dangerous_name(entries):
        m=json.loads(entries['manifest.json']);m['source']['source_name']='../science.py'
        entries['manifest.json']=e.canonical_json(m)
    rewrite(archive,bad,dangerous_name)
    with pytest.raises(ValueError,match='safe JSON'):e.restore_directory(bad,tmp_path/'fresh')
    assert not (tmp_path/'fresh').exists()
    bad2=tmp_path/'bad2.zip'
    rewrite(archive,bad2,lambda entries:entries.update({'../outside.npy':b'no'}))
    with pytest.raises(ValueError,match='unsafe'):e.unpack_record(bad2)
    assert not (tmp_path/'outside.npy').exists()


def test_corruption_missing_and_extra_entries_cannot_be_restored(tmp_path):
    source=write(tmp_path/'000.json.gz',sample());archive=tmp_path/'record.zip';e.pack_record(source,archive)
    def corrupt(entries):
        name=next(k for k in entries if k.endswith('.npy'));entries[name]=entries[name][:-1]+b'x'
    bad=tmp_path/'corrupt.zip';rewrite(archive,bad,corrupt)
    with pytest.raises(ValueError,match='checksum'):e.unpack_record(bad)
    extra=tmp_path/'extra.zip';rewrite(archive,extra,lambda entries:entries.update({'extra.txt':b'x'}))
    with pytest.raises(ValueError,match='exact manifest'):e.unpack_record(extra)


def test_exact_comparison_detects_equality_masked_type_and_sign_changes():
    assert [1]==[1.] and not e.exact_equal([1],[1.])[0]
    assert [True]==[1] and not e.exact_equal([True],[1])[0]
    assert [-0.]==[0.] and not e.exact_equal([-0.],[0.])[0]


def test_small_and_heterogeneous_lists_stay_json_without_type_coercion(tmp_path):
    record=dict(small=[1.,2.,3.],mixed=[1,1.]*200,nones=[None]*300)
    source=write(tmp_path/'000.json.gz',record);archive=tmp_path/'record.zip'
    result=e.pack_record(source,archive);restored,_=e.unpack_record(archive)
    assert result['statistics']['unique_arrays']==0
    assert e.exact_equal(record,restored)[0]


def test_collection_exact_ten_data_paths_restore_without_sources(tmp_path):
    source=tmp_path/'source'
    for phase,index in e._declared_records():
        path=source/phase/f'history_{index:03d}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
        write(path,dict(phase=phase,index=index,data=[float(index),-0.]*128))
    collection=tmp_path/'packed';result=e.pack_collection(source,collection)
    assert result['record_count']==10 and all(r['verification']['passed'] for r in result['records'])
    restored=tmp_path/'restored';report=e.restore_collection(collection,restored)
    assert report['record_count']==10 and not report['scientific_sources_extracted']
    assert len(list(restored.rglob('history_*.json.gz')))==10
    for phase,index in e._declared_records():
        relative=f'{phase}/history_{index:03d}.json.gz'
        assert e.exact_equal(e.read_record(source/relative)[0],e.read_record(restored/relative)[0])[0]
    with pytest.raises(FileExistsError):e.restore_collection(collection,restored)
    bad=json.loads((collection/'collection_manifest.json').read_bytes())
    bad['records'][0]['source_relative']='../science.py'
    (collection/'collection_manifest.json').write_bytes(e.canonical_json(bad))
    with pytest.raises(ValueError,match='allowed data paths'):e.restore_collection(collection,tmp_path/'bad_restore')
    assert not (tmp_path/'bad_restore').exists()
