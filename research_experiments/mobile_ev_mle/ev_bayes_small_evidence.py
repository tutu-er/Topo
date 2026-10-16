"""Pure-IO, deterministic, lossless JSON-record evidence archives.

Large rectangular lists with one exact Python leaf type become non-pickled
NumPy arrays.  Mixed int/float/bool lists stay typed JSON.  Reference paths in
the manifest, not magic dictionary keys, identify arrays when restoring.
Scientific sources are neither imported nor extracted by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
import zipfile

import numpy as np


FORMAT='topo-bayes-json-npy-v1'
COLLECTION_FORMAT='topo-bayes-record-collection-v1'
REFERENCE_KEY='$topo_evidence_npy'
ZIP_DATE=(1980,1,1,0,0,0)
DTYPES={'<f8':'float','<i8':'int','<u8':'int','|b1':'bool'}
COMPRESSIONS={'deflate':zipfile.ZIP_DEFLATED,'lzma':zipfile.ZIP_LZMA}


def canonical_json(record):
    """Finite JSON floats use Python's round-trip IEEE-754 representation."""
    return json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _digest(data):
    return sha256(data).hexdigest()


def read_record(source):
    path=Path(source);original=path.read_bytes()
    raw=gzip.decompress(original) if original.startswith(b'\x1f\x8b') else original
    record=json.loads(raw.decode('utf-8-sig'))
    # This also explicitly rejects nonfinite JSON-extension constants.
    canonical=canonical_json(record)
    return record,dict(source_name=path.name,original_bytes=len(original),
        original_compressed_sha256=_digest(original),original_json_bytes=len(raw),
        original_json_sha256=_digest(raw),canonical_json_bytes=len(canonical),
        canonical_json_sha256=_digest(canonical))


def _rectangular(value):
    """Return shape/kind/int limits without coercing heterogeneous values."""
    if type(value) is not list:
        if type(value) is float:return (), 'float', None, None
        if type(value) is bool:return (), 'bool', None, None
        if type(value) is int:return (), 'int', value, value
        return None
    if not value:return (0,),None,None,None
    first=_rectangular(value[0])
    if first is None:return None
    shape,kind,low,high=first
    for item in value[1:]:
        result=_rectangular(item)
        if result is None or result[:2]!=(shape,kind):return None
        if kind=='int':low=min(low,result[2]);high=max(high,result[3])
    return (len(value),)+shape,kind,low,high


def _array(value, threshold):
    info=_rectangular(value)
    if info is None:return None
    shape,kind,low,high=info;count=1
    for width in shape:count*=width
    if count<threshold or not shape or kind is None:return None
    if kind=='float':dtype='<f8'
    elif kind=='bool':dtype='|b1'
    elif -(2**63)<=low and high<2**63:dtype='<i8'
    elif low>=0 and high<2**64:dtype='<u8'
    else:return None  # Arbitrary-size JSON integers remain exact Python ints.
    result=np.asarray(value,dtype=dtype)
    if result.shape!=shape:raise ValueError('rectangular-array conversion disagrees with its shape')
    return result,kind


def _encode(record, threshold):
    arrays={};references=[];statistics=Counter()
    def walk(value,path):
        if type(value) is list:
            converted=_array(value,threshold)
            if converted is not None:
                array,kind=converted;stream=BytesIO();np.save(stream,array,allow_pickle=False)
                data=stream.getvalue();digest=_digest(data);name='arrays/'+digest+'.npy'
                if name not in arrays:
                    arrays[name]=dict(data=data,sha256=digest,bytes=len(data),dtype=array.dtype.str,
                        shape=list(array.shape),kind=kind,values=int(array.size))
                references.append(dict(path=list(path),name=name))
                statistics['array_references']+=1;statistics['referenced_values']+=array.size
                statistics['referenced_npy_bytes']+=len(data);statistics[kind+'_array_references']+=1
                return {REFERENCE_KEY:name}
            return [walk(item,path+(i,)) for i,item in enumerate(value)]
        if type(value) is dict:
            if any(type(k) is not str for k in value):raise ValueError('JSON object keys must be strings')
            return {k:walk(item,path+(k,)) for k,item in value.items()}
        if value is None or type(value) in (str,int,float,bool):return value
        raise ValueError('unsupported parsed JSON type: '+type(value).__name__)
    encoded=walk(record,())
    statistics['unique_arrays']=len(arrays)
    statistics['unique_npy_bytes']=sum(v['bytes'] for v in arrays.values())
    statistics['deduplicated_references']=statistics['array_references']-len(arrays)
    return encoded,arrays,references,dict(statistics)


def _safe_entry(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name:
        raise ValueError('unsafe archive entry name')
    path=PurePosixPath(name)
    if path.is_absolute() or any(p in ('','..','.') for p in name.split('/')):
        raise ValueError('unsafe archive entry path')
    return name


def _write_zip(path, entries, *, overwrite=False, compression='deflate'):
    """Write only the explicitly named archive; existing files are protected."""
    path=Path(path)
    if path.exists() and not overwrite:raise FileExistsError(str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(descriptor);temporary=Path(temporary)
    try:
        codec=COMPRESSIONS[compression]
        with zipfile.ZipFile(temporary,'w',compression=codec,compresslevel=9) as archive:
            for name in sorted(entries):
                _safe_entry(name);info=zipfile.ZipInfo(name,ZIP_DATE)
                info.create_system=3;info.external_attr=0o600<<16;info.compress_type=codec
                archive.writestr(info,entries[name],compress_type=codec,compresslevel=9)
        if overwrite:os.replace(temporary,path)
        else:
            # Hard linking gives atomic no-overwrite publication on NTFS/POSIX.
            try:os.link(temporary,path)
            except FileExistsError:raise
            except OSError:
                # Filesystems without hard links still get exclusive creation.
                created=False
                try:
                    with path.open('xb') as dest, temporary.open('rb') as src:
                        created=True;shutil.copyfileobj(src,dest)
                except BaseException:
                    if created:path.unlink(missing_ok=True)
                    raise
    finally:temporary.unlink(missing_ok=True)


def pack_record(source, output, *, threshold=256, overwrite=False, compression='deflate'):
    """Pack one parsed JSON/gzip record; never modify the raw source."""
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError('archive output must differ from raw source')
    if output.exists() and not overwrite:raise FileExistsError(str(output))
    if isinstance(threshold,bool) or int(threshold)!=threshold or threshold<1:
        raise ValueError('positive integer numeric-array threshold required')
    if compression not in COMPRESSIONS:raise ValueError('unknown ZIP compression')
    record,provenance=read_record(source)
    encoded,arrays,references,statistics=_encode(record,int(threshold))
    payload=json.dumps(encoded,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    metadata={name:{k:v for k,v in item.items() if k!='data'} for name,item in sorted(arrays.items())}
    manifest=dict(format=FORMAT,source=provenance,threshold=int(threshold),compression=compression,
        record_entry='record.json',record_sha256=_digest(payload),arrays=metadata,references=references,
        statistics=statistics,numpy_version=np.__version__,
        lossless_scope='parsed JSON types and finite float64 IEEE bits; original text/gzip formatting is not reproduced')
    entries={'manifest.json':canonical_json(manifest),'record.json':payload}
    entries.update({name:item['data'] for name,item in arrays.items()})
    _write_zip(output,entries,overwrite=overwrite,compression=compression)
    archive=output.read_bytes()
    return dict(archive=str(output.resolve()),archive_bytes=len(archive),archive_sha256=_digest(archive),
        original_bytes=provenance['original_bytes'],original_compressed_sha256=provenance['original_compressed_sha256'],
        canonical_json_sha256=provenance['canonical_json_sha256'],
        size_fraction_of_original=len(archive)/provenance['original_bytes'],compression=compression,statistics=statistics)


def unpack_record(archive_path):
    """Read arrays in memory only; never extract ZIP paths into the filesystem."""
    with zipfile.ZipFile(archive_path,'r') as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate ZIP entries')
        for name in names:_safe_entry(name)
        if 'manifest.json' not in names:raise ValueError('missing manifest')
        manifest=json.loads(archive.read('manifest.json'))
        if manifest.get('format')!=FORMAT or manifest.get('record_entry')!='record.json':
            raise ValueError('unsupported evidence format')
        listed={'manifest.json','record.json'}|set(manifest['arrays'])
        if set(names)!=listed:raise ValueError('ZIP entries differ from the exact manifest')
        payload=archive.read('record.json')
        if _digest(payload)!=manifest['record_sha256']:raise ValueError('record checksum mismatch')
        encoded=json.loads(payload);arrays={}
        for name,metadata in manifest['arrays'].items():
            if not name.startswith('arrays/') or not name.endswith('.npy'):_safe_entry('..')
            data=archive.read(name)
            if _digest(data)!=metadata['sha256'] or len(data)!=metadata['bytes']:
                raise ValueError('array checksum mismatch')
            array=np.load(BytesIO(data),allow_pickle=False)
            if (array.dtype.str not in DTYPES or array.dtype.str!=metadata['dtype']
                    or DTYPES[array.dtype.str]!=metadata['kind'] or list(array.shape)!=metadata['shape']
                    or array.size!=metadata['values']):
                raise ValueError('array dtype/shape differs from manifest')
            arrays[name]=array
    references={}
    for row in manifest['references']:
        path=tuple(row['path'])
        if any(type(part) not in (str,int) for part in path) or path in references or row['name'] not in arrays:
            raise ValueError('invalid or duplicate array-reference path')
        references[path]=row['name']
    seen=set()
    def walk(value,path):
        if path in references:
            name=references[path]
            if value!={REFERENCE_KEY:name}:raise ValueError('array reference differs from manifest')
            seen.add(path);return arrays[name].tolist()
        if type(value) is list:return [walk(item,path+(i,)) for i,item in enumerate(value)]
        if type(value) is dict:return {k:walk(item,path+(k,)) for k,item in value.items()}
        return value
    restored=walk(encoded,())
    if seen!=set(references):raise ValueError('array reference path not found in record')
    if _digest(canonical_json(restored))!=manifest['source']['canonical_json_sha256']:
        raise ValueError('restored canonical JSON checksum mismatch')
    return restored,manifest


def exact_equal(left,right,path=()):
    """Check Python types and float bits, including negative zero."""
    if type(left) is not type(right):return False,list(path)
    if type(left) is float:return (struct.pack('>d',left)==struct.pack('>d',right)),list(path)
    if type(left) is list:
        if len(left)!=len(right):return False,list(path)
        for i,(a,b) in enumerate(zip(left,right)):
            same,where=exact_equal(a,b,path+(i,))
            if not same:return False,where
        return True,None
    if type(left) is dict:
        if left.keys()!=right.keys():return False,list(path)
        for key in left:
            same,where=exact_equal(left[key],right[key],path+(key,))
            if not same:return False,where
        return True,None
    return left==right,None if left==right else list(path)


def verify_record(archive,source=None):
    restored,manifest=unpack_record(archive)
    result=dict(passed=True,canonical_json_sha256=manifest['source']['canonical_json_sha256'],
        array_references=len(manifest['references']),unique_arrays=len(manifest['arrays']))
    if source is not None:
        original,provenance=read_record(source);same,where=exact_equal(original,restored)
        result.update(python_record_equal=original==restored,types_and_float_bits_equal=same,
            first_mismatch=where if not same else None,
            original_compressed_sha256_matches=provenance['original_compressed_sha256']==manifest['source']['original_compressed_sha256'])
        result['passed']=bool(same and result['python_record_equal'] and result['original_compressed_sha256_matches'])
    return result


def restore_directory(archive,output_dir):
    """Restore into a fresh data directory, never overwrite or extract sources."""
    target=Path(output_dir)
    if target.exists():raise FileExistsError('restore requires a fresh data directory: '+str(target))
    restored,manifest=unpack_record(archive)
    name=manifest['source']['source_name']
    if Path(name).name!=name or '/' in name or '\\' in name or ':' in name or not (name.endswith('.json') or name.endswith('.json.gz')):
        raise ValueError('restored record must have a safe JSON data filename')
    target.mkdir(parents=True,exist_ok=False)
    path=target/name;data=canonical_json(restored)
    if name.endswith('.gz'):data=gzip.compress(data,mtime=0)
    with path.open('xb') as stream:stream.write(data)
    with (target/'restore_manifest.json').open('xb') as stream:
        stream.write(canonical_json(dict(archive_sha256=_digest(Path(archive).read_bytes()),source=manifest['source'],
            restored_record=name,canonical_json_sha256=manifest['source']['canonical_json_sha256'])))
    return dict(restored_record=str(path.resolve()),canonical_json_sha256=manifest['source']['canonical_json_sha256'],
        exact_manifest_entries_only=True,scientific_sources_extracted=False)


def _declared_records():
    return [(phase,index) for phase,count in (('development',2),('evaluation',8)) for index in range(count)]


def pack_collection(source_root,output_dir,*,threshold=256,compression='lzma'):
    """Pack only the ten explicitly named final JSON records, in separate ZIPs."""
    source_root,target=Path(source_root),Path(output_dir)
    if target.exists():raise FileExistsError('collection requires a fresh directory: '+str(target))
    declared=_declared_records()
    for phase,index in declared:
        path=source_root/phase/f'history_{index:03d}.json.gz'
        if not path.is_file():raise FileNotFoundError(str(path))
    target.mkdir(parents=True,exist_ok=False);rows=[]
    for phase,index in declared:
        source=source_root/phase/f'history_{index:03d}.json.gz'
        relative=f'archives/{phase}_history_{index:03d}.npz.zip'
        packed=pack_record(source,target/relative,threshold=threshold,compression=compression)
        verified=verify_record(target/relative,source)
        if not verified['passed']:raise ValueError('collection round trip failed: '+relative)
        packed.pop('archive')
        rows.append(dict(phase=phase,index=index,source_relative=f'{phase}/{source.name}',
                         archive_relative=relative,verification=verified,**packed))
        print(json.dumps(dict(packed_record=f'{phase}/{source.name}',archive_bytes=packed['archive_bytes'],
                              bit_exact=verified['types_and_float_bits_equal'])),flush=True)
    manifest=dict(format=COLLECTION_FORMAT,threshold=threshold,compression=compression,records=rows,
        record_count=len(rows),original_bytes=sum(r['original_bytes'] for r in rows),
        archive_bytes=sum(r['archive_bytes'] for r in rows),
        packer_sha256=_digest(Path(__file__).read_bytes()),
        scope='only ten history data records; sidecars and source snapshots stay separate and unchanged')
    with (target/'collection_manifest.json').open('xb') as stream:stream.write(canonical_json(manifest))
    return manifest


def restore_collection(collection_dir,output_dir):
    """Create a new data tree containing only the exact ten manifest records."""
    source,target=Path(collection_dir),Path(output_dir)
    if target.exists():raise FileExistsError('restore requires a fresh data directory: '+str(target))
    manifest_path=source/'collection_manifest.json';manifest=json.loads(manifest_path.read_bytes())
    if manifest.get('format')!=COLLECTION_FORMAT:raise ValueError('unsupported collection format')
    rows=manifest.get('records',[]);expected=set(_declared_records());seen=set()
    for row in rows:
        phase,index=row.get('phase'),row.get('index')
        if type(index) is not int or (phase,index) not in expected or (phase,index) in seen:
            raise ValueError('collection must list exactly the ten declared records')
        seen.add((phase,index))
        relative=f'{phase}/history_{index:03d}.json.gz'
        archive=f'archives/{phase}_history_{index:03d}.npz.zip'
        if row.get('source_relative')!=relative or row.get('archive_relative')!=archive:
            raise ValueError('collection output path differs from exact allowed data paths')
        if _digest((source/archive).read_bytes())!=row['archive_sha256']:
            raise ValueError('collection archive checksum mismatch')
    if seen!=expected:raise ValueError('collection must list exactly the ten declared records')
    target.mkdir(parents=True,exist_ok=False);restored_rows=[]
    for row in rows:
        record,individual=unpack_record(source/row['archive_relative'])
        if individual['source']['canonical_json_sha256']!=row['canonical_json_sha256'] or individual['source']['original_compressed_sha256']!=row['original_compressed_sha256']:
            raise ValueError('individual manifest differs from collection content identity')
        path=target/row['source_relative'];path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(gzip.compress(canonical_json(record),mtime=0))
        read_back,provenance=read_record(path);same,where=exact_equal(record,read_back)
        if not same or provenance['canonical_json_sha256']!=row['canonical_json_sha256']:
            raise ValueError('restored file differs in types or float bits: '+str(where))
        restored_rows.append(dict(path=row['source_relative'],canonical_json_sha256=row['canonical_json_sha256'],
            original_compressed_sha256=row['original_compressed_sha256'],restored_compressed_sha256=provenance['original_compressed_sha256'],
            types_and_float_bits_equal=True))
    result=dict(format=COLLECTION_FORMAT,record_count=len(restored_rows),records=restored_rows,
        collection_manifest_sha256=_digest(manifest_path.read_bytes()),scientific_sources_extracted=False,
        exact_allowed_data_paths_only=True)
    with (target/'restore_manifest.json').open('xb') as stream:stream.write(canonical_json(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    pack=commands.add_parser('pack');pack.add_argument('source');pack.add_argument('--output',required=True)
    pack.add_argument('--threshold',type=int,default=256);pack.add_argument('--overwrite',action='store_true')
    pack.add_argument('--compression',choices=tuple(COMPRESSIONS),default='deflate')
    unpack=commands.add_parser('unpack');unpack.add_argument('archive');unpack.add_argument('--output-dir',required=True)
    verify=commands.add_parser('verify');verify.add_argument('archive');verify.add_argument('--source')
    pack_set=commands.add_parser('pack-set');pack_set.add_argument('source_root');pack_set.add_argument('--output-dir',required=True)
    pack_set.add_argument('--threshold',type=int,default=256);pack_set.add_argument('--compression',choices=tuple(COMPRESSIONS),default='lzma')
    unpack_set=commands.add_parser('unpack-set');unpack_set.add_argument('collection_dir');unpack_set.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.command=='pack':result=pack_record(args.source,args.output,threshold=args.threshold,overwrite=args.overwrite,compression=args.compression)
    elif args.command=='unpack':result=restore_directory(args.archive,args.output_dir)
    elif args.command=='pack-set':result=pack_collection(args.source_root,args.output_dir,threshold=args.threshold,compression=args.compression)
    elif args.command=='unpack-set':result=restore_collection(args.collection_dir,args.output_dir)
    else:result=verify_record(args.archive,args.source)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))
    if args.command=='verify' and not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
