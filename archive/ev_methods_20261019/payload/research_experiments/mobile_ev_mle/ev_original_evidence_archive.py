"""Lossless JSON-payload solid archives; raw evidence is never removed.

gzip is decoded as bytes, not parsed or reformatted JSON. Its original
headers/trailers and hash are retained; restoration distinguishes exact gzip
bytes from an exact payload reproduced by another gzip/zlib runtime.
"""
from __future__ import annotations
import argparse,base64,fnmatch,gzip,hashlib,io,json,platform,struct,sys,tarfile,zlib
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone

SCHEMA='ev_evidence_payload_tar_xz_v1'
MANIFEST='archive_manifest.json'
DEFAULT_PATTERNS=('freeze.json','source_snapshot.json.gz','trial_*.json.gz','review*.json',
    'analysis*.json','analysis*.csv','*.csv','checkpoint.json','environment.json')
MAX_FILE=512*1024*1024
MAX_TOTAL=4*1024*1024*1024


def digest(value):return hashlib.sha256(value).hexdigest()


def runtime():
    return dict(python=sys.version,implementation=platform.python_implementation(),
        zlib_compile=zlib.ZLIB_VERSION,zlib_runtime=zlib.ZLIB_RUNTIME_VERSION)


def safe_name(name):
    if (not isinstance(name,str) or not name or name in ('.','..') or
            any(c in name for c in ('/','\\',':','\0','<','>','"','|','?','*')) or
            any(ord(c)<32 for c in name) or name[-1] in (' ','.')):
        raise ValueError('only safe flat filenames are allowed: '+repr(name))
    stem=name.split('.')[0].rstrip(' .').upper()
    if stem in ('CON','PRN','AUX','NUL') or stem in {'COM'+str(i) for i in range(1,10)} or stem in {'LPT'+str(i) for i in range(1,10)}:
        raise ValueError('reserved Windows filename: '+name)
    return name


def _gzip_header_end(raw,offset):
    if len(raw)-offset<18 or raw[offset:offset+3]!=b'\x1f\x8b\x08':raise ValueError('invalid gzip member')
    flags=raw[offset+3]
    if flags&0xe0:raise ValueError('reserved gzip flags are not supported')
    p=offset+10
    if flags&4:
        if len(raw)<p+2:raise ValueError('truncated gzip extra length')
        length=int.from_bytes(raw[p:p+2],'little');p+=2+length
    for flag in (8,16):
        if flags&flag:
            end=raw.find(b'\0',p)
            if end<0:raise ValueError('truncated gzip string')
            p=end+1
    if flags&2:p+=2
    if p>len(raw)-8:raise ValueError('truncated gzip header')
    return p


def _gzip_description(raw,payload):
    members=[];offset=0;poffset=0
    while offset<len(raw) and raw[offset:offset+1]!=b'\0':
        start=_gzip_header_end(raw,offset);decoder=zlib.decompressobj(-zlib.MAX_WBITS)
        member=decoder.decompress(raw[start:])+decoder.flush()
        if not decoder.eof:raise ValueError('truncated gzip deflate stream')
        used=len(raw)-start-len(decoder.unused_data);end=start+used
        if len(raw)<end+8:raise ValueError('truncated gzip trailer')
        header=raw[offset:start];footer=raw[end:end+8];compressed=raw[start:end]
        if payload[poffset:poffset+len(member)]!=member:raise ValueError('gzip member payload mismatch')
        level=None
        # Required current-runtime round trip starts from gzip.compress(...,
        # mtime=0). Preserve the original header including filename and OS.
        for candidate in (9,6,1,2,3,4,5,7,8,0):
            generated=gzip.compress(member,compresslevel=candidate,mtime=0)
            if generated[10:-8]==compressed:
                level=candidate;break
        members.append(dict(header_base64=base64.b64encode(header).decode('ascii'),
            trailer_base64=base64.b64encode(footer).decode('ascii'),flags=header[3],
            mtime=int.from_bytes(header[4:8],'little'),xfl=header[8],os=header[9],
            payload_offset=poffset,payload_size=len(member),payload_sha256=digest(member),
            deflate_size=len(compressed),deflate_sha256=digest(compressed),
            recompression_level=level,current_runtime_deflate_exact=level is not None))
        poffset+=len(member);offset=end+8
    padding=raw[offset:]
    if padding.strip(b'\0') or poffset!=len(payload):raise ValueError('unsupported trailing gzip bytes')
    return dict(codec='gzip RFC1952; deflate RFC1951',members=members,
        trailing_zero_padding_base64=base64.b64encode(padding).decode('ascii'),
        current_runtime_deflate_exact=all(m['current_runtime_deflate_exact'] for m in members))


def _restore_gzip(payload,description):
    result=[]
    for member in description['members']:
        start=int(member['payload_offset']);stop=start+int(member['payload_size']);part=payload[start:stop]
        if len(part)!=member['payload_size'] or digest(part)!=member['payload_sha256']:raise ValueError('gzip member payload hash mismatch')
        level=member['recompression_level'];level=9 if level is None else int(level)
        if level not in range(10):raise ValueError('invalid gzip recompression level')
        generated=gzip.compress(part,compresslevel=level,mtime=0)
        header=base64.b64decode(member['header_base64'],validate=True)
        footer=base64.b64decode(member['trailer_base64'],validate=True)
        if _gzip_header_end(header+b'\0'*8,0)!=len(header) or len(footer)!=8:raise ValueError('invalid archived gzip header/trailer')
        result.append(header+generated[10:-8]+footer)
    padding=base64.b64decode(description.get('trailing_zero_padding_base64',''),validate=True)
    if padding.strip(b'\0'):raise ValueError('invalid archived gzip padding')
    raw=b''.join(result)+padding
    if gzip.decompress(raw)!=payload:raise ValueError('restored gzip payload differs')
    return raw


def select_files(directory,*,filenames=None,patterns=None):
    directory=Path(directory).resolve(strict=True)
    if not directory.is_dir():raise ValueError('source must be a directory')
    if filenames is not None and patterns is not None:raise ValueError('use filenames or patterns, not both')
    if filenames is None:
        patterns=DEFAULT_PATTERNS if patterns is None else tuple(patterns)
        for pattern in patterns:
            if not isinstance(pattern,str) or any(c in pattern for c in ('/','\\',':')):raise ValueError('patterns must select flat files')
        names=sorted(p.name for p in directory.iterdir() if any(fnmatch.fnmatchcase(p.name,q) for q in patterns))
    else:names=list(filenames)
    if not names:raise ValueError('no evidence files selected')
    seen=set()
    for name in names:
        safe_name(name)
        if name.casefold() in seen:raise ValueError('duplicate or case-colliding evidence filename')
        seen.add(name.casefold());path=directory/name
        if path.is_symlink() or not path.is_file() or path.resolve().parent!=directory:raise ValueError('selected evidence must be an ordinary file inside source')
    return directory,sorted(names)


def _add(tar,name,value):
    item=tarfile.TarInfo(name);item.size=len(value);item.mtime=0;item.mode=0o600
    item.uid=item.gid=0;item.uname=item.gname='';tar.addfile(item,io.BytesIO(value))


def pack_directory(directory,archive,*,filenames=None,patterns=None,preset=6,max_file_bytes=MAX_FILE,max_total_bytes=MAX_TOTAL):
    began=perf_counter();directory,names=select_files(directory,filenames=filenames,patterns=patterns)
    if not isinstance(preset,int) or preset not in range(10):raise ValueError('xz preset must be 0..9')
    archive=Path(archive)
    if archive.exists() or archive.is_symlink():raise FileExistsError('archive destination already exists')
    records=[];total=0;original_total=0
    for name in names:
        raw=(directory/name).read_bytes();original_total+=len(raw)
        if len(raw)>max_file_bytes:raise ValueError('source file exceeds size limit')
        payload=gzip.decompress(raw) if name.endswith('.gz') else raw
        total+=len(payload)
        if len(payload)>max_file_bytes or total>max_total_bytes:raise ValueError('payload size limit exceeded')
        rec=dict(name=name,member='payload/'+name,codec='gzip' if name.endswith('.gz') else 'identity',
            original_size=len(raw),original_sha256=digest(raw),payload_size=len(payload),payload_sha256=digest(payload))
        if rec['codec']=='gzip':rec['gzip']=_gzip_description(raw,payload)
        records.append(rec)
    manifest=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),runtime=runtime(),
        source_directory=str(directory),flat_files_only=True,raw_inputs_removed=False,
        payload_is_original_bytes=True,json_reformatted=False,xz_preset=preset,
        xz_dictionary_note='preset 6 uses an 8 MiB LZMA2 dictionary; no extreme preset',files=records)
    archive.parent.mkdir(parents=True,exist_ok=True)
    # The output path is new and opened exclusively before handing it to tar.
    with archive.open('xb') as target:
        with tarfile.open(fileobj=target,mode='w:xz',preset=preset,format=tarfile.PAX_FORMAT) as tar:
            _add(tar,MANIFEST,json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True).encode('utf-8'))
            for rec in records:
                raw=(directory/rec['name']).read_bytes()
                if digest(raw)!=rec['original_sha256']:raise RuntimeError('source changed during archive construction')
                payload=gzip.decompress(raw) if rec['codec']=='gzip' else raw
                if digest(payload)!=rec['payload_sha256']:raise RuntimeError('payload changed during archive construction')
                _add(tar,rec['member'],payload)
    for rec in records:
        if digest((directory/rec['name']).read_bytes())!=rec['original_sha256']:raise RuntimeError('source changed during archive construction')
    size=archive.stat().st_size
    return dict(archive=str(archive.resolve()),archive_sha256=digest(archive.read_bytes()),files=len(records),
        original_total_bytes=original_total,payload_total_bytes=total,archive_bytes=size,
        archive_over_original_ratio=size/original_total if original_total else None,
        raw_inputs_unchanged=True,raw_inputs_removed=False,seconds=perf_counter()-began)


def _read(tar,member,limit):
    if member.size<0 or member.size>limit:raise ValueError('archive member exceeds size limit')
    handle=tar.extractfile(member)
    if handle is None:raise ValueError('archive member is not readable')
    with handle:value=handle.read(member.size+1)
    if len(value)!=member.size:raise ValueError('truncated archive payload')
    return value


def restore_archive(archive,destination,*,max_file_bytes=MAX_FILE,max_total_bytes=MAX_TOTAL):
    began=perf_counter();destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError('restore requires a new destination directory')
    records=[]
    with tarfile.open(archive,mode='r:xz') as tar:
        members=tar.getmembers();byname={};folded=set()
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():raise ValueError('only ordinary archive files are allowed')
            if member.name.casefold() in folded:raise ValueError('duplicate or case-colliding archive member')
            folded.add(member.name.casefold());byname[member.name]=member
        if MANIFEST not in byname:raise ValueError('archive manifest missing')
        manifest=json.loads(_read(tar,byname[MANIFEST],4*1024*1024))
        if manifest.get('schema')!=SCHEMA or not isinstance(manifest.get('files'),list):raise ValueError('unsupported evidence archive schema')
        if not manifest['files'] or len(manifest['files'])>10000:raise ValueError('invalid archive file count')
        expected={MANIFEST};seen=set();total=0
        for rec in manifest['files']:
            name=safe_name(rec['name'])
            if name.casefold() in seen:raise ValueError('duplicate or case-colliding manifest filename')
            seen.add(name.casefold())
            if rec['member']!='payload/'+name or rec['codec'] not in ('identity','gzip'):raise ValueError('invalid archive payload path or codec')
            for key in ('original_sha256','payload_sha256'):
                if not isinstance(rec[key],str) or len(rec[key])!=64 or any(c not in '0123456789abcdef' for c in rec[key]):raise ValueError('invalid manifest hash')
            size=rec['payload_size'];total+=size
            if not isinstance(size,int) or size<0 or size>max_file_bytes or total>max_total_bytes:raise ValueError('invalid payload size')
            expected.add(rec['member'])
        if set(byname)!=expected:raise ValueError('unexpected or missing archive members')
        # Reject the complete archive structure before creating any output.
        destination.mkdir(parents=True,exist_ok=False);root=destination.resolve(strict=True)
        for rec in manifest['files']:
            member=byname[rec['member']]
            if member.size!=rec['payload_size']:raise ValueError('manifest/tar payload size mismatch')
            payload=_read(tar,member,max_file_bytes)
            if digest(payload)!=rec['payload_sha256']:raise ValueError('payload hash mismatch')
            raw=_restore_gzip(payload,rec['gzip']) if rec['codec']=='gzip' else payload
            path=root/rec['name']
            if path.resolve().parent!=root:raise ValueError('restore path escapes destination')
            with path.open('xb') as target:target.write(raw)
            actual=path.read_bytes();after=gzip.decompress(actual) if rec['codec']=='gzip' else actual
            if digest(after)!=rec['payload_sha256']:raise RuntimeError('written payload verification failed')
            exact=digest(actual)==rec['original_sha256']
            records.append(dict(name=rec['name'],codec=rec['codec'],payload_sha256=digest(after),
                payload_sha256_match=True,original_sha256=rec['original_sha256'],
                restored_sha256=digest(actual),original_bytes_match=exact,
                note=None if exact else 'Exact payload verified; gzip bytes differ. The original gzip hash is not claimed for this restored file.'))
    return dict(schema=SCHEMA,destination=str(destination.resolve()),runtime=runtime(),
        archived_runtime=manifest['runtime'],files=records,all_payloads_match=True,
        all_original_bytes_match=all(r['original_bytes_match'] for r in records),
        raw_inputs_removed=False,seconds=perf_counter()-began)


def compare_restored(source,destination,filenames):
    source=Path(source).resolve(strict=True);destination=Path(destination).resolve(strict=True);records=[]
    for name in filenames:
        safe_name(name);a=(source/name).read_bytes();b=(destination/name).read_bytes()
        records.append(dict(name=name,source_sha256=digest(a),restored_sha256=digest(b),byte_equal=a==b))
    return dict(all_byte_equal=all(r['byte_equal'] for r in records),files=records)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('pack');pack.add_argument('directory',type=Path);pack.add_argument('archive',type=Path)
    pack.add_argument('--file',action='append',dest='filenames');pack.add_argument('--pattern',action='append',dest='patterns')
    pack.add_argument('--preset',type=int,default=6)
    restore=sub.add_parser('restore');restore.add_argument('archive',type=Path);restore.add_argument('destination',type=Path)
    restore.add_argument('--require-exact',action='store_true')
    for command in (pack,restore):command.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.command=='pack':answer=pack_directory(args.directory,args.archive,filenames=args.filenames,patterns=args.patterns,preset=args.preset)
    else:answer=restore_archive(args.archive,args.destination)
    if args.report:
        with args.report.open('x',encoding='utf-8') as f:json.dump(answer,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in answer.items() if k!='files'},ensure_ascii=False,allow_nan=False))
    if args.command=='restore' and args.require_exact and not answer['all_original_bytes_match']:raise SystemExit(2)


if __name__=='__main__':main()
