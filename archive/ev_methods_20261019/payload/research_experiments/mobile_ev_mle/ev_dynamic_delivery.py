"""Stream a compact, verified dynamic-EV evidence archive without running code.

pack --phase dev=DIR --phase calibration=DIR --phase evaluation=DIR --output FILE.tar.xz
restore --archive FILE.tar.xz --output NEW_EMPTY_DIRECTORY

Gzip payloads are stored decompressed in the outer xz stream. Repeated decoded
bytes share one SHA256 object. Exact gzip reconstruction is verified, not assumed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import zlib

SCHEMA = "ev_dynamic_streamed_delivery_v1"
CHUNK = 1024*1024
MAX_MANIFEST_BYTES = 8*1024*1024
FIXED_NAMES = {"freeze.json", "source_snapshot.json.gz", "checkpoint.json"}
SHA = re.compile(r"^[0-9a-f]{64}$")
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RESERVED = {"CON", "PRN", "AUX", "NUL",
            *("COM"+str(i) for i in range(1,10)), *("LPT"+str(i) for i in range(1,10))}


class ByteRestoreError(RuntimeError):
    """The raw evidence was recovered but some gzip files differ byte-for-byte."""
    def __init__(self, report):
        self.report = report
        super().__init__("gzip byte reconstruction failed; verified decompressed raw saved: "+
                         ", ".join(item["raw_fallback_path"] for item in report["failures"]))


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def _hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def _environment():
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                zlib_compile=zlib.ZLIB_VERSION, zlib_runtime=zlib.ZLIB_RUNTIME_VERSION)


def _label(label):
    if (not isinstance(label,str) or not LABEL.fullmatch(label)
            or label.endswith((".", " ")) or label.split(".")[0].upper() in RESERVED):
        raise ValueError("unsafe phase label")
    return label


def _allowed_name(name):
    return name in FIXED_NAMES or (name.startswith("trial") and name.endswith(".json.gz"))


def _logical(path, *, fallback=False):
    if not isinstance(path,str) or not path or "\\" in path or ":" in path or "\x00" in path:
        raise ValueError("unsafe logical path")
    parts = PurePosixPath(path).parts
    if (len(parts) != 2 or PurePosixPath(path).is_absolute()
            or any(part in (".","..","") or part.endswith((".", " ")) for part in parts)
            or str(PurePosixPath(path)) != path):
        raise ValueError("unsafe logical path")
    _label(parts[0])
    name = parts[1]
    if fallback:
        suffix = ".uncompressed.raw"
        if not name.endswith(suffix):
            raise ValueError("unsafe raw fallback path")
        name = name[:-len(suffix)]
    if not _allowed_name(name):
        raise ValueError("file outside evidence whitelist")
    return path


@contextmanager
def _decoded(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rb") as stream:
        yield stream


def _file_record(label, path):
    before = path.stat()
    original_sha = _sha_file(path)
    decoded_sha = hashlib.sha256()
    size = 0
    with _decoded(path) as stream:
        for block in iter(lambda:stream.read(CHUNK), b""):
            decoded_sha.update(block)
            size += len(block)
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError("evidence changed during scan: "+str(path))
    logical = _logical(label+"/"+path.name)
    record = dict(logical_path=logical, original_sha256=original_sha,
                  original_bytes=before.st_size, decoded_sha256=decoded_sha.hexdigest(),
                  decoded_bytes=size, encoding="gzip_mtime_0" if path.name.endswith(".gz") else "identity")
    if record["encoding"] != "identity":
        with path.open("rb") as stream:
            header = stream.read(10)
        record.update(raw_fallback_path=_logical(logical+".uncompressed.raw",fallback=True),
                      original_gzip_header_hex=header.hex())
    return record


class _HashReader:
    def __init__(self, stream):
        self.stream, self.hash, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hash.update(block)
        self.count += len(block)
        return block


def _tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def pack_archive(phases, output):
    """Pack only whitelisted top-level files. Memory is bounded by stream buffers."""
    if not isinstance(phases,dict) or not phases:
        raise ValueError("at least one named phase is required")
    labels = [_label(label) for label in phases]
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("phase labels collide on Windows")
    output = Path(output).absolute()
    sidecar = output.with_name(output.name+".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("archive or sidecar already exists")
    entries, objects, sources = [], {}, {}
    for label in sorted(labels):
        directory = Path(phases[label])
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("phase must be a real directory")
        files = [p for p in directory.iterdir() if _allowed_name(p.name)]
        if not files:
            raise ValueError("phase contains no whitelisted evidence")
        for path in sorted(files):
            if path.is_symlink() or not path.is_file():
                raise ValueError("evidence must be a regular non-symlink file")
            entry = _file_record(label,path)
            entries.append(entry)
            sha = entry["decoded_sha256"]
            obj = dict(member="objects/"+sha, decoded_bytes=entry["decoded_bytes"])
            if sha in objects and objects[sha] != obj:
                raise ValueError("inconsistent content-addressed object")
            objects[sha] = obj
            sources.setdefault(sha,path)
    manifest = dict(schema=SCHEMA, phases=sorted(labels), entries=entries, objects=objects,
        reconstruction=dict(gzip_rule="gzip.compress(decoded, compresslevel=9, mtime=0)",
            streaming_rule="runtime gzip header + zlib raw deflate level 9 + CRC32/ISIZE trailer",
            exact_bytes_verified_after_reconstruction=True,
            failure_rule="save decoded bytes only at the manifest-listed raw_fallback_path and report failure"),
        pack_environment=_environment(), logical_files=len(entries), unique_objects=len(objects),
        logical_decoded_bytes=sum(e["decoded_bytes"] for e in entries),
        unique_decoded_bytes=sum(o["decoded_bytes"] for o in objects.values()))
    manifest["manifest_id"] = _hash(manifest)
    _validate_manifest(manifest)
    data = _json_bytes(manifest)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest too large")
    output.parent.mkdir(parents=True,exist_ok=True)
    created = sidecar_created = False
    try:
        with output.open("xb") as target:
            created = True
            with lzma.LZMAFile(target,"w",preset=6) as compressed:
                with tarfile.open(fileobj=compressed,mode="w|") as archive:
                    archive.addfile(_tar_info("manifest.json",len(data)),io.BytesIO(data))
                    for sha,obj in sorted(objects.items()):
                        path = sources[sha]
                        with _decoded(path) as stream:
                            reader = _HashReader(stream)
                            archive.addfile(_tar_info(obj["member"],obj["decoded_bytes"]),reader)
                            if reader.read(1) or reader.count != obj["decoded_bytes"] or reader.hash.hexdigest() != sha:
                                raise ValueError("decoded evidence changed while packing")
        # Verify every original, including duplicate objects, before publishing the sidecar.
        for entry in entries:
            phase,name = entry["logical_path"].split("/")
            if _sha_file(Path(phases[phase])/name) != entry["original_sha256"]:
                raise ValueError("original evidence changed while packing")
        summary = dict(manifest=manifest,archive_sha256=_sha_file(output),
                       archive_bytes=output.stat().st_size)
        with sidecar.open("xb") as stream:
            sidecar_created = True
            stream.write(_json_bytes(summary))
        return dict(archive=str(output),sidecar=str(sidecar),**summary)
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        if sidecar_created:
            sidecar.unlink(missing_ok=True)
        raise


def _validate_manifest(manifest):
    if not isinstance(manifest,dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unknown archive manifest")
    if _hash({k:v for k,v in manifest.items() if k != "manifest_id"}) != manifest.get("manifest_id"):
        raise ValueError("manifest content/hash mismatch")
    entries,objects = manifest.get("entries"),manifest.get("objects")
    if not isinstance(entries,list) or not entries or not isinstance(objects,dict) or not objects:
        raise ValueError("empty archive manifest")
    all_paths, phases, referenced = set(), set(), set()
    for entry in entries:
        logical = _logical(entry.get("logical_path"))
        phases.add(logical.split("/")[0])
        paths = [logical]
        encoding = entry.get("encoding")
        if encoding == "gzip_mtime_0":
            if not logical.endswith(".gz") or entry.get("raw_fallback_path") != logical+".uncompressed.raw":
                raise ValueError("inconsistent gzip fallback path")
            paths.append(_logical(entry["raw_fallback_path"],fallback=True))
        elif encoding == "identity":
            if logical.endswith(".gz") or entry.get("original_sha256") != entry.get("decoded_sha256"):
                raise ValueError("inconsistent identity encoding")
        else:
            raise ValueError("unknown evidence encoding")
        for path in paths:
            if path.casefold() in all_paths:
                raise ValueError("duplicate or Windows-colliding output path")
            all_paths.add(path.casefold())
        for key in ("original_sha256","decoded_sha256"):
            if not isinstance(entry.get(key),str) or not SHA.fullmatch(entry[key]):
                raise ValueError("invalid evidence SHA256")
        for key in ("original_bytes","decoded_bytes"):
            if not isinstance(entry.get(key),int) or isinstance(entry[key],bool) or entry[key] < 0:
                raise ValueError("invalid evidence byte count")
        if encoding == "identity" and entry["original_bytes"] != entry["decoded_bytes"]:
            raise ValueError("inconsistent identity byte counts")
        sha = entry["decoded_sha256"]
        referenced.add(sha)
        if objects.get(sha) != dict(member="objects/"+sha,decoded_bytes=entry["decoded_bytes"]):
            raise ValueError("missing/inconsistent decoded object")
    if set(objects) != referenced or manifest.get("phases") != sorted(phases):
        raise ValueError("unreferenced object or inconsistent phases")
    if len({p.casefold() for p in phases}) != len(phases):
        raise ValueError("phase labels collide on Windows")
    return manifest


@contextmanager
def _open_tar(path):
    with lzma.open(path,"rb") as stream:
        with tarfile.open(fileobj=stream,mode="r|") as archive:
            yield archive


def _manifest_from_tar(archive):
    members = iter(archive)
    first = next(members,None)
    if first is None or first.name != "manifest.json" or not first.isfile() or first.linkname:
        raise ValueError("first archive member must be a regular manifest.json")
    if not 0 <= first.size <= MAX_MANIFEST_BYTES:
        raise ValueError("manifest too large")
    stream = archive.extractfile(first)
    return _validate_manifest(json.loads(stream.read(MAX_MANIFEST_BYTES+1))),members


def _check_member(member, allowed, seen):
    if (member.name not in allowed or member.name in seen or not member.isfile()
            or member.linkname or member.size != allowed[member.name] or member.issparse()):
        raise ValueError("unlisted, duplicate, nonregular, or invalid archive member")
    seen.add(member.name)


def _preflight(archive_path):
    with _open_tar(archive_path) as archive:
        manifest,members = _manifest_from_tar(archive)
        allowed = {obj["member"]:obj["decoded_bytes"] for obj in manifest["objects"].values()}
        seen = set()
        for member in members:
            _check_member(member,allowed,seen)
        if seen != set(allowed):
            raise ValueError("archive is missing decoded objects")
    return manifest


def _destination(root, logical, *, fallback=False):
    _logical(logical,fallback=fallback)
    target = root.joinpath(*PurePosixPath(logical).parts)
    resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("output path leaves restore directory")
    if target.exists():
        raise FileExistsError("restore refuses an existing file")
    target.parent.mkdir(parents=True,exist_ok=True)
    return target


def _copy_verified(source, target, expected):
    created = False
    try:
        with target.open("xb") as dst:
            created = True
            with Path(source).open("rb") as src:
                shutil.copyfileobj(src,dst,CHUNK)
        if _sha_file(target) != expected:
            raise ValueError("restored byte SHA256 mismatch")
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise


def _gzip_to(raw_path, output_path):
    """Stream the current runtime's gzip.compress(level=9,mtime=0) convention."""
    header = gzip.compress(b"",compresslevel=9,mtime=0)[:10]
    compressor = zlib.compressobj(9,zlib.DEFLATED,-zlib.MAX_WBITS)
    crc,total = 0,0
    with Path(raw_path).open("rb") as source,Path(output_path).open("wb") as target:
        target.write(header)
        for block in iter(lambda:source.read(CHUNK), b""):
            crc = zlib.crc32(block,crc)
            total += len(block)
            target.write(compressor.compress(block))
        target.write(compressor.flush())
        target.write(struct.pack("<II",crc & 0xffffffff,total & 0xffffffff))


def restore_archive(archive, output):
    """Restore only manifest-listed files into an empty directory; never extractall."""
    archive,root = Path(archive),Path(output).absolute()
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise ValueError("restore output must be a new or empty real directory")
    manifest = _preflight(archive)
    root.mkdir(parents=True,exist_ok=True)
    groups = {}
    for entry in manifest["entries"]:
        groups.setdefault(entry["decoded_sha256"],[]).append(entry)
    restored,failures = [],[]
    with _open_tar(archive) as stream:
        repeated,members = _manifest_from_tar(stream)
        if repeated != manifest:
            raise ValueError("archive manifest changed after preflight")
        allowed = {obj["member"]:obj["decoded_bytes"] for obj in manifest["objects"].values()}
        seen = set()
        for member in members:
            _check_member(member,allowed,seen)
            sha = member.name.removeprefix("objects/")
            fd,temp = tempfile.mkstemp(prefix=".ev-delivery-raw-",dir=root.parent)
            os.close(fd)
            temp = Path(temp)
            try:
                digest = hashlib.sha256()
                size = 0
                with temp.open("wb") as dst:
                    src = stream.extractfile(member)
                    for block in iter(lambda:src.read(CHUNK), b""):
                        dst.write(block);digest.update(block);size += len(block)
                if digest.hexdigest() != sha or size != manifest["objects"][sha]["decoded_bytes"]:
                    raise ValueError("decoded object SHA256 mismatch")
                for entry in groups[sha]:
                    if entry["encoding"] == "identity":
                        target = _destination(root,entry["logical_path"])
                        _copy_verified(temp,target,entry["original_sha256"])
                    else:
                        fd,compressed = tempfile.mkstemp(prefix=".ev-delivery-gzip-",dir=root.parent)
                        os.close(fd);compressed = Path(compressed)
                        try:
                            _gzip_to(temp,compressed)
                            reconstructed_sha = _sha_file(compressed)
                            if reconstructed_sha != entry["original_sha256"]:
                                target = _destination(root,entry["raw_fallback_path"],fallback=True)
                                _copy_verified(temp,target,entry["decoded_sha256"])
                                failures.append(dict(logical_path=entry["logical_path"],
                                    raw_fallback_path=entry["raw_fallback_path"],
                                    expected_original_sha256=entry["original_sha256"],
                                    reconstructed_sha256=reconstructed_sha))
                                continue
                            target = _destination(root,entry["logical_path"])
                            _copy_verified(compressed,target,entry["original_sha256"])
                        finally:
                            compressed.unlink(missing_ok=True)
                    restored.append(dict(logical_path=entry["logical_path"],
                                         sha256=entry["original_sha256"]))
            finally:
                temp.unlink(missing_ok=True)
        if seen != set(allowed):
            raise ValueError("archive changed or lost objects during restore")
    report = dict(manifest_id=manifest["manifest_id"],archive_sha256=_sha_file(archive),
        exact_byte_restore=not failures,restored_files=restored,failures=failures,
        pack_environment=manifest["pack_environment"],restore_environment=_environment())
    if failures:
        raise ByteRestoreError(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--phase",action="append",required=True,metavar="LABEL=DIRECTORY")
    pack.add_argument("--output",type=Path,required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--archive",type=Path,required=True)
    restore.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.command == "pack":
            phases = {}
            for value in args.phase:
                if "=" not in value:
                    parser.error("--phase must be LABEL=DIRECTORY")
                label,path = value.split("=",1)
                if label in phases:
                    parser.error("duplicate --phase label")
                phases[label] = Path(path)
            result = pack_archive(phases,args.output)
            result = {k:v for k,v in result.items() if k != "manifest"}
        else:
            result = restore_archive(args.archive,args.output)
    except ByteRestoreError as exc:
        print(json.dumps(dict(error=str(exc),**exc.report),ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
