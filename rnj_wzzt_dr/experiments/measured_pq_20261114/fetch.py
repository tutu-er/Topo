"""Retrieve selected public FeederBW files, then build a small P/Q archive.

Only selected ZIP members are fetched with HTTP byte ranges. CRC32 validates
each complete uncompressed member; SHA256 records the retained input bytes.
No network topology or identification result is used to choose the feeders.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import struct
import urllib.request
import zlib

from ... import io

RECORD = "19550073"
DOI = "10.5281/zenodo.19550073"
API = f"https://zenodo.org/api/records/{RECORD}"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def request_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def byte_range(url, start, stop, total):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        expected = f"bytes {start}-{stop}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not return the requested byte range")
        data = response.read(stop - start + 2)
    if len(data) != stop - start + 1:
        raise ValueError("Incomplete or oversized range response")
    return data


def zip_members(url, total):
    start = max(0, total - 131072)
    tail = byte_range(url, start, total - 1, total)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError("ZIP directory terminator missing")
    fields = struct.unpack_from("<4s4H2IH", tail, end)
    if fields[1] or fields[2] or fields[3] != fields[4]:
        raise ValueError("Multi-volume ZIP is unsupported")
    size, offset = fields[5:7]
    directory = (tail[offset-start:offset-start+size] if offset >= start else
                 byte_range(url, offset, offset+size-1, total))
    members, pos = [], 0
    while pos < len(directory):
        f = struct.unpack_from("<4s6H3I5H2I", directory, pos)
        if f[0] != b"PK\x01\x02" or f[3] & 1 or f[4] not in (0, 8):
            raise ValueError("Unsupported ZIP entry")
        name = directory[pos+46:pos+46+f[10]].decode("utf-8")
        members.append(dict(name=name, method=f[4], crc32=f[7], compressed_bytes=f[8],
                            bytes=f[9], offset=f[16]))
        pos += 46 + f[10] + f[11] + f[12]
    if len(members) != fields[4]:
        raise ValueError("ZIP directory entry count differs")
    return members


def valid_member(path, member):
    path = Path(path)
    if not path.exists() or path.stat().st_size != member["bytes"]:
        return False
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff == member["crc32"]


def fetch_member(member, url, total, destination, cache=None):
    name = Path(member["name"]).name
    path = destination / name
    origin = "existing_verified_file"
    if not valid_member(path, member):
        if cache is not None and valid_member(cache / name, member):
            shutil.copyfile(cache / name, path)
            origin = "verified_local_cache"
        else:
            offset = member["offset"]
            header = byte_range(url, offset, offset+29, total)
            f = struct.unpack("<4s5H3I2H", header)
            if f[0] != b"PK\x03\x04":
                raise ValueError("Local ZIP header missing")
            start = offset + 30 + f[-2] + f[-1]
            compressed = byte_range(url, start, start+member["compressed_bytes"]-1, total)
            data = zlib.decompress(compressed, -15) if member["method"] == 8 else compressed
            if len(data) != member["bytes"] or zlib.crc32(data) & 0xffffffff != member["crc32"]:
                raise ValueError("ZIP member integrity failure")
            temporary = path.with_suffix(".parquet.part")
            temporary.write_bytes(data)
            temporary.replace(path)
            origin = "public_http_range"
    record = dict(**member, path=str(path.resolve()), sha256=sha256(path), source_url=url,
                  acquisition=origin)
    print(json.dumps(dict(file=name, acquisition=origin, bytes=member["bytes"])), flush=True)
    return record


def download(destination, feeders, *, cache=None, workers=3):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw = destination / "raw"
    raw.mkdir(exist_ok=True)
    if len(set(feeders)) != len(feeders) or not feeders or any(i < 1 or i > 200 for i in feeders):
        raise ValueError("Unique feeder numbers in 1..200 required")
    record = request_json(API)
    files = {f["key"]: f for f in record["files"]}
    tasks = []
    for first in sorted({((i-1)//40)*40+1 for i in feeders}):
        key = f"feeder_measurement_data_{first:03}-{first+39:03}.zip"
        archive = files[key]
        url, size = archive["links"]["self"], archive["size"]
        for member in zip_members(url, size):
            name = Path(member["name"]).name
            if any(name == f"feeder_{i:03}.parquet" for i in feeders):
                tasks.append((member, url, size))
    if len(tasks) != len(feeders):
        raise ValueError("Requested feeder files missing from published archive")
    receipts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(fetch_member, m, u, s, raw, cache) for m, u, s in tasks]
        for job in as_completed(pending):
            receipts.append(job.result())
    meta_file = files["feeder_metadata.csv"]
    with urllib.request.urlopen(meta_file["links"]["self"], timeout=60) as response:
        metadata = response.read(meta_file["size"]+1)
    if len(metadata) != meta_file["size"]:
        raise ValueError("Metadata download size differs")
    checksum = meta_file.get("checksum", "")
    if checksum.startswith("md5:") and hashlib.md5(metadata).hexdigest() != checksum[4:]:
        raise ValueError("Published metadata checksum differs")
    (destination / "feeder_metadata.csv").write_bytes(metadata)
    manifest = dict(dataset="FeederBW", version="v1.1", doi=DOI, record_url=f"https://zenodo.org/records/{RECORD}",
                    license=record["metadata"]["license"], feeder_ids=sorted(feeders),
                    selection="Predeclared numbered feeders; no topology recovery or P/Q correlation screening",
                    files=sorted(receipts, key=lambda x: x["name"]),
                    metadata=io.ref(destination / "feeder_metadata.csv"))
    io.write(destination / "sources.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--feeders", default=",".join(map(str, range(1, 19))))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use 1..4 download workers")
    if (args.destination / "paired_pq_15min.npz").exists():
        parser.error("Preserve the completed archive; choose a new destination")
    if not args.aggregate_only:
        download(args.destination, [int(i) for i in args.feeders.split(",")],
                 cache=args.cache, workers=args.workers)
    from .profiles import make_archive
    archive = make_archive(args.destination)
    print(json.dumps(dict(archive=str(archive), sha256=sha256(archive))), flush=True)


if __name__ == "__main__":
    main()
