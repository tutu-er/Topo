"""Publish sealed strict-96 runs as independently valid, size-limited ZIP volumes.

Standard library only. Every volume extracts below each input run's basename;
extract all volumes into one folder to restore the complete runs. The default
target is 60 MiB per ZIP, with a 90 MiB hard ceiling and no byte-split archives.
An individually large member gets its own volume; raw members above 90 MiB are
refused. Existing destinations are preserved rather than overwritten.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import struct
import traceback
import zipfile
import zlib


MIB = 1 << 20
TARGET_BYTES = 60*MIB
HARD_BYTES = 90*MIB
TERMINAL = {"fitted", "complete", "failed", "interrupted", "budget_exceeded", "common_failed"}
BLOCK = 1 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def local(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root), "A sealed path escapes its run")
    return path


def npy_header(stream):
    require(stream.read(6) == b"\x93NUMPY", "Invalid NPY header")
    version = tuple(stream.read(2)); require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY version")
    size = 2 if version == (1, 0) else 4
    length = struct.unpack("<H" if size == 2 else "<I", stream.read(size))[0]
    require(length <= 100000, "NPY header is too large")
    header = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
    require(isinstance(header["descr"], str) and "O" not in header["descr"], "Object/structured arrays are not observed inputs")
    return dict(shape=tuple(header["shape"]), dtype=header["descr"])


def check96(path, labels):
    with zipfile.ZipFile(path) as archive:
        required = {name+".npy" for name in ("P", "Q", "V", "root", "labels")}
        require(set(archive.namelist()) == required and len(archive.namelist()) == len(required), "Unexpected observed NPZ fields")
        require(archive.testzip() is None, "Observed NPZ CRC failure")
        headers = {name[:-4]: npy_header(archive.open(name)) for name in archive.namelist()}
    n = len(labels)
    require(n > 0 and headers["labels"]["shape"] == (n,), "Terminal count differs from protocol")
    require(all(headers[k]["shape"] in ((96, n), (1, 96, n)) for k in ("P", "Q", "V")), "An input is not exactly one 96-point day")
    require(headers["P"]["shape"] == headers["Q"]["shape"] == headers["V"]["shape"], "Observed arrays differ in shape")
    require(headers["root"]["shape"] in ((96,), (1, 96)), "Root observation count differs")
    return dict(points=96, terminals=n, array_headers=headers)


def excluded(path):
    name = path.name.lower()
    return (name == "run.lock" or name.endswith(".tmp") or ".tmp." in name
            or name.endswith((".pyc", ".pyo")) or "__pycache__" in path.parts)


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not excluded(path):
            require(not path.is_symlink() and path.resolve().is_relative_to(root), "A linked file escapes the run")
            result[path.relative_to(root).as_posix()] = path
    return result


def validate_run(root):
    root = Path(root).resolve(); runtime = root/"runtime"
    protocol, seal, completion = [read(root/name) for name in ("protocol.json", "selection_seal.json", "completion.json")]
    require(protocol.get("kind") == "large96_single_day_v1", "Unexpected batch protocol kind")
    require(seal.get("sealed") and seal.get("all_planned_terminal") and completion.get("all_planned_terminal"), "Only sealed, fully terminal batches may be published")
    require(seal["protocol_sha256"] == sha(root/"protocol.json") == sha(runtime/"protocol.json"), "Protocol seal SHA differs")
    require(completion["selection_seal_sha256"] == sha(root/"selection_seal.json"), "Completion seal SHA differs")
    require(protocol["split"] == dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), "Only strict total-96 runs are permitted")
    require(seal.get("source_sha256", protocol["source_sha256"]) == protocol["source_sha256"], "Sealed running source differs")
    inputs, methods = protocol["inputs"], protocol["methods"]
    expected_results = {f"runs/{job}/{method}/result.json" for job in inputs for method in methods}
    expected_common = {f"common/{job}/common.json" for job in inputs}
    require(set(seal["result_sha256"]) == expected_results, "The result seal omits planned methods")
    require(set(seal["common_sha256"]) == expected_common, "The common seal omits planned inputs")
    runtime_expected = set(protocol["source_sha256"]) | {"protocol.json"}
    input_receipts = []
    for name, checksum in protocol["source_sha256"].items():
        require(sha(local(runtime, name)) == checksum, "Frozen running source changed: "+name)
    for job, info in inputs.items():
        if info.get("generation_failed"):
            require(info.get("path") is None, "Failed generation has an unexpected observed file")
            continue
        name = info["path"]; file = local(runtime, name)
        require(name == f"inputs/{job}.npz" and sha(file) == info["sha256"], "Frozen sole input changed")
        runtime_expected.add(name)
        input_receipts.append(dict(job=job, path="runtime/"+name, sha256=info["sha256"], **check96(file, info["labels"])))
    require(set(inventory(runtime)) == runtime_expected, "The worker runtime includes unexpected files")
    files = inventory(root)
    allowed_npz = {r["path"] for r in input_receipts}
    require({name for name in files if name.lower().endswith(".npz")} == allowed_npz, "An unapproved or extra-day NPZ exists in the run")
    for name, checksum in seal["common_sha256"].items():
        require(sha(local(root, name)) == checksum, "A sealed common record changed")
        record = read(local(root, name))
        require(record["status"] in TERMINAL and record["protocol_sha256"] == seal["protocol_sha256"], "Invalid common terminal record")
        if record["status"] in {"fitted", "complete"}:
            require(record["input_sha256"] == inputs[record["job"]]["sha256"], "Common used different observations")
            require(record["source_sha256_before"] == record["source_sha256_after"] == protocol["source_sha256"], "Common used different sources")
    status_counts = {}
    for name, checksum in seal["result_sha256"].items():
        require(sha(local(root, name)) == checksum, "A sealed result changed")
        record = read(local(root, name)); status = record["status"]
        require(status in TERMINAL and record["protocol_sha256"] == seal["protocol_sha256"], "Invalid result terminal record")
        status_counts[status] = status_counts.get(status, 0)+1
        if status in {"fitted", "complete"}:
            job = record["job"]
            require(record["input_sha256"] == inputs[job]["sha256"], "A method used different observations")
            require(record["source_sha256_before"] == record["source_sha256_after"] == protocol["source_sha256"], "A method used different running sources")
            common_name = f"common/{job}/common.json"
            require(record["common_sha256"] == seal["common_sha256"][common_name], "A method used a different common record")
    for key, name in (("evaluation_map_sha256", "evaluation_map.json"), ("generation_sha256", "generation.json")):
        if protocol.get(key):
            require(sha(root/name) == protocol[key], "Frozen metadata changed: "+name)
    for name, checksum in protocol.get("evaluation_case_sha256", {}).items():
        require(sha(local(root/"evaluation_cases", name)) == checksum, "Frozen physical evaluation case changed")
    return dict(name=root.name, source_directory=str(root), protocol_sha256=sha(root/"protocol.json"),
                selection_seal_sha256=sha(root/"selection_seal.json"), completion_sha256=sha(root/"completion.json"),
                planned_inputs=len(inputs), planned_methods=len(expected_results), status_counts=status_counts,
                single_day_npz_count=len(input_receipts), generation_failures=len(inputs)-len(input_receipts),
                running_source_sha256=protocol["source_sha256"], inputs=input_receipts), files


def compressed_record(path, member):
    require(path.stat().st_size <= HARD_BYTES, "A raw member is larger than 90 MiB: "+member)
    digest = hashlib.sha256(); compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    compressed, raw, crc = 0, 0, 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            raw += len(block); digest.update(block); crc = zlib.crc32(block, crc)
            compressed += len(compressor.compress(block))
    compressed += len(compressor.flush())
    # Seekable, non-ZIP64 output: local header (30) + central header (46).
    overhead = 76+2*len(member.encode("utf-8"))
    return dict(sha256=digest.hexdigest(), size_bytes=raw, compressed_size_bytes=compressed,
                crc32=crc & 0xffffffff, zip_contribution_bytes=compressed+overhead)


def partition(members, *, target_bytes=TARGET_BYTES, hard_bytes=HARD_BYTES):
    require(0 < target_bytes <= hard_bytes <= HARD_BYTES, "Invalid ZIP size limits")
    volumes, current, size = [], [], 22
    for name, record in members.items():
        contribution = record["zip_contribution_bytes"]
        require(22+contribution <= hard_bytes, "A member cannot fit a legal volume below the hard limit: "+name)
        if current and (size+contribution > target_bytes or len(current) >= 60000):
            volumes.append(current); current, size = [], 22
        current.append(name); size += contribution
        if size > target_bytes:
            volumes.append(current); current, size = [], 22
    if current:
        volumes.append(current)
    return volumes


def verify_volume(path, names, members, *, target_bytes, hard_bytes):
    require(path.stat().st_size <= hard_bytes, "A completed ZIP exceeds its hard size limit")
    with zipfile.ZipFile(path) as archive:
        actual = archive.namelist()
        require(actual == names and len(set(actual)) == len(names), "ZIP member order/set differs")
        require(archive.testzip() is None, "ZIP CRC verification failed")
        for name in names:
            info, expected = archive.getinfo(name), members[name]
            require(info.file_size == expected["size_bytes"] and info.compress_size == expected["compressed_size_bytes"] and info.CRC == expected["crc32"], "ZIP sizes/CRC differ from compression planning")
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(BLOCK), b""):
                    digest.update(block)
            require(digest.hexdigest() == expected["sha256"], "ZIP member SHA differs: "+name)
    oversized = path.stat().st_size > target_bytes
    require(not oversized or len(names) == 1, "A multi-member ZIP exceeds the rolling target")
    return dict(path=path.name, sha256=sha(path), size_bytes=path.stat().st_size, members=names,
                member_count=len(names), crc_passed=True, all_member_sha256_passed=True,
                target_exceeded_single_member=oversized)


def publish(runs, output, *, target_bytes=TARGET_BYTES, hard_bytes=HARD_BYTES):
    roots = [Path(run).resolve() for run in runs]; output = Path(output).resolve()
    require(roots and len({r.name for r in roots}) == len(roots), "Input runs must have unique directory basenames")
    require(not output.exists(), "Destination already exists; refusing to overwrite")
    require(all(not output.is_relative_to(root) for root in roots), "Destination must be outside the input runs")
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for a, b in combinations(roots, 2)), "Input runs overlap")
    run_receipts, paths, inventories = [], {}, {}
    for root in roots:
        require(root.is_dir(), "An input run directory does not exist")
        receipt, files = validate_run(root); run_receipts.append(receipt); inventories[root] = set(files)
        paths.update({root.name+"/"+name: file for name, file in files.items()})
    script = Path(__file__).resolve(); paths["publication/publish_large96.py"] = script
    members = {name: compressed_record(path, name) for name, path in sorted(paths.items())}
    planned = partition(members, target_bytes=target_bytes, hard_bytes=hard_bytes)
    output.mkdir(parents=True)
    volumes = []
    try:
        for index, names in enumerate(planned, 1):
            path = output/f"records-{index:03d}.zip"
            with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED, compresslevel=6, strict_timestamps=False) as archive:
                for name in names:
                    archive.write(paths[name], arcname=name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            volume = verify_volume(path, names, members, target_bytes=target_bytes, hard_bytes=hard_bytes)
            for name in names:
                members[name]["volume"] = path.name
            volumes.append(volume)
        for name, path in paths.items():
            require(sha(path) == members[name]["sha256"], "An original source changed during publication: "+name)
        for root, initial in zip(roots, run_receipts, strict=True):
            require(set(inventory(root)) == inventories[root], "Run inventory changed during publication")
            final_receipt, _ = validate_run(root)
            require(final_receipt == initial, "A sealed run changed between validation and archive completion")
        manifest = dict(format_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
            publisher_source_sha256=members["publication/publish_large96.py"]["sha256"],
            compression="independent ZIP_DEFLATED level6 archives; no byte-split ZIP",
            target_volume_bytes=target_bytes, hard_volume_bytes=hard_bytes, hard_raw_member_bytes=HARD_BYTES,
            excluded="run.lock, *.tmp, *.tmp.*, __pycache__, *.pyc, *.pyo",
            runs=run_receipts, single_day_npz_count=sum(r["single_day_npz_count"] for r in run_receipts),
            old_multiday_npz_included=False, original_sources_unchanged=True,
            total_members=len(members), total_raw_bytes=sum(r["size_bytes"] for r in members.values()),
            total_zip_bytes=sum(r["size_bytes"] for r in volumes), volume_count=len(volumes),
            volumes=volumes, members=members,
            extraction="Extract every records-*.zip into the same destination; run basenames are retained as prefixes")
        write(output/"manifest.json", manifest)
        write(output/"receipt.json", dict(verified=True, manifest_sha256=sha(output/"manifest.json"),
            publisher_source_sha256=manifest["publisher_source_sha256"], volumes=len(volumes), members=len(members),
            single_day_npz_count=manifest["single_day_npz_count"],
            zip_crc_passed=True, all_member_sha256_passed=True, original_sources_unchanged=True))
        return manifest
    except Exception:
        write(output/"publication_failure.json", dict(error=traceback.format_exc(), completed_volumes=volumes,
            scope="Incomplete publication preserved; choose a new destination for a reviewed retry"))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path, help="sealed run directory; may be repeated")
    parser.add_argument("--output", required=True, type=Path, help="new archive directory outside all input runs")
    args = parser.parse_args(); result = publish(args.run, args.output)
    print(json.dumps({key: result[key] for key in ("volume_count", "total_members", "single_day_npz_count", "total_raw_bytes", "total_zip_bytes")}, indent=2))
