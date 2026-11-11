"""Archive both strict96 attempts and verify every byte; standard library only.

The failed attempt is retained. Existing archives are checked, never replaced.
Only the two named output trees and the two postprocessing scripts are included.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import zipfile


BASE = Path(__file__).resolve().parents[1]
RUNS = ("strict96_20261112_001", "strict96_20261112_002")
SCRIPTS = ("tools/analyze_strict96.py", "tools/publish_strict96.py")
DESTINATION = BASE/"experiments/strict96_20261112"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def npy_shape(stream):
    require(stream.read(6) == b"\x93NUMPY", "Invalid NPY magic")
    version = tuple(stream.read(2))
    require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY version")
    length_size = 2 if version == (1, 0) else 4
    length = struct.unpack("<H" if length_size == 2 else "<I", stream.read(length_size))[0]
    require(length <= 100000, "Unexpectedly large NPY header")
    header = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
    require("O" not in str(header["descr"]), "Object arrays are not archive inputs")
    return tuple(header["shape"])


def sole_input(name, payload):
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        require(set(archive.namelist()) == {k+".npy" for k in ("P", "Q", "V", "root", "labels")}, "Unexpected sole-input fields")
        require(archive.testzip() is None, "Input NPZ CRC failure")
        shapes = {key[:-4]: npy_shape(archive.open(key)) for key in archive.namelist()}
    require(len(shapes["labels"]) == 1 and shapes["labels"][0] > 0, "Invalid input terminal labels")
    n = shapes["labels"][0]
    require(all(shapes[key] in ((96, n), (1, 96, n)) for key in ("P", "Q", "V")), "Input is not one 96-point day")
    require(shapes["P"] == shapes["Q"] == shapes["V"] and shapes["root"] in ((96,), (1, 96)), "Unaligned input arrays")
    return dict(member=name, points=96, terminals=n, array_shapes=shapes, sha256=digest(payload))


def collect():
    paths = []
    for run in RUNS:
        root = (BASE/"outputs"/run).resolve()
        require(root.is_dir(), "Missing output attempt: "+run)
        for path in root.rglob("*"):
            if path.is_file():
                require(not path.is_symlink() and path.resolve().is_relative_to(root), "Linked file escapes output scope")
                paths.append(path)
    paths.extend(BASE/name for name in SCRIPTS)
    payloads = {p.relative_to(BASE).as_posix(): p.read_bytes() for p in sorted(paths)}
    expected = {f"outputs/{run}/runtime/inputs/input{i:02d}.npz" for run in RUNS for i in range(4)}
    require({name for name in payloads if name.lower().endswith(".npz")} == expected, "Archive must contain exactly eight sole inputs")
    inputs = [sole_input(name, payloads[name]) for name in sorted(expected)]
    for run in RUNS:
        prefix = f"outputs/{run}/"
        protocol = json.loads(payloads[prefix+"protocol.json"])
        require(len(protocol["inputs"]) == 4, "An attempt has an unexpected input denominator")
        for info in protocol["inputs"].values():
            require(digest(payloads[prefix+"runtime/"+info["path"]]) == info["sha256"], "Input differs from its frozen protocol")
        for source, checksum in protocol["source_sha256"].items():
            require(digest(payloads[prefix+"runtime/"+source]) == checksum, "Frozen native source changed")
    members = {name: dict(sha256=digest(data), size_bytes=len(data)) for name, data in sorted(payloads.items())}
    return payloads, members, inputs


def verify(archive_path, members):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(members), "ZIP membership changed")
        require(archive.testzip() is None, "ZIP CRC verification failed")
        for name in names:
            data = archive.read(name); expected = members[name]
            require(len(data) == expected["size_bytes"] and digest(data) == expected["sha256"], "ZIP member SHA mismatch: "+name)
    data = archive_path.read_bytes()
    return dict(name=archive_path.name, sha256=digest(data), size_bytes=len(data))


def publish():
    payloads, members, inputs = collect()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archive, manifest_path = DESTINATION/"records.zip", DESTINATION/"records_manifest.json"
    if archive.exists() or manifest_path.exists():
        require(archive.is_file() and manifest_path.is_file(), "Incomplete existing archive; refusing to overwrite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest["members"] == members, "Existing archive differs from current sources; refusing to overwrite")
        checked = verify(archive, members)
        require(checked == manifest["zip"], "Existing ZIP hash/size differs from its manifest")
        return dict(action="verified_existing", members=len(members), inputs=len(inputs), **checked)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for name, data in sorted(payloads.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 12, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED; entry.external_attr = 0o100644 << 16
            stream.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    checked = verify(archive, members)
    for name, expected in members.items():
        require(digest((BASE/name).read_bytes()) == expected["sha256"], "Source changed while archiving: "+name)
    manifest = dict(format_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        source_roots=["outputs/"+run for run in RUNS], analysis_sources=list(SCRIPTS),
        failed_attempt_preserved=RUNS[0], member_count=len(members), members=members, zip=checked,
        input_npz_count=8, inputs=inputs, old_validation_test_inputs_included=False,
        verification=dict(zip_crc_passed=True, every_member_sha256_passed=True, original_sources_unchanged=True),
        publisher_sha256=digest(Path(__file__).read_bytes()))
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
    return dict(action="created", members=len(members), inputs=len(inputs), **checked)


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2))
