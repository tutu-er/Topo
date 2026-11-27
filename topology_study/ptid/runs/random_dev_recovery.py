"""Rebuild only randomly generated development cases and verify the original SHA-256.

The original builder writes all four variants to a temporary directory. Only byte-
verified pickle files are installed in the development cache. No source manifests
are modified. Run from topology_study with the same environment as the generator:

python -B -m ptid.runs.random_dev_recovery --cases-root /path/to/case_library_20261125 \
    --groups dev_small_001 --report recovery.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

from ptid.cases import library

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

import numpy as np
from rnj_wzzt_dr.experiments.case_library_20261125 import build, loads


def verified_write(target, contents):
    """Install a verified case atomically. Preserve any different existing file."""
    if target.exists():
        if target.read_bytes() != contents:
            raise ValueError(f"Existing file differs at {target}")
        return
    target.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as out:
        out.write(contents)
        temp = Path(out.name)
    try:
        if not target.exists():
            temp.rename(target)
        elif target.read_bytes() != contents:
            raise ValueError(f"Concurrent file differs at {target}")
    finally:
        temp.unlink(missing_ok=True)


def reconstruct(root, job):
    stream, idx, specification = job
    if stream != "dev" or specification[0] != "random":
        raise ValueError("Expected a random development job")
    loads.CACHE = root/"customers.npz"
    if not loads.CACHE.is_file():
        raise FileNotFoundError(loads.CACHE)
    expected = {r["name"]: r for r in library.manifest()}
    start = perf_counter()
    with tempfile.TemporaryDirectory(dir=root) as work:
        build.OUT = Path(work)
        encoded = {}
        def encode(record):
            # Python 3.0-3.7 used default 3, 3.8-3.13 used 4, and 3.14
            # uses 5. Compare the same in-memory object in each protocol.
            name = record["name"]
            by_protocol = {}
            target = expected[name]["sha256"]
            verified = None
            for protocol in (3, 4, 5):
                data = pickle.dumps(record, protocol=protocol)
                digest = hashlib.sha256(data).hexdigest()
                by_protocol[protocol] = digest
                if digest == target:
                    verified = data
                if protocol == pickle.DEFAULT_PROTOCOL:
                    primary = data
            encoded[name] = by_protocol
            return verified if verified is not None else primary
        prior_pickle = build.pickle
        build.pickle = SimpleNamespace(dumps=encode)
        try:
            rows = build.build_job(job)
        finally:
            build.pickle = prior_pickle
        report = []
        for r in rows:
            name = r["name"]
            row = expected[name]
            contents = (build.OUT/stream/(name+".pkl")).read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            fields_match = all(row[k] == r[k] for k in row if k != "sha256")
            match = digest == row["sha256"] and fields_match
            # Pickle bytes can differ across runtime versions. A metadata match
            # is recorded separately and never qualifies a case for installation.
            if match:
                verified_write(root/stream/(name+".pkl"), contents)
            record = pickle.loads(contents)  # generated in this invocation
            arr_hashes = {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in record["true"].items()}
            report.append(dict(name=name, n=r["n"], expected_meters=row["n"], sha256=digest,
                               expected_sha256=row["sha256"], metadata_match=fields_match, verified=match,
                               protocol_header=contents[1], tested_pickle_sha256=encoded[name],
                               quarter_hour_sha256=arr_hashes,
                               rejected=int(record["rejected"])))
    return dict(group=rows[0]["group"], generator_index=idx, seconds=perf_counter()-start, cases=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", required=True, help="dev_small_001 etc., or all-random")
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.cases_root.resolve()
    if not (root/"customers.npz").is_file():
        parser.error(f"No source customer cache at {root}")
    expected_manifest = Path(library.__file__).with_name("manifest_dev.json")
    build_manifest = build.HERE/"manifest_dev.json"
    if json.loads(build_manifest.read_text(encoding="utf-8")) != library.manifest():
        raise ValueError("Generation and identification development manifests disagree")
    recorded = {r["group"] for r in library.manifest() if r["name"].startswith("dev_")}
    requested = recorded if args.groups == ["all-random"] else set(args.groups)
    if not requested or not requested <= recorded:
        parser.error("Invalid random development group")
    jobs = []
    for idx, spec in enumerate(build.jobs_of("dev")):
        if spec[0] == "random" and f"dev_{spec[1]}_{spec[2]:03d}" in requested:
            jobs.append(("dev", idx, spec))
    report = dict(source_commit="739863c17d91d914ee6dea62a4ea8c99fba64b7f",
                  customer_cache_sha256=hashlib.sha256((root/"customers.npz").read_bytes()).hexdigest(),
                  studies_manifest_sha256=hashlib.sha256(expected_manifest.read_bytes()).hexdigest(),
                  original_manifest_sha256=hashlib.sha256(build_manifest.read_bytes()).hexdigest(),
                  platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                  pickle_default=pickle.DEFAULT_PROTOCOL, groups=[])
    args.report.parent.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = [pool.submit(reconstruct, root, job) for job in jobs]
        for job, future in zip(jobs, outputs):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(job=job, error=f"{type(exc).__name__}: {exc}", cases=[])
            report["groups"].append(result)
            temp = args.report.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            temp.replace(args.report)
            print(result.get("group", job), sum(r["verified"] for r in result["cases"]),
                  "verified /", len(result["cases"]), result.get("seconds"), result.get("error", ""), flush=True)
    return int(any(len(g["cases"]) != 4 or not all(r["verified"] for r in g["cases"]) for g in report["groups"]))


if __name__ == "__main__":
    raise SystemExit(main())
