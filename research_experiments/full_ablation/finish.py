"""Wait for this finite batch, then audit and archive its existing results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def finish(output):
    output = Path(output).resolve()
    protocol = read(output / "protocol.json")
    assert protocol["planned"] == 960
    print("Waiting for the frozen 960-evaluation batch; no new optimization is launched here.", flush=True)
    while not (output / "completion.json").exists():
        sleep(10.)
    expected = read(output / "analysis_source/manifest.json")["files"]
    for name, digest in expected.items():
        assert sha(ROOT / "research_experiments/full_ablation" / name) == digest, name
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "rnj_wzzt_core")
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    with (output / "postprocess.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-B", "-m", "research_experiments.full_ablation.analyze", str(output), "--workers", "4"],
                                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        write(output / "postprocess.json", dict(passed=False, analysis_exit_code=result.returncode))
        raise RuntimeError("Postprocessing failed; see the preserved postprocess.log")
    receipt = read(output / "analysis/verification.json")
    delivery = ROOT / "docs" / output.name
    archive = delivery / "records.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".py", ".json", ".csv", ".md", ".log"}:
                bundle.write(path, path.relative_to(output).as_posix())
    status = dict(passed=receipt["passed"], complete=receipt["complete"], returned=receipt["returned"],
                  verified=receipt["verified"], planned=960, analysis_exit_code=result.returncode,
                  archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
                  updated_utc=datetime.now(timezone.utc).isoformat(), visual_review_pending=True)
    write(output / "postprocess.json", status)
    write(delivery / "postprocess.json", status)
    write(delivery / "manifest.json", {"files": {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
          for p in sorted(delivery.iterdir()) if p.is_file() and p.name != "manifest.json"}})
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    finish(parser.parse_args().output)
