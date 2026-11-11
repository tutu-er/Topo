"""Check local data/tests, optionally run all methods from a detached copy."""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


def check(output, full_smoke=False):
    started = perf_counter(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    sources = list(ROOT.glob("*.py"))
    for file in sources:
        tree = ast.parse(file.read_text(encoding="utf8"))
        for node in ast.walk(tree):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split(".")[0] in {"research_experiments", "rnj_wzzt", "terminal_case33"} for name in names):
                raise RuntimeError("External project import in "+file.name)
    data_check = data.verify_bundled_data()
    environment = dict(python=sys.version, executable=sys.executable, packages={})
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            environment["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            environment["packages"][name] = None
    unit = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (output/"unittest.log").write_text(unit.stdout, encoding="utf8")
    if unit.returncode:
        raise RuntimeError("Unit verification failed; inspect "+str(output/"unittest.log"))
    smoke = None
    if full_smoke:
        with tempfile.TemporaryDirectory(prefix="topology-rte-detached-") as temporary:
            temporary = Path(temporary).resolve()
            if temporary.parent != Path(tempfile.gettempdir()).resolve():
                raise RuntimeError("Unexpected temporary-directory boundary")
            copied = temporary/"renamed_standalone"
            shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns("outputs", "validation", "__pycache__", ".git"))
            target = temporary/"results"
            env = os.environ.copy(); env.pop("PYTHONPATH", None)
            bootstrap = """import runpy, sys
class RejectParentProject:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'research_experiments', 'rnj_wzzt', 'terminal_case33', 'topology_identification_standalone'}:
            raise ImportError('Original-project import forbidden: '+fullname)
        return None
sys.meta_path.insert(0, RejectParentProject())
entry = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(entry, run_name='__main__')
"""
            command = [sys.executable, "-I", "-c", bootstrap, str(copied/"run.py"),
                       "ablation", "--time-limit", "30", "--output", str(target)]
            run = subprocess.run(command, cwd=temporary, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            (output/"detached_smoke.log").write_text(run.stdout, encoding="utf8")
            if run.returncode:
                raise RuntimeError("Detached full-MIQP smoke failed; inspect "+str(output/"detached_smoke.log"))
            summary = json.loads((target/"summary.json").read_text(encoding="utf8"))
            records = [json.loads(p.read_text(encoding="utf8")) for p in (target/"runs").rglob("*.json")]
            if len(records) != 10 or any(r["status"] != "complete" for r in records):
                raise RuntimeError("Detached smoke did not complete all ten algorithms")
            calls = [call for record in records for call in record["result"]["MIQP"] if call.get("solve_started")]
            if any(call["status"] != 2 or call["gap"] > 1.0001e-4 for call in calls):
                raise RuntimeError("Detached smoke has an unresolved structural subproblem")
            destination = output/"detached_smoke"
            if destination.exists():
                raise FileExistsError("Use a new verification output directory")
            shutil.copytree(target, destination)
            smoke = dict(summary=summary, algorithms=len(records), MIQP_calls=len(calls),
                         maximum_MIP_gap=max((r["gap"] for r in calls), default=0.),
                         copied_and_renamed=True, python_isolated_mode=True,
                         original_project_not_on_pythonpath=True, runtime_project_imports_forbidden=True,
                         bundled_data_copied=True, temporary_parent_checked=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = dict(passed=True, environment=environment, data=data_check, unit_returncode=unit.returncode,
                  structural_smoke=smoke, source_sha256=hashes, seconds=perf_counter()-started,
                  skipped_test_lines=[line.strip() for line in unit.stdout.splitlines() if "skipped" in line.lower()],
                  scope="Implementation, finite toy domains and directory independence; not a new full benchmark")
    write(output/"check_receipt.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/check")
    parser.add_argument("--full-smoke", action="store_true", help="also run all ten methods in an isolated copied directory")
    args = parser.parse_args()
    check(args.output, args.full_smoke)
