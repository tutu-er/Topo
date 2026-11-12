"""Post-seal reproducibility audit, including the original coverage ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io as string_io
import json
from pathlib import Path
import unittest

from ... import io
from . import test_scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    protocol = io.read(out/"protocol.json"); seal = io.read(out/"selection_seal.json")
    rows = io.read(out/"evaluation.json"); issues = []
    root = Path(__file__).resolve().parents[3]
    ledger_path = root/"topology_identification_standalone/experiments/full48_mse/coverage.csv"
    ledger = {(r["job"], r["method"]): r for r in csv.DictReader(ledger_path.open(encoding="utf-8"))}
    if io.ref(out/"protocol.json")["sha256"] != seal["protocol_sha256"]:
        issues.append("protocol seal changed")
    references = {}
    for request in protocol["requests"]:
        for role in ("result", "train", "validation", "observed"):
            if role in request: references[request[role]["path"]] = request[role]
        if io.ref(out/"scores"/(request["id"]+".json"))["sha256"] != seal["scores"][request["id"]]:
            issues.append("weights changed: "+request["id"])
    for reference in references.values():
        if io.ref(reference["path"])["sha256"] != reference["sha256"]:
            issues.append("input changed: "+reference["path"])
    for relative, reference in protocol["source_references"].items():
        if io.ref(reference["path"])["sha256"] != reference["sha256"]:
            issues.append("scoring source changed: "+relative)
        if io.ref(out/"source_snapshot"/relative)["sha256"] != reference["sha256"]:
            issues.append("source snapshot changed: "+relative)
    for reference in io.read(out/"evaluation_sources.json"):
        if io.ref(reference["path"])["sha256"] != reference["sha256"]:
            issues.append("evaluation source changed: "+reference["path"])
    compared = 0
    for row in rows:
        if row["cohort"] == "full48" and row["status"] == "complete":
            job = row["id"].removeprefix("full48__").rsplit("__", 1)[0]
            old = ledger[job, row["method"]]
            compared += 1
            if (row["native_correct"] != (old["selected_exact"] == "True")
                    or row["truth_covered"] != (old["output_bank_contains_true"] == "True")):
                issues.append("parent truth/coverage differs: "+row["id"])
    stream = string_io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_scoring)
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not tests.wasSuccessful(): issues.append("formula/interface tests failed")
    (out/"audit_tests.txt").write_text(stream.getvalue(), encoding="utf-8", newline="\n")
    near = [r for r in rows if r.get("near_pair_003")]
    complete = [r for r in rows if r["status"] == "complete"]
    validation = dict(status="passed" if not issues else "failed", libraries=len(rows),
        formula_and_interface_tests=tests.testsRun, primary_parent_coverage_comparisons=compared,
        selection_seal_checks=len(protocol["requests"]), input_references=len(references),
        scoring_source_files=len(protocol["source_references"]), audit_source=io.ref(__file__),
        original_coverage_ledger=io.ref(ledger_path), issues=issues,
        max_probability_error=max((r["max_probability_error"] for r in complete), default=None),
        near_method_input_evaluations=len(near),
        near_distinct_histories=len({r["id"].removeprefix("full48__").rsplit("__", 1)[0] for r in near}),
        near_truth_covered=sum(r["truth_covered"] for r in near),
        fits=0, optimizers=0, new_random_draws=0)
    io.write(out/"public/VALIDATION.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
