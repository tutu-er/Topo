"""Independent replay of the saved-candidate audit, using the standard library.

Provenance: organized from the stdin fragment actually executed on 2026-10-30
at 00:55 UTC.  The original execution receipt is audit_independent.json.  This
saved version was prepared afterwards; it must not be represented as a script
whose hash was frozen before that original execution.

The numerical selection/contraction and crossing checks reproduce that fragment.
The only organizational changes are a main function, command-line paths and a
separate default replay receipt, so the original receipt is not overwritten.

Reads JSON metadata, source hashes and the original 160 result JSON files.
Checks only the existence of NPZ paths: no trajectory loads, fitting, solver,
truth metrics or test values.  Does not import either the parent audit script
or the archived selection implementation.
"""

from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family(supports):
    return tuple(sorted(tuple(sorted(int(x) for x in s)) for s in supports))


def contracted(point, cal):
    return family([
        s for s, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
        if len(s) == 1 or cal["a"] * r + cal["b"] * x + 1e-10
        >= .16 * cal["median_root_depth"]
    ])


def eligible(points):
    value = min(p["validation_mae"] for p in points)
    return [i for i, p in enumerate(points)
            if p["validation_mae"] <= 1.03 * value + 1e-12]


def choose(points):
    return min(eligible(points), key=lambda i: (
        len(points[i]["supports"]), points[i]["validation_mae"], i))


def crossing(families):
    for i, j in itertools.combinations(range(len(families)), 2):
        for a in families[i]:
            left = set(a)
            for b in families[j]:
                right = set(b)
                if left & right and not left <= right and not right <= left:
                    return True
    return False


def run(root, output):
    root = Path(root)
    inventory = read(root / "source_inventory.json")
    audit = read(root / "candidate_audit.json")
    cross = read(root / "crossing_support_audit.json")
    protocol = read(root / "candidate_protocol.json")
    archive_protocol_path = next(
        p for p in inventory["source_sha256"]
        if Path(p).name == "protocol.json")
    archive_protocol = read(archive_protocol_path)
    errors = []
    checks = collections.Counter()
    by_sigma = collections.defaultdict(collections.Counter)
    by_case = collections.defaultdict(collections.Counter)

    def check(condition, label, job=None):
        checks[label] += 1
        if not condition:
            errors.append({"check": label, "job_id": job})

    ids = {r["job_id"] for r in inventory["rows"]}
    expected_ids = {x for x in archive_protocol["jobs"]
                    if x.startswith("main__") and x.endswith("__N288")}
    check(ids == expected_ids, "complete_prespecified_main_N288_index")
    check(len(ids) == len(inventory["rows"]), "no_duplicate_inventory_ids")
    for path, digest in inventory["source_sha256"].items():
        check(sha(path) == digest, "inventory_source_hash", path)
    for path, digest in protocol["source_sha256"].items():
        check(sha(path) == digest, "candidate_source_hash", path)
    check(sha(root / "candidate_protocol.json") == audit["protocol_sha256"],
          "candidate_protocol_hash")
    parent_records = {r["job_id"]: r for r in audit["records"]}
    parent_cross = {r["job_id"]: r for r in cross["records"]}
    check(set(parent_records) == ids, "candidate_record_coverage")
    multiple = difference = 0
    crossing_ids = set()
    outputs = []

    for row in inventory["rows"]:
        job = row["job_id"]
        result_path = Path(row["existing_result"]["path"])
        result = read(result_path)
        saved = parent_records[job]
        calibration = result["calibration"]
        points = result["base_points"]
        rnj = result["rnj_bank"]
        checks["result_files_read"] += 1
        check(sha(result_path) == row["existing_result"]["catalog_sha256"]
              == saved["result_sha256"], "result_hash", job)
        check(result["N"] == 288 and result["panel"] == "main"
              and result["job_id"] == job, "result_index_match", job)
        check(result["case"] == row["case"]
              and result["bundle"] == row["bundle"]
              and result["voltage_sigma"] == row["voltage_sigma"],
              "condition_match", job)
        for channel in ("train", "validation", "test"):
            check(Path(row[channel]["path"]).is_file(),
                  "observation_path_exists_only", job)
        check(all(math.isfinite(p["validation_mae"]) for p in points + rnj),
              "finite_saved_validation_scores", job)
        accepted = eligible(points)
        index = choose(points)
        rnj_index = choose(rnj)
        families = sorted(set(contracted(points[i], calibration) for i in accepted))
        raw_families = set(family(points[i]["supports"]) for i in accepted)
        chosen = contracted(points[index], calibration)
        rnj_chosen = contracted(rnj[rnj_index], calibration)
        check(len(points) == saved["raw_candidate_points"], "raw_point_count", job)
        check(accepted == saved["eligible_point_indices"],
              "eligible_point_indices", job)
        check(len(raw_families) == saved["eligible_raw_families"],
              "eligible_raw_family_count", job)
        check(len(families) == saved["eligible_contracted_families"],
              "eligible_contracted_family_count", job)
        check(set(families) == {family(x) for x in saved["contracted_families"]},
              "contracted_family_contents", job)
        check(index == saved["selected_index"], "selected_index", job)
        check(points[index].get("origin") == saved["selected_origin"],
              "selected_origin", job)
        check(chosen == family(saved["multistart_final_support"])
              == family(result["models"]["multistart"]["supports"]),
              "multistart_output", job)
        check(rnj_chosen == family(saved["RNJ_final_support"])
              == family(result["models"]["RNJ_L1_bank"]["supports"]),
              "RNJ_output", job)
        check((rnj_chosen == chosen) == saved["RNJ_multistart_same_final_support"],
              "RNJ_multistart_same_flag", job)
        is_multiple = len(families) > 1
        is_different = rnj_chosen != chosen
        is_crossing = crossing(families)
        multiple += is_multiple
        difference += is_different
        if is_crossing:
            crossing_ids.add(job)
        for counter in (by_sigma[str(row["voltage_sigma"])], by_case[row["case"]]):
            counter["planned"] += 1
            counter["multiple_historical_outputs"] += is_multiple
            counter["RNJ_multistart_different"] += is_different
            counter["crossing_candidate_conditions"] += is_crossing
        outputs.append(dict(job_id=job, eligible_family_count=len(families),
                            selected_index=index, RNJ_multistart_different=is_different,
                            crossing_candidate_conditions=is_crossing))

    check(crossing_ids == set(parent_cross), "crossing_condition_set")
    for job, record in parent_cross.items():
        left, right = map(set, record["witness"]["crossing_support_indices"])
        check(bool(left & right) and not left <= right and not right <= left,
              "crossing_witness_is_true_overlap", job)
        families = parent_records[job]["contracted_families"]
        i, j = record["witness"]["family_indices"]
        check(tuple(sorted(left)) in family(families[i])
              and tuple(sorted(right)) in family(families[j]),
              "crossing_witness_membership", job)
    check(multiple == audit["multiple_historical_outputs"], "multiple_total")
    check(difference == audit["RNJ_multistart_different"], "RNJ_difference_total")
    check(len(crossing_ids) == cross["crossing_candidate_conditions"], "crossing_total")
    for sigma, counter in by_sigma.items():
        saved = audit["by_voltage_sigma"][sigma]
        for key in ("planned", "multiple_historical_outputs", "RNJ_multistart_different"):
            check(counter[key] == saved[key], "sigma_" + key, sigma)
        check(counter["crossing_candidate_conditions"]
              == cross["by_voltage_sigma"].get(sigma, 0), "sigma_crossing", sigma)

    receipt = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="passed" if not errors else "failed",
        independent_implementation=True, imported_parent_audit=False,
        imported_archive_selection=False, NPZ_loads=0, new_fits=0, solver_calls=0,
        truth_metrics_used=False, test_values_loaded=False,
        observation_hashes_verified=False,
        source_index="all original predeclared main N288 inputs",
        implementation_provenance=("organized after execution from the stdin fragment "
                                   "that produced audit_independent.json; this is a replay"),
        replay_script_sha256=sha(__file__),
        inputs_sha256={p.name: sha(p) for p in (
            root / "source_inventory.json", root / "candidate_audit.json",
            root / "crossing_support_audit.json", root / "candidate_protocol.json")},
        planned=len(ids), audited=checks["result_files_read"],
        multiple_historical_outputs=multiple, RNJ_multistart_different=difference,
        crossing_candidate_conditions=len(crossing_ids),
        by_voltage_sigma={k: dict(v) for k, v in by_sigma.items()},
        by_case={k: dict(v) for k, v in by_case.items()}, checks=dict(checks),
        error_count=len(errors), errors=errors,
        scope=("Saved finite-family point diversity after original selection/contraction "
               "only; no parameter coverage, correctness or DR improvement claim"),
        records=outputs)
    Path(output).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("records", "checks")}, indent=2))
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path,
                        help="Default: <root>/audit_independent_replay.json")
    args = parser.parse_args()
    output = args.output or args.root / "audit_independent_replay.json"
    if output.resolve() == (args.root / "audit_independent.json").resolve():
        parser.error("preserve the original execution receipt; choose a replay filename")
    return run(args.root, output)


if __name__ == "__main__":
    raise SystemExit(main())
