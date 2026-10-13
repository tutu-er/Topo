"""Recompute archived H-only importance diagnostics without new Monte Carlo.

This verifies artifact bytes, frozen source bytes and the seven-mode weight
algebra. It never accepts a posterior, creates EIG or starts a new run.
"""
from __future__ import annotations
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tarfile
import numpy as np
from scipy.special import logsumexp


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_full(compact):
    compact = Path(compact)
    manifest = json.loads((compact / "compact_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = compact / row["relative"]
        if digest(path) != row["sha256"]:
            raise RuntimeError("compact artifact hash mismatch: " + row["relative"])
    with tarfile.open(compact / "frozen_source_history.tar.gz", "r:gz") as archive:
        snapshots = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    for row in manifest["frozen_snapshot_files"]:
        if sha256(snapshots[row["archive_member"]]).hexdigest() != row["sha256"]:
            raise RuntimeError("frozen source snapshot mismatch: " + row["archive_member"])
    runs = []
    for row in manifest["diagnostic_banks"]:
        with gzip.open(compact / row["relative"], "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        weights, modes, max_error = [], [], 0.
        for mode in data["modes"]:
            lh, lq, lp, lw = (np.asarray(mode[k], float) for k in ("loghistory", "logq", "logprior", "logweight"))
            expected = np.log(mode["mass"]) - np.log(len(lw)) + lp - lq + lh
            finite = np.isfinite(lw)
            if not np.array_equal(np.isfinite(expected), finite):
                raise RuntimeError("weight finite pattern mismatch")
            if np.any(finite):
                error = float(np.max(np.abs(expected[finite] - lw[finite])))
                max_error = max(max_error, error)
                if error > 2e-10:
                    raise RuntimeError("archived target weight algebra mismatch")
            total = float(logsumexp(lw))
            ess = float(np.exp(2*total-logsumexp(2*lw))) if np.isfinite(total) else 0.
            modes.append(dict(component=mode["component"], conditional_ESS=ess,
                              zero_MC_estimate=not np.isfinite(total)))
            weights.append(lw)
        if len(modes) != 7:
            raise RuntimeError("seven-mode denominator missing")
        flat = np.concatenate(weights)
        logz = float(logsumexp(flat))
        ess = float(np.exp(2*logz-logsumexp(2*flat)))
        maximum = float(np.exp(flat.max()-logz))
        failed = ess < 100 or maximum > .05 or any(m["conditional_ESS"] < 20 for m in modes)
        if not failed:
            raise RuntimeError("an archived failure unexpectedly passes original ESS gates")
        runs.append(dict(version=data["version"], bank=data["bank"], status="failed",
                         global_ESS=ess, maximum_weight=maximum, log_evidence=logz,
                         modes=modes, weight_recomputation_max_error=max_error))
    result = dict(postprocessing_only=True, new_MC_executed=False,
                  preserved_versions=sorted({r["version"] for r in runs}), diagnostic_banks=len(runs),
                  all_original_gates_failed=True, posterior_accepted=False,
                  predictive_EIG_executed=False, frozen_snapshot_files=len(snapshots), runs=runs)
    return result


def verify(compact):
    """Integrity and sufficient-statistic checks; does not reread particle NPZ."""
    compact = Path(compact)
    manifest = json.loads((compact / "compact_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] == "continuous_eig_compact_v1":
        return verify_full(compact)
    for row in manifest["files"]:
        if digest(compact / row["relative"]) != row["sha256"]:
            raise RuntimeError("compact artifact hash mismatch: " + row["relative"])
    with tarfile.open(compact / "frozen_source_history.tar.gz", "r:gz") as archive:
        blobs = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    for row in manifest["frozen_snapshot_files"]:
        if sha256(blobs[row["archive_member"]]).hexdigest() != row["sha256"]:
            raise RuntimeError("frozen source blob hash mismatch")
    source_hashes = {r["source_relative"]: r["sha256"] for r in manifest["frozen_snapshot_files"]}
    source_config_checks = 0
    for version in ("v1", "v2", "v3", "v4", "v5"):
        directory = "continuous_eig_Honly_" + version
        config = json.loads((compact / directory / "config_before_MC.json").read_text(encoding="utf-8"))
        for field in ("source_sha256", "current_source_sha256", "math_sha256"):
            for name, expected in config.get(field, {}).items():
                if source_hashes.get(directory + "/source_snapshot/" + name) != expected:
                    raise RuntimeError("source snapshot differs from frozen configuration: " + directory + "/" + name)
                source_config_checks += 1
    with gzip.open(compact / "aggregate_diagnostics.json.gz", "rt", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    maximum_error = 0.
    for row in aggregate["banks"]:
        if len(row["modes"]) != 7:
            raise RuntimeError("seven-mode failure denominator missing")
        moments = row["weight_sufficient_statistics"]
        logz = float(logsumexp([m["logsumweight"] for m in moments]))
        ess = float(np.exp(2*logz-logsumexp([m["logsumweight_squared"] for m in moments])))
        maxw = float(np.exp(max(m["maximum_logweight"] for m in moments)-logz))
        errors = [abs(logz-row["log_evidence"]), abs(ess-row["global_ESS"]), abs(maxw-row["maximum_weight"])]
        mode_ess = []
        for moment, mode in zip(moments, row["modes"]):
            value = float(np.exp(2*moment["logsumweight"]-moment["logsumweight_squared"]))
            mode_ess.append(value)
            errors.append(abs(value-mode["conditional_particle_ESS"]))
        maximum_error = max(maximum_error, *errors)
        if max(errors) > 2e-7:
            raise RuntimeError("aggregate sufficient statistics disagree with archived diagnostics")
        if not (ess < 100 or maxw > .05 or min(mode_ess) < 20):
            raise RuntimeError("archived failure unexpectedly passes initial ESS gates")
    return dict(postprocessing_only=True,new_MC_executed=False,
                scope="artifact integrity and consistency of archived aggregate sufficient statistics",
                particle_ESS_independently_recomputed_from_git_summary=False,
                raw_particle_recheck_record="full_bank_recheck_record.json.gz",
                banks_checked=len(aggregate["banks"]),preserved_versions=["v1","v2","v3","v4","v5"],
                frozen_source_unique_blobs=len(blobs),maximum_aggregate_consistency_error=maximum_error,
                frozen_config_source_hashes_checked=source_config_checks,
                all_original_gates_failed=True,posterior_accepted=False,predictive_EIG_executed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, default=Path(__file__).parent / "evidence/continuous_eig_compact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.compact)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "runs"}))
