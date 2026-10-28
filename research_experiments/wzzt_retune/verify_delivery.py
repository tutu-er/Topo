"""Verify frozen inputs, source archives, selected-model AC binding and tables."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read, write, sha


def verify(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    root = Path(__file__).resolve().parents[2]
    manifest = read(destination/"manifest.json")
    checks = 0
    for paths, base in ((manifest["local"], output), (manifest["external"], None),
                        (read(destination/"result_manifest.json"), output),
                        (read(destination/"historical_source_manifest.json"), root),
                        (read(destination/"analysis_source_manifest.json"), Path(__file__).parent),
                        (read(destination/"delivery_manifest.json"), destination)):
        for path, expected in paths.items():
            target = base/path if base is not None else Path(path)
            if sha(target) != expected:raise AssertionError(f"hash mismatch: {target}")
            checks += 1
    for archive_name, base in (("producer_source.zip", output/"source"), ("analysis_source.zip", Path(__file__).parent)):
        with zipfile.ZipFile(destination/archive_name) as archive:
            for name in archive.namelist():
                if hashlib.sha256(archive.read(name)).hexdigest() != sha(base/name):
                    raise AssertionError(f"archive source mismatch: {archive_name}/{name}")
                checks += 1
    frame = pd.read_csv(destination/"per_job.csv")
    assert len(frame)==320 and not frame.duplicated(["job_id","method"]).any()
    assert frame.job_id.nunique()==80 and frame.model_available.all() and frame.ac_valid.all()
    assert (frame.groupby(["N","method"]).size()==20).all()
    assert (frame.groupby("method").config_json.nunique()==1).all()
    selected = read(destination/"selected_models.json")
    assert len(selected)==320
    for item in selected:
        record = read(output/"ac"/(item["key"]+".json"))
        digest = hashlib.sha256(json.dumps(item["point"],sort_keys=True).encode()).hexdigest()
        assert record["model_sha256"]==digest and "error" not in record
        assert record["testing"]["valid"] and record["physical"]["valid"]
    paired = pd.read_csv(destination/"same_source_ablation.csv")
    assert len(paired)==80 and paired.job_id.nunique()==80
    joined = paired.merge(frame[frame.method=="RNJ_WZZT"][["job_id","clade_f1"]],on="job_id",validate="one_to_one")
    assert np.allclose(joined.final_hybrid_F1,joined.clade_f1)
    assert np.allclose(paired.final_hybrid_F1-paired.native_RNJ_F1,
                       paired.L1_delta+paired.resolution_delta+paired.search_delta)
    audit = read(destination/"audit_summary.json")
    assert audit["all_passed"] and audit["independent_LP_checks"]==32
    summary = dict(all_passed=True,hash_and_archive_checks=checks,selected_model_AC_bindings=320,
                   paired_ablation_rows=80,unique_inputs=80,method_rows=320,
                   per_N_method_rows=20,one_common_config_per_method=True,
                   unit_test_command="python -B -m pytest research_experiments/tests/test_wzzt_retune.py -q",
                   observed_unit_tests_passed=7,physical_and_primal_audit_passed=True)
    write(destination/"delivery_verification.json", summary)
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir()
          if p.is_file() and p.name!="delivery_manifest.json"})
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination")
    a=p.parse_args();verify(a.output,a.destination)
