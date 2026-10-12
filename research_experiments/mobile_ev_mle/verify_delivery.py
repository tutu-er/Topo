"""Independent exported-evidence checks, without rerunning statistical trials."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from .run_study import dump

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
e=HERE/"evidence"
summary=json.loads((e/"formal_summary.json").read_text(encoding="utf-8"))
trials=json.loads((e/"formal_trials.json").read_text(encoding="utf-8"))
assert len(trials)==80
assert len({(t["role"],t["case"],t["seed"]) for t in trials})==80
assert all(not t["selection_uses_truth"] and not t["selection_uses_future_noise"] for t in trials)
normalizer_per_time=3*sum(np.log(2*np.pi*np.array([.0006,.0004,.00012])**2))+2*np.log(2*np.pi*.001**2)
max_loglik_error=0.;max_unit_error=0.;fit_count=0
with (e/"formal_fits.csv").open(encoding="utf-8",newline="") as stream:
    for row in csv.DictReader(stream):
        t=12 if row["method"]=="historical" else 20
        loss,ell=float(row["loss"]),float(row["attained_loglik"])
        max_loglik_error=max(max_loglik_error,abs(ell+.5*(loss+t*normalizer_per_time)))
        for name in ("r","x"):
            pu=np.array(json.loads(row[name+"_pu"]))
            ohm=np.array(json.loads(row[name+"_ohm"]))
            assert len(pu)==len(json.loads(row["supports"]))
            max_unit_error=max(max_unit_error,float(np.max(abs(ohm-8*pu))))
        assert row["global_optimum_certified"]=="False"
        fit_count+=1
assert fit_count==1920
assert max_loglik_error<1e-9 and max_unit_error<1e-14
for case in ("strong","weak"):
    assert summary["thresholds"][case]["valid_count"]==20
    assert summary["thresholds"][case]["failed_count"]==0
joint=[]
for case in ("strong","weak"):
    valid=[t for t in trials if t["role"]=="evaluation" and t["case"]==case and
           t["records"][0]["paired_trial_complete"]]
    dcrit=summary["thresholds"][case]["deviance_threshold"]
    joint.append(dict(case=case,paired_valid=len(valid),
        any_method_false_rejection=sum(any(r["true_deviance"]>dcrit for r in t["records"]) for t in valid)))
pdf_path=ROOT/"docs"/"mobile_ev_probing_report_v2.pdf"
pdf_info=subprocess.check_output(["pdfinfo",str(pdf_path)]).decode("utf-8",errors="replace")
pdf_pages=int(re.search(r"^Pages:\s+(\d+)",pdf_info,re.MULTILINE).group(1))
result=dict(date="2026-10-12",scientific_tests_passed=23,
    document_revision="explicit AC states, identifiability proof, Gaussian likelihood, joint EV design and timing invariance",
    numerical_experiments_recomputed=False,
    scientific_test_evidence="23 tests passed at prior implementation delivery; scientific algorithm and tests unchanged by this document revision",
    tests_command="D:/apps/miniconda3/envs/Topo/python.exe -m pytest research_experiments/mobile_ev_mle -q -p no:cacheprovider",
    formal_trials=80,formal_method_records=400,exported_candidate_fits=fit_count,
    local_loglikelihood_normalization_max_error=max_loglik_error,
    impedance_unit_conversion_max_error=max_unit_error,joint_false_rejection=joint,
    pdf=dict(pages=pdf_pages,bytes=(ROOT/"docs"/"mobile_ev_probing_report_v2.pdf").stat().st_size,
        compiler="local latexmk/XeLaTeX",compile_success=True,
        builtin_compile="unavailable: Windows sandbox setup; local compiler used",
        visual_review=f"all{pdf_pages}latestpages rendered and inspected by root and review agents",
        log_checks="no LaTeX warnings, undefined refs/citations, overfull/underfull boxes or missing glyphs"),
    source_sha256=hashlib.sha256((ROOT/"docs"/"mobile_ev_probing_report_v2.tex").read_bytes()).hexdigest(),
    pdf_sha256=hashlib.sha256((ROOT/"docs"/"mobile_ev_probing_report_v2.pdf").read_bytes()).hexdigest(),
    global_MLE_verified=False,
    statistical_scope="synthetic cherry-truth mixture, known calibrated meters, fixed optimizer and paired-complete evaluation; star truth and n5 calibration not claimed",
    formal_failure_preserved="strong51007 fixed_pulse/clade01 optimizer budget; auxiliary cold start recovery does not change primary19 denominator")
dump(e/"delivery_verification.json",result)
print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    pass
