"""Run disjoint seed jobs of the frozen scientific pipeline; no model changes."""
import argparse
import hashlib
import json
from pathlib import Path
from .run_study import one_trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    jobs=json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    out=Path(args.output)
    config=json.loads((out/"config.json").read_text(encoding="utf-8-sig"))
    here=Path(__file__).resolve().parent
    for name,digest in config["code_sha256"].items():
        actual=hashlib.sha256((here/name).read_bytes()).hexdigest()
        if actual!=digest:
            raise RuntimeError("Frozen scientific source changed: "+name)
    for job in jobs:
        case,seed=job["case"],job["seed"]
        path=out/f"{case}_{seed}.json"
        if path.exists():
            old=json.loads(path.read_text(encoding="utf-8"))
            if old["case"]!=case or old["seed"]!=seed or len(old["records"])!=5:
                raise RuntimeError("Incompatible cached trial: "+str(path))
            print(f"cached {case} {seed}",flush=True)
            continue
        t=one_trial(seed,case,run_wzzt=not config["skip_wzzt"],
            multistart=config["multistart"],max_nfev=config["max_nfev"],output=out)
        print(f"{job['role']} {case} {seed}: {t['seconds']:.2f}s "+
              f"design={t['design']['status']} paired_complete="+
              str(t["records"][0]["paired_trial_complete"]),flush=True)


if __name__=="__main__":
    main()
