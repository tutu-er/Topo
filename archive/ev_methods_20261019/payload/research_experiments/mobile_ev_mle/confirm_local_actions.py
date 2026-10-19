"""Confirm local search proposals by the original full AC joint design fit.

Only history and the historical-side proposed action names enter. Realized
future noise and evaluator truth are never read. This confirmation is still
reference-conditional and locally optimized, not a global design certificate.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .audit_common import build_bank,unvectorize
from .likelihood import fit_candidate
from .design import select_action
from .run_study import dump


def main():
    here=Path(__file__).resolve().parent;bank=build_bank();records=[]
    for path in sorted((here/"outputs"/"refinement").glob("history_*.json")):
        proposed=json.loads(path.read_text());seed=proposed["seed"]
        original=json.loads((here/"outputs"/"three_directions"/f"history_{seed}.json").read_text())
        yh=np.asarray(original["history_y"]);history=unvectorize(yh,bank.meter,12)
        fits={}
        for h,c in enumerate(bank.candidates):
            indices=np.flatnonzero(bank.labels==h)
            i=int(indices[np.argmin(np.sum((bank.history_means[indices]-yh)**2,axis=1))])
            atom=bank.atoms[i]
            init={k:(atom[k][:12] if k in ("p","q","source_p","source_q") else atom[k])
                  for k in ("r","x","p","q","source_p","source_q")}
            fits[c.name]=fit_candidate(c,history,bank.meter,initial=init,multistart=2,max_nfev=150,seed=seed+81000+h)
        names={"constant_site_0","none"}
        names.update(name for key,name in proposed["choices"].items() if key.startswith("symmetric_") and name is not None)
        design=select_action(bank.candidates,history,fits,bank.meter,
            [bank.actions[n] for n in sorted(names)],multistart=2,max_nfev=150,seed=seed+85000)
        # Keep the complete candidate-action outcomes and every invalid fit.
        record=dict(seed=seed,proposed=sorted(names),design=design,
                    selector_uses_truth=False,selector_uses_future_noise=False,
                    scope="complete continuous AC rival refit of H plus predicted future means; local optimizer")
        records.append(record)
        print(f"nonlinear confirmation {seed}: selected={design.get('selected')}",flush=True)
    payload=dict(records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 design_sha256=hashlib.sha256((here/"design.py").read_bytes()).hexdigest(),
                 likelihood_sha256=hashlib.sha256((here/"likelihood.py").read_bytes()).hexdigest(),
                 is_global_MLE=False,continuous_statistical_calibration=False)
    dump(here/"outputs"/"refinement"/"nonlinear_confirmation.json",payload)
    dump(here/"evidence"/"nonlinear_confirmation.json",payload)


if __name__=="__main__":main()
