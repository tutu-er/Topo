"""Development cases for the draft v12 code (identification_v8): reading a case, and the F1 score. Kept out of the
method package, which knows nothing of the case library. The sealed set is never opened here."""
import hashlib
import json
import pickle

from . import build


def load_dev_case(name):
    """A development case's record and readings (build.to_input), checked against the development manifest's hash."""
    rows = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    manifest = {r["name"]: r["sha256"] for r in rows}
    if name not in manifest:
        raise SystemExit(f"{name} is not a development case")
    data = (build.OUT/"dev"/f"{name}.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[name]:
        raise SystemExit(f"{name}: the case file does not match its manifest hash")
    rec = pickle.loads(data)
    return rec, build.to_input(rec)


def f1(est, truth):
    """Precision, recall and F1 of a set of lines against the true set."""
    tp = len(est & truth)
    p = tp/len(est) if est else 0.
    r = tp/len(truth) if truth else 1.
    return dict(tp=tp, est=len(est), truth=len(truth), precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)
