import json, numpy as np
from pathlib import Path
from collections import Counter
ROOT = Path(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120")
for c in ("main", "unbalanced", "meter_bias", "shortline", "low_adoption", "rmin_records", "lead0"):
    p = ROOT/c/"runs.jsonl"
    if not p.exists(): continue
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    bind = Counter(min(b["parts"], key=b["parts"].get) for r in recs for b in r["blocks"] if b.get("parts"))
    ac = sum(r["violations"]["ac_band_hours"] for r in recs); db = sum(r["violations"]["deadband_hours"] for r in recs)
    vmin = min(r["violations"]["min_v"] for r in recs); vmax = max(r["violations"]["max_v"] for r in recs)
    fails = sum(r.get("refit_failures", 0) for r in recs); has = sum("refit_failures" in r for r in recs)
    cost = max(r["cost_cny"] for r in recs)
    secs = [g["seconds"] for r in recs for g in r["gates"]]
    print(f"{c:13s} n={len(recs)} binding={dict(bind)} AC_band_h={ac} deadband_h={db} v=[{vmin:.4f},{vmax:.4f}] "
          f"refit_failures={fails} (logged in {has}) max_cost={cost:.1f} gate_s mean={np.mean(secs):.2f} max={max(secs):.1f}")
