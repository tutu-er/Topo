"""Regression of the simplified measured-only path against immutable M7 rows."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "theft_wzzt"
OUT = PACKAGE / "outputs/theft_simplification"
HISTORY = PACKAGE / "outputs/theft"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


entry = module("measured_entry", PACKAGE / "detect_simple.py")
old = module("historical_entry", PACKAGE / "experiments/run_theft.py")
context = old._m7_context()
manifest = json.loads((HISTORY / "m7_run_manifest.json").read_text(encoding="utf-8"))
assert context["signature"] == manifest["signature"], "Historical M7 signature changed"
identified = entry.load_identified(HISTORY / "identified_tree.json")
checks = []
for exp, replicate in (("null", 30), ("exp2", 100), ("exp3", 100)):
    _, scenario, _, _ = old._scenario(exp, replicate)
    # Export ONLY measured fields; this file also exercises the public JSON API.
    wire = {"index": scenario["P_terminal"].index.astype(str).tolist(),
            "terminals": scenario["P_terminal"].columns.tolist(),
            "scenario_settings": scenario["scenario_settings"]}
    wire.update({k: scenario[k].to_numpy().tolist() for k in (*entry.MATRIX_CHANNELS, *entry.SERIES_CHANNELS)})
    input_path = OUT / f"measurements_{exp}.json"
    input_path.write_text(json.dumps(wire, indent=1), encoding="utf-8")
    measured = entry.load_measurements(input_path)
    assert not any("true" in k or "loss" in k or "theft" in k for k in measured)
    started = time.perf_counter()
    result = entry.detect(measured, identified, window=(44, 52))
    elapsed = time.perf_counter() - started
    baseline = json.loads((HISTORY / f"m7_L1_{exp}_rep{replicate}.json").read_text(encoding="utf-8"))
    gain_error = result["gain"] - baseline["gain"]
    amplitude_error = float(np.max(np.abs(np.array(result["amplitude_input_pu"]) - baseline["amplitude_envelope"][1])))
    assert abs(gain_error) < 1e-5 and amplitude_error < 1e-12
    assert result["candidate_locations_by_time"] == baseline["h1_locations_by_time"]
    assert abs(result["gain"] - result["gain_voltage"] - result["gain_balance"]) < 1e-8
    assert result["decision"]["alarm"] is None and result["reported_locations_by_time"] is None
    (OUT / f"simple_{exp}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    checks.append({"experiment": exp, "gain_difference": gain_error,
                   "amplitude_max_difference": amplitude_error,
                   "locations_identical": True, "seconds": elapsed,
                   "gain": result["gain"], "gain_voltage": result["gain_voltage"],
                   "gain_balance": result["gain_balance"], "decision": result["decision"]})
    print(json.dumps(checks[-1]), flush=True)
# The known five L1 nulls cannot resolve a 5% rank test, even for these large gains.
summary = json.loads((HISTORY / "m7_summary.json").read_text(encoding="utf-8"))
l1_null = next(g["null_gains"] for g in summary["groups"] if g["loss_model"] == "L1")
verdict = entry.decision(result["gain"], {"protocol_signature": result["protocol_signature"],
                                       "null_gains": l1_null}, result["protocol_signature"])
assert verdict["status"] == "insufficient_calibration" and verdict["alarm"] is None
before = json.loads((OUT / "before.json").read_text(encoding="utf-8"))
changed = [p for p, digest in before.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest]
assert not changed, changed
report = {"status": "passed", "checks": checks, "changed_historical_files": changed,
          "protected_count": len(before), "m7_signature_unchanged": True,
          "signature": context["signature"], "five_L1_null_decision": verdict,
          "entry_sha256": hashlib.sha256((PACKAGE / "detect_simple.py").read_bytes()).hexdigest()}
(OUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("All measured-only regression checks passed; protected files:", len(before), flush=True)
