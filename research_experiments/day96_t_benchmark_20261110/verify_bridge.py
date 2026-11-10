"""Synthetic replay against the neighboring frozen-predictor API; no fitting."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NEIGHBOR = Path("C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo/research_experiments/wzzt_dr_selection")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    sys.path[:0] = [str(NEIGHBOR.parents[1]), str(NEIGHBOR.parents[1] / "rnj_wzzt_core")]
    own = module("own_day96_candidates", Path(__file__).with_name("candidates.py"))
    package = types.ModuleType("neighbor_bridge_audit")
    package.__path__ = [str(NEIGHBOR / "dr_pool_v2")]
    sys.modules[package.__name__] = package
    observations_path = NEIGHBOR / "dr_pool_v2/observations.py"
    selection_path = NEIGHBOR / "dr_pool_v2/selection.py"
    frozen_path = NEIGHBOR / "dr_frozen_distance/frozen_pool.py"
    module(package.__name__ + ".observations", observations_path)
    selection = module(package.__name__ + ".selection", selection_path)
    frozen = module("neighbor_frozen_bridge_audit", frozen_path)
    labels = [201, 202, 203]
    point = dict(supports=[[0], [1], [2], [1, 2]], r=[.1, .1, .1, .2], x=[.02, .02, .02, .05])
    alternative = dict(supports=[[0], [1], [2]], r=[.1, .1, .1], x=[.02, .02, .02])
    R, X = own.response_matrices(point, 3)
    rng = np.random.default_rng(2026111001)
    def data(rows):
        p, q = rng.uniform(.01, .1, (rows, 3)), rng.uniform(.003, .03, (rows, 3))
        return dict(p=p, q=q, y=p@R.T + q@X.T)
    train, valid = data(96), data(288)
    for model in (point, alternative):
        model["objective"] = own.loss(model, train, 3)
    provenance = dict(weights_fit_split="train", truth_used_in_fitting=False,
                      test_used_in_generation=False, future_DR_used=False,
                      input_hashes=dict(train="a"*64, validation="b"*64))
    calibration = dict(a=1., b=1., median_root_depth=1.)
    exported = own.export_candidates([point, point, alternative], labels=labels, train=train,
                                    validation=valid, provenance=provenance, calibration=calibration,
                                    root_label=1, base_kw=200.)
    assert len(exported["candidates"]) == 2
    own.verify_export(exported, train=train, validation=valid)
    predicted_before = own.digest([dict(R=item["R"], X=item["X"]) for item in exported["candidates"]])
    scene = dict(P_terminal=valid["p"], Q_terminal=valid["q"], drop_target=valid["y"], labels=labels)
    max_error = 0.
    for item in exported["candidates"]:
        prediction = frozen.predict(item, scene)
        expected = valid["p"]@np.asarray(item["R"]).T + valid["q"]@np.asarray(item["X"]).T
        max_error = max(max_error, float(np.max(np.abs(prediction-expected))))
        assert abs(float(np.mean(abs(prediction-valid["y"]))) - item["validation_mae"]) < 1e-14
    chosen = selection.select([item["point"] for item in exported["candidates"]], calibration)
    assert chosen["selection_index"] == exported["selections"]["neighbor_style_band3"]["index"]
    assert predicted_before == own.digest([dict(R=item["R"], X=item["X"]) for item in exported["candidates"]])
    rejects = {}
    changed = copy.deepcopy(exported)
    changed["root_label"] = 7
    try:
        own.verify_export(changed)
    except ValueError:
        rejects["changed_contract"] = True
    try:
        frozen.predict(exported["candidates"][0], dict(scene, labels=list(reversed(labels))))
    except ValueError:
        rejects["changed_terminal_order"] = True
    assert all(rejects.get(key) for key in ("changed_contract", "changed_terminal_order"))
    receipt = dict(passed=True, train_rows=96, validation_rows=288, exported_models=2,
                   duplicate_origins_preserved=len(exported["candidates"][0]["origins"]) == 2,
                   neighbor_prediction_max_error=max_error, neighbor_selection_same=True,
                   rejected=rejects, new_fit_calls=0, new_search_calls=0,
                   source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), Path(__file__).with_name("candidates.py"),
                                               observations_path, selection_path, frozen_path)})
    output = ROOT / "docs/day96_t_benchmark_20261110/bridge_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
