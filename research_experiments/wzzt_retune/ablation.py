"""Paired RNJ-source ablations using saved fits and identical observations."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from research_experiments.wzzt_systematic.common import read, write, observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics
from .evaluate import choose_index
from .selection import configurations, select


def resolved(point, calibration, resolution):
    threshold = calibration["median_root_depth"] * resolution
    return [s for s, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
            if len(s) == 1 or calibration["a"]*r + calibration["b"]*x + 1e-10 >= threshold]


def source_point(result, config, selected):
    points = result["methods"][config["preprocess"]]["RNJ_WZZT"]
    if selected.get("origin") == "tree_bank":
        return points[selected["selection_index"]]
    key = f"{config['preprocess']}_RX75_0.16"
    return next(p for p in points if p.get("origin") == "tree_bank" and p["config"] == key)


def ablation(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    protocol = read(output/"protocol.json")
    base = Path(protocol["baseline"])
    hybrid_cfg = read(destination/"chosen_global.json")["RNJ_WZZT"]["config"]
    configs = configurations("NJ")
    scores = np.zeros((len(protocol["jobs"]), len(configs)))
    counts = np.zeros_like(scores)
    details, truth_cache = [], {}
    for i, job in enumerate(protocol["jobs"]):
        directory = output/"jobs"/job
        payload, result = read(directory/"payload.json"), read(directory/"result.json")
        labels = result["labels"]
        key = payload["case"], payload["condition"]
        if key not in truth_cache:
            truth_cache[key] = physical_truth(base/"data"/key[0]/key[1]/"truth_network", labels)["clades"]
        truth = truth_cache[key]

        def family_sets(family):
            return {frozenset(labels[k] for k in s) for s in family if 2 <= len(s) < len(labels)}

        def f1(family):
            return clade_metrics(family_sets(family), truth, labels)["clade_f1"]

        for j, cfg in enumerate(configs):
            name = f"{cfg['preprocess']}_{cfg['geometry']}_{cfg['collapse']:g}"
            point = next(p for p in result["methods"][cfg["preprocess"]]["RNJ_WZZT"]
                         if p.get("origin") == "tree_bank" and p["config"] == name)
            scores[i, j] = f1(point["native_family"])
            counts[i, j] = len(point["native_family"])

        selected = select(result, "RNJ_WZZT", hybrid_cfg)
        initial = source_point(result, hybrid_cfg, selected)
        contracted = resolved(initial, result["calibrations"][hybrid_cfg["preprocess"]], hybrid_cfg["resolution"])
        before, after = family_sets(contracted), family_sets(selected["supports"])
        default = next(p for p in result["methods"]["raw"]["RNJ_WZZT"]
                       if p.get("origin") == "tree_bank" and p["config"] == "raw_RX75_0.16")
        native, lp, shrink, final = map(f1, (initial["native_family"], initial["supports"], contracted, selected["supports"]))
        details.append(dict(job_id=job, case=payload["case"], condition=payload["condition"], N=payload["N"],
            selected_origin=selected["origin"], matched_source=initial["config"],
            native_RNJ_F1=native, same_family_L1_F1=lp, L1_resolution_F1=shrink, final_hybrid_F1=final,
            L1_delta=lp-native, resolution_delta=shrink-lp, search_delta=final-shrink,
            added_true=len((after-before)&truth), added_false=len((after-before)-truth),
            removed_true=len((before-after)&truth), removed_false=len((before-after)-truth),
            default_RNJ_F1=f1(default["native_family"]),
            default_RNJ_L1_resolution_F1=f1(resolved(default, result["calibrations"]["raw"], .16))))
    # RNJ is tuned independently; its tie break uses structural complexity and
    # deterministic config order, never the L1 validation score of another model.
    best = choose_index(scores, counts=counts)
    frame = pd.DataFrame(details)
    frame["standalone_tuned_RNJ_F1"] = scores[:, best]
    frame.to_csv(destination/"same_source_ablation.csv", index=False, encoding="utf-8-sig")
    columns = ["standalone_tuned_RNJ_F1", "default_RNJ_F1", "default_RNJ_L1_resolution_F1",
               "native_RNJ_F1", "same_family_L1_F1", "L1_resolution_F1", "final_hybrid_F1",
               "L1_delta", "resolution_delta", "search_delta"]
    frame.groupby("N")[columns].mean().to_csv(destination/"same_source_by_N.csv", encoding="utf-8-sig")
    pd.DataFrame([dict(config=cfg, mean_F1=float(scores[:, j].mean())) for j, cfg in enumerate(configs)]).to_json(
        destination/"standalone_RNJ_grid.json", orient="records", indent=2)
    summary = dict(rows=len(frame), mean=frame[columns].mean().to_dict(),
        standalone_RNJ_common_config=configs[best], selected_origin=frame.selected_origin.value_counts().to_dict(),
        structural_edits=frame[["added_true", "added_false", "removed_true", "removed_false"]].sum().astype(int).to_dict(),
        correction_rows=frame[frame.selected_origin != "tree_bank"].to_dict("records"),
        scope="same saved inputs/fits; matched RNJ sources adaptively chosen by hybrid validation; independently tuned standalone RNJ reported separately",
        truth_used_for_support_generation=False, truth_used_for_global_parameter_selection=True)
    if not np.allclose(frame.final_hybrid_F1-frame.native_RNJ_F1,
                       frame.L1_delta+frame.resolution_delta+frame.search_delta):
        raise AssertionError("paired decomposition failed")
    write(destination/"same_source_summary.json", summary)
    print({k:v for k,v in summary.items() if k != "correction_rows"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("output"); p.add_argument("destination")
    a = p.parse_args(); ablation(a.output, a.destination)
