"""Combine completed GTLS posterior sweeps without rerunning simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.run_gtls_topology_posterior_sweep import METHOD_COLUMNS
from terminal_case33.utils.io import ensure_dir, write_json


def _long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in frame.iterrows():
        for method, column in METHOD_COLUMNS.items():
            if column not in row.index:
                continue
            records.append(
                {
                    "condition_id": row["condition_id"],
                    "case": row["case"],
                    "data_regime": row["data_regime"],
                    "noise_regime": row["noise_regime"],
                    "trial": row["trial"],
                    "method": method,
                    "f1": row[column],
                    "exact": row[column.replace("_f1", "_exact")],
                }
            )
    return pd.DataFrame(records)


def run(input_directories: list[str], output_directory: str) -> None:
    """Merge unique successful conditions and write aggregate tables."""

    output = ensure_dir(output_directory)
    frames = []
    for directory in input_directories:
        path = Path(directory) / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["status"].eq("ok")]
    frame = frame.drop_duplicates("condition_id", keep="last")
    frame = frame.sort_values("condition_id").reset_index(drop=True)
    long = _long_frame(frame)
    overall = (
        long.groupby("method", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            min_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    by_case = long.groupby(["case", "method"], as_index=False).agg(
        mean_f1=("f1", "mean"), min_f1=("f1", "min"), exact_rate=("exact", "mean")
    )
    by_regime = long.groupby(["data_regime", "noise_regime", "method"], as_index=False).agg(
        mean_f1=("f1", "mean"), min_f1=("f1", "min"), exact_rate=("exact", "mean")
    )
    frame.to_csv(output / "metrics.csv", index=False)
    long.to_csv(output / "method_metrics_long.csv", index=False)
    overall.to_csv(output / "summary_by_method.csv", index=False)
    by_case.to_csv(output / "summary_by_case.csv", index=False)
    by_regime.to_csv(output / "summary_by_regime.csv", index=False)
    write_json(
        output / "summary.json",
        {
            "unique_conditions": len(frame),
            "input_directories": input_directories,
            "all_measurements_noisy": True,
            "voltage_generation": "radial_ac_power_flow",
        },
    )
    lines = [
        "# Combined structured-GTLS posterior evaluation",
        "",
        f"Unique noisy AC conditions: {len(frame)}.",
        "",
        "| method | mean F1 | minimum F1 | exact rate |",
        "|---|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['method']} | {row['mean_f1']:.4f} | "
            f"{row['min_f1']:.4f} | {row['exact_rate']:.2%} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/gtls_structured_combined")
    args = parser.parse_args()
    run(args.inputs, args.output)


if __name__ == "__main__":
    main()
