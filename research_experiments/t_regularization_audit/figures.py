"""Plot saved lambda intervals; no fitting or model selection."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/t_regularization_audit_20261102/analysis/lambda_intervals.csv"
TARGET = ROOT / "docs/t_regularization_audit_20261102"


def main():
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = ["soumalas b2 / sigma .0002 / N24", "soumalas b3 / sigma .0002 / N48",
              "soumalas b3 / sigma .0005 / N48", "soumalas b4 / sigma .0002 / N24",
              "flynn b3 / sigma .0002 / N24", "soumalas / root noise / N48"]
    fig, ax = plt.subplots(figsize=(10.1, 4.6), layout="constrained")
    for i, row in enumerate(rows):
        if row["same_count_counterexample"] == "True":
            ax.text(.0042, i, "No count penalty can select the true raw tree",
                    va="center", color="#a33432", fontsize=9)
        else:
            lo = float(row["cross_cardinality_sufficient_alpha_lower"])
            hi = float(row["cross_cardinality_sufficient_alpha_upper"])
            ax.plot([lo, hi], [i, i], lw=7, color="#29756d", solid_capstyle="butt")
            ax.plot([lo, hi], [i, i], "|", color="#174e48", ms=13)
    ax.axvline(.003, color="#b55b31", ls="--", lw=1.5, label="Current alpha = .003")
    ax.axvline(.01, color="#356aa0", ls=":", lw=1.8, label="Diagnostic alpha = .01")
    ax.set(xscale="log", xlim=(.0023, .23), ylim=(5.6, -.7), yticks=range(len(labels)),
           yticklabels=labels, xlabel="alpha = lambda / f0 (log scale)",
           title="True-tree optimality intervals in six fixed T domains")
    ax.grid(axis="x", alpha=.18, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    TARGET.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(TARGET / f"lambda_intervals.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
