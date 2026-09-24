"""Blueprint Automation entry: aggregation phase-diagram full experiment (144 conditions).

Runs 4 case-parallel subprocesses with the conda Topo Python, then the plot
script. The managed runner calls run(ctx) and uses the returned dict as the
artifact payload.
"""

import csv
import json
import subprocess
import time
from pathlib import Path

PROJECT = Path(r"D:\0-github_workspace\Topo")
PY = r"D:\apps\miniconda3\envs\Topo\python.exe"
CASES = ["flynn16", "grid16_k2", "grid16_k4", "grid16_k8"]


def run(ctx):
    t0 = time.time()
    outdirs = []
    procs = []
    logs = []
    for case in CASES:
        outdir = PROJECT / "outputs" / f"agg_phase_{case}"
        outdir.mkdir(parents=True, exist_ok=True)
        log = open(PROJECT / "outputs" / f"agg_phase_{case}.log", "w", encoding="utf-8")
        procs.append(
            subprocess.Popen(
                [PY, "-m", "experiments.run_aggregation_phase_diagram",
                 "--cases", case, "--output", str(outdir), "--resume"],
                cwd=str(PROJECT), stdout=log, stderr=subprocess.STDOUT,
            )
        )
        logs.append(log)
        outdirs.append(outdir)

    rc = [p.wait() for p in procs]
    for log in logs:
        log.close()

    fig_dir = PROJECT / "outputs" / "agg_phase_figures"
    plot_cmd = [PY, "-m", "experiments.plot_aggregation_phase_diagram",
                "--input", *[str(d) for d in outdirs],
                "--output", str(fig_dir)]
    plot = subprocess.run(plot_cmd, cwd=str(PROJECT),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")

    total = ok = 0
    details = []
    for case, code, outdir in zip(CASES, rc, outdirs):
        metrics = outdir / "metrics.csv"
        n = 0
        if metrics.exists():
            with open(metrics, newline="", encoding="utf-8") as f:
                n = sum(1 for _ in csv.DictReader(f))
        total += 36
        ok += n
        details.append(f"{case}: rc={code}, rows={n}/36")
    details.append(f"plot rc={plot.returncode}: {plot.stdout[-300:]} {plot.stderr[-300:]}")

    duration = (time.time() - t0) / 60.0
    status = "success" if all(c == 0 for c in rc) and plot.returncode == 0 and ok == total else "partial"
    artifact = {
        "status": status,
        "duration_minutes": round(duration, 1),
        "conditions_ok": ok,
        "conditions_total": total,
        "figures_dir": str(fig_dir),
        "details": " | ".join(details),
    }
    return {"artifact": artifact}
