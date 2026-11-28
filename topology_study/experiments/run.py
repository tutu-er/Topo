"""Run a topology study experiment with its original command-line options."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


STUDY = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "offline": ("four_offline", "Four-feeder passive comparison, stage audit and end-of-day replay"),
    "probe": ("four_probe", "One- or two-day dispatch on one named feeder"),
    "probe-summary": ("four_probe_summary", "Summarize saved dispatch trajectories and draw the example"),
    "sensitivity": ("four_sensitivity", "Paired changes to history length, sampling and meter noise"),
    "node": ("node_mechanism", "Exhaustive fixed-node comparison and controlled missing-line test"),
    "root-ablation": ("root_ablation", "Trace root-side lines missed by NJ or RNJ through the identification stages"),
    "root-variants": ("root_variants", "Check the same failure mode on alternate hidden-load variants"),
    "baselines-check": ("passive_baselines", "Verify the passive baselines on a known topology"),
    "wide-feeders": ("wide_feeders", "Compare identification stages on larger feeders with wide junctions"),
    "line-exchange": ("exchange_cases", "Compare adding or replacing one line on the same anchored tree"),
    "exchange-reference": ("exchange_reference", "Verify a ten-child line replacement against exhaustive enumeration"),
    "random-restore": ("random_dev_recovery", "Rebuild development feeders and verify the original file hashes"),
    "random": ("random_dev_compare", "Compare five methods on verified small to huge development feeders"),
    "random-budget": ("random_budget", "Repeat time-limited random-feeder solves with a larger budget"),
    "pool-audit": ("pool_audit", "Check four small cases by independent enumeration"),
    "node-pairs": ("node_pairs", "Run the earlier paired node-refinement and anchor tests"),
    "identify-cases": ("identify_cases", "Run, summarize or compare development and EU-LV identification"),
    "probe-cases": ("probe_cases", "Run the earlier simulated-day probing protocol"),
    "checks": ("checks", "Inspect extra lines and wide nodes in saved identification runs"),
    "node-exact": ("node_exact", "N1: node MIQP and its flip test against enumeration, 6 to 24 children"),
    "node-recovery": ("node_recovery", "N2: hidden shared lines the builders miss, found by the node MIQP"),
    "node-pipeline": ("node_pipeline", "N3: Algorithm 1 with and without node search, from one first solve"),
    "node-probe": ("node_probe", "N4: probing an open node question (one-meter, aggregated and random orders)"),
    "node-roots": ("node_roots", "N5: root lines (chain and siblings) found by the node MIQP, local search, builders"),
    "node-probe-roots": ("node_probe_roots", "N6: probing an open root line, three settings, raised caps"),
    "node-price": ("node_price", "N2b: extended-BIC price for lines found at wide nodes (post hoc, saved rows)"),
    "node-report": ("node_report", "Rebuild all node MIQP tables into experiments/node_miqp/TABLES.md"),
}


def output_root():
    configured = os.environ.get("PTID_OUTPUTS")
    if configured:
        return Path(configured).resolve()
    # A linked working tree can share the original checkout's large input files.
    for parent in STUDY.parents:
        if (parent / "outputs" / "case_library_20261125").is_dir():
            return parent / "outputs"
    return STUDY.parent / "outputs"


def default_options(command, outputs):
    results = outputs / "topology_study"
    case_root = outputs / "case_library_20261125"
    if command == "random-restore":
        return ["--cases-root", case_root, "--groups", "all-random", "--report", results / "random" / "recovery.json"]
    if command == "random":
        return ["--cases-root", case_root, "--cache", outputs / "topology_study_random_dev" / "arms",
                "--out", results / "random" / "comparison.json"]
    if command == "random-budget":
        return ["--outputs", outputs, "--out", results / "random" / "time_limits.json"]
    if command in ("node-pairs", "pool-audit"):
        return ["--out", results / (command + ".json")]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Pass --help after an experiment name for its inputs and options. "
               "Each experiment runs from topology_study. Use an absolute path for an external destination.")
    subs = parser.add_subparsers(dest="experiment", required=True, metavar="EXPERIMENT")
    for name, (_, description) in EXPERIMENTS.items():
        subs.add_parser(name, help=description, add_help=False)
    args = parser.parse_args(sys.argv[1:2])
    options = sys.argv[2:]
    root = output_root()
    defaults = default_options(args.experiment, root)
    if defaults and not any(flag in options for flag in ("-h", "--help")):
        # The original runners accept file paths whose parent already exists.
        (root / "topology_study").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    import_paths = [str(STUDY), str(STUDY.parent)]
    if env.get("PYTHONPATH"):
        import_paths.append(env["PYTHONPATH"])
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(import_paths),
               PTID_OUTPUTS=str(root))
    module = "ptid.runs." + EXPERIMENTS[args.experiment][0]
    return subprocess.call([sys.executable, "-B", "-m", module, *map(str, defaults), *options],
                           cwd=STUDY, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
