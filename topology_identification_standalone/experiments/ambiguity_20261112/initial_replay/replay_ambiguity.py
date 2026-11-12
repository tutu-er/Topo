"""Replay conditional one/two-topology output from sealed scores, standard library only.

Reads JSON scores and source text only: no observed NPZ, truth, solver or refit.
The old primary and every old result file remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
KAPPAS = (0., .003, .01, .03, .1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gate(primary, alternative, kappa, tie):
    """Independent scalar implementation of the declared relative-MSE rule."""
    if (not all(math.isfinite(v) for v in (primary, alternative, kappa, tie))
            or min(primary, alternative, tie) < 0 or not 0 <= kappa < 1):
        raise ValueError("Invalid score or tolerance")
    gap = alternative-primary
    require(gap >= -tie, "Primary violates the declared MSE tie rule")
    return dict(ambiguous=gap <= max(tie, kappa*alternative), kappa=float(kappa),
                score_gap=float(gap), score_threshold=float(max(tie, kappa*alternative)),
                relative_mse_gap=float(max(0., gap)/alternative) if alternative else 0.,
                denominator="best_distinct_alternative_selection_mse")


def compare_helper(source):
    """Extract just the pure helper; importing competition/NumPy is unnecessary."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ambiguity_gate"]
    require(len(functions) == 1, "Expected one public ambiguity_gate helper")
    namespace = dict(np=SimpleNamespace(isfinite=math.isfinite))
    exec(compile(ast.Module(body=functions, type_ignores=[]), "isolated_ambiguity_gate", "exec"), namespace)
    return namespace["ambiguity_gate"]


def identity(point, labels, zero):
    active = sorted(sorted(labels[i] for i in C)
                    for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
                    if 1 < len(C) < len(labels) and max(r, x) > zero)
    return sha(json.dumps(dict(labels=sorted(labels), clades=active), sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


def choose(rows, n, tie):
    best = min(row["validation_scaled_mse"] for row in rows)
    return min((row for row in rows if row["validation_scaled_mse"] <= best+tie),
               key=lambda row: (sum(1 < len(C) < n for C in row["point"]["supports"]),
                                row["validation_scaled_mse"], row["index"]))


def replay(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    consumed = {}

    def load(path):
        data = path.read_bytes(); consumed[str(path)] = sha(data)
        return json.loads(data)

    protocol, seal, completion = [load(source/name) for name in ("protocol.json", "selection_seal.json", "completion.json")]
    require(seal["sealed"] and seal["all_planned_terminal"] and completion["all_planned_terminal"], "Unsealed or incomplete source")
    require(seal["protocol_sha256"] == consumed[str(source/"protocol.json")], "Protocol hash mismatch")
    require(completion["selection_seal_sha256"] == consumed[str(source/"selection_seal.json")], "Completion seal mismatch")
    require(protocol["split"] == dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), "Source is not the strict 96-point study")
    planned = {f"runs/{i}/{k}.json" for i in protocol["inputs"] for k in protocol["kappas"]}
    require(len(planned) == 12 and set(seal["result_sha256"]) == planned, "Unexpected source denominator")
    helper_path = BASE/"competition.py"; helper_source = helper_path.read_bytes()
    consumed[str(helper_path)] = sha(helper_source)
    helper = compare_helper(helper_source.decode("utf-8"))
    valid_probes = [(0., 0., 0., 1e-12), (0., 1., .999, 1e-12), (1., 1., 0., 1e-12),
                    (.875, 1., .125, 0.), (.874, 1., .125, 0.), (.876, 1., .125, 0.),
                    (1., 1.-5e-13, .003, 1e-12), (0., 5e-13, .003, 1e-12)]
    invalid_probes = [(-1., 1., .003, 1e-12), (2., 1., .003, 1e-12),
                      (1., 1., -.01, 1e-12), (1., 1., 1., 1e-12),
                      (1., 1., float("nan"), 1e-12), (1., float("inf"), .003, 1e-12),
                      (1., 1., .003, -1e-12)]
    for p, a, k, t in valid_probes:
        require(gate(p, a, k, t) == helper(p, a, kappa=k, tie=t), "Boundary probe differs from public helper")
    for p, a, k, t in invalid_probes:
        for function in (lambda: gate(p, a, k, t), lambda: helper(p, a, kappa=k, tie=t)):
            try:
                function()
            except ValueError:
                continue
            raise ValueError("Invalid-input probe was unexpectedly accepted")
    rows = []
    for name in sorted(planned):
        path = source/name; record = load(path)
        require(consumed[str(path)] == seal["result_sha256"][name], "A sealed source result changed")
        require(record["status"] == "fitted", "This replay requires the twelve completed fits")
        result = record["result"]; saved = result["competition"]; cfg = result["config"]
        reference, tie, zero = float(result["reference_mse"]), float(cfg["validation_tie"]), float(cfg["zero_tol"])
        require(math.isfinite(reference) and reference > 0, "Invalid training reference")
        bank = result["frozen_candidates"]; labels = bank[0]["labels"]; n = len(labels)
        require(all(row["index"] == i and row["labels"] == labels for i, row in enumerate(bank)), "Changed candidate order/labels")
        first = choose(bank, n, tie)
        require(first["index"] == saved["primary_index"] == result["selected"]["index"], "Primary does not follow the saved MSE rule")
        alternatives = [row for row in bank if identity(row["point"], labels, zero) != identity(first["point"], labels, zero)]
        require(bool(alternatives), "Source has no distinct alternative")
        second = choose(alternatives, n, tie)
        require(second["index"] == saved["secondary_index"], "Old secondary is not best distinct topology")
        models = saved["models"]
        require(len(models) == 2 and [m["index"] for m in models] == [first["index"], second["index"]], "Old comparison is not the declared pair")
        for model, candidate in zip(models, (first, second), strict=True):
            require(math.isclose(model["validation_mse"]/reference, candidate["validation_scaled_mse"], rel_tol=1e-10, abs_tol=1e-12), "Saved comparison MSE differs from the frozen bank")
        s1, s2 = [float(m["validation_mse"])/reference for m in models]
        previous = False
        for kappa in KAPPAS:
            decision = gate(s1, s2, kappa, tie)
            require(decision == helper(s1, s2, kappa=kappa, tie=tie), "Replay differs from current helper")
            require(not previous or decision["ambiguous"], "Increasing kappa removed a retained pair")
            previous = decision["ambiguous"]
            rows.append(dict(input=Path(name).parts[1], search_relative_gain=record["config"]["relative_gain"],
                ambiguity_kappa=kappa, primary_index=first["index"], best_distinct_index=second["index"],
                secondary_index=second["index"] if decision["ambiguous"] else None,
                returned_models=2 if decision["ambiguous"] else 1, primary_unchanged=True,
                primary_mse=models[0]["validation_mse"], alternative_mse=models[1]["validation_mse"],
                mse_gap=models[1]["validation_mse"]-models[0]["validation_mse"],
                normalized_mse_gap=decision["score_gap"], normalized_threshold=decision["score_threshold"],
                relative_mse_gap=decision["relative_mse_gap"]))
    script = Path(__file__).resolve(); consumed[str(script)] = sha(script.read_bytes())
    for path, checksum in consumed.items():
        require(sha(Path(path).read_bytes()) == checksum, "Read source changed during replay")
    stream = StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows); payload = stream.getvalue().encode("utf-8")
    summary = [dict(ambiguity_kappa=k, records=12, single=sum(r["returned_models"] == 1 for r in rows if r["ambiguity_kappa"] == k),
                    double=sum(r["returned_models"] == 2 for r in rows if r["ambiguity_kappa"] == k)) for k in KAPPAS]
    receipt = dict(verified=True, created_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
        sealed_results=12, replay_rows=len(rows), ambiguity_kappas=list(KAPPAS), summary=summary,
        actual_primary_unchanged=True, old_secondary_independently_verified_best_distinct=True,
        observations_read=0, truth_read=False, solver_calls=0, refits=0, source_files_modified=False,
        source_json_sha256={str(Path(p).relative_to(source)): s for p, s in consumed.items() if Path(p).is_relative_to(source)},
        replay_source_sha256=consumed[str(script)], competition_source_sha256=consumed[str(helper_path)],
        public_helper_check=dict(mode="AST-extracted pure helper; math.isfinite surrogate, no module imports",
                                 valid_boundary_probes=len(valid_probes), invalid_boundary_probes=len(invalid_probes), real_decisions=len(rows), passed=True),
        replay_csv_sha256=sha(payload), scope="post-hoc output-policy replay on frozen selection scores; no F1-based kappa choice; saved finite library only")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/"replay.csv").exists() and not (output/"receipt.json").exists(), "Existing replay evidence is preserved")
    (output/"replay.csv").write_bytes(payload)
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE/"outputs/strict96_20261112_002")
    parser.add_argument("--output", type=Path, default=BASE/"experiments/ambiguity_20261112")
    answer = replay(**vars(parser.parse_args()))
    print(json.dumps({k: answer[k] for k in ("verified", "sealed_results", "replay_rows", "summary", "public_helper_check")}, indent=2))
