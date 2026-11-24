"""Build strategies.tex from strategies_template.tex, copying the code listings from the source files.

    python build_strategies_tex.py

Every listing is located by function or method name (or by start and end marker lines for code
inside a function), so the listings and their line numbers always match the current code.
"""
from __future__ import annotations

import ast
import datetime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent

# (section title, [(file, qualified name) or (file, (start marker, end marker))])
LISTINGS = [
    ("GLS whitening, offsets and the hourly refit", [
        ("model.py", "root_gamma"), ("model.py", "whiten_slots"), ("model.py", "demean_rows"),
        ("model.py", "Fit.refit"), ("model.py", "Fit.covariance")]),
    ("Response model: design and operating-point sensitivity", [
        ("model.py", "Candidate.design"), ("model.py", "Candidate.sensitivity")]),
    ("Scores, equivalence and confirmation", [
        ("scores.py", "bic"), ("scores.py", "posterior_ok"),
        ("run.py", ("        N = y.size", "        rivals = sorted("))]),
    ("When a block may be bought; dispatch to the strategies; no-progress rule", [
        ("run.py", ("            elif probing and not ok", "                                status[\"stopped\"] = True"))]),
    ("Committed spend and battery state", [("run.py", "committed")]),
    ("Prediction rows and GLS in the planner", [("planner.py", "Gate.rows"), ("planner.py", "Gate.wh")]),
    ("Dictionary of shapes", [("planner.py", "Gate.shapes"), ("planner.py", "Gate.differing_nodes")]),
    ("Ceilings, budget and cost", [
        ("planner.py", "Gate.headroom_hour"), ("planner.py", "Gate.cexp"), ("planner.py", "Gate.level_cost"),
        ("planner.py", "Gate.level_for_cost"), ("planner.py", "Gate.unit_cost"), ("planner.py", "Gate.ceilings_hour"),
        ("planner.py", "Gate.window"), ("planner.py", "block_cost")]),
    ("Random block", [("planner.py", "random_block")]),
    ("Crude greedy and query by committee", [("planner.py", "Gate.plan_heuristic")]),
    ("v3 dictionary planner", [
        ("planner.py", "Gate.threshold"), ("planner.py", "Gate.c_tail"), ("planner.py", "Gate.gamma"),
        ("planner.py", "Gate.eta"), ("planner.py", "Gate.delta_mu"), ("planner.py", "Gate.q_full"),
        ("planner.py", "Gate.q_frozen_hour"), ("planner.py", "Gate.reach"), ("planner.py", "Gate.active_set"),
        ("planner.py", "Gate.plan")]),
    ("Certified planner", [("planner.py", "Gate.cert_coef"), ("planner.py", "Gate.plan_certified")]),
    ("Strong greedy", [("planner.py", "Gate.plan_bayes_greedy")]),
    ("Broad check", [("planner.py", "check_block")]),
]


def _span(path: Path, target):
    lines = path.read_text(encoding="utf-8").splitlines()
    if isinstance(target, tuple):
        start = next(i for i, l in enumerate(lines) if l.startswith(target[0]))
        end = next(i for i in range(start, len(lines)) if lines[i].startswith(target[1]))
        while end+1 < len(lines) and lines[end].rstrip().endswith((",", "(", "\\")):
            end += 1                                   # finish a statement that continues on the next line
        return start+1, end+1, lines[start:end+1]
    tree = ast.parse("\n".join(lines))
    node = tree
    for part in target.split("."):
        node = next(n for n in ast.iter_child_nodes(node)
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    first = min([node.lineno]+[d.lineno for d in node.decorator_list])
    return first, node.end_lineno, lines[first-1:node.end_lineno]


def _ascii(text: str, where: str) -> str:
    bad = sorted({ch for ch in text if ord(ch) > 126})
    if bad:
        print(f"warning: non-ASCII characters {bad} in {where}; replaced by '?'", file=sys.stderr)
        text = "".join(ch if ord(ch) <= 126 else "?" for ch in text)
    return text


def _tex(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")


def build():
    parts = []
    for title, items in LISTINGS:
        parts.append(f"\\subsection{{{_tex(title)}}}")
        for fname, target in items:
            first, last, body = _span(SRC/fname, target)
            label = target if isinstance(target, str) else "excerpt"
            code = _ascii("\n".join(body), f"{fname}:{first}")
            parts.append(f"\\begin{{lstlisting}}[firstnumber={first},title={{\\code{{{_tex(fname)}}}, "
                         f"\\code{{{_tex(label)}}}, lines {first}--{last}}}]\n{code}\n\\end{{lstlisting}}")
    template = (HERE/"strategies_template.tex").read_text(encoding="utf-8")
    out = template.replace("%%CODE%%", "\n\n".join(parts))
    out = out.replace("%%BUILD_DATE%%", datetime.date.today().isoformat())
    out = out.replace("%  GENERATED FILE: edit strategies_template.tex", "%  GENERATED FILE: do not edit; edit strategies_template.tex")
    left = [l for l in out.splitlines() if "%%" in l and not l.lstrip().startswith("%")]
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (HERE/"strategies.tex").write_text(out, encoding="utf-8")
    print(f"wrote {HERE/'strategies.tex'} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    build()
