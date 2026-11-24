"""Count the cells of every tabular row against its column specification; flag stray single backslashes."""
import re
import sys

t = open(sys.argv[1], encoding="utf-8").read()
nolst = re.sub(r"(?s)\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "LISTING", t)


def ncols(spec):
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r">\{[^}]*\}", "", spec)
    spec = re.sub(r"<\{[^}]*\}", "", spec)
    spec = re.sub(r"p\{[^}]*\}|m\{[^}]*\}|b\{[^}]*\}", "p", spec)
    return len(re.findall(r"[lcrXp]", spec))


bad = 0
for m in re.finditer(r"\\begin\{(tabularx?)\}(\{[^}]*\})?\{(.*?)\}\n(.*?)\\end\{\1\}", nolst, re.S):
    env, spec, body = m.group(1), m.group(3), m.group(4)
    if env == "tabularx":
        spec = re.match(r"(.*)", spec).group(1)
    n = ncols(spec)
    rows = [r for r in re.split(r"\\\\", body)]
    for r in rows:
        r2 = re.sub(r"\\(toprule|midrule|bottomrule)", "", r).strip()
        if not r2:
            continue
        cells = r2.count("&") - r2.count("\\&") + 1
        if cells != n:
            bad += 1
            print(f"{env} with {n} columns: row has {cells} cells: {r2[:100]!r}")
print("tables checked; bad rows:", bad)
for i, l in enumerate(nolst.splitlines(), 1):
    s = l.rstrip()
    if s.endswith("\\") and not s.endswith("\\\\"):
        print(f"line {i} ends with a single backslash: {s[-60:]!r}")
