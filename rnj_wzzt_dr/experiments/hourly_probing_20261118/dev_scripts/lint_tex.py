"""Heuristic LaTeX lint (no TeX installed): bare special characters, braces, environments, $ parity."""
import re
import sys
from pathlib import Path

t = Path(sys.argv[1]).read_text(encoding="utf-8")
body = t.split("\\begin{document}")[1]
# drop code listings entirely
body_nc = re.sub(r"(?s)\\begin\{lstlisting\}.*?\\end\{lstlisting\}", " ", body)
s = re.sub(r"(?s)\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}", " ", body_nc)
s = re.sub(r"(?s)\\\[.*?\\\]", " ", s)
s = re.sub(r"(?s)\$[^$]*\$", " ", s)
s = re.sub(r"\\(label|ref|eqref|cite)\{[^}]*\}", " ", s)
s = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("%"))
for l in s.splitlines():
    for ch in ("_", "^", "#"):
        for m in re.finditer(re.escape(ch), l):
            if m.start() == 0 or l[m.start()-1] != "\\":
                print("bare", ch, "|", l.strip()[:140])
    for m in re.finditer("%", l):
        if m.start() > 0 and l[m.start()-1] != "\\":
            print("comment % mid-line |", l.strip()[:140])
opens = len(re.findall(r"(?<!\\)\{", body_nc)); closes = len(re.findall(r"(?<!\\)\}", body_nc))
print("brace balance (outside listings):", opens - closes)
stack = []
for kind, name in re.findall(r"\\(begin|end)\{([a-zA-Z*]+)\}", body_nc):
    if kind == "begin":
        stack.append(name)
    else:
        top = stack.pop() if stack else None
        if top != name:
            print("environment mismatch:", top, name)
print("unclosed environments:", stack)
for para in re.split(r"\n\s*\n", body_nc):
    if (para.count("$") - para.count("\\$")) % 2:
        print("odd $ in paragraph:", para.strip()[:120])
print("left/right:", body_nc.count("\\left"), body_nc.count("\\right"))
print("Big( vs Big):", len(re.findall(r"\\Big\(", body_nc)), len(re.findall(r"\\Big\)", body_nc)))
print("big( vs big):", len(re.findall(r"\\big\(", body_nc)), len(re.findall(r"\\big\)", body_nc)))
print("big[ vs big]:", len(re.findall(r"\\big\[", body_nc)), len(re.findall(r"\\big\]", body_nc)))
# lstlisting bodies: check for the end marker inside code
for code in re.findall(r"(?s)\\begin\{lstlisting\}\[.*?\]\n(.*?)\\end\{lstlisting\}", body):
    if "\\end{lstlisting}" in code:
        print("listing contains end marker")
print("listings:", len(re.findall(r"\\begin\{lstlisting\}", body)))
