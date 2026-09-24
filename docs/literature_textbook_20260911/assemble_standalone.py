"""Embed chapters and BibTeX output into one portable XeLaTeX source."""
from pathlib import Path
import re

root = Path(__file__).resolve().parent
main = (root / 'main.tex').read_text(encoding='utf-8-sig')

def expand(match):
    path = root / (match.group(1) + '.tex')
    return '\n% BEGIN ' + path.name + '\n' + path.read_text(encoding='utf-8-sig') + '\n% END ' + path.name + '\n'

flat = re.sub(r'\\input\{([^}]+)\}', expand, main)
flat = re.sub(r'\\bibliographystyle\{[^}]+\}', '', flat)
bbl = (root / 'build' / 'main.bbl').read_text(encoding='utf-8')
flat = re.sub(r'\\bibliography\{[^}]+\}', lambda m: bbl, flat)
flat = flat.replace('% Compile from this directory: latexmk -xelatex -outdir=build main.tex', '% Standalone: repeat XeLaTeX until references stabilize; bibliography and all chapters are embedded.')
if '\\input{' in flat or '\\bibliography{' in flat:
    raise RuntimeError('External LaTeX input remains')
path = root / 'topology_literature_textbook_20260911.tex'
path.write_text(flat, encoding='utf-8')
print(path)
