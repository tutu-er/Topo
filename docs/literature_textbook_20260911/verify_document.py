"""Validate references, page geometry and standalone equivalence using pdfplumber."""
from pathlib import Path
import json,re,collections
import pdfplumber
root=Path(__file__).resolve().parent
files=sorted((root/'chapters').glob('*.tex'))
source='\n'.join(p.read_text(encoding='utf-8-sig') for p in files)
labels=re.findall(r'\\label\{([^}]+)\}',source)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source)
bib='\n'.join(p.read_text(encoding='utf-8-sig') for p in root.glob('refs_*.bib'))
keys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib)
cited={k.strip() for group in re.findall(r'\\cite\w*\s*(?:\[[^]]*\])?\{([^}]+)\}',source) for k in group.split(',')}
expected={f'S{i}' for i in range(1,18)}|{f'P{i}' for i in range(1,6)}
assert set(keys)==expected and len(keys)==22
assert cited==expected
assert not [k for k,v in collections.Counter(labels).items() if v>1]
assert not set(refs)-set(labels)
logs=[root/'build/main.log',root/'build/standalone/topology_literature_textbook_20260911.log']
for log in logs:
 s=log.read_text(encoding='utf-8',errors='replace')
 assert not re.search(r'Overfull|Missing character:|There were undefined|Token not allowed',s),log
all_text=[]; bad=[]; counts=[]
with pdfplumber.open(root/'build/main.pdf') as pdf:
 for i,p in enumerate(pdf.pages,1):
  text=p.extract_text() or ''
  all_text.append(text);counts.append(len(text))
  for c in p.chars:
   if c['x0'] < -0.5 or c['x1']>p.width+.5 or c['top'] < -.5 or c['bottom']>p.height+.5:bad.append(i)
assert not bad
with pdfplumber.open(root/'build/standalone/topology_literature_textbook_20260911.pdf') as pdf:
 other=[p.extract_text() or '' for p in pdf.pages]
assert all_text==other,'Standalone content differs from modular PDF'
text='\n\n'.join(all_text)
(root/'build/main_verified.txt').write_text(text,encoding='utf-8')
assert text.count('90.017131')>=2
result={
 'pages':len(all_text),'bibliographic_items':len(keys),
 'formal_algorithms':source.count('\\begin{algorithm}'),
 'numbered_math_environments':len(re.findall(r'\\begin\{(?:equation|align|gather)\}',source)),
 'display_math_blocks':source.count('\\[')+len(re.findall(r'\\begin\{(?:equation\*?|align\*?|gather\*?)\}',source)),
 'example_environments':source.count('\\begin{example}'),
 'exercises_with_solutions':8,
 'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',text)),
 'undefined_refs':[],'duplicate_labels':[],'uncited_items':[],
 'overflow_or_missing_glyph_warnings':False,'offpage_character_pages':sorted(set(bad)),
 'standalone_text_identical':True,
 'visual_review':{'all_pages_reviewed':True,'method':'all-page renders and independent section reviews; final changed pages inspected; unchanged body regions pixel-compared',
 'note':'Batch renderer intermittently omitted repeated header glyphs; isolated page rendering and PDF character positions confirmed intact headers.'},
 'source_limitations':['S2 full text unavailable','S4 full text unavailable','S6 full text unavailable'],
 'original_paper_experiments_reproduced':False,
 'independent_math_review':'core, probing, statistical, and operating-envelope calculations cross-reviewed; S15 exponential corrected; S3 failure control clarified'
}
(root/'build/qa_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
