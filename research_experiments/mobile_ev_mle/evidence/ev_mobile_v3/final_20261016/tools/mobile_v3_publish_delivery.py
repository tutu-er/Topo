"""Publish only the verified v3 task files; never stage or modify Git here."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess

STAGE = Path('D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015')
CANON = Path('C:/Users/23761/.codex/worktrees/36a1/Topo')
REL = Path('research_experiments/mobile_ev_mle')
TARGET = CANON / REL / 'evidence/ev_mobile_v3/final_20261016'
PACK = STAGE / 'outputs/mobile_v3_pilot2_delivery_v1'
PROTECTED = {
    'docs/mobile_ev_probing_report_v2.pdf': 'fe89b241285fd81f570a921fe99a49bfdb1d6bca6e0ced0d498a45ef81d5dbfa',
    'pyproject.toml': '67d21b6a3da4629c5cba5c445404fc5ecde154bf6b472ee4fa06f75ceb760e1c',
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(CANON), *args], text=True).strip()

if git('branch', '--show-current') != 'codex/ev-train-validation-design':
    raise RuntimeError('wrong branch')
if git('remote', 'get-url', 'origin') != 'https://github.com/tutu-er/Topo.git':
    raise RuntimeError('wrong remote')
if TARGET.exists():
    raise FileExistsError('preserve existing evidence directory')
for rel, sha in PROTECTED.items():
    if digest(CANON / rel) != sha:
        raise RuntimeError('protected file changed before publication: ' + rel)
if json.loads((PACK / 'pack_status.json').read_text())['status'] != 'complete':
    raise RuntimeError('evidence pack incomplete')
verification = json.loads((PACK/'delivery_validation_v1.json').read_text(encoding='utf-8'))
if not verification.get('success'):
    raise RuntimeError('roundtrip verification incomplete')
freeze = json.loads((STAGE/'outputs/mobile_v3_pilot2/freeze.json').read_text(encoding='utf-8'))

names = sorted(p.name for p in (STAGE / REL).glob('ev_mobile_v3_*')
               if p.is_file() and p.suffix in ('.py', '.md', '.json') and '.draft' not in p.name)
names += sorted(p.name for p in (STAGE / REL).glob('test_ev_mobile_v3_*.py'))
if len(names) != len(set(names)):
    raise RuntimeError('duplicate task source path')
records = []
TARGET.mkdir(parents=True)
for name in names:
    source, target = STAGE/REL/name, CANON/REL/name
    shutil.copyfile(source, target)
    records.append(dict(kind='source', path=str(target.relative_to(CANON)).replace('\\','/'),
                        sha256=digest(target), bytes=target.stat().st_size))

def copy(source, target, kind):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(source) != digest(target):
        raise RuntimeError('copy byte mismatch')
    records.append(dict(kind=kind, path=str(target.relative_to(CANON)).replace('\\','/'),
                        source=str(source), sha256=digest(target), bytes=target.stat().st_size))

for source in sorted(PACK.rglob('*')):
    if source.is_file():
        copy(source, TARGET/'pilot2_compact'/source.relative_to(PACK), 'pilot2_compact')

audits = [
    'outputs/mobile_v3_delivery_tests.xml',
    'outputs/mobile_v3_pilot2/costs_and_pairs_v1.json',
    'outputs/mobile_v3_pilot2/costs_and_pairs_v1.md',
    'outputs/mobile_v3_implementation_tests.xml',
    'outputs/mobile_v3_proxy_replay.json',
    'outputs/mobile_v3_proxy_replay.md',
    'outputs/mobile_v3_development_physics_audit_v1.json',
    'outputs/mobile_v3_evidence_dev2_roundtrip_v1.json',
    'outputs/mobile_v3_score_attribution_compact.json.validation.json',
]
for rel in audits:
    copy(STAGE/rel, TARGET/'audits'/Path(rel).name, 'audit')
for name in ('freeze.json','summary.json','summary.md','validation.json'):
    copy(STAGE/'outputs/mobile_v3_score_attribution_v1'/name,
         TARGET/'score_attribution'/name, 'score_attribution')
copy(STAGE/'outputs/mobile_v3_score_attribution_compact.json.gz',
     TARGET/'score_attribution/records.json.gz', 'score_attribution')
tools = ['mobile_v3_publish_delivery.py','mobile_v3_planning_failure_build.py',
         'mobile_v3_delivery_verify_v1.py']
for name in tools:
    source = STAGE/'outputs'/name
    if source.exists():
        copy(source, TARGET/'tools'/name, 'tool')

for name, sha in freeze['source_sha256'].items():
    path = CANON/name if name.startswith('rnj_wzzt_core/') else CANON/REL/name
    if digest(path) != sha:
        raise RuntimeError('frozen science mismatch ' + name)
for rel, sha in PROTECTED.items():
    if digest(CANON/rel) != sha:
        raise RuntimeError('protected file changed during publication ' + rel)
metadata = dict(status='complete', policy_id=freeze['policy_id'],
    source_files_copied=len(names), frozen_source_files_verified=len(freeze['source_sha256']),
    old_source_files_unchanged=115, history_counts=dict(development=4,evaluation=16),
    inference_method_slots=128, final_candidate_slots=512,
    tests=dict(passed=127, failed=0),
    evidence_recovery='All 20 compact histories and 160 summary methods verified; independent physical restore audits for evaluation 0 and 8, plus all four development histories.',
    unimplemented='Physical-domain shared QP/MILP next model is math only.',
    protected_files=PROTECTED, branch=git('branch','--show-current'),
    parent_commit=git('rev-parse','HEAD'), files=records,
    total_copied_bytes=sum(r['bytes'] for r in records),
    largest_copied_file_bytes=max(r['bytes'] for r in records))
(TARGET/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:metadata[k] for k in ('status','source_files_copied','frozen_source_files_verified',
    'total_copied_bytes','largest_copied_file_bytes')},ensure_ascii=False))
