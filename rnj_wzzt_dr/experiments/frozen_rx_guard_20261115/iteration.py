"""One predeclared response-budget sensitivity; keep the first run untouched."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

from ... import io
from . import run as base


def prepare(parent,out,diagnostic,independent_audit,preflight):
    parent,out=Path(parent).resolve(),Path(out).resolve()
    first=io.read(parent/'protocol.json')
    base.check_refs([first['inputs'],*first['sources']])
    bound=io.read(diagnostic)
    base.old.checked(bound['source_protocol'])
    if bound['source_protocol']!=io.ref(parent/'protocol.json'):
        raise ValueError('Budget bound has another parent')
    conflicts=[r for r in bound['records'] if r['response_rms_fraction']==.25 and r['worst_gate']['sufficient_conflict']]
    if not conflicts:
        raise ValueError('A predeclared planning-side conflict is required')
    audit=io.read(independent_audit)
    if audit.get('status')!='passed':
        raise ValueError('Independent budget-bound audit must pass')
    # This timing change is explicitly recorded before any parent new-day opens.
    parent_seals=list((parent/'cases').glob('*/seal.json'))
    if parent_seals:
        raise ValueError('This parallel-start amendment requires unopened parent execution')
    p=base.prepare(first['source_run'],first['source_audit']['path'],out,
        response_rms=.35,pq_rho=first['pq_rho'],seconds=first['solver_seconds'],
        parent_run=parent,modes=tuple(first['modes']),preflight_audit=preflight)
    entries=io.read(p['inputs']['path'])
    for path in (Path(__file__).resolve(),Path(__file__).with_name('ROUND2.md').resolve()):
        destination=out/'source_snapshot'/path.relative_to(base.old.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        if not any(ref['path']==str(path) for ref in p['sources']):p['sources'].append(io.ref(path))
    p.update(phase='frozen_rx_guard_development_iteration1',kinds=['response'],
        expected_quotes=len(entries)*len(p['modes']),expected_rows=len(entries)*len(p['modes'])*2*p['repeats'],
        parent_protocol=io.ref(parent/'protocol.json'),planning_bound_diagnostic=io.ref(diagnostic),
        independent_bound_audit=io.ref(independent_audit),iteration_protocol=io.ref(Path(__file__).with_name('ROUND2.md')),
        iteration_number=1,only_changed_model_parameter='response_rms .25 to .35 for every case and every base mode',
        trigger='Planning-only negative upper bounds; no new AC outcomes opened',
        parent_execution_seals_at_iteration_lock=[],
        parent_completed_quotes_at_iteration_lock=[io.ref(f) for f in sorted((parent/'cases').glob('*/*_plan.json'))],
        co_running_parent=True,combined_max_workers=6,
        runtime_comparison_scope='Same per-quote wall limit, shared hardware with concurrent parent; solver work and actual wall time retained')
    p.update(base.old._git_receipt([Path(r['path']) for r in p['sources']]))
    io.write(out/'protocol.json',p)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','out','diagnostic','independent-audit','preflight'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    prepare(a.parent,a.out,a.diagnostic,a.independent_audit,a.preflight)
    print(base.run(a.out))


if __name__=='__main__':main()
