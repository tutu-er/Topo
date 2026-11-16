"""Check the normal output floor against sealed caches without new solves.

Current 56 libraries are exported in native frozen format; earlier 400 are
checked only. All fitting data, candidates and QP points are already recorded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from ... import io
from ...history import freeze_history
from ...identification.algorithm import SearchConfig, _output
from ...identification.competition import competition
from ...identification.optimization import QPFitter
from ...identification.output_floor import contract_bank
from ..impedance_floor_20261116 import sources


ROOT = Path(__file__).resolve().parents[3]


class CacheOnlyFitter(QPFitter):
    """A missing family is a replay failure, never permission to solve it."""
    def fit(self, family):
        from ...identification.trees import canonical
        key = canonical(family)
        if key not in self.cache:
            raise RuntimeError('Sealed cache lacks a requested family; new QP forbidden')
        return self.cache[key]


def parameters(points):
    return [{k: p[k] for k in ('supports', 'r', 'x')} for p in points]


def execute(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing attempt: '+str(out))
    completion = io.read(run/'complete.json')
    if completion['status'] != 'complete' or completion['successful_inputs'] != completion['planned_inputs'] or completion['failed_grid_cells']:
        raise ValueError('Source replay is incomplete')
    protocol = io.read(run/'protocol.json')
    entries = io.read(run/'inputs.json')
    if io.ref(run/'inputs.json')['sha256'] != protocol['inputs']['sha256']:
        raise ValueError('Source input index changed')
    out.mkdir(parents=True)
    refs = [io.ref(Path(__file__)), io.ref(run/'protocol.json'), io.ref(run/'complete.json'), io.ref(run/'inputs.json')]
    from ...identification import output_floor, algorithm
    from ... import history
    refs += [io.ref(Path(m.__file__)) for m in (output_floor, algorithm, history)]
    io.write(out/'protocol.json', dict(
        source_run=str(run), planned_inputs=len(entries), threshold=SearchConfig().internal_floor,
        metric='rx75', scope='non_singleton', new_QP_allowed=False, new_MIQP_allowed=False,
        new_AC_allowed=False, sources=refs, environment=io.environment(),
        source_document=io.ref(Path(__file__).with_name('PROTOCOL.md'))))
    began = perf_counter(); records = []
    for index, entry in enumerate(entries):
        directory = run/'inputs'/entry['id']; started = perf_counter()
        record = dict(input_id=entry['id'], source_group=entry['source_group'], case=entry['case'])
        try:
            seal = io.read(directory/'candidates_sealed.json')
            for key in ('banks','calibration','qp_records','qp_cache_points'):
                ref = seal[key]
                if io.ref(ref['path'])['sha256'] != ref['sha256']:
                    raise ValueError('Sealed artifact changed: '+key)
            banks = io.read(directory/'banks.json'); native = io.read(directory/'original_native.json')
            if io.ref(directory/'original_native.json')['sha256'] != banks['original_native']['sha256']:
                raise ValueError('Original native snapshot changed')
            data = sources.load_training(entry)
            cfg = SearchConfig(**dict(native['config'], internal_floor=SearchConfig().internal_floor))
            fitter = CacheOnlyFitter(data['train'], reference_mse=data['reference_mse'], bound=data['bound'])
            fitter.records = io.read(directory/'qp_records.json')
            fitter.cache = {tuple(tuple(c) for c in r['family']): r['point']
                            for r in io.read(directory/'qp_cache_points.json')}
            calibration = io.read(directory/'calibration.json')['calibration']
            original = [r['point'] for r in native['frozen_candidates']]
            points, receipt = contract_bank(original, fitter, cfg.internal_floor, calibration)
            expected = next(p for p in banks['grid'] if p['setting']['metric']=='rx75'
                and p['setting']['scope']=='non_singleton' and p['setting']['threshold']==cfg.internal_floor)
            if parameters(points) != parameters([r['point'] for r in expected['candidates']]):
                raise ArithmeticError('Normal output floor differs from the sealed grid replay')
            selected, mae, candidates = _output(points, data['train'], data['validation'], data['reference_mse'], cfg)
            if selected['index'] != expected['historical_scoring']['selection']['selected']:
                raise ArithmeticError('Normal strict selection differs from the sealed grid replay')
            applied = deepcopy(native)
            applied.update(config=vars(cfg), output_contraction=receipt,
                uncontracted_output_path=original, frozen_candidates=candidates,
                selected=selected, selected_mse_view=deepcopy(selected), selected_mae_view=mae)
            applied['handoff'] = dict(native.get('handoff', {}), schema='rnj_wzzt_dr_v2',
                parameter_rule='All original output candidates receive training-only non-terminal RX75 contraction')
            applied['competition'] = competition(applied, data['validation'])
            applied['output_replay'] = dict(source_native=io.ref(directory/'original_native.json'),
                source_cache=seal['qp_cache_points'], previous_search_reused=True, new_QP_solves=0,
                source_grid_setting=expected['setting'], software_receipt=io.ref(out/'protocol.json'))
            frozen = freeze_history(applied, data['validation'])
            record.update(status='passed', candidates=len(points), selected=frozen.current,
                parameter_sha256=frozen.parameter_sha256, new_QP_solves=0,
                source_bank=seal['banks'], original_internal_floor=native['config'].get('internal_floor',0.))
            if entry['source_group'] in ('history288','history10'):
                dest = out/'inputs'/entry['id']/'native.json'; io.write(dest,applied)
                record['native']=io.ref(dest)
        except Exception as exc:
            record.update(status='failed', error=type(exc).__name__+': '+str(exc))
        record['seconds']=perf_counter()-started; records.append(record)
        if (index+1)%50==0:
            print(dict(processed=index+1, failed=sum(r['status']!='passed' for r in records)),flush=True)
    result = dict(status='passed' if all(r['status']=='passed' for r in records) else 'failed',
        planned_inputs=len(entries), checked_inputs=len(records),
        failed_inputs=sum(r['status']!='passed' for r in records),
        exported_native_libraries=sum('native' in r for r in records), new_QP_solves=0,
        seconds=perf_counter()-began, records=records, protocol=io.ref(out/'protocol.json'))
    result['source_unchanged'] = all(io.ref(ref['path'])['sha256']==ref['sha256'] for ref in refs)
    if not result['source_unchanged']:
        result['status'] = 'failed'
    io.write(out/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/impedance_floor_20261116_v1')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=execute(args.run,args.out)
    print({k:v for k,v in result.items() if k!='records'})
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
