"""Compact v3 evidence with explicit omissions and lossless final fit evidence.

The original raw files remain authoritative.  This format deduplicates realized
evaluations and removes only named planning caches/repeated predictive fits.
``restore`` reconstructs an auditor/summary-readable record, not the full raw.
No parameter fitting, AC optimization, or action selection is performed here.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = 'mobile_v3_compact_evidence_v1'


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _hash(value):
    return sha256(_bytes(value)).hexdigest()


def _read(path):
    path = Path(path); payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def _write(path, value):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _bytes(value) + b'\n'
    path.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)


def _failure_nodes(value, path):
    """Keep complete failed nodes before omitting their surrounding cache.

    The first explicit failed parent is retained in full, including its nested
    states/restarts/errors.  Descendants need not be duplicated a second time.
    Paths are field occurrences, not independent statistical failure events.
    """
    if isinstance(value, dict):
        failed = (value.get('success') is False or value.get('solver_success') is False
                  or value.get('optimization_failure') is True or bool(value.get('optimizer_error'))
                  or bool(value.get('failure')) or bool(value.get('failures')))
        if failed:
            return [dict(path=path, complete_failed_node=deepcopy(value))]
        result = []
        for key, item in value.items(): result.extend(_failure_nodes(item, path + '/' + key))
        return result
    if isinstance(value, list):
        result = []
        for i, item in enumerate(value): result.extend(_failure_nodes(item, path + '/' + str(i)))
        return result
    return []


def compact_record(record, *, raw_sha256=None, raw_path=None, raw_bytes=None):
    if not isinstance(record, dict): raise TypeError('record must be a dictionary')
    omissions, failures = [], []

    def omit(value, path, reason):
        payload = _bytes(value)
        omissions.append(dict(path=path, reason=reason, canonical_bytes=len(payload),
                              content_sha256=sha256(payload).hexdigest()))
        failures.extend(_failure_nodes(value, path))

    def planning(value, path='planning', cache=False):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                field_path = path + '/' + key
                if key == 'references':
                    omit(item, field_path, 'repeated complete predictive reference-fit states')
                elif cache and key in ('background', 'tables'):
                    omit(item, field_path, 'scalar planning table/reference cache; summary and failures retained')
                else:
                    result[key] = planning(item, field_path,
                        cache=(key == 'table' and isinstance(item, dict) and 'tables' in item))
            return result
        if isinstance(value, list):
            return [planning(v, path + '/' + str(i), cache=False) for i,v in enumerate(value)]
        return value

    body = {k:v for k,v in record.items() if k not in ('methods', 'planning')}
    if 'planning' in record: body['planning'] = planning(record['planning'])
    actions, methods = {}, {}
    raw_methods = record.get('methods', {})
    if not isinstance(raw_methods, dict): raise ValueError('v3 methods must be a dictionary')
    for name, method in raw_methods.items():
        if not isinstance(method, dict): raise ValueError('method must be a dictionary')
        retained = {k:v for k,v in method.items() if k != 'evaluation'}
        if 'evaluation' in method:
            key = method.get('action_key')
            if not isinstance(key, str) or not key: raise ValueError('evaluation requires a saved action_key')
            evaluation = method['evaluation']
            if key in actions and _hash(actions[key]) != _hash(evaluation):
                raise ValueError('same action_key has different saved evaluations')
            actions.setdefault(key, evaluation)
            retained['_evidence_evaluation_key'] = key
        methods[name] = retained
    if 'methods' in record: body['methods'] = methods
    result = dict(schema=SCHEMA, record=body, actions=actions, exclusions=omissions,
        failure_evidence=failures,
        source_record=dict(raw_sha256=raw_sha256, raw_path=raw_path, raw_compressed_bytes=raw_bytes,
                           raw_object_sha256=_hash(record)),
        scope='compact retained final evidence; selected planning caches omitted; not the complete raw record')
    result['payload_sha256'] = _hash(result)
    return result


normalize_record = compact_record


def restore(compact):
    if not isinstance(compact, dict) or compact.get('schema') != SCHEMA:
        raise ValueError('unsupported compact evidence schema')
    if compact.get('payload_sha256') != _hash({k:v for k,v in compact.items() if k != 'payload_sha256'}):
        raise ValueError('compact payload digest mismatch')
    result = deepcopy(compact['record'])
    used = set()
    for method in result.get('methods', {}).values():
        key = method.pop('_evidence_evaluation_key', None)
        if key is not None:
            if key not in compact['actions']: raise ValueError('referenced action evaluation missing')
            method['evaluation'] = compact['actions'][key]
            used.add(key)
    if used != set(compact['actions']): raise ValueError('unreferenced action evaluation')
    result['_evidence_provenance'] = dict(schema=SCHEMA, source_record=compact['source_record'],
        exclusions=compact['exclusions'], failure_evidence=compact['failure_evidence'],
        scope=compact['scope'])
    return result


def validate_record(record, compact, *, physical_replay=True, candidate_count=4):
    from . import ev_mobile_v3_summary as summary
    restored = restore(compact)
    comparisons = {}
    for name in summary.METHODS:
        a = summary.project_method(record, name, candidate_count=candidate_count)
        b = summary.project_method(restored, name, candidate_count=candidate_count)
        comparisons[name] = dict(equal=a == b, original_sha256=_hash(a), restored_sha256=_hash(b))
    result = dict(summary_methods=comparisons, summary_equal=all(r['equal'] for r in comparisons.values()),
                  restored_is_complete_raw=False)
    if physical_replay:
        from . import ev_mobile_v3_physics_audit as physics
        original, replay = physics.audit_record(record), physics.audit_record(restored)
        keys = ('success', 'complete_physical_coverage', 'expected', 'metrics', 'errors',
                'missing', 'states', 'methods', 'recorded_failures')
        result.update(physical_replay_equal=all(original.get(k) == replay.get(k) for k in keys),
                      original_physical_audit_success=original['success'],
                      restored_physical_audit=replay)
    result['success'] = result['summary_equal'] and result.get('physical_replay_equal', True)
    return result


def _phase_files(directory, frozen, phase):
    specs = frozen['specifications'][phase]
    expected = {str(s['index']).zfill(3) + '.json.gz':s for s in specs}
    files = {p.name:p for p in (directory / phase).glob('*.json.gz')}
    if set(files) != set(expected):
        raise ValueError('phase not complete: ' + phase + '; missing=' + str(sorted(set(expected)-set(files)))
                         + '; extra=' + str(sorted(set(files)-set(expected))))
    return [(files[name], expected[name]) for name in sorted(expected)]


def validate_directory(directory, output, *, phase='development', physical_replay=True):
    directory, output = Path(directory), Path(output)
    if output.exists(): raise FileExistsError(output)
    frozen = _read(directory/'freeze.json'); rows=[]
    for file, spec in _phase_files(directory, frozen, phase):
        payload = file.read_bytes(); record = json.loads(gzip.decompress(payload))
        if record.get('policy_id') != frozen['policy_id'] or record.get('spec') != spec:
            raise ValueError('case identity differs from frozen policy/specification')
        compact = compact_record(record, raw_sha256=sha256(payload).hexdigest(),
                                 raw_path=str(file.resolve()), raw_bytes=len(payload))
        check = validate_record(record, compact, physical_replay=physical_replay,
                                candidate_count=frozen['config']['candidate_count'])
        check.update(spec=spec, raw_sha256=sha256(payload).hexdigest(), raw_gzip_bytes=len(payload),
            compact_gzip_bytes=len(gzip.compress(_bytes(compact),mtime=0)),
            exclusions=compact['exclusions'], failure_nodes_preserved=len(compact['failure_evidence']))
        rows.append(check); print('ROUNDTRIP', file.name, check['success'], flush=True)
    result = dict(success=all(r['success'] for r in rows), phase=phase, policy_id=frozen['policy_id'],
        case_count=len(rows), cases=rows, raw_gzip_bytes=sum(r['raw_gzip_bytes'] for r in rows),
        compact_gzip_bytes=sum(r['compact_gzip_bytes'] for r in rows),
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='summary equality and independent saved-state replay; no claim that excluded planning caches are recoverable')
    _write(output,result); return result


def _check_snapshot(directory, frozen):
    snapshot = _read(directory/'source_snapshot.json.gz')
    if snapshot.get('source_sha256') != frozen['source_sha256']:
        raise ValueError('source snapshot hash map differs from freeze')
    content = snapshot['source_base64']
    if set(content) != set(frozen['source_sha256']): raise ValueError('source snapshot coverage differs')
    for name, expected in frozen['source_sha256'].items():
        if sha256(base64.b64decode(content[name],validate=True)).hexdigest() != expected:
            raise ValueError('source snapshot bytes/hash differ: ' + name)


def pack(directory, output, *, phases=('development','evaluation'), attachments=()):
    """Require complete requested frozen phases; never launch a missing case."""
    directory, output = Path(directory), Path(output)
    if output.exists(): raise FileExistsError(output)
    frozen = _read(directory/'freeze.json')
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= {'development','evaluation'}:
        raise ValueError('distinct requested phases required')
    selected = [(phase,file,spec) for phase in phases for file,spec in _phase_files(directory,frozen,phase)]
    _check_snapshot(directory,frozen)
    output.mkdir(parents=True)
    status=output/'pack_status.json'
    _write(status,dict(status='writing',source_directory=str(directory.resolve())))
    copied, records = [], []
    try:
        def copy(file, target):
            payload = file.read_bytes(); target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise FileExistsError(target)
            target.write_bytes(payload)
            copied.append(dict(source=str(file.resolve()),file=str(target.relative_to(output)),
                               bytes=len(payload),sha256=sha256(payload).hexdigest()))
        for file in sorted(directory.iterdir()):
            if file.is_file() and (file.name in ('freeze.json','source_snapshot.json.gz')
                    or file.suffix in ('.json','.md','.log','.exitcode')):
                copy(file,output/file.name)
        if not (output/'environment.json').exists(): _write(output/'environment.json',frozen.get('environment',{}))
        for path in attachments:
            file=Path(path);copy(file,output/'attachments'/file.name)
        for phase,file,spec in selected:
            payload=file.read_bytes();record=json.loads(gzip.decompress(payload))
            if record.get('spec') != spec or record.get('policy_id') != frozen['policy_id']:
                raise ValueError('record identity differs: '+str(file))
            compact=compact_record(record,raw_sha256=sha256(payload).hexdigest(),raw_path=str(file.resolve()),raw_bytes=len(payload))
            check=validate_record(record,compact,physical_replay=False,candidate_count=frozen['config']['candidate_count'])
            if not check['success']:raise ValueError('summary roundtrip differs: '+str(file))
            target=output/phase/file.name;_write(target,compact)
            records.append(dict(phase=phase,spec=spec,file=str(target.relative_to(output)),
                raw_sha256=sha256(payload).hexdigest(),raw_gzip_bytes=len(payload),
                compact_sha256=sha256(target.read_bytes()).hexdigest(),compact_gzip_bytes=target.stat().st_size,
                excluded_fields=compact['exclusions'],failure_nodes_preserved=len(compact['failure_evidence']),
                all_summary_methods_equal=True))
        manifest=dict(status='complete',schema=SCHEMA,policy_id=frozen['policy_id'],phases=list(phases),
            case_count=len(records),records=records,context_files=copied,
            raw_gzip_bytes=sum(r['raw_gzip_bytes'] for r in records),
            compact_gzip_bytes=sum(r['compact_gzip_bytes'] for r in records),
            source_snapshot_all_bytes_verified=True,tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='compact final physical/statistical evidence; original raw retained at source paths; exclusions explicit')
        _write(output/'manifest.json',manifest)
        status.write_bytes(_bytes(dict(status='complete',manifest_sha256=sha256((output/'manifest.json').read_bytes()).hexdigest()))+b'\n')
        return manifest
    except Exception as exc:
        status.write_bytes(_bytes(dict(status='failed',error=type(exc).__name__+': '+str(exc)))+b'\n')
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--validate-phase',choices=('development','evaluation'))
    parser.add_argument('--phase',choices=('development','evaluation'),action='append')
    parser.add_argument('--attachment',action='append',default=[])
    args=parser.parse_args()
    result=(validate_directory(args.directory,args.output,phase=args.validate_phase)
        if args.validate_phase else pack(args.directory,args.output,phases=args.phase or ('development','evaluation'),
                                         attachments=args.attachment))
    print(json.dumps({k:result[k] for k in ('success','status','case_count','raw_gzip_bytes','compact_gzip_bytes') if k in result}))


if __name__ == '__main__':main()
