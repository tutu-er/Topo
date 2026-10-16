"""Frozen postprocessing reliability comparisons, with no fitting or planning.

Every rule uses its own alpha. These are separate procedures, not simultaneous
coverage claims. Structural complexity is read from actual edge supports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from . import ev_systematic_analyze as base


RULES = ('global_R', 'H_R', 'label_R', 'global_AIC', 'label_AIC')
H_GROUPS = (8, 32, 96)


def declared_methods(phase):
    values=phase.get('methods',phase.get('manifest',{}).get('config',{}).get('methods'))
    if values is None:
        # In-memory fixtures and version-one evidence have the explicit rows.
        values=[row['name'] for row in phase['records'][0]['methods']]
    if not isinstance(values,(list,tuple)) or not values or any(not isinstance(v,str) or not v for v in values) or len(set(values))!=len(values):
        raise ValueError('explicit unique method list required')
    for record in phase['records']:
        actual=[row.get('name') for row in record['methods']]
        if len(actual)!=len(values) or set(actual)!=set(values):
            raise ValueError('record must retain exactly the declared methods')
        for method in record['methods']:
            if set(method.get('inference',{}))!=set(base.INFERENCE):
                raise ValueError('record must retain all three inference modes')
    return tuple(values)


def read_phase(directory, *, expected_phase=None):
    """Read the methods declared in freeze.config, without the v1 ten-policy gate."""
    directory=Path(directory);metadata={};expected_phase=base._phase(expected_phase)
    for name in ('freeze.json','manifest.json'):
        path=directory/name
        if path.exists():
            value=base._read(path)
            for key,row in value.items():
                if key in metadata and metadata[key]!=row and key in ('source_id','config_id','policy_id','source_sha256'):
                    raise ValueError('freeze/manifest identity mismatch')
                metadata[key]=row
    if not metadata.get('config',{}).get('methods'):
        raise ValueError('freeze.config.methods declaration required')
    paths=sorted({p for pattern in ('case_*.json','case_*.json.gz','trial_*.json','trial_*.json.gz') for p in directory.glob(pattern)})
    records=[];files=[];seen=set()
    for path in paths:
        raw=base._read(path)
        if not isinstance(raw,dict) or 'methods' not in raw:continue
        required=('case_id','seed','phase','true_name','candidate_miss','status','methods')
        if any(key not in raw for key in required):raise ValueError('case record missing required schema fields')
        if expected_phase is not None and base._phase(raw['phase'])!=expected_phase:
            raise ValueError('record phase differs from requested phase')
        if raw.get('source_unchanged') is False:raise ValueError('record reports source drift')
        record=base.project_analysis_record(raw)
        key=(record['case_id'],record['seed'])
        if key in seen:raise ValueError('duplicate case/history record')
        seen.add(key);records.append(record)
        files.append(dict(path=str(path.resolve()),sha256=sha256(path.read_bytes()).hexdigest()))
    if not records:raise ValueError('no case/trial records found')
    phases={base._phase(r['phase']) for r in records}
    if len(phases)!=1:raise ValueError('mixed phases in one directory')
    phase=dict(directory=str(directory.resolve()),phase=next(iter(phases)),manifest=metadata,
               identity=base._stable_identity(metadata,records),records=records,files=files)
    phase['methods']=list(declared_methods(phase))
    return phase


def candidate_parameters(record):
    """Return independently adjustable R/X counts and exact support signatures.

    Names are opaque identifiers; the edge count is never guessed from a name.
    Missing or malformed structural metadata yields an explicit error and the
    deployed scoring procedure keeps every candidate.
    """
    bank = base._bank(record)
    rows = record.get('candidates')
    if not isinstance(rows, list) or len(rows) != len(bank):
        return None, 'missing candidate support metadata'
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('name') not in bank:
            return None, 'candidate support metadata differs from bank'
        name = row['name']; supports = row.get('supports')
        if name in result or not isinstance(supports, list) or not supports:
            return None, 'missing or duplicate candidate supports'
        normalized = []
        for support in supports:
            if not isinstance(support, (list, tuple)) or not support or any(
                not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in support
            ) or len(set(support)) != len(support):
                return None, 'malformed edge support'
            normalized.append(tuple(sorted(support)))
        if len(set(normalized)) != len(normalized):
            return None, 'duplicate edge support'
        terminals=set().union(*(set(s) for s in normalized));n=max(terminals)+1
        if terminals!=set(range(n)) or tuple(range(n)) not in normalized or any((i,) not in normalized for i in range(n)):
            return None,'edge supports do not declare a complete terminal bank'
        ordered=sorted(normalized)
        key=json.dumps(dict(n=n,supports=ordered),sort_keys=True,separators=(',',':'))
        result[name] = dict(n=n,edge_count=len(normalized), line_parameters=2*len(normalized),
                            supports=ordered, structural_key=key)
    if set(result) != set(bank):
        return None, 'candidate support metadata differs from bank'
    if len({v['n'] for v in result.values()})!=1 or len({v['structural_key'] for v in result.values()})!=len(bank):
        return None,'candidate bank has incompatible terminals or duplicate structures'
    return result, None


def group_key(record, rule, candidate):
    if rule == 'H_R':
        value = record.get('config', {}).get('training_points')
        return str(value) if value in H_GROUPS and not isinstance(value, bool) else 'unknown'
    if rule.startswith('label_'):
        parameters,_=candidate_parameters(record)
        return parameters[candidate]['structural_key'] if parameters is not None and candidate in parameters else 'unknown:'+str(candidate)
    return 'all'


def scored_mode(record, method, inference, rule):
    if rule not in RULES:
        raise ValueError('unknown reliability rule')
    value = base._mode(record, method, inference)
    parameters, error = candidate_parameters(record)
    complete = bool(value['complete'] and parameters is not None)
    gamma = 2. if rule.endswith('AIC') else 0.
    scores = ({k: s + gamma*parameters[k]['line_parameters']
               for k, s in value['losses'].items()} if complete else {})
    minimum = min(scores.values()) if scores else None
    relative = {k: max(0., s-minimum) for k, s in scores.items()}
    return dict(complete=complete, candidate_miss=value['candidate_miss'],
                losses=value['losses'], scores=scores, relative=relative,
                bank=value['bank'], parameters=parameters, structural_error=error,
                gamma=gamma)


def _threshold(scores, alpha):
    if not scores:
        return dict(n=0, alpha=float(alpha), order_rank=1, threshold=None, infinite=True)
    return base.conformal_threshold(scores, alpha)


def _signatures(records):
    signatures = {};bank_keys=None
    for record in records:
        values, _ = candidate_parameters(record)
        if values is None:
            continue
        keys={row['structural_key'] for row in values.values()}
        if bank_keys is not None and keys!=bank_keys:
            raise ValueError('formal phase mixes structural candidate banks')
        bank_keys=keys
        for name, row in values.items():
            normalized = dict(n=row['n'],supports=[list(s) for s in row['supports']])
            signatures[row['structural_key']] = normalized
    return signatures


def calibrate(phase, output, *, alpha=.1, expected_count=None):
    records = phase['records']
    if phase['phase'] != 'calibration':
        raise ValueError('calibration phase required')
    if expected_count is not None and len(records) != expected_count:
        raise ValueError('calibration count differs from declared batch')
    if len({r['seed'] for r in records}) != len(records):
        raise ValueError('calibration requires independent history seeds')
    if any(r.get('candidate_miss') or r['true_name'] not in base._bank(r) for r in records):
        raise ValueError('formal calibration prohibits candidate miss')
    if not 0 < alpha < 1:
        raise ValueError('0<alpha<1 required')
    identity=phase['identity']
    if not any(k in identity for k in ('source_id','source_sha256_digest')) or not any(
        k in identity for k in ('config_id','policy_id','manifest_config_digest')):
        raise ValueError('formal calibration requires declared source and configuration identities')
    method_names=declared_methods(phase);signatures = _signatures(records)
    result = dict(schema='ev_reliability_v2_thresholds_v1', identity=phase['identity'],
        alpha=float(alpha), calibration_count=len(records), calibration_files=phase['files'],
        calibration_seeds=[r['seed'] for r in records], candidate_supports=signatures,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        analyzer_source_sha256=sha256(Path(base.__file__).read_bytes()).hexdigest(),
        fixed_rules=list(RULES), gamma_AIC=2., structural_dimension='2 * actual edge count',
        methods=list(method_names),
        failure_rule='incomplete inference, illegal action or unknown structural metadata: keep all; calibration score zero',
        absent_group_rule='infinite threshold; keep this candidate',
        thresholds_saved_before_evaluation_read=True, simultaneous_coverage_claimed=False,
        coverage_scope='each frozen rule separately; label rules conditional on the candidate label; H_R conditional on declared H',
        AIC_scope='working penalty only; no asymptotic AIC or Wilks claim', rules={})
    for rule in RULES:
        methods = {}
        for method in method_names:
            methods[method] = {}
            for inference in base.INFERENCE:
                groups = defaultdict(list); failures = Counter()
                for record in records:
                    value = scored_mode(record, method, inference, rule)
                    group = group_key(record, rule, record['true_name'])
                    groups[group].append(value['relative'].get(record['true_name'], 0.))
                    failures[group] += not value['complete']
                declared_groups = (set(signatures) if rule.startswith('label_') else
                                   set(map(str, H_GROUPS)) if rule == 'H_R' else {'all'})
                methods[method][inference] = {group: dict(**_threshold(groups[group], alpha),
                    scores=groups[group], incomplete_calibration=failures[group])
                    for group in sorted(set(groups)|declared_groups)}
        result['rules'][rule] = methods
    base._write(Path(output)/'reliability_v2_thresholds.json', result, immutable=True)
    return result


def decision(record, method, inference, rule, thresholds):
    value = scored_mode(record, method, inference, rule)
    bank = value['bank']; true = record['true_name']
    if value['candidate_miss']:
        return dict(category='candidate_miss', retained=bank, false_reject=None, **value)
    if not value['complete']:
        return dict(category='failed_keep_all', retained=bank, false_reject=False, **value)
    retained = []
    for candidate in bank:
        threshold = thresholds.get(group_key(record, rule, candidate))
        limit = math.inf if threshold is None or threshold['infinite'] else threshold['threshold']
        if value['relative'][candidate] <= limit:
            retained.append(candidate)
    category = ('unique_correct' if retained == [true] else 'unique_wrong' if len(retained) == 1
                else 'unresolved')
    return dict(category=category, retained=retained, false_reject=true not in retained, **value)


def _summary(records, thresholds):
    result = {}
    for rule in RULES:
        methods = {}
        for method in thresholds['methods']:
            methods[method] = {}
            for inference in base.INFERENCE:
                rows = [decision(r, method, inference, rule,
                        thresholds['rules'][rule][method][inference]) for r in records]
                counts = Counter(r['category'] for r in rows)
                methods[method][inference] = dict(denominator=len(records),
                    **{key:counts[key] for key in ('unique_correct','unique_wrong','unresolved',
                                                  'failed_keep_all','candidate_miss')},
                    false_reject=sum(r['false_reject'] is True for r in rows),
                    empty_set=sum(not r['retained'] for r in rows),
                    retained_count=base._stats(len(r['retained']) for r in rows))
        result[rule] = methods
    return result


def evaluate(phase, thresholds, output, *, expected_count=None, development=False):
    records = phase['records']; output = Path(output)
    if tuple(declared_methods(phase))!=tuple(thresholds['methods']):
        raise ValueError('calibration and evaluation declared methods differ')
    if not development and phase['phase'] != 'evaluation':
        raise ValueError('evaluation phase required')
    if thresholds['tool_sha256'] != sha256(Path(__file__).read_bytes()).hexdigest() or thresholds[
        'analyzer_source_sha256'] != sha256(Path(base.__file__).read_bytes()).hexdigest():
        raise ValueError('postprocessing source changed since calibration')
    if expected_count is not None and len(records) != expected_count:
        raise ValueError('evaluation count differs from declared batch')
    base._identity_match(thresholds['identity'], phase['identity'])
    if not development and set(thresholds['calibration_seeds']) & {r['seed'] for r in records}:
        raise ValueError('calibration and evaluation histories overlap')
    if not development and len({r['seed'] for r in records}) != len(records):
        raise ValueError('evaluation requires independent history seeds')
    if not development and any(r.get('candidate_miss') for r in records):
        raise ValueError('formal evaluation prohibits candidate miss')
    signatures = _signatures(records)
    if not development and signatures and set(signatures)!=set(thresholds['candidate_supports']):
        raise ValueError('calibration and evaluation structural candidate banks differ')
    path = output/'reliability_v2_thresholds.json'
    if not path.exists() or base._read(path) != thresholds:
        raise ValueError('matching thresholds must be durably saved before evaluation')
    truth_groups = defaultdict(list); H_groups = defaultdict(list); cases = defaultdict(list)
    for r in records:
        truth_groups[r['true_name']].append(r)
        H_groups[str(r.get('config',{}).get('training_points','unknown'))].append(r)
        cases[r['case_id']].append(r)
    result = dict(schema='ev_reliability_v2_evaluation_v1', phase=phase['phase'],
        identity=phase['identity'], source_files=phase['files'],
        threshold_sha256=sha256(path.read_bytes()).hexdigest(), thresholds_development_only=development,
        overall=_summary(records, thresholds),
        by_truth={k:_summary(v,thresholds) for k,v in sorted(truth_groups.items())},
        by_H={k:_summary(v,thresholds) for k,v in sorted(H_groups.items())},
        by_case={k:_summary(v,thresholds) for k,v in sorted(cases.items())},
        by_history=[dict(seed=r['seed'],case_id=r['case_id'],true_name=r['true_name'],
            rules={rule:{method:{mode:decision(r,method,mode,rule,thresholds['rules'][rule][method][mode])
                for mode in base.INFERENCE} for method in thresholds['methods']} for rule in RULES})
            for r in records])
    base._write(output/('development_reliability_v2.json' if development else 'evaluation_reliability_v2.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--expected-calibration', type=int)
    parser.add_argument('--expected-evaluation', type=int)
    args = parser.parse_args()
    cal = read_phase(args.calibration, expected_phase='calibration')
    thresholds = calibrate(cal,args.output,alpha=args.alpha,expected_count=args.expected_calibration)
    # This is deliberately the first evaluation read.
    if args.evaluation:
        ev = read_phase(args.evaluation, expected_phase='evaluation')
        evaluate(ev,thresholds,args.output,expected_count=args.expected_evaluation)


if __name__ == '__main__':
    main()
