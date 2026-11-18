"""Read-only decomposition of completed flexibility actions against physical R/X.

This entry point reads saved AC observations. It never calls an AC solver,
optimizer, candidate fitter, or random generator. Physical lines are used only
for this post-hoc diagnosis, not to change any plan, candidate, or score.

Example (two completed cells, the same chosen target and comparison indices)::

    python -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.response_bias_diagnosis \
      --run data/flexibility_grid_execution_20261115_v1 \
      --ids soumalas11__2024-01-04__s2__m0p5 soumalas11__2024-01-04__s2__m1p0 \
      --target-native 5 --competitors 6 3 --out outputs/new_readonly_diagnosis

Indices are explicitly selected by the caller for offline explanation. The
entry point does not infer which candidate is true or select favorable pairs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from rnj_wzzt_dr.physical import backend as backend
from ..measured_pq_20261114.physical import _network_hash
from ..measured_pq_flex_20261114.analyze import physical_response


class Sources:
    def __init__(self):
        self.refs = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        got = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if expected is not None and (got['sha256'] != expected['sha256']
                                    or path != Path(expected['path']).resolve()):
            raise ValueError(f'Sealed source differs: {path}')
        if str(path) in self.refs and self.refs[str(path)] != got:
            raise ValueError(f'Source changed during diagnosis: {path}')
        self.refs[str(path)] = got
        return got

    def json(self, path, expected=None):
        self.check(path, expected)
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))

    def arrays(self, path, expected=None):
        self.check(path, expected)
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key].copy() for key in z.files}

    def finish(self):
        for item in list(self.refs.values()):
            self.check(item['path'], item)
        return list(self.refs.values())


def decompose(power, y, baseline_power, baseline_y, observed_power, observed_y,
              baseline_observed_power, baseline_observed_y, target, competitor,
              physical, sigma2, historical_log_odds):
    """Exact signed score decomposition; matrices include P and Q columns."""
    n = target.shape[0]
    difference = target - competitor
    error = physical - target
    delta_power = power - baseline_power
    d, d0 = power @ difference.T, baseline_power @ difference.T
    dd = delta_power @ difference.T
    ep, ep0 = power @ error.T, baseline_power @ error.T
    dep = delta_power @ error.T
    remainder, remainder0 = y - power @ physical.T, baseline_y - baseline_power @ physical.T

    def score(x, voltage):
        return historical_log_odds + float((np.square(voltage-x@competitor.T).sum()
                                           - np.square(voltage-x@target.T).sum())/(2*sigma2))

    parameter = float(np.sum(ep*d)/sigma2)
    parameter0 = float(np.sum(ep0*d0)/sigma2)
    ac = float(np.sum(remainder*d)/sigma2)
    ac0 = float(np.sum(remainder0*d0)/sigma2)
    clean = historical_log_odds + float(np.sum(d*d)/(2*sigma2)) + parameter + ac
    clean0 = historical_log_odds + float(np.sum(d0*d0)/(2*sigma2)) + parameter0 + ac0
    observed = score(observed_power, observed_y)
    observed0 = score(baseline_observed_power, baseline_observed_y)
    output = dict(
        clean_L=clean, observed_L=observed,
        baseline_clean_L=clean0, baseline_observed_L=observed0,
        delta_J=float((np.sum(d*d)-np.sum(d0*d0))/(2*sigma2)),
        delta_response_bias=parameter-parameter0, delta_AC_remainder=ac-ac0,
        delta_meter=observed-observed0-(clean-clean0), delta_L=observed-observed0,
        parameter_background_projection=float(np.sum(ep0*dd)/sigma2),
        parameter_response_on_background=float(np.sum(dep*d0)/sigma2),
        parameter_action_interaction=float(np.sum(dep*dd)/sigma2))
    blocks = [('P', 'R', slice(0, n)), ('Q', 'X', slice(n, 2*n))]
    for name, error_name, index in blocks:
        error_part = power[:, index] @ error[:, index].T
        error_part0 = baseline_power[:, index] @ error[:, index].T
        output[f'delta_{error_name}_response_bias'] = float(
            (np.sum(error_part*d)-np.sum(error_part0*d0))/sigma2)
        for other_name, _, other_index in blocks:
            output[f'quadratic_{name}{other_name}'] = float(np.sum(
                (delta_power[:, index]@error[:, index].T)
                * (delta_power[:, other_index]@difference[:, other_index].T))/sigma2)
            difference_part = power[:, other_index] @ difference[:, other_index].T
            difference_part0 = baseline_power[:, other_index] @ difference[:, other_index].T
            output[f'{error_name}_error_against_{other_name}_difference'] = float(
                (np.sum(error_part*difference_part)-np.sum(error_part0*difference_part0))/sigma2)
    discrepancy = max(
        abs(clean-score(power, y)), abs(clean0-score(baseline_power, baseline_y)),
        abs(output['delta_L']-sum(output[k] for k in
            ('delta_J', 'delta_response_bias', 'delta_AC_remainder', 'delta_meter'))),
        abs(output['delta_response_bias']-sum(output[k] for k in
            ('parameter_background_projection', 'parameter_response_on_background', 'parameter_action_interaction'))),
        abs(output['delta_response_bias']-output['delta_R_response_bias']-output['delta_X_response_bias']),
        abs(output['parameter_action_interaction']-sum(output[f'quadratic_{a}{b}'] for a in 'PQ' for b in 'PQ')))
    return output, discrepancy


def run(run_directory, ids, target_native, competitors, *, fraction=.10, forecast_index=0):
    run_directory = Path(run_directory).resolve()
    sources = Sources()
    for path in (__file__, backend.__file__,
                 Path(__file__).parents[1]/'measured_pq_20261114/physical.py',
                 Path(__file__).parents[1]/'measured_pq_flex_20261114/analyze.py'):
        sources.check(path)
    entries = sources.json(run_directory/'inputs.json')
    if len(ids) != len(set(ids)) or len(competitors) != len(set(competitors)):
        raise ValueError('Duplicate input or competitor indices')
    records, cases = [], []
    identity_error = score_error = 0.
    for identity in ids:
        entry = next(e for e in entries if e['id'] == identity)
        folder = run_directory/f'fraction{fraction:.2f}'/identity/f'forecast{forecast_index:02d}'
        complete = sources.json(folder/'complete.json')
        if complete['status'] != 'complete':
            raise ValueError(f'Incomplete forecast: {identity}')
        seal = sources.json(folder/'seal.json')
        score = sources.json(seal['score']['path'], seal['score'])
        if entry['score'] != seal['score']:
            raise ValueError('Input score and sealed score differ')
        results = sources.json(folder/'results.json', complete['results'])
        if len(results) != complete['rows'] or any(r['status'] != 'complete' for r in results):
            raise ValueError('Missing or failed execution: diagnose completed rows explicitly elsewhere')
        H = np.asarray([m['H'] for m in score['native_bank']], float)
        history = np.asarray([m['history_sse'] for m in score['native_bank']], float)
        if target_native < 0 or target_native >= len(H) or any(j < 0 or j >= len(H) for j in competitors):
            raise ValueError('Native candidate index out of bounds')
        topology = score['native_bank'][target_native]['topology_id']
        if any(score['native_bank'][j]['topology_id'] == topology for j in competitors):
            raise ValueError('Requested competitor is in the target topology family')
        sigma2 = float(score['sigma2'])
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError('Invalid frozen score scale')
        network = backend.load_case(entry['case'])
        if _network_hash(network) != entry['physical_cache']['network_hash']:
            raise ValueError('Physical network differs from executed version')
        if network['base_kw'] != entry['base_kw']:
            raise ValueError('Executed P/Q and network power bases differ')
        for name in ('metadata.json', 'buses.csv', 'branches.csv'):
            sources.check(backend.ROOT/'cases'/entry['case']/name)
        physical = physical_response(entry['case'], score['labels'], entry['physical_cache']['network_hash'])
        n = len(score['labels'])
        if H.shape[1:] != (n, 2*n) or physical.shape != (n, 2*n):
            raise ValueError('Response dimensions differ')
        policies = list(dict.fromkeys(r['strategy'] for r in results))
        repeats = sorted({r['repeat'] for r in results})
        if len(results) != len(policies)*len(repeats):
            raise ValueError('Unbalanced saved policy/repeat panel')

        def observation(policy, repeat):
            ob = sources.arrays(folder/f'execution{repeat:02d}'/policy/'observations.npz')
            if list(map(int, ob['labels'])) != list(map(int, score['labels'])):
                raise ValueError('Terminal order changed')
            true_x = np.c_[ob['true_P'], ob['true_Q']]
            true_y = ob['true_root'][:, None]**2-ob['true_V']**2
            observed_x = np.c_[ob['P'], ob['Q']]
            return ob, true_x, true_y, observed_x

        zero = {repeat: observation('no_dfs', repeat) for repeat in repeats}
        summaries = {}
        for policy in policies:
            policy_rows = []
            for repeat in repeats:
                ob, power, y, observed_power = zero[repeat] if policy == 'no_dfs' else observation(policy, repeat)
                z, power0, y0, observed_power0 = zero[repeat]
                total = history + np.square(ob['Y'][None]-np.einsum('tc,knc->ktn', observed_power, H)).sum((1, 2))
                score_error = max(score_error, float(np.max(abs(total-ob['native_total_sse']))))
                for j in competitors:
                    values, error = decompose(power, y, power0, y0, observed_power, ob['Y'],
                        observed_power0, z['Y'], H[target_native], H[j], physical, sigma2,
                        float((history[j]-history[target_native])/(2*sigma2)))
                    identity_error = max(identity_error, error)
                    row = dict(id=identity, forecast_index=forecast_index, policy=policy,
                               repeat=repeat, target=target_native, competitor=j, **values)
                    records.append(row)
                    policy_rows.append(row)
            summaries[policy] = {str(j): {key: dict(
                mean=float(np.mean([r[key] for r in policy_rows if r['competitor'] == j])),
                minimum=float(min(r[key] for r in policy_rows if r['competitor'] == j)),
                maximum=float(max(r[key] for r in policy_rows if r['competitor'] == j)))
                for key in values} for j in competitors}
        error_matrix = physical-H[target_native]
        cases.append(dict(id=identity, target_native=target_native, competitors=list(competitors),
            forecast_index=forecast_index, strategies=len(policies), repeats=len(repeats),
            sigma2=sigma2, base_kva=network['base_mva']*1000., base_kv=network['base_kv'],
            zbase_ohm=network['base_kv']**2/network['base_mva'],
            network_hash=entry['physical_cache']['network_hash'],
            physical_response_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
            R_relative_response_error=float(np.linalg.norm(error_matrix[:, :n])/np.linalg.norm(physical[:, :n])),
            X_relative_response_error=float(np.linalg.norm(error_matrix[:, n:])/np.linalg.norm(physical[:, n:])),
            policies=summaries))
    if identity_error > 1e-7 or score_error > 1e-7:
        raise ArithmeticError('Saved-score or decomposition identity check failed')
    return dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post hoc saved-data diagnosis; no optimization, AC, fit, random simulation or plan changes.',
        attribution_scope='Full response-matrix mismatch includes parameter and full-support differences. R/X blocks keep the same complete P/Q candidate contrast; they are not P-only/Q-only counterfactuals.',
        observed_file_scope='Observation arrays receive content hashes here and frozen native SSE is independently replayed; this does not replace the separate execution/provenance audit.',
        uncertainty_scope='Min/max and means describe paired execution paths; not independent-date confidence intervals.',
        maximum_identity_error=identity_error, maximum_native_SSE_replay_error=score_error,
        source_refs=sources.finish(), cases=cases, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--target-native', type=int, required=True)
    parser.add_argument('--competitors', nargs='+', type=int, required=True)
    parser.add_argument('--fraction', type=float, default=.10)
    parser.add_argument('--forecast-index', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(args.run.resolve()),
        ids=args.ids, target_native=args.target_native, competitors=args.competitors,
        fraction=args.fraction, forecast_index=args.forecast_index, source=Sources().check(__file__),
        scope='Existing completed execution arrays only; physical truth used solely for offline explanation')
    def save(name, value):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    save('protocol.json', protocol)
    result = run(args.run, args.ids, args.target_native, args.competitors,
                 fraction=args.fraction, forecast_index=args.forecast_index)
    save('response_bias.json', result)
    save('complete.json', dict(status='complete', inputs=len(result['cases']),
        directed_records=len(result['records']), output=Sources().check(args.out/'response_bias.json'),
        maximum_identity_error=result['maximum_identity_error'],
        maximum_native_SSE_replay_error=result['maximum_native_SSE_replay_error']))
    print(json.dumps(dict(output=str((args.out/'response_bias.json').resolve()),
        directed_records=len(result['records']), maximum_identity_error=result['maximum_identity_error'])))


if __name__ == '__main__':
    main()
