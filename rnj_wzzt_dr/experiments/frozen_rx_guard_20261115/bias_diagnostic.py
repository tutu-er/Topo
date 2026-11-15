"""Read-only three-term bias-direction decomposition of completed actions.

No fitting, planning, optimization or AC is called. Physical arrays are opened
only for cases with a completion receipt. Existing execution files are hashed
before and after reading. The frozen historical calibration is replayed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ... import io


def decompose(X0, Y0, X, Y, Hi, Hj, K, sigma2, history_logodds=0.):
    X0, Y0, X, Y, Hi, Hj, K = map(lambda x: np.asarray(x, float), (X0, Y0, X, Y, Hi, Hj, K))
    if X0.shape != X.shape or Y0.shape != Y.shape or Hi.shape != Hj.shape or Hi.shape != K.shape:
        raise ValueError('Unaligned response decomposition arrays')
    if sigma2 <= 0 or not all(np.isfinite(x).all() for x in (X0, Y0, X, Y, Hi, Hj, K)):
        raise ValueError('Finite arrays and positive variance required')
    D = Hi-Hj
    dx, dy = X-X0, Y-Y0
    d0, dd = X0@D.T, dx@D.T
    b0, db = Y0-X0@K.T, dy-dx@K.T
    pieces = [float(np.sum(b0*dd)/sigma2), float(np.sum(db*d0)/sigma2), float(np.sum(db*dd)/sigma2)]
    def odds(x, y):
        return float(history_logodds+(np.sum((y-x@Hj.T)**2)-np.sum((y-x@Hi.T)**2))/(2*sigma2))
    L0, L = odds(X0, Y0), odds(X, Y)
    M0, M = odds(X0, X0@K.T), odds(X, X@K.T)
    error = abs(sum(pieces)-((L-L0)-(M-M0)))
    if error > 1e-8*(1+abs(sum(pieces))):
        raise ArithmeticError('Direct SSE difference fails the three-term identity')
    def rms(a):
        return float(np.sqrt(np.mean(a*a)/sigma2))
    def cosine(a, b):
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        return None if denominator == 0 else float(np.sum(a*b)/denominator)
    return dict(A_background_bias_rotated=float(pieces[0]),
        B1_response_error_against_old_difference=float(pieces[1]),
        B2_response_error_against_added_difference=float(pieces[2]),
        sum_model_gap_increment=float(sum(pieces)), identity_error=error,
        clean_logodds_before=L0, clean_logodds_after=L, clean_increment=L-L0,
        K_logodds_before=M0, K_logodds_after=M, K_increment=M-M0,
        J_before=float(np.sum(d0*d0)/(2*sigma2)),
        J_after=float(np.sum((d0+dd)**2)/(2*sigma2)),
        baseline_bias_rms_over_sigma=rms(b0),
        incremental_K_error_rms_over_sigma=rms(db),
        incremental_H_error_rms_over_sigma=rms(dy-dx@Hi.T),
        incremental_K_error_mse_over_sigma2=rms(db)**2,
        incremental_H_error_mse_over_sigma2=rms(dy-dx@Hi.T)**2,
        old_difference_rms_over_sigma=rms(d0), added_difference_rms_over_sigma=rms(dd),
        cos_background_added_difference=cosine(b0, dd),
        cos_incremental_error_old_difference=cosine(db, d0),
        cos_incremental_error_added_difference=cosine(db, dd))


def calibrated_means(score, validation):
    H = np.asarray([p['H'] for p in score['parameters']], float)
    X = np.c_[validation['P'], validation['Q']]
    prediction = np.einsum('tc,knc->ktn', X, H)
    factors = 1+np.sum(prediction*(validation['Y'][None]-prediction), axis=1)/(
        np.sum(prediction**2, axis=1)+len(X)*score['sigma2'])
    return factors[:, :, None]*H, factors


class Sources:
    def __init__(self):
        self.refs = {}

    def path(self, item):
        ref = item if isinstance(item, dict) else io.ref(item)
        path = Path(ref['path']).resolve()
        if io.ref(path) != ref:
            raise ValueError('Source content hash mismatch: '+str(path))
        if str(path) in self.refs and self.refs[str(path)] != ref:
            raise ValueError('Source changed during diagnosis: '+str(path))
        self.refs[str(path)] = ref
        return path

    def json(self, item):
        return io.read(self.path(item))

    def arrays(self, item):
        with np.load(self.path(item), allow_pickle=False) as z:
            return {key:z[key].copy() for key in z.files}

    def verify(self):
        return all(io.ref(path) == ref for path, ref in self.refs.items())


def run(run_root, out, ids=None, target_native=None, competitor_native=None):
    root, out = Path(run_root).resolve(), Path(out).resolve()
    if root.parent.name == 'cases':
        ids, root = [root.name], root.parent.parent
    if out.exists():
        raise FileExistsError(out)
    reader = Sources()
    protocol = reader.json(root/'protocol.json')
    entries = reader.json(protocol['inputs'])
    selected = set(ids) if ids else {e['id'] for e in entries}
    if selected-set(e['id'] for e in entries):
        raise ValueError('Unknown diagnostic case')
    out.mkdir(parents=True)
    cases = []
    for entry in entries:
        if entry['id'] not in selected:
            continue
        directory = root/'cases'/entry['id']
        complete = reader.json(directory/'complete.json')
        if complete['status'] != 'complete':
            raise ValueError('Only completed cases may be diagnosed')
        rows = reader.json(complete['results'])
        if len(rows) != complete['rows']:
            raise ValueError('Completed case row count mismatch')
        seal = reader.json(directory/'seal.json')
        score = reader.json(entry['score'])
        if seal['score'] != entry['score']:
            raise ValueError('Completed plan score differs from input score')
        validation = reader.arrays(entry['files']['validation.npz'])
        H = np.asarray([p['H'] for p in score['native_bank']], float)
        hist = np.asarray([p['history_sse'] for p in score['native_bank']], float)
        sigma2 = float(score['sigma2'])
        i = int(score['native_index'] if target_native is None else target_native)
        alternatives = [j for j, b in enumerate(score['native_bank'])
                        if b['topology_id'] != score['native_bank'][i]['topology_id']]
        j = min(alternatives, key=lambda j:(hist[j], j)) if competitor_native is None else int(competitor_native)
        if j not in alternatives:
            raise ValueError('Competitor must have a different topology')
        reps = [int(p['index']) for p in score['parameters']]
        if i not in reps:
            raise ValueError('Target has no frozen historical calibrated mean')
        means, factors = calibrated_means(score, validation)
        K = means[reps.index(i)]
        h = float((hist[j]-hist[i])/(2*sigma2))
        prior = Path(entry['prior_directory'])
        prior_done = reader.json(prior/'complete.json')
        if prior_done['status'] != 'complete':
            raise ValueError('Baseline source is incomplete')
        # Baseline native rows are reused, not rerun or counted as new samples.
        old_rows = reader.json(prior_done['results'])
        policies = [('old_'+name, prior, name, old_rows) for name in
                    ('no_dfs', 'anchor', 'normal_fixed', 'mgf_fixed', 'mgf_forecast')]
        new_names = sorted({r['strategy'] for r in rows if not r['strategy'].endswith('_rotation')})
        policies += [('new_'+name, directory, name, rows) for name in new_names]
        repeats = sorted({int(r['repeat']) for r in rows})
        records, summaries = [], {}
        zero_cache = {}
        for repeat in repeats:
            zero_cache[repeat] = reader.arrays(prior/f'execution{repeat:02d}'/'no_dfs'/'observations.npz')
        for label, location, strategy, result_rows in policies:
            this = []
            lookup = {int(r['repeat']):r for r in result_rows if r['strategy'] == strategy}
            if set(lookup) != set(repeats):
                raise ValueError('Policy does not have the same paired denominator')
            for repeat in repeats:
                r = lookup[repeat]
                if r['status'] != 'complete':
                    raise ValueError('Failed execution retained in source; cannot produce a complete numeric decomposition')
                ob = reader.arrays(location/f'execution{repeat:02d}'/strategy/'observations.npz')
                z = zero_cache[repeat]
                if not np.array_equal(ob['labels'], score['labels']):
                    raise ValueError('Terminal order changed')
                X0, X = np.c_[z['true_P'], z['true_Q']], np.c_[ob['true_P'], ob['true_Q']]
                Y0 = z['true_root'][:, None]**2-z['true_V']**2
                Y = ob['true_root'][:, None]**2-ob['true_V']**2
                result = decompose(X0, Y0, X, Y, H[i], H[j], K, sigma2, h)
                observed = h+(np.sum((ob['Y']-np.c_[ob['P'], ob['Q']]@H[j].T)**2)
                              -np.sum((ob['Y']-np.c_[ob['P'], ob['Q']]@H[i].T)**2))/(2*sigma2)
                result.update(id=entry['id'], policy=label, repeat=repeat,
                    target_native=i, competitor_native=j, observed_logodds_after=float(observed),
                    recovered=bool(r['native_map_correct']), native_selected=int(r['native_selected_index']))
                this.append(result); records.append(result)
            numeric = [key for key in this[0] if isinstance(this[0][key], (int, float))
                       and key not in ('repeat', 'target_native', 'competitor_native', 'native_selected', 'recovered')]
            summaries[label] = dict(n=len(this), recovered=sum(r['recovered'] for r in this),
                selected_counts=dict(Counter(str(r['native_selected']) for r in this)),
                means={key:float(np.mean([r[key] for r in this])) for key in numeric},
                maximum_identity_error=max(r['identity_error'] for r in this))
            print(entry['id'], label, summaries[label]['recovered'],
                {key:round(summaries[label]['means'][key], 6) for key in
                 ('A_background_bias_rotated', 'B1_response_error_against_old_difference',
                  'B2_response_error_against_added_difference', 'clean_increment', 'K_increment',
                  'incremental_H_error_mse_over_sigma2')}, flush=True)
        cases.append(dict(id=entry['id'], target_native=i, competitor_native=j, sigma2=sigma2,
            selection_rule='Historical native winner versus lowest historical SSE different topology, unless explicit CLI override',
            row_scale_minimum=float(factors[reps.index(i)].min()), row_scale_maximum=float(factors[reps.index(i)].max()),
            summaries=summaries, records=records))
    unchanged = reader.verify()
    if not unchanged:
        raise ValueError('A saved source changed while reading')
    result = dict(status='complete', sources_unchanged=unchanged, source=io.ref(__file__),
        scope='Saved-array decomposition at identical actual P/Q; fixed H/K/history; baseline reuse; no new fitting/optimization/AC',
        formula='Delta(Lclean-LK) = <b0,D deltaX>/sigma2 + <delta b,D X0>/sigma2 + <delta b,D deltaX>/sigma2',
        model='b0=Y0-KX0; delta b=(Y-Y0)-K(X-X0); D=Hi-Hj; all inner products over new-day time and terminal cells',
        sources=list(reader.refs.values()), cases=cases)
    io.write(out/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--target-native', type=int)
    parser.add_argument('--competitor-native', type=int)
    args = parser.parse_args()
    run(args.run, args.out, args.ids, args.target_native, args.competitor_native)
