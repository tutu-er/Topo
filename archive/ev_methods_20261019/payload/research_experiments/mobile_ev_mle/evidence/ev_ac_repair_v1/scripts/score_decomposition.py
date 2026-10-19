"""Read-only Gaussian score factorization; no prediction, fit or plan changes.

The conditional-only classifier is a diagnostic, not a predeclared policy.
Restoring log p(P/Q) gives exactly the original joint classifier.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

WORK = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(WORK))
from research_experiments.mobile_ev_mle import ev_ac_repair_analysis as analysis


def log_normal(y, mean, cov):
    factor = cho_factor(cov, lower=True)
    delta = np.asarray(y) - mean
    return float(-.5*(delta @ cho_solve(factor, delta)
        + 2*np.log(np.diag(factor[0])).sum() + len(delta)*np.log(2*np.pi)))


def split_score(y, mean, cov, count):
    """Exact log p(x,y)=log p(x)+log p(y|x) for one Gaussian."""
    x, z = np.asarray(y)[:count], np.asarray(y)[count:]
    fx = cho_factor(cov[:count, :count], lower=True)
    cross = cov[count:, :count]
    conditional_mean = mean[count:] + cross @ cho_solve(fx, x-mean[:count])
    conditional_cov = cov[count:, count:] - cross @ cho_solve(fx, cross.T)
    marginal = log_normal(x, mean[:count], cov[:count, :count])
    conditional = log_normal(z, conditional_mean, .5*(conditional_cov+conditional_cov.T))
    total = log_normal(y, mean, cov)
    return dict(PQ_marginal=marginal, electrical_conditional=conditional,
                total=total, identity_error=abs(total-marginal-conditional))


def audit(root, phase):
    rows = []; statuses = []; hashes = {}; maximum_error = 0.
    for path in sorted((Path(root)/phase).glob('history_*.json.gz')):
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        r = analysis.load(path)
        statuses.append(dict(index=r['spec']['index'], status=r.get('status'),
                             failures=r.get('failures', {})))
        names = r.get('candidate_names', [])
        if r.get('true_name') not in names:
            continue
        true = names.index(r['true_name'])
        for plan, action in analysis._actions(r):
            n = len(r['observed_training']['p'][0])
            horizon = len(action['power']); count = 2*horizon*n
            for model, bank in analysis.usable_banks(action).items():
                means, covs, prior = analysis.scaled_bank(bank)
                for regime, actual in action.get('regimes', {}).items():
                    for kind, data in (('clean', actual['clean']), ('noisy', actual['values'])):
                        for repeat, y in enumerate(np.atleast_2d(data)):
                            scores = [split_score(y, mu, cov, count)
                                      for mu, cov in zip(means, covs)]
                            maximum_error = max(maximum_error,
                                max(s['identity_error'] for s in scores))
                            joint = np.array([s['total'] for s in scores])+prior
                            pq = np.array([s['PQ_marginal'] for s in scores])
                            electrical = np.array([s['electrical_conditional'] for s in scores])
                            competitors = np.flatnonzero(np.arange(len(names)) != true)
                            other = int(competitors[np.argmax(joint[competitors])])
                            prior_gap = float(prior[true]-prior[other])
                            pq_gap = float(pq[true]-pq[other])
                            electrical_gap = float(electrical[true]-electrical[other])
                            rows.append(dict(index=r['spec']['index'],plan=plan,
                                model=model,regime=regime,kind=kind,repeat=repeat,
                                true_name=names[true],competitor=names[other],
                                joint_prediction=names[int(np.argmax(joint))],
                                diagnostic_conditional_only_prediction=
                                    names[int(np.argmax(prior+electrical))],
                                joint_margin=float(joint[true]-joint[other]),
                                prior_gap=prior_gap,PQ_marginal_gap=pq_gap,
                                electrical_conditional_gap=electrical_gap,
                                reconstructed_margin=prior_gap+pq_gap+electrical_gap))
        if sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
            raise RuntimeError('raw record changed during read-only audit')
    if maximum_error > 1e-7:
        raise AssertionError('Gaussian factorization failed: '+str(maximum_error))
    return dict(scope='read-only score audit; conditional-only rule diagnostic, not tuned/evaluated policy',
        root=str(root),phase=phase,independent_histories=len(statuses),
        maximum_factorization_error=maximum_error,rows=rows,
        statuses=statuses,raw_sha256=hashes,
        AC_calls=0,fit_calls=0,planning_calls=0,
        caveat='correlated PQ and electrical channels split by Gaussian conditioning; '
            'prior+PQ+conditional is original joint score; omitting PQ changes inference')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--phase',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); result=audit(a.root,a.phase)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
                   encoding='utf-8',newline='\n')
    print(json.dumps(dict(histories=result['independent_histories'],
        rows=len(result['rows']),identity_error=result['maximum_factorization_error'])))


if __name__ == '__main__':
    main()
