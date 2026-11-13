"""Review-only arithmetic on sealed histories and AC observations.

No identifier, procurement optimizer, AC simulation, or new random paths.
Raw scores/arrays stay private. All baselines are declared before replay.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs/customer_flexibility_sweep_20261113_v1'
EXTRA = ROOT/'outputs/customer_flexibility_intensity_20261113_v1'
IDENTITY = 's0_c013'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def binary_mean(history_logodds, J, order=192, bias=0., variance=None):
    nodes, weights = roots_hermitenorm(order)
    variance = 2*J if variance is None else variance
    return float(weights@expit(history_logodds+J+bias+np.sqrt(max(0., variance))*nodes)/np.sqrt(2*np.pi))


def binary_integral(history_logodds, J):
    value, error = quad(lambda z: expit(history_logodds+J+np.sqrt(2*J)*z)
        *np.exp(-z*z/2)/np.sqrt(2*np.pi), -12., 12., epsabs=1e-12, epsrel=1e-12)
    return float(value), float(error)


def reject_penalty(budget, history_sse, sigma2, new_cells):
    boundary = (budget-history_sse)/sigma2
    lo, hi = chi2.ppf(1e-13, new_cells), chi2.isf(1e-13, new_cells)
    points = [v for v in (boundary-20, boundary, boundary+20) if lo < v < hi]
    value, error = quad(lambda x: expit((x-boundary)/2)*chi2.pdf(x, new_cells),
                        lo, hi, points=points, epsabs=1e-12)
    return dict(expected_reject_penalty=float(value), integration_error_estimate=float(error),
                omitted_tail_mass_upper=2e-13, expected_target_weight_ceiling=float(1-value))


def replay_history():
    truth = read(BASE/'offline_truth.json')['truth_id']
    jobs = [j for f in sorted(BASE.glob('history_jobs_s*.json')) for j in read(f)]
    counts = {name: dict(fitted=0, failed=0, correct=0, wrong=0, reject=0,
        truth_present=0, truth_present_rejected=0, missing_truth_rejected=0)
        for name in ('original', 'noise_chi95', 'fixed_125pct_noise_MSE')}
    for job in jobs:
        receipt = read(Path(job['directory'])/'fit_complete.json')
        if receipt['status'] != 'fitted':
            for row in counts.values():
                row['failed'] += 1
            continue
        s = read(receipt['score']['path'])
        assert ref(receipt['score']['path'])['sha256'] == receipt['score']['sha256']
        var = 8*job['voltage_sigma']**2
        correct = s['map_topology_id'] == truth
        present = any(m['topology_id'] == truth for m in s['models'])
        for name, row in counts.items():
            B = var*chi2.ppf(.95, s['cells']) if name == 'noise_chi95' else s['cells']*1.25*var
            rejected = name != 'original' and s['cells']*s['sigma2'] >= B
            row['fitted'] += 1
            row['truth_present'] += int(present)
            row['truth_present_rejected'] += int(present and rejected)
            row['missing_truth_rejected'] += int(not present and rejected)
            row['reject' if rejected else 'correct' if correct else 'wrong'] += 1
    return counts


def run(out):
    if out.exists():
        raise ValueError('Preserve existing review arithmetic; use a new output directory')
    out.mkdir(parents=True)
    scorefile = BASE/'histories'/IDENTITY/'score.json'
    s = read(scorefile)
    H = np.asarray([p['H'] for p in s['parameters']])
    target, sigma2 = s['map_index'], s['sigma2']
    others = [j for j in range(len(H)) if j != target]
    L = np.array([s['models'][target]['log_probability']-s['models'][j]['log_probability'] for j in others])
    nearest = int(np.argmin(L))
    runner = others[nearest]
    m, total_cells = 1440, s['cells']+1440
    variants = dict(noise_chi95=float(2e-6*chi2.ppf(.95, total_cells)),
                    fixed_125pct_noise_MSE=float(total_cells*2.5e-6))
    with np.load(BASE/'histories'/IDENTITY/'validation.npz') as z:
        historical_power = np.c_[z['P'], z['Q']]
        historical_y = z['root'][:, None]**2-z['V']**2
    prediction = historical_power@H[target].T
    residual = historical_y-prediction
    bias_mean = residual.mean(0)
    mean_scale = 1+np.sum(prediction*residual, axis=0)/(np.sum(prediction**2, axis=0)+len(prediction)*sigma2)
    corrected_H = mean_scale[:, None]*H[target]
    protocol = dict(kind='sealed_data_review_only', identity=IDENTITY, source=ref(__file__),
        score=ref(scorefile), history_validation=ref(BASE/'histories'/IDENTITY/'validation.npz'),
        resource_config=ref(BASE/'resource_configs/higher.json'),
        parent_protocols=[ref(p/'protocol.json') for p in (BASE, EXTRA)],
        new_fits=0, new_optimized_plans=0, new_AC_days=0, new_random_paths=0,
        residual_reference_variance=2e-6, baselines=variants,
        history_baselines='noise_chi95 uses8*voltage_sigma²*chi2(.95,cells); fixed alternative uses1.25*8*voltage_sigma²*cells',
        decision='argmax over frozen distinct-candidate weights plus rejection class; rejection counts as no correct output',
        model='given sealed point forecast and nominal linear delivery, Y=Htarget*P+epsilon, epsilon~N(0,sigma2 I)',
        mathematics='Binary logistic-normal expectation by192-node quadrature;384 nodes checked independently; null expectation by scalar chi-square quadrature',
        diagnostic='Historical residual constant mean and existing multiplicative mean correction, each with independent errors and a1pu approximation of common root plus terminal meter errors; each channel relative voltage SD0.0005. These are single-pair mean diagnostics, not actual recovery tests.',
        scope='Re-scoring of previously opened observations; not an independent test of the new criterion. No c0 selected using replay accuracy.')
    write(out/'protocol.json', protocol)
    c = read(BASE/'resource_configs/higher.json')
    labels = c['expected_terminal_labels']
    nodes = [labels.index(r['node_label']) for r in c['resources']]
    ratios = [r.get('q_over_p', np.tan(np.arccos(r.get('power_factor', 1.)))) for r in c['resources']]
    null = {name: reject_penalty(B, s['parameters'][target]['history_sse'], sigma2, m) for name, B in variants.items()}
    result = dict(protocol=ref(out/'protocol.json'), history=replay_history(),
        constants=dict(sigma2=sigma2, history_cells=s['cells'], new_cells=m, total_cells=total_cells,
            candidates=len(H), historical_target_probability=s['p_map'],
            historical_target_runner_logodds=float(L[nearest])),
        null_model_checks=null, original_rule_replay={}, actual_replay={}, model_checks={},
        closed_form_binary_J_sufficient={str(beta): float(max(0., 4*np.log(1/(2*(1-beta)))-2*L[nearest])) for beta in (.9, .95)},
        exact_binary_J_threshold={str(beta): float(brentq(lambda J:binary_mean(L[nearest], J)-beta, 0, 100)) for beta in (.9, .95)})
    policies = [('no_dfs', BASE, 'no_dfs'), ('small', BASE, 'optimized_g0'),
                ('two', EXTRA, 'optimized_g0'), ('four', EXTRA, 'optimized_g1')]
    verification_error, probability_error, integral_error = 0., 0., 0.
    input_refs, rows = [], []
    for label, source, strategy in policies:
        metrics = {name: [] for name in variants}
        forecasts, diagnostic, original_values = [], [], []
        for day in range(32):
            directory = source/'confirmation'/IDENTITY/f'day{day:03d}'
            seal = read(directory/'policy_seal.json')
            assert ref(directory/'planner.npz')['sha256'] == seal['planner']['sha256']
            input_refs.extend([ref(directory/(strategy+'.npz')), ref(directory/'planner.npz')])
            with np.load(directory/(strategy+'.npz')) as z:
                observed_power = np.c_[z['P'], z['Q']]
                y = z['root'][:, None]**2-z['V']**2
                saved = np.exp(z['log_probability'][target])
            SSE = np.array([p['history_sse'] for p in s['parameters']])+np.sum(
                (y[None]-np.einsum('tc,knc->ktn', observed_power, H))**2, axis=(1, 2))
            loglike = -SSE/(2*sigma2)
            original_p = float(np.exp(loglike[target]-logsumexp(loglike)))
            original_values.append([original_p, int(np.argmax(loglike) == target)])
            probability_error = max(probability_error, abs(original_p-saved))
            for name, B in variants.items():
                normal = logsumexp(np.r_[-B/(2*sigma2), loglike])
                ptrue = float(np.exp(loglike[target]-normal))
                pzero = float(np.exp(-B/(2*sigma2)-normal))
                rejected = -B/(2*sigma2) >= float(max(loglike))
                correct = not rejected and int(np.argmax(loglike)) == target
                metrics[name].append([ptrue, pzero, int(rejected), int(correct), int(not rejected and not correct), int(ptrue >= .9)])
                rows.append(dict(strategy=label, day=day, baseline=name, p_target=ptrue, rejection_weight=pzero,
                    rejected=bool(rejected), correct=bool(correct), original_target_weight=original_p))
            with np.load(directory/'planner.npz') as z:
                forecast, u = np.array(z['forecast']), np.array(z[strategy])
            shift = np.zeros((24, 30))
            for r, (node, ratio) in enumerate(zip(nodes, ratios, strict=True)):
                shift[:, node] += u[:, r]/200
                shift[:, 15+node] += ratio*u[:, r]/200
            power = forecast+np.repeat(shift, 4, axis=0)
            J = np.array([np.sum((power@(H[target]-H[j]).T)**2)/(2*sigma2) for j in others])
            difference = power@(H[target]-H[runner]).T
            shared_variance = (1e-6*np.sum(difference**2)+1e-6*np.sum(difference.sum(1)**2))/sigma2**2
            additive = float(np.sum(bias_mean[None]*difference)/sigma2)
            multiplicative = float(np.sum((power@(corrected_H-H[target]).T)*difference)/sigma2)
            diagnostic.append([shared_variance/(2*J[nearest]),
                binary_mean(L[nearest], J[nearest], order=384, bias=additive),
                binary_mean(L[nearest], J[nearest], order=384, bias=multiplicative),
                binary_mean(L[nearest], J[nearest], order=384, bias=additive, variance=shared_variance),
                binary_mean(L[nearest], J[nearest], order=384, bias=multiplicative, variance=shared_variance)])
            binary = binary_mean(L[nearest], J[nearest])
            checked = binary_mean(L[nearest], J[nearest], order=384)
            verification_error = max(verification_error, abs(binary-checked))
            integrated, integration_error = binary_integral(L[nearest], J[nearest])
            integral_error = max(integral_error, integration_error, abs(integrated-checked))
            tail = float(sum(.5*np.exp(-L[i]/2-J[i]/4) for i in range(len(others)) if i != nearest))
            forecasts.append([J[nearest], integrated, tail])
        original = np.asarray(original_values)
        result['original_rule_replay'][label] = dict(days=len(original),
            mean_target_weight=float(original[:, 0].mean()), correct=int(original[:, 1].sum()))
        result['actual_replay'][label] = {}
        for name, values in metrics.items():
            a = np.asarray(values)
            result['actual_replay'][label][name] = dict(days=len(values), mean_target_weight=float(a[:, 0].mean()),
                mean_rejection_weight=float(a[:, 1].mean()), rejected=int(a[:, 2].sum()),
                correct=int(a[:, 3].sum()), wrong=int(a[:, 4].sum()), pass90=int(a[:, 5].sum()))
        a = np.asarray(forecasts)
        loose_q = (null['fixed_125pct_noise_MSE']['expected_reject_penalty']+
                   null['fixed_125pct_noise_MSE']['integration_error_estimate']+2e-13)
        result['model_checks'][label] = dict(mean_J=float(a[:, 0].mean()), min_J=float(a[:, 0].min()), max_J=float(a[:, 0].max()),
            mean_binary_expected_weight=float(a[:, 1].mean()), max_other_candidate_bound=float(a[:, 2].max()),
            full_pool_expected_weight_lower_mean=float(np.maximum(0, a[:, 1]-a[:, 2]-loose_q).mean()),
            full_pool_expected_weight_upper_mean=float(a[:, 1].mean()),
            zero_or_existing_policy_certified_days={str(beta): int(np.sum(a[:, 1]-a[:, 2]-loose_q >= beta)) for beta in (.9, .95)})
        diag = np.asarray(diagnostic).mean(0)
        result['model_checks'][label]['single_pair_diagnostics'] = dict(
            shared_root_direction_variance_ratio=float(diag[0]),
            additive_mean_independent_error=float(diag[1]), multiplicative_mean_independent_error=float(diag[2]),
            additive_mean_shared_error=float(diag[3]), multiplicative_mean_shared_error=float(diag[4]),
            interpretation='Binary expected-weight upper bounds; not full-bank expected weights or AC recovery rates')
    checks = []
    for history in (-2., 0., float(L[nearest]), 5.):
        for J in (0., .1, 1., 3., 10.):
            f = binary_mean(history, J, order=384)
            lower = 1-.5*np.exp(-history/2-J/4)
            assert f >= lower-1e-8
            checks.append(float(f-lower))
    assert probability_error < 1e-9 and verification_error < 1e-7
    result['verification'] = dict(original_probability_max_error=probability_error,
        quadrature_192_vs_384_max_error=verification_error, scalar_lower_bound_checks=len(checks),
        quadrature_vs_adaptive_integral_max_error=integral_error,
        minimum_bound_slack=min(checks), source_arrays_changed=False,
        numerical_bounds='Ordinary floating-point quadrature estimates; not interval-arithmetic certificates')
    write(out/'input_refs.json', input_refs)
    write(out/'private_rows.json', rows)
    write(out/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.out.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
