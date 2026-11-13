"""Independent frozen-input arithmetic and two-second solver smoke checks."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
from scipy.stats import ncx2, norm

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..flexibility_sweep_20261113.execution import fixed_rotation
from .solver import Recovery, plan


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs/customer_flexibility_sweep_20261113_v1'


def run(out):
    if out.exists():
        raise ValueError('Preserve completed checks; choose a new directory')
    out.mkdir(parents=True)
    paths = dict(score=BASE/'histories/s0_c013/score.json',
        validation=BASE/'histories/s0_c013/validation.npz',
        forecast=BASE/'confirmation/s0_c013/day000/planner.npz',
        resources=BASE/'resource_configs/higher.json')
    io.write(out/'protocol.json', dict(inputs={k:io.ref(v) for k,v in paths.items()},
        solver=io.ref(Path(__file__).with_name('solver.py')), tests=io.ref(__file__),
        new_AC_days=0, new_random_paths=0, smoke_seconds_per_mode=2.))
    score = io.read(paths['score'])
    validation = load(paths['validation'])
    case = CustomerCase(io.read(paths['resources']))
    with np.load(paths['forecast']) as z:
        forecast = z['forecast'].copy()
    pattern, _ = fixed_rotation(case)
    report = dict(expansion={}, smoke={})
    for mode in ('nominal', 'calibrated'):
        model = Recovery(case, score, validation, forecast, 200., mode, .0005)
        max_mu_error, max_var_error, max_bias_error = 0., 0., 0.
        for fraction in (0., .3, 1.):
            u = fraction*pattern
            assert case.audit(u)['passed']
            P = forecast.copy()
            # Independent explicit device/hour mapping, not model.shift_pq_pu.
            for h in range(case.H):
                for r, node in enumerate(case.node):
                    P[h*4:(h+1)*4, node] += u[h,r]/200.
                    P[h*4:(h+1)*4, case.N+node] += u[h,r]*case.qratio[r]/200.
            meanY = P@model.K.T+model.offset
            target_residual = meanY-P@model.H.T
            mean, variance = [], []
            for other in model.competitors:
                error = meanY-P@np.asarray(other['H']).T
                mean.append((other['history_sse']-model.history+np.sum(error**2)-np.sum(target_residual**2))/(2*model.sigma2))
                coef = (error-target_residual)/model.sigma2
                variance.append(model.variance*np.sum(coef**2))
            direct = model.verify_expansion(u)
            max_mu_error = max(max_mu_error, float(np.max(np.abs(np.array(mean)-direct['mu']),initial=0)))
            max_var_error = max(max_var_error, float(np.max(np.abs(np.array(variance)-2*model.ratio*direct['J']),initial=0)))
            max_bias_error = max(max_bias_error, abs(np.sum(target_residual**2)-direct['bias_energy']))
            if len(mean):
                expected = norm.cdf((model.tie-np.array(mean))/np.sqrt(variance))
                assert abs(expected.sum()-direct['pair_failure_sum']) < 1e-10
            expected_reject = ncx2.sf((model.B-model.history)/model.variance,
                meanY.size, np.sum(target_residual**2)/model.variance)
            assert abs(expected_reject-direct['reject_failure']) < 1e-12
        assert max_mu_error < 1e-8 and max_var_error < 1e-7 and max_bias_error < 1e-12
        report['expansion'][mode] = dict(max_mean_error=max_mu_error,max_variance_error=max_var_error,
            max_bias_energy_error=float(max_bias_error),competitors=len(model.competitors),
            variance=model.variance,tie=model.tie)
    # Regression: tolerance must not certify a completely tied different tree.
    clone = deepcopy(score)
    t = clone['map_index']
    rival = dict(clone['native_bank'][0], H=clone['parameters'][t]['H'],
        history_sse=clone['parameters'][t]['history_sse'],topology_id='test_identical_predictions_wrong_tree')
    clone['native_bank'] = [rival]
    model = Recovery(case,clone,validation,forecast,200.,'nominal',.0005)
    tied = model.direct(np.zeros(case.shape))
    assert not tied['feasible'] and tied['pair_failure_sum'] == 1.
    report['identical_prediction_regression'] = 'passed'
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0)
        env.start()
        for mode in ('nominal','calibrated'):
            receipt, point = plan(case,score,validation,forecast,200.,mode=mode,seconds=2.,env=env)
            assert (point is not None) == receipt['feasible']
            if point is not None:
                assert case.audit(point)['passed']
                assert Recovery(case,score,validation,forecast,200.,mode,.0005).verify_expansion(point)['feasible']
            io.write(out/(mode+'_smoke.json'),receipt)
            report['smoke'][mode] = {k:receipt.get(k) for k in ('status','feasible','seconds','cost_yuan','relative_gap','global_optimality_verified')}
    report['status']='passed'
    io.write(out/'summary.json',report)
    return report


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.out.resolve()),indent=2))
