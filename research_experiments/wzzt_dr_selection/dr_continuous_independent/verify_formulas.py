"""Record independent numerical checks of each continuous-inversion formula.

Inputs are declared synthetic coefficients only. No archive, actual power,
future observation, candidate pool, or topology truth is opened. A protocol
and source fingerprints are written before any calculation. Quadrature
agreement is empirical numerical evidence, not interval certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from continuous import IndependentResiduals, _truncated_fourier, exact_single_probability
from gauss_frequency import gauss_probability


INTERVALS = ((-2., -.3), (.2, 2.5), (-1.5, 2.), (0., math.inf))
FREQUENCIES = (0., .5, 4., 20., 100.)
CELLS = ((1., 0., 1.), (.2, -.8, .6), (-.5, .7, 1.2),
         (.4, .4, .7), (-2., -.3, 0.), (8., 5., .1))
KAPPA = 1.03


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def density(z):
    return math.exp(-z*z/2) / math.sqrt(2*math.pi)


def value(z, c, f, sigma):
    return abs(c + sigma*z) - KAPPA*abs(f + sigma*z)


def interval_mass(left, right):
    return float(ndtr(-left)-ndtr(-right) if left >= 0 else ndtr(right)-ndtr(left))


def root_probability(c, f, sigma, threshold):
    """Reference uses original abs and two monotone root brackets, no A/B code."""
    if sigma == 0:
        return float(abs(c)-KAPPA*abs(f) > threshold)
    maximum = abs(c-f)
    if threshold >= maximum:
        return 0.
    peak = -f/sigma
    width = max(1., (abs(threshold)+abs(c)+KAPPA*abs(f)+1.) / ((KAPPA-1)*sigma))
    fun = lambda z: value(z,c,f,sigma)-threshold
    left, right = peak-width, peak+width
    while fun(left) >= 0:
        width *= 2
        left = peak-width
    while fun(right) >= 0:
        width *= 2
        right = peak+width
    lo = brentq(fun, left, peak, xtol=1e-12)
    hi = brentq(fun, peak, right, xtol=1e-12)
    return interval_mass(lo, hi)


def direct_integral(c, f, sigma, function):
    """Integrate original absolute-value expression over [-12,12]."""
    points = sorted({-12., 12.} | ({r for r in (-c/sigma,-f/sigma) if -12<r<12}
                                  if sigma else set()))
    answer = error = 0.
    for left, right in zip(points[:-1],points[1:]):
        part, estimate = quad(lambda z: function(value(z,c,f,sigma))*density(z),
                              left,right,epsabs=2e-11,epsrel=2e-11,limit=500)
        answer += part
        error += estimate
    return answer,error


class Checks:
    def __init__(self, out):
        self.out,self.rows=out,[]

    def compare(self, name, actual, expected, tolerance, inputs=None, **extra):
        started=time.perf_counter()
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        row=dict(name=name,inputs=inputs,actual=actual,expected=expected,
                 absolute_error=error,tolerance=tolerance,passed=bool(error<=tolerance),**extra)
        row['record_seconds']=time.perf_counter()-started
        self.rows.append(row)
        write(self.out/'checks.json',self.rows)


def base_checks(checks):
    for interval in INTERVALS:
        left,right=interval
        frequencies=np.asarray([FREQUENCIES])
        calculated=_truncated_fourier(np.asarray([left]),np.asarray([right]),frequencies)[0]
        for frequency,actual in zip(FREQUENCIES,calculated):
            if frequency == 0:
                real,error=quad(density,left,right,epsabs=1e-12,limit=500)
                imaginary,imag_error=0.,0.
            else:
                real,error=quad(density,left,right,weight='cos',wvar=frequency,
                                epsabs=2e-11,limit=500,limlst=200)
                imaginary,imag_error=quad(density,left,right,weight='sin',wvar=frequency,
                                          epsabs=2e-11,limit=500,limlst=200)
            checks.compare('scaled_erfcx_vs_real_integral',actual,complex(real,imaginary),2e-9,
                           inputs=dict(left=left,right=right,frequency=frequency),
                           reference_quad_error_estimate=error+imag_error)

    means,variances=[] ,[]
    for c,f,sigma in CELLS:
        model=IndependentResiduals([c],[f],[sigma])
        expected_mean,mean_error=direct_integral(c,f,sigma,lambda d:d)
        expected_second,second_error=direct_integral(c,f,sigma,lambda d:d*d)
        expected_variance=max(0.,expected_second-expected_mean**2)
        means.append(expected_mean);variances.append(expected_variance)
        checks.compare('folded_normal_mean_vs_original_abs_integral',model.sum_mean,expected_mean,2e-9,
                       inputs=dict(coarse=c,fine=f,sigma=sigma),reference_quad_error_estimate=mean_error)
        checks.compare('piecewise_second_moment_vs_original_abs_integral',model.sum_sd**2,expected_variance,2e-8,
                       inputs=dict(coarse=c,fine=f,sigma=sigma),reference_quad_error_estimate=second_error)
        for t in (0.,.5,4.,20.):
            actual=model.characteristic([t],normalization=(0.,1.))[0]
            real,err_real=direct_integral(c,f,sigma,lambda d:math.cos(t*d))
            imag,err_imag=direct_integral(c,f,sigma,lambda d:math.sin(t*d))
            checks.compare('piecewise_cf_vs_original_abs_integral',actual,complex(real,imag),3e-9,
                           inputs=dict(coarse=c,fine=f,sigma=sigma,frequency=t),
                           reference_quad_error_estimate=err_real+err_imag,
                           reference_normal_tail_probability=2*float(ndtr(-12.)))
        for threshold in (-.2,0.,.2):
            actual=exact_single_probability(c,f,sigma,threshold)
            expected=root_probability(c,f,sigma,threshold)
            checks.compare('single_event_piecewise_vs_original_abs_root',actual,expected,2e-11,
                           inputs=dict(coarse=c,fine=f,sigma=sigma,threshold=threshold))

    c=np.array([row[0] for row in CELLS]);f=np.array([row[1] for row in CELLS])
    sigma=np.array([row[2] for row in CELLS]);model=IndependentResiduals(c,f,sigma)
    checks.compare('sum_mean_from_independent_local_integrals',model.sum_mean,sum(means),2e-8)
    checks.compare('sum_variance_from_independent_local_integrals',model.sum_sd**2,sum(variances),2e-8)
    frequencies=np.array([-20.,-4.,-.5,0.,.5,4.,20.])
    cf=model.characteristic(frequencies,normalization=(0.,1.))
    checks.compare('cf_at_zero',cf[3],1.+0j,2e-13)
    checks.compare('cf_conjugate_symmetry',cf,np.conj(cf[::-1]),2e-13)
    checks.compare('cf_unit_modulus_bound',max(0.,float(np.abs(cf).max())-1.),0.,2e-13)
    center,scale=model.sum_mean+.17,model.sum_sd*1.3
    actual=model.characteristic(frequencies,normalization=(center,scale))
    product=np.ones(frequencies.size,complex)
    for a,b,s in CELLS:
        product*=IndependentResiduals([a],[b],[s]).characteristic(frequencies/scale,normalization=(0.,1.))
    expected=np.exp(-1j*frequencies*center/scale)*product
    checks.compare('fixed_normalization_is_exact_affine_change',actual,expected,3e-13,
                   inputs=dict(center=center,scale=scale,frequency=frequencies))

    c=np.array([.5,.8]);f=np.array([0.,.2]);sigma=np.array([1.,.7]);threshold=0.
    def nested(z):
        remaining=2*threshold-value(z,c[0],f[0],sigma[0])
        return root_probability(c[1],f[1],sigma[1],remaining)*density(z)
    expected,error=quad(nested,-10.,10.,points=[-.5,0.],epsabs=2e-9,limit=800)
    model=IndependentResiduals(c,f,sigma)
    actual=model.probability(threshold,cutoff=1024.,step=.05,refinements=1)
    checks.compare('two_cell_fourier_inversion_vs_nested_scalar_roots',actual['probability'],expected,1e-5,
                   inputs=dict(coarse=c,fine=f,sigma=sigma,threshold=threshold),
                   direct_quad_error_estimate=error,actual_probability_diagnostics=actual,
                   certification='empirical agreement; analytic tail bound separate')
    checks.compare('deterministic_strict_threshold',IndependentResiduals([1.],[0.],[0.]).probability(1.)['probability'],0.,0.)


def adjoint_checks(checks):
    from full_adjoint import probability_with_adjoint
    c=np.array([.5,.8,-.3]);f=np.array([0.,.2,.4]);sigma=np.array([1.,.7,.9])
    model=IndependentResiduals(c,f,sigma)
    norm=(model.sum_mean+.12,model.sum_sd)
    options=dict(cutoff=128.,order=64,refinements=0,normalization=norm)
    result=probability_with_adjoint(model,0.,**options)
    original=gauss_probability(model,0.,**options)
    checks.compare('coefficient_adjoint_preserves_original_finite_probability',result['probability'],original['probability'],2e-13)
    for which,gradient in [('coarse',result['gradient_coarse']),('fine',result['gradient_fine'])]:
        for j in range(c.size):
            step=1e-6
            left_c,right_c=c.copy(),c.copy();left_f,right_f=f.copy(),f.copy()
            if which=='coarse':left_c[j]-=step;right_c[j]+=step
            else:left_f[j]-=step;right_f[j]+=step
            left=gauss_probability(IndependentResiduals(left_c,left_f,sigma),0.,**options)['probability']
            right=gauss_probability(IndependentResiduals(right_c,right_f,sigma),0.,**options)['probability']
            checks.compare('coefficient_adjoint_vs_fixed_quadrature_difference',gradient[j],(right-left)/(2*step),2e-7,
                           inputs=dict(candidate=which,cell=j,step=step),
                           scope='derivative of same finite quadrature, not infinite-integral derivative certification')


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'complete.json').exists():
        print('Already complete',flush=True);return
    lock=args.out/'running.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(__import__('os').getpid()))
    try:
        sources=[Path(__file__),HERE/'continuous.py',HERE/'gauss_frequency.py']
        if args.stage in ('adjoint','all'):sources.append(HERE/'full_adjoint.py')
        request=dict(stage=args.stage,intervals=INTERVALS,frequencies=FREQUENCIES,cells=CELLS,kappa=KAPPA,
                     python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                     sources={p.name:sha(p) for p in sources},synthetic_only=True,
                     no_topology_or_future_inputs=True,reference_truncation=12,
                     numeric_scope='float64 comparisons; explicit tolerances; not interval-certified')
        path=args.out/'request.json'
        if path.exists() and json.loads(path.read_text())!=plain(request):
            raise ValueError('Existing verification stage has different source/configuration')
        write(path,request)
        checks=Checks(args.out);started=time.perf_counter()
        if args.stage in ('base','all'):base_checks(checks)
        if args.stage in ('adjoint','all'):adjoint_checks(checks)
        summary=dict(checks=len(checks.rows),passed=sum(row['passed'] for row in checks.rows),
                     failed=sum(not row['passed'] for row in checks.rows),seconds=time.perf_counter()-started,
                     max_absolute_error=max((row['absolute_error'] for row in checks.rows),default=0.),
                     request_sha256=sha(path),checks_sha256=sha(args.out/'checks.json'),
                     scope=request['numeric_scope'])
        write(args.out/'summary.json',summary)
        write(args.out/'complete.json',dict(status='complete' if summary['failed']==0 else 'failed_checks',
                                           summary_sha256=sha(args.out/'summary.json')))
        print(json.dumps(summary,ensure_ascii=True),flush=True)
        if summary['failed']:raise SystemExit(1)
    except Exception as exc:
        write(args.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=('base','adjoint','all'),default='base')
    main(parser.parse_args())
