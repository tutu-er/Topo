"""Toy checks of Gaussian paired refit moments; no grid data or selector."""
from pathlib import Path
import argparse,hashlib,json,math,platform
import numpy as np


def moments(ca,Aa,cb,Ab,E):
    m=len(ca);eye=np.eye(m)
    Pa=eye-Aa@np.linalg.pinv(Aa,rcond=1e-12) if Aa.shape[1] else eye
    Pb=eye-Ab@np.linalg.pinv(Ab,rcond=1e-12) if Ab.shape[1] else eye
    d=float(cb@Pb@cb-ca@Pa@ca);h=E.T@(Pa@ca-Pb@cb);G=E.T@(Pb-Pa)@E
    return dict(noiseless_difference=d,trace_correction=float(np.trace(G)),
        mean=d+float(np.trace(G)),variance=float(4*h@h+2*np.trace(G@G)))


def refit_cost(c,A,noise):
    residual=c[:,None]-noise
    if A.shape[1]:residual=residual-A@np.linalg.lstsq(A,residual,rcond=1e-12)[0]
    return np.sum(residual**2,axis=0)


def compare(sample,mu,var):
    n=len(sample);mean=float(np.mean(sample));variance=float(np.var(sample))
    se_mean=math.sqrt(var/n)
    fourth=float(np.mean((sample-mu)**4));se_var=math.sqrt(max(0.,fourth-var**2)/n)
    ok=abs(mean-mu)<=8*se_mean+1e-10 and abs(variance-var)<=8*se_var+1e-10
    return dict(passed=ok,samples=n,expected_mean=mu,expected_variance=var,
        measured_mean=mean,measured_variance=variance,mean_standard_error=se_mean,
        variance_standard_error=se_var,tolerance_standard_errors=8)


def run(samples=200000,seed=20261015):
    e=np.random.default_rng(seed).normal(size=(2,samples));zero2=np.zeros(2);empty=np.empty((2,0))
    records=[]
    for delta in (0.,.5,2.):
        for repeated in (False,True):
            A=np.array([[0.,0.],[1.,1.]]) if repeated else np.array([[0.],[1.]])
            c=np.array([delta,0.]);formula=moments(zero2,empty,c,A,np.eye(2))
            assert np.isclose(formula['mean'],delta**2-1) and np.isclose(formula['variance'],4*delta**2+2)
            actual=refit_cost(c,A,e)-refit_cost(zero2,empty,e)
            records.append(dict(case='extra_future_freedom',delta=delta,duplicate_column=repeated,
                formula=formula,check=compare(actual,delta**2-1,4*delta**2+2)))
    Aa=np.array([[1.],[0.]]);Ab=np.array([[0.],[1.]]);E=Ab;cb=np.array([.5,0.])
    formula=moments(zero2,Aa,cb,Ab,E);noise=E@e[:1]
    actual=refit_cost(cb,Ab,noise)-refit_cost(zero2,Aa,noise)
    records.append(dict(case='fixed_history_equal_parameter_counts',formula=formula,
        check=compare(actual,-.75,2.)))
    actual=-(e[0]**2)*(e[0]>=0)
    records.append(dict(case='nonnegative_boundary_counterexample',
        unbounded_prediction=dict(mean=-1.,variance=2.),check=compare(actual,-.5,1.25)))
    delta=.5;cb=np.array([delta,0.]);actual=refit_cost(cb,empty,e)-refit_cost(zero2,empty,e)
    formula=moments(zero2,empty,cb,empty,np.eye(2));p=.5*(1+math.erf(delta/(2*math.sqrt(2))))
    measured=float(np.mean(actual>0));pse=math.sqrt(p*(1-p)/samples)
    probability_check=abs(measured-p)<=8*pse
    records.append(dict(case='fixed_means_normal_difference',formula=formula,
        check=compare(actual,delta**2,4*delta**2),model_a_generates_probability=p,
        measured_fraction=measured,fraction_standard_error=pse,probability_check=probability_check))
    return dict(success=all(r['check']['passed'] for r in records) and probability_check,
        cases=records,seed=seed,samples_per_case=samples,shared_noise_across_cases=True,
        independent_grid_histories=0,AC_calls=0,selector_calls=0,grid_records_read=0,
        numpy=np.__version__,python=platform.python_version(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='unbounded frozen linear toys and an explicit boundary counterexample; no new EV strategy')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('new evidence file required')
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(success=result['success'],toy_cases=len(result['cases']),samples_per_case=result['samples_per_case'])))
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
