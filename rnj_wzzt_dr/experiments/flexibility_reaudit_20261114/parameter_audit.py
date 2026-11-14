"""Bounded audit of two frozen support families; no candidate-bank mutation.

Every new coefficient estimate uses only the original measured training data.
All fits are saved before validation, new-day, physical-line, or action arrays
are loaded for offline evaluation. No new identifier, AC, or DFS solve occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import lsq_linear

from ... import io
from ...observations import Observation, load, fingerprint
from ...identification import optimization
from ...identification.optimization import QPFitter, feature_matrices, response_matrices

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = ROOT/'outputs/measured_pq_pengwah_enriched_20261114_v1/qualified_enriched_input.json'
DEFAULT_RUN = ROOT/'outputs/reactive_flex_confirmation_20261114_v1'
DEFAULT_OUT = ROOT/'outputs/flexibility_parameter_audit_20261114_v1'
VARIANTS = ('native_replay', 'trf_same_bounds', 'signed_box_diagnostic',
            'positive_X_floor_001', 'first_training_day', 'second_training_day')


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, data):
    io.write(path, clean(data))


def checked(reference):
    path = Path(reference['path'])
    if io.ref(path)['sha256'] != reference['sha256']:
        raise ValueError(f'Changed source: {path}')
    return path


def subset(data, start, stop):
    return Observation(data.p[start:stop], data.q[start:stop], data.y[start:stop], data.labels)


def independent_design(data, family):
    """Explicit output/time/support construction, independent of native helper."""
    columns = []
    for values in (data.p, data.q):
        for support in family:
            col = np.zeros_like(data.y)
            col[:, support] = values[:, support].sum(axis=1)[:, None]
            columns.append(col.ravel())
    result = np.stack(columns, axis=1)
    native = np.column_stack(feature_matrices(data, family))
    if not np.array_equal(result, native):
        if np.max(np.abs(result-native)) > 1e-14:
            raise ArithmeticError('Independent support design mismatch')
    return result


def matrix(point, n):
    r, x = response_matrices(point, n)
    return np.c_[r, x]


def box_certificate(A, y, theta, lower, upper):
    Al, yl, tl = A.astype(np.longdouble), y.astype(np.longdouble), theta.astype(np.longdouble)
    residual = Al@tl-yl
    value = float(np.mean(residual**2))
    gradient = np.asarray(2*Al.T@residual/len(y), float)
    bound = max(0., value+float(np.minimum(gradient*(lower-theta), gradient*(upper-theta)).sum()))
    violation = max(0., float(np.max(lower-theta)), float(np.max(theta-upper)))
    lower_active = theta <= lower+1e-10
    upper_active = theta >= upper-1e-10
    projected = gradient.copy()
    projected[lower_active] = np.minimum(gradient[lower_active], 0.)
    projected[upper_active] = np.maximum(gradient[upper_active], 0.)
    return dict(objective=value, supporting_plane_lower=bound, gap=value-bound,
                maximum_bound_violation=violation, projected_gradient_inf=float(np.max(np.abs(projected))),
                gradient=gradient, lower_active=lower_active, upper_active=upper_active)


def fit_variant(data, family, reference, bound, variant):
    """Estimator boundary: measured history, family, fixed numerical settings only."""
    started = perf_counter()
    ncoef = 2*len(family)
    lower, upper = np.zeros(ncoef), np.full(ncoef, bound)
    estimate_data = data
    if variant == 'first_training_day': estimate_data = subset(data, 0, 96)
    if variant == 'second_training_day': estimate_data = subset(data, 96, 192)
    F = independent_design(estimate_data, family)
    A, y = F/np.sqrt(reference), estimate_data.y.ravel()/np.sqrt(reference)
    if variant in ('native_replay', 'first_training_day', 'second_training_day'):
        fitter = QPFitter(estimate_data, reference_mse=reference, bound=bound, qp_seconds=30.)
        point = fitter.fit(family)
        theta = np.r_[point['r'], point['x']]
        receipt = fitter.records[-1]
    else:
        method = 'trf' if variant == 'trf_same_bounds' else 'bvls'
        if variant == 'signed_box_diagnostic': lower[:] = -bound
        if variant == 'positive_X_floor_001':
            for k, support in enumerate(family):
                if len(support) < data.n: lower[len(family)+k] = .001
        solved = lsq_linear(A, y, bounds=(lower, upper), method=method,
                            lsq_solver='exact', tol=1e-12, max_iter=5000)
        theta = solved.x
        if not solved.success:
            raise RuntimeError(f'{variant}: {solved.message}')
        point = dict(supports=family, r=theta[:len(family)].tolist(), x=theta[len(family):].tolist())
        receipt = dict(solver=f'SciPy {method}', status=int(solved.status), message=solved.message,
                       iterations=int(solved.nit), success=bool(solved.success))
    certificate = box_certificate(A, y, theta, lower, upper)
    if certificate['maximum_bound_violation'] > 1e-9:
        raise ArithmeticError('Returned point violates its declared box')
    if certificate['gap'] > 1e-6*(1+certificate['objective']):
        raise ArithmeticError('Returned point lacks the declared convex objective certificate')
    return dict(status='complete', variant=variant, point=point, receipt=receipt,
                certificate=certificate, seconds=perf_counter()-started,
                training_rows=estimate_data.points, training_fingerprint=fingerprint(estimate_data),
                lower=lower, upper=upper,
                negative_R=int((theta[:len(family)] < -1e-9).sum()),
                negative_X=int((theta[len(family):] < -1e-9).sum()),
                zero_R=int((np.abs(theta[:len(family)]) <= 1e-9).sum()),
                zero_X=int((np.abs(theta[len(family):]) <= 1e-9).sum()),
                maximum_coefficient=float(np.max(np.abs(theta))))


def spectrum(A):
    sv = np.linalg.svd(A, compute_uv=False)
    norms = np.linalg.norm(A, axis=0)
    normalized = A/np.maximum(norms, 1e-300)
    ns = np.linalg.svd(normalized, compute_uv=False)
    return dict(shape=A.shape, rank=int(np.linalg.matrix_rank(A)), singular_values=sv,
                condition=float(sv[0]/sv[-1]), column_normalized_condition=float(ns[0]/ns[-1]),
                column_norms=norms, smallest_over_largest=float(sv[-1]/sv[0]))


def design_audit(data, family):
    F = independent_design(data, family)
    size = len(family)
    U, sv, Vt = np.linalg.svd(F, full_matrices=False)
    inv_diagonal = np.sum((Vt.T/sv)**2, axis=1)
    fractions = 1/np.sqrt(inv_diagonal)/np.linalg.norm(F, axis=0)
    correlations = []
    for k, support in enumerate(family):
        a, b = data.p[:, support].sum(axis=1), data.q[:, support].sum(axis=1)
        ac, bc = a-a.mean(), b-b.mean()
        correlations.append(dict(support=support, node_labels=[data.labels[i] for i in support],
            PQ_cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            PQ_centered_correlation=float(ac@bc/(np.linalg.norm(ac)*np.linalg.norm(bc))),
            P_rms=float(np.sqrt(np.mean(a*a))), Q_rms=float(np.sqrt(np.mean(b*b))),
            P_std=float(a.std()), Q_std=float(b.std()),
            R_conditional_feature_fraction=float(fractions[k]),
            X_conditional_feature_fraction=float(fractions[size+k])))
    centered = F.reshape(data.points, data.n, 2*size)
    centered = (centered-centered.mean(axis=0, keepdims=True)).reshape(-1, 2*size)
    return dict(raw=spectrum(F), within_output_time_centered=spectrum(centered),
                support_diagnostics=correlations,
                interpretation='Full rank tests uniqueness in the chosen least-squares model; it is not a parameter accuracy guarantee. Conditional feature fraction removes all other support/channel columns.')


def offline_physical(entry, labels):
    # Called only after all history-only estimates have been sealed.
    from topology_identification_standalone import data as backend
    from ..measured_pq_20261114.physical import _network_hash
    network = backend.load_case(entry['case'])
    if _network_hash(network) != entry['physical_cache']['network_hash']:
        raise ValueError('Physical reference changed since saved execution')
    order, parent, incoming = backend._radial_order(network)
    descendant = {bus: set() for bus in order}
    for k, label in enumerate(labels): descendant[label].add(k)
    coefficients = {}
    H = np.zeros((len(labels), 2*len(labels)))
    zbase = network['base_kv']**2/network['base_mva']
    for bus in reversed(order[1:]):
        support = tuple(sorted(descendant[bus])); edge = incoming[bus]
        vals = np.array([2*edge['r_ohm']/zbase, 2*edge['x_ohm']/zbase])
        coefficients[support] = coefficients.get(support, np.zeros(2))+vals
        z = np.array([i in support for i in range(len(labels))], float)
        H[:, :len(labels)] += vals[0]*np.outer(z, z)
        H[:, len(labels):] += vals[1]*np.outer(z, z)
        descendant[parent[bus]].update(descendant[bus])
    return H, coefficients, dict(base_kw=network['base_kw'], base_kv=network['base_kv'],
                                 base_mva=network['base_mva'], impedance_base_ohm=zbase,
                                 coefficient_to_ohm_factor=zbase/2,
                                 original_network_sha256=_network_hash(network))


def error_ratio(predicted, actual):
    denominator = np.linalg.norm(actual)
    return float(np.linalg.norm(predicted-actual)/denominator) if denominator > 1e-30 else None


def action_arrays(run):
    results = []
    for path in sorted(run.glob('fraction*/*/forecast*/results.json')):
        for row in io.read(path):
            if row['resource_variant'] != 'eligible_sites' or row['strategy'] not in ('old_gamma2','symmetric_guard'):
                continue
            if row['status'] != 'complete':
                results.append(dict(status=row['status'], source=io.ref(path)))
                continue
            folder = path.parent/f"execution{row['repeat']:02d}"
            def observed(strategy):
                source = folder/strategy/'observations.npz'
                with np.load(source, allow_pickle=False) as z:
                    x = np.c_[z['true_P'], z['true_Q']]
                    y = z['true_root'][:, None]**2-z['true_V']**2
                return x, y, io.ref(source)
            x0, y0, ref0 = observed('no_dfs'); x1, y1, ref1 = observed(row['strategy'])
            results.append(dict(status='complete', strategy=row['strategy'], forecast=row['forecast_index'],
                                repeat=row['repeat'], dx=x1-x0, dy=y1-y0, source=ref1, baseline=ref0))
    return results


def run(index, execution, out):
    if out.exists(): raise FileExistsError('Never overwrite a completed or partial audit')
    out.mkdir(parents=True)
    index_ref = io.ref(index); source_index = io.read(index)
    if source_index['input_count'] != 1: raise ValueError('This audit is locked to one Pengwah history')
    entry = source_index['inputs'][0]
    native_path = checked(entry['files']['101_native.json'])
    train_path = checked(entry['files']['train.npz'])
    native, train = io.read(native_path), load(train_path)
    reference, bound = native['reference_mse'], native['config']['response_bound']
    sources = [io.ref(__file__), io.ref(optimization.__file__),
               io.ref(Path(optimization.__file__).with_name('algorithm.py')),
               io.ref(Path(optimization.__file__).with_name('trees.py')),
               io.ref(Path(optimization.__file__).parents[1]/'observations.py')]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), index=index_ref,
        input=dict(native=io.ref(native_path), train=io.ref(train_path)), source=sources,
        environment=io.environment(), scipy_version=scipy.__version__, variants=VARIANTS,
        families=[2,4], original_reference_mse=reference, original_coefficient_bound=bound,
        positive_X_floor=.001, floor_scope='all non-universal supports; diagnostic, not proposed deployment bound',
        numerical_limit=dict(tol=1e-12, max_iter=5000, native_soft_seconds=30.),
        estimation_information='Original measured 192 training points and frozen supports only; each chronological day uses its own original 96 points. No truth, validation, future input, or executed response enters any new fit.',
        evaluation='Previously observed validation/new day and already completed confirmation actions. No new AC/DFS/identifier run. Exploratory diagnostics after seeing the original failure.',
        parameter_regularizer='None in native fixed-family QP; structural alpha adds a constant for a fixed family.',
        no_candidate_mutation=True)
    write(out/'protocol.json', protocol)
    for source in sources:
        (out/'source').mkdir(exist_ok=True)
        shutil.copy2(source['path'], out/'source'/Path(source['path']).name)
    fits, designs, failures = {}, {}, []
    for native_id in (2,4):
        point = native['frozen_candidates'][native_id]['point']; family = point['supports']
        designs[native_id] = design_audit(train, family)
        fits[native_id] = {}
        for variant in VARIANTS:
            try:
                fitted = fit_variant(train, family, reference, bound, variant)
                fitted['coefficient_change_from_frozen_max'] = float(np.max(np.abs(
                    np.r_[fitted['point']['r'],fitted['point']['x']]-np.r_[point['r'],point['x']])))
                fits[native_id][variant] = fitted
            except Exception as exc:
                fits[native_id][variant] = dict(status='failed', error_type=type(exc).__name__, error=str(exc))
                failures.append(dict(native=native_id, variant=variant, error=str(exc)))
            write(out/'progress.json', dict(phase='history_only_fitting', native=native_id, variant=variant,
                                            completed=sum(len(v) for v in fits.values()), total=12, failures=failures))
            print('history fit', native_id, variant, fits[native_id][variant]['status'], flush=True)
    write(out/'fitted_history_only.json', dict(protocol=io.ref(out/'protocol.json'), fits=fits, designs=designs))
    fitted_ref = io.ref(out/'fitted_history_only.json')
    # The estimation phase has ended: load evaluation-only sources now.
    obs = dict(train=train, validation=load(checked(entry['files']['validation.npz'])),
               new_day=load(checked(entry['files']['new_day.npz'])))
    eval_entries = io.read(execution/'inputs.json')
    physical_entry = next(e for e in eval_entries if e['resource_variant']=='eligible_sites')
    if checked(physical_entry['score']).name != 'score.json': raise ValueError('Unexpected execution score')
    physical, true_coeff, units = offline_physical(physical_entry, train.labels)
    actions = action_arrays(execution)
    coefficients, evaluation = {}, {}
    for native_id in (2,4):
        family = native['frozen_candidates'][native_id]['point']['supports']
        coefficients[native_id] = [dict(support=c, labels=[train.labels[k] for k in c],
            offline_physical=true_coeff.get(tuple(c),np.zeros(2)),
            fitted={variant:[f['point']['r'][j],f['point']['x'][j]] for variant,f in fits[native_id].items() if f['status']=='complete'})
            for j,c in enumerate(family)]
        evaluation[native_id] = {}
        for variant, fit in fits[native_id].items():
            if fit['status'] != 'complete': continue
            H = matrix(fit['point'], train.n)
            losses = {name:float(np.mean((data.y-np.c_[data.p,data.q]@H.T)**2)) for name,data in obs.items()}
            response = {}
            for strategy in ('old_gamma2','symmetric_guard'):
                selected = [a for a in actions if a['status']=='complete' and a['strategy']==strategy]
                errors = [error_ratio(a['dx']@H.T,a['dy']) for a in selected]
                pred = np.concatenate([a['dx']@H.T for a in selected]); actual=np.concatenate([a['dy'] for a in selected])
                response[strategy] = dict(paths=len(selected), mean_path_relative_error=float(np.mean(errors)),
                                          joint_relative_error=error_ratio(pred,actual),
                                          min_path_relative_error=min(errors), max_path_relative_error=max(errors))
            n=train.n
            node_response = {str(label):dict(P=error_ratio(H[:,k],physical[:,k]),
                                             Q=error_ratio(H[:,n+k],physical[:,n+k])) for k,label in enumerate(train.labels)}
            evaluation[native_id][variant] = dict(observed_mse=losses, action_response=response,
                physical_linear_R_error=error_ratio(H[:,:n],physical[:,:n]),
                physical_linear_X_error=error_ratio(H[:,n:],physical[:,n:]),
                node_response_relative_error=node_response)
    original_receipts = {}
    for native_id in (2,4):
        family=native['frozen_candidates'][native_id]['point']['supports']
        original_receipts[native_id] = [r for r in native['QP'] if r['family']==family]
    search = checked(entry['candidate_enrichment']['source_search'])
    original_receipts[4] += io.read(search)['QP']
    eval_sources = [entry['files']['validation.npz'],entry['files']['new_day.npz'],
                   io.ref(execution/'inputs.json'),io.ref(execution/'protocol.json'),io.ref(execution/'complete.json'),io.ref(search)]
    if not all(io.ref(r['path'])['sha256']==r['sha256'] for r in [*sources, index_ref, *protocol['input'].values(),*eval_sources]):
        raise RuntimeError('A source changed during the audit')
    if io.ref(out/'fitted_history_only.json') != fitted_ref:
        raise RuntimeError('Sealed history estimates changed during evaluation')
    physical_action = {strategy:dict(paths=sum(a['status']=='complete' and a['strategy']==strategy for a in actions),
        mean_path_relative_error=float(np.mean([error_ratio(a['dx']@physical.T,a['dy']) for a in actions if a['status']=='complete' and a['strategy']==strategy])))
        for strategy in ('old_gamma2','symmetric_guard')}
    report=dict(protocol=io.ref(out/'protocol.json'), fitted_history_only=fitted_ref, units=units,
        designs=designs, fits=fits, original_QP_receipts=original_receipts, evaluation=evaluation,
        coefficient_comparison=coefficients, evaluation_sources=eval_sources,
        physical_linear_action_error=physical_action,
        executed_paths=[{k:v for k,v in a.items() if k not in ('dx','dy')} for a in actions],
        failures=failures, source_unchanged=True, candidate_bank_unchanged=True,
        scope='Convex coefficient fit audit at two already-frozen support families; neither a new topology search nor a validated replacement estimator.')
    write(out/'REPORT.json',report)
    write(out/'complete.json',dict(status='complete',planned_fits=12,completed_fits=12-len(failures),
        failed_fits=len(failures), failed_action_records=sum(a['status']!='complete' for a in actions),
        report=io.ref(out/'REPORT.json'), source_unchanged=True))
    print(json.dumps(dict(output=str(out),fits=12-len(failures),failures=failures)),flush=True)
    return report


def run_ridge(index, execution, out):
    """Separate, fully retained raw-coefficient ridge exploration.

    The completed first audit and its source snapshot are never overwritten.
    Lambda is a numerical response-coefficient penalty, not a physical R/X prior.
    """
    if out.exists(): raise FileExistsError('Preserve every ridge supplement')
    out.mkdir(parents=True)
    source_index=io.read(index); entry=source_index['inputs'][0]
    native_path=checked(entry['files']['101_native.json']); train_path=checked(entry['files']['train.npz'])
    native,train=io.read(native_path),load(train_path)
    reference,bound=native['reference_mse'],native['config']['response_bound']
    lambdas=(.01,.1,1.)
    sources=[io.ref(__file__),io.ref(optimization.__file__)]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),index=io.ref(index),
        source=sources,environment=io.environment(),input=dict(native=io.ref(native_path),train=io.ref(train_path)),
        family_indices=[2,4],lambdas=lambdas,original_reference_mse=reference,coefficient_bound=bound,
        objective='MSE_train/reference_mse + lambda*sum(all support response r^2+x^2)',
        rationale='Predeclared six-fit sensitivity after the original failure; not a selected replacement estimator.',
        scaling='Raw response coefficients in the current 200kVA/0.4kV per-unit model. The penalty depends on parameterization and support scaling; it encodes no measured line prior.',
        information='Original measured 192 training points only. No validation, executed action, physical line, or new-day values enter estimation or lambda choice.',
        evaluation='Old validation/new-day and completed confirmation actions, after all six coefficient vectors are saved.',
        no_RX_ratio_fixed=True,no_candidate_mutation=True,no_new_AC=True,no_DFS_solve=True)
    write(out/'protocol.json',protocol)
    (out/'source').mkdir()
    for source in sources:shutil.copy2(source['path'],out/'source'/Path(source['path']).name)
    fits,failures={},[]
    for native_id in (2,4):
        family=native['frozen_candidates'][native_id]['point']['supports']; size=len(family)
        F=independent_design(train,family); A=F/np.sqrt(train.y.size*reference)
        y=train.y.ravel()/np.sqrt(train.y.size*reference); fits[native_id]={}
        for penalty in lambdas:
            name=f'ridge_{penalty:g}';started=perf_counter()
            try:
                stacked=np.vstack([A,np.sqrt(penalty)*np.eye(2*size)])
                rhs=np.r_[y,np.zeros(2*size)]
                solved=lsq_linear(stacked,rhs,bounds=(0.,bound),method='bvls',tol=1e-12,max_iter=5000)
                if not solved.success:raise RuntimeError(solved.message)
                theta=solved.x; residual=A@theta-y
                mse_part=float(residual@residual);regularizer=float(penalty*(theta@theta))
                objective=mse_part+regularizer
                gradient=2*(A.T@residual+penalty*theta)
                lower=max(0.,objective+float(np.minimum(-gradient*theta,gradient*(bound-theta)).sum()))
                gap=objective-lower
                if gap>1e-6*(1+objective):raise ArithmeticError('Ridge convex certificate gap too large')
                fits[native_id][name]=dict(status='complete',lambda_value=penalty,
                    point=dict(supports=family,r=theta[:size],x=theta[size:]),
                    original_normalized_training_mse=mse_part,penalty_contribution=regularizer,
                    ridge_objective=objective,supporting_plane_lower=lower,certificate_gap=gap,
                    solver='SciPy BVLS on augmented least-squares design',solver_status=int(solved.status),
                    iterations=int(solved.nit),seconds=perf_counter()-started,
                    zero_R=int((theta[:size]<=1e-9).sum()),zero_X=int((theta[size:]<=1e-9).sum()),
                    maximum_coefficient=float(theta.max()))
            except Exception as exc:
                fits[native_id][name]=dict(status='failed',error_type=type(exc).__name__,error=str(exc))
                failures.append(dict(native=native_id,variant=name,error=str(exc)))
            write(out/'progress.json',dict(phase='history_only_ridge',completed=sum(len(x) for x in fits.values()),total=6,failures=failures))
            print('history ridge',native_id,name,fits[native_id][name]['status'],flush=True)
    write(out/'fitted_history_only.json',dict(protocol=io.ref(out/'protocol.json'),fits=fits))
    fitted_ref=io.ref(out/'fitted_history_only.json')
    obs=dict(train=train,validation=load(checked(entry['files']['validation.npz'])),new_day=load(checked(entry['files']['new_day.npz'])))
    physical_entry=next(e for e in io.read(execution/'inputs.json') if e['resource_variant']=='eligible_sites')
    physical,true_coeff,units=offline_physical(physical_entry,train.labels)
    actions=action_arrays(execution);evaluation={}
    for native_id,variants in fits.items():
        evaluation[native_id]={}
        for name,fit in variants.items():
            if fit['status']!='complete':continue
            H=matrix(fit['point'],train.n);n=train.n
            losses={key:float(np.mean((data.y-np.c_[data.p,data.q]@H.T)**2)) for key,data in obs.items()}
            response={}
            for strategy in ('old_gamma2','symmetric_guard'):
                selected=[a for a in actions if a['status']=='complete' and a['strategy']==strategy]
                errors=[error_ratio(a['dx']@H.T,a['dy']) for a in selected]
                response[strategy]=dict(paths=len(selected),mean_path_relative_error=float(np.mean(errors)),
                    min_path_relative_error=min(errors),max_path_relative_error=max(errors))
            evaluation[native_id][name]=dict(observed_mse=losses,action_response=response,
                physical_linear_R_error=error_ratio(H[:,:n],physical[:,:n]),
                physical_linear_X_error=error_ratio(H[:,n:],physical[:,n:]),
                node_response_relative_error={str(label):dict(P=error_ratio(H[:,k],physical[:,k]),Q=error_ratio(H[:,n+k],physical[:,n+k])) for k,label in enumerate(train.labels)})
    if not all(io.ref(r['path'])['sha256']==r['sha256'] for r in [*sources,*protocol['input'].values()]):
        raise RuntimeError('Changed ridge source')
    if io.ref(out/'fitted_history_only.json')!=fitted_ref:raise RuntimeError('Changed sealed estimates')
    write(out/'REPORT.json',dict(protocol=io.ref(out/'protocol.json'),fits=fits,evaluation=evaluation,
        fitted_history_only=fitted_ref,units=units,failures=failures,source_unchanged=True,
        evaluation_sources=[entry['files']['validation.npz'],entry['files']['new_day.npz'],io.ref(execution/'inputs.json'),io.ref(execution/'complete.json')],
        executed_paths=[{k:v for k,v in a.items() if k not in ('dx','dy')} for a in actions],
        interpretation='All lambda values retained; future-action error is offline diagnosis, not a selection score available to deployment.'))
    write(out/'complete.json',dict(status='complete',planned_fits=6,completed_fits=6-len(failures),failed_fits=len(failures),
        report=io.ref(out/'REPORT.json'),source_unchanged=True))
    print(json.dumps(dict(output=str(out),fits=6-len(failures),failures=failures)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=DEFAULT_INDEX)
    parser.add_argument('--execution',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--ridge',action='store_true',help='Run the separate fixed three-lambda supplement under out/ridge_supplement')
    args=parser.parse_args()
    if args.ridge:run_ridge(args.index.resolve(),args.execution.resolve(),args.out.resolve()/'ridge_supplement')
    else:run(args.index.resolve(),args.execution.resolve(),args.out.resolve())
