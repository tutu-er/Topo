"""Training-only full-AC assembly for a frozen shared-line tangent proxy.

Reference lines are immutable. Competitor line changes are shared across all
times. P/Q inputs have no fitted columns. Free tangent projection is a linear
relaxation; operation bounds and full AC acceptance are separate.
"""
from time import perf_counter
import numpy as np
from . import ev_fixed_train_design as fixed
from .symmetric_ac_local import mean_and_jacobian, whiten_observations


def source_columns(T,n):
    return np.array([t*(2*n+2)+2*n+j for t in range(T) for j in range(2)])


def geometry_inputs(context, parent_kw):
    """Return primitive a,B,J per pair, using the same H/E nuisance columns.

    Residual rows use H channel-major followed by E channel-major; controls
    flatten time-major. Lines use physical all-R/all-X coordinates.
    """
    began = perf_counter()
    if not context.get('success'):
        raise ValueError('successful training-only context required')
    args = context['_rebuild_inputs']; Hobs = args['training']; tm=args['train_meter']
    T,n=context['control_shape']; u=np.asarray(parent_kw,float)
    if u.shape!=(T,n) or not np.isfinite(u).all():raise ValueError('matching parent power')
    H=Hobs.t; observed=whiten_observations(Hobs,tm,fixed.CHANNELS)
    backgroundH=dict(p=Hobs.p-fixed._broadcast(tm.bias_p,Hobs.p.shape,'P bias'),
        q=Hobs.q-fixed._broadcast(tm.bias_q,Hobs.q.shape,'Q bias'))
    totalE=dict(p=context['background']['p']+u/context['base_kw'],q=context['background']['q'])
    Hgeo={}; Egeo={}; nuisance={}; hist_residual={}; diagnostics={}
    cc=np.array([t*(2*n+2)+j for t in range(T) for j in range(n)])
    for b in context['bank']:
        name=b.name; fit=context['train_fits'][name]; line=context['lines'][name]
        if fixed._line_id(fit.r,fit.x)!=line['sha256']:raise ValueError('training lines changed')
        gh=mean_and_jacobian(b,line['r'],line['x'],backgroundH['p'],backgroundH['q'],
            fit.source_p,fit.source_q,Hobs.root_v,tm,channels=fixed.CHANNELS)
        fs=context['forecasts'][name]
        ge=fixed._geometry(b,line,totalE,context['root_v'],context['future_meter'],
            fixed._source_vector(fs['source_p'],fs['source_q']))
        nh,ne=len(gh['mean']),len(ge['mean']); m=b.m
        J=np.zeros((nh+ne,2*m+2*H+2*T))
        J[:nh,:2*m]=gh['line_jacobian']; J[nh:,:2*m]=ge['line_jacobian']
        J[:nh,2*m:2*m+2*H]=gh['state_jacobian'][:,source_columns(H,n)]
        J[nh:,2*m+2*H:]=ge['state_jacobian'][:,source_columns(T,n)]
        Hgeo[name]=gh;Egeo[name]=ge;nuisance[name]=J
        hist_residual[name]=gh['mean']-observed
        # Source projection of historical line columns diagnoses R/X support;
        # it is not itself a design objective or a confidence construction.
        sh=J[:nh,2*m:2*m+2*H]
        hp=gh['line_jacobian']-sh@np.linalg.lstsq(sh,gh['line_jacobian'],rcond=1e-10)[0]
        singular=np.linalg.svd(hp,compute_uv=False)
        ratio=fit.x/fit.r
        diagnostics[name]=dict(history_SSE=float(hist_residual[name]@hist_residual[name]),
            training_record_SSE=float(fit.loss),history_profile_singular_values=singular,
            X_over_R=ratio,one_trained_line_sha256=line['sha256'],
            future_min_voltage=float(np.min(ge['physical']['v'])))
    inputs=[]; own=[]
    for a in context['bank']:
        for b in context['bank']:
            gb,ga=Egeo[b.name],Egeo[a.name]; nh=len(hist_residual[b.name])
            residual=np.r_[hist_residual[b.name],gb['mean']-ga['mean']]
            B=np.vstack((np.zeros((nh,T*n)),
                (gb['state_jacobian'][:,cc]-ga['state_jacobian'][:,cc])/context['base_kw']))
            row=dict(a=a.name,b=b.name,residual=residual,B=B,J=nuisance[b.name],
                reference_u=u.copy())
            (own if a.name==b.name else inputs).append(row)
    return dict(pairs=inputs,own=own,diagnostics=diagnostics,seconds=perf_counter()-began,
        control_shape=(T,n),reference_kw=u.copy(),history_included_once=True,
        line_adjustment_shared_across_time=True,PQ_direct_inputs=True,
        future_source_interface=1,uses_truth=False,uses_future_noise=False)


def projected_context(context,parent_kw):
    """Assemble all directed pairs and subtract each reference's own profile."""
    from . import ev_mobile_v3_projection as proxy
    data=geometry_inputs(context,parent_kw)
    projectors={r['b']:proxy.build_projector(r['J']) for r in data['own']}
    own={}
    for row in data['own']:
        pair=proxy.build_projected_pair(row['a'],row['b'],row['residual'],row['B'],row['J'],
            row['reference_u'],projector=projectors[row['b']])
        own[row['a']]=proxy.evaluate_pair(pair,row['reference_u'])
    pairs=[]
    for row in data['pairs']:
        pairs.append(proxy.build_projected_pair(row['a'],row['b'],row['residual'],row['B'],row['J'],
            row['reference_u'],const_minus=own[row['a']],projector=projectors[row['b']]))
    return pairs,dict(diagnostics=data['diagnostics'],own_profile=own,seconds=data['seconds'],
        scope='frozen unrestricted shared-line/source tangent relaxation; not AC bound')
