"""Independent residual/normalizer replay check without fitting."""
import numpy as np
import pytest
from .ev_physical_spatial_fit_audit import generated_history
from .ev_physical_spatial_fit_review import replay_fit
from .likelihood import ac_predict


def test_replay_cost_normalizer_and_H_N_split_match_direct_white_errors():
    g=generated_history(987201);case=g['case'];obs=g['history'];m=g['meter']
    physical=ac_predict(case.candidate,case.r,case.x,g['p'][:12],g['q'][:12],g['source_p'][:12],g['source_q'][:12],g['root_v'][:12])
    prediction={k:physical[k]+(getattr(m,'bias_'+k) if k in ('p','q','v') else 0.)
        for k in ('p','q','v','master_p','master_q')}
    expected=0.;norm=0.
    for k in prediction:
        sigma=np.broadcast_to(getattr(m,k+'_sigma'),np.asarray(getattr(obs,k)).shape)
        expected+=float(np.sum(((prediction[k]-getattr(obs,k))/sigma)**2))
        norm+=float(np.sum(np.log(2*np.pi*sigma*sigma)))
    fit=dict(name='clade_02',supports=[list(s) for s in case.candidate.supports],r=case.r,x=case.x,
        p=g['p'][:12],q=g['q'][:12],source_p=g['source_p'][:12],source_q=g['source_q'][:12],predicted=prediction,
        loss=expected,loglik=-.5*(expected+norm),success=True,
        diagnostics=dict(restarts=[dict(success=True),dict(success=False,status=0)],
            local_shared_rank=10,shared_parameter_count=10,active_bound_count=0,source_node=1))
    record=dict(config=dict(history_count=12),historical_fits=[fit])
    result=replay_fit(record,fit,obs,m)
    assert result['cost_replay_abs_error']<1e-10
    assert result['loglik_replay_abs_error']<1e-10
    assert result['history_cost']==pytest.approx(expected)
    assert result['future_cost']==0.
    assert result['converged_restarts']==1 and len(result['failed_restarts'])==1
