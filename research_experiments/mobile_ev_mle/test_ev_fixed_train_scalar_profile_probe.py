"""Input isolation and empirical interpolation bookkeeping only."""
from copy import deepcopy

import pytest

from . import ev_fixed_train_scalar_profile_probe as probe


def test_payload_whitelist_does_not_change_when_unseen_fields_change():
    source=dict(training={'p':[[1.]]},training_meter={'fixed':'training'},
        future_meter={'public':'precision'},candidates=[dict(name='public_candidate')],
        training_result={'fits':[]},seeds={'policy':12,'noise':500},validation={'secret':1},
        truth={'r':'secret'},future_white={'secret':2},methods=[{'future':'secret'}])
    a=probe.training_only_payload(source)
    changed=deepcopy(source)
    for k in ('validation','truth','future_white','methods'):changed[k]={'different':'unseen'}
    changed['seeds']['noise']=900
    b=probe.training_only_payload(changed)
    assert a==b
    assert set(a)=={'training','training_meter','future_meter','candidates','training_fits','forecast_horizon','optimizer_seed'}
    with pytest.raises(ValueError,match='whitelist'):
        probe.restore_inputs(dict(a,validation='forbidden'))


def test_midpoint_error_and_rank_counts_keep_the_full_task_denominator():
    queries=[]
    for t in probe.CONFIG['slots']:
        for j in probe.CONFIG['sites']:
            for p in sorted(probe.CONFIG['knots_kw']+probe.CONFIG['midpoints_kw']):
                pairs=[]
                for k in range(12):
                    d=(t+1)*(j+1)*(p-.3)**2+k
                    pairs.append(dict(name=str(k),success=True,added=d,
                        proxy_QP=dict(success=True,primal_upper=2*(d+100))))
                queries.append(dict(slot=t,site=j,power_kw=p,result=dict(pairs=pairs)))
    rows=probe.interpolation_comparison(queries)
    assert len(rows)==288 and all(r['success'] for r in rows)
    assert sum(r['interpolation_absolute_error'] for r in rows)/288==pytest.approx(6.091875)
    assert all(r['proxy_absolute_error']==pytest.approx(100.) for r in rows)
    ranks=probe.ranking_agreement(rows)
    assert all(x['comparisons']==432 for x in ranks.values())
    aggregate=probe.ranking_agreement(rows,aggregate=True)
    assert all(x['comparisons']==132 for x in aggregate.values())
    queries[0]['result']['pairs'][0]['success']=False
    failed=probe.interpolation_comparison(queries)
    assert len(failed)==288 and sum(r['success'] for r in failed)==287
    assert any(not r['success'] and r['failure'] for r in failed)
