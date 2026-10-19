from ..analysis import rank,comparison,stats


def test_ranking_ties_failures_and_denominator():
    a=rank(dict(star=1.,pair01=1.+1e-7,pair02=2.,pair12=3.))
    assert a['first']==['star','pair01']
    assert not rank(dict(star=1.),True)['available']
    assert not rank(dict(star=1.,pair01=2.,pair02=3.,pair12=4.),False)['available']
    b=rank(dict(star=1.,pair01=2.,pair02=3.,pair12=4.))
    c=comparison(a,b)
    assert c['pairwise_denominator']==6 and c['pairwise_agreement']==5
    assert not c['top_same']
    assert stats([1.,2.,float('nan'),None])['count']==2
