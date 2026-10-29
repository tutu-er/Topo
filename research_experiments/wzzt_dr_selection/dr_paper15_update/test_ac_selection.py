import unittest
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.dr_paper15_update.ac_candidate_study import select_ac
from ac_family_refit import predict_ac


class ACSelectionTests(unittest.TestCase):
    def test_post_contraction_is_rescored_with_AC(self):
        idx=pd.date_range('2020-01-01',periods=6,freq='h');labels=[101,102]
        p=pd.DataFrame([[.04,.06],[.08,.03],[.06,.05],[.09,.03],[.05,.1],[.03,.07]],index=idx,columns=labels)
        q=p*.3;v=p*0+1;root=pd.Series(1.,index=idx);blank=(p,q,v,root)
        point=dict(supports=[[0],[1],[0,1]],r=[.03,.04,.12],x=[.01,.02,.05])
        physical=dict(base_kv=.4,base_mva=.2,root_label=1)
        true=predict_ac(point,blank,**physical);obs=(p,q,true['voltage'],root)
        point.update(objective=0.,validation_mae=0.,origin='test',candidate_index=0)
        # Deliberately exercise the existing output contraction's actual effect.
        cal=dict(a=1.,b=1.,median_root_depth=10.)
        selected=select_ac([point],cal,obs,obs,physical)
        self.assertEqual(len(selected['supports']),2)
        self.assertEqual(selected['selection_validation_mae'],0.)
        self.assertGreater(selected['validation_mae'],1e-4)
        self.assertAlmostEqual(selected['validation_mae'],predict_ac(selected,obs,**physical)['mae'],places=14)


if __name__=='__main__':unittest.main()
