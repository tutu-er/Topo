from copy import deepcopy
import unittest

from . import ev_mobile_v3_parent_initial_audit as audit


def fixture():
    return dict(methods=[dict(name='scalar_fixed',power_kw=[[1.,0.],[0.,2.]])],
        selection=dict(methods=dict(scalar_mobile=dict(raw=dict(initial_candidates=[],grids=[])))))


def profile(success=True):
    return dict(power_kw=[[1.,0.],[0.,2.]],success=success,pairs=[{}]*12)


class ParentTests(unittest.TestCase):
    def test_actual_duplicate_before_parent(self):
        r=fixture();r['selection']['methods']['scalar_mobile']['raw']['initial_candidates']=[dict(origin='square_mobile',success=True,profile=profile(),score=2.)]
        x=audit.inspect_parent(r)
        self.assertEqual(x['category'],'complete_other_initial_origin_same_curve')
        self.assertTrue(x['complete_parent_curve_scored'])

    def test_skip_with_scored_duplicate(self):
        r=fixture();r['selection']['methods']['scalar_mobile']['raw']['initial_candidates']=[dict(origin='square_mobile',success=True,profile=profile(),score=2.),dict(origin='fixed_parent',attempted=False,failure='initial plan budget exhausted')]
        x=audit.inspect_parent(r)
        self.assertEqual(x['category'],'complete_other_initial_origin_same_curve')
        self.assertTrue(x['explicit_parent_deadline_skip'])

    def test_deadline_no_profile(self):
        r=fixture();r['selection']['methods']['scalar_mobile']['raw']['initial_candidates']=[dict(origin='fixed_parent',attempted=False,failure='initial plan budget exhausted')]
        self.assertEqual(audit.inspect_parent(r)['category'],'deadline_skipped_no_scored_duplicate')

    def test_failed_profile_is_not_complete(self):
        r=fixture();r['selection']['methods']['scalar_mobile']['raw']['initial_candidates']=[dict(origin='fixed_parent',success=False,profile=profile(False))]
        self.assertEqual(audit.inspect_parent(r)['category'],'failed_profiles_only')

    def test_selected_copy_alone_cannot_prove_scoring(self):
        r=fixture();r['selection']['methods']['scalar_mobile']['raw']['selected_profile']=profile()
        self.assertEqual(audit.inspect_parent(r)['category'],'unknown')


if __name__=='__main__':unittest.main()
