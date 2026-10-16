from itertools import product
import unittest

from . import ev_mobile_v3_error_decomposition as audit


class ShapleyTests(unittest.TestCase):
    def test_additive_exact(self):
        errors={bits:sum(w*(1-b) for w,b in zip((2.,3.,5.),bits)) for bits in product((0,1),repeat=3)}
        result=audit.shapley_reduction(errors)
        for name,value in dict(lines=2.,background=3.,source=5.).items():
            self.assertAlmostEqual(result['contributions'][name],value)

    def test_interactions_and_negative_corrections(self):
        # At 000 two error terms cancel; correcting one first increases SSE.
        errors={bits:((1-bits[0])-2*(1-bits[1]))**2 for bits in product((0,1),repeat=3)}
        result=audit.shapley_reduction(errors)
        self.assertAlmostEqual(sum(result['contributions'].values()),1.)
        self.assertAlmostEqual(result['contributions']['lines'],-1.)
        self.assertAlmostEqual(result['contributions']['background'],2.)
        self.assertEqual(result['contributions']['source'],0.)

    def test_missing_state_retained_as_failure(self):
        with self.assertRaises(ValueError):audit.shapley_reduction({(0,0,0):1.})

    def test_rank_ties_and_reversal(self):
        result=audit.pairwise_ranking(dict(a=dict(predicted=2.,true=1.),b=dict(predicted=1.,true=2.),c=dict(predicted=2.,true=1.)))
        self.assertEqual(result['counts'],dict(reversed=2,both_tied=1))


if __name__=='__main__':unittest.main()
