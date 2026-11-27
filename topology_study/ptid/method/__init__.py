"""The method of draft v16 (paper/procedure.tex). Self-contained: numpy, scipy and gurobipy only.

    sets.py       supports, families, compatibility, a node's children and unions        Section I, Lemma 2
    model.py      balance, load gate, M, noise variance, price, bounded fit, crit         eqs 1-5
    builders.py   the nine starting trees and the initial candidate lines                 Appendix B
    miqp.py       the tree MIQP (scaled coefficients, per-meter bounds, checked claims)   eq. 9, eq. 6
    search.py     local search to convergence; node pricing (enumeration, node MIQP)      eq. 10, Lemma 2
    labels.py     local flips, flip tests, labels of meter groupings                      Definition 1, Lemma 1, eqs 6-8
    anchors.py    units and the anchored universe                                         Definition 2, Proposition 4
    identify.py   Algorithm 1: gate, tree, anchors, labels, load places and evidence      Section IV, eq. 12
    probe.py      Algorithm 2: one day of cost-aware probing                              Section V, eqs 13-15

Units: the code scores in crit = RSS / (N sigma2) + alpha k, alpha = (p_s ln N - 2 lambda) / N; the draft's score is
H = crit N / 2 (half-BIC units), so a margin of m half-BIC units is m / (N / 2) in crit.
"""
