"""Identification and probing of method draft v12 (experiments/case_library_20261125/paper/procedure.tex), development
only.

Self-contained: it imports only numpy, scipy and gurobipy, so the folder works on its own. One module per element of
the draft, so it can be checked by hand:
    sets.py        supports, trees, children, unions                                  Section I
    model.py       balance, load gate, M, noise variance, price, bounded fit, crit     eqs 1-5
    miqp.py        the tree MIQP with optional per-meter bounds (also the flip test)   eq. 9, eq. 6
    search.py      local search; node pricing (enumeration, node MIQP for wide nodes)  eq. 10, Lemma 2
    labels.py      local flips, flip tests, labels of meter groupings                  eqs 6-8, Definition 1, Lemma 1
    anchors.py     units of anchors, the anchored universe                             eq. 11, Proposition 4
    identify.py    Algorithm 1 (gate, search, stopping rule, load evidence); refine (lines 4-11) for every gate
    probe.py       Algorithm 2 (one day of probing; target variants T0, T1, T2)        eqs 13-15, Lemmas 4-5
    builders.py    the nine starting trees and R-hat (Appendix B), the initial candidate lines
    test_v8.py     brute-force checks of the set logic; the gate, the bounds and the node MIQP on data; matches
                   against the registered code (numbers, builder trees); kappa and the block gradients of probing
The case library (development cases, simulated probing days) is outside, in experiments/case_library_20261125.
identification/ (the registered predecessor of the sealed test) is never edited here.
"""
