"""Identification of method draft v9 (experiments/case_library_20261125/paper/procedure.tex), development only.

Self-contained (numpy and gurobipy), one module per element of the draft, so it can be checked by hand:
    sets.py        supports, trees, children, unions                                  Section I
    model.py       balance, load gate, M, noise variance, price, bounded fit, crit     eqs 1-5
    miqp.py        the tree MIQP with optional per-meter bounds (also the flip test)   eq. 11, eq. 6
    search.py      local search; node pricing (enumeration, node MIQP for wide nodes)  eq. 12, Lemma 4
    labels.py      local flips, flip tests, labels of meter groupings                  eqs 6-8, Definition 1, Lemma 1
    anchors.py     units of anchors, the anchored universe                             eq. 13, Proposition 4
    identify.py    Algorithm 1 (gate, search, stopping rule, load evidence), development runs (never the sealed set)
    candidates.py  the initial candidate lines (Appendix B): the only module that uses identification/
    test_v8.py     brute-force checks of the set logic; the gate, the bounds and the node MIQP on data; a match
                   against the registered code's numbers
identification/ (the registered predecessor of the sealed test) is never edited here.
"""
