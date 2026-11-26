"""Initial candidate lines (draft v8, Appendix B): the nine builder trees of the registered code.

Algorithm 1 takes these trees as an input; any builders could supply them. This is the only module of the package that
uses identification/ (the frozen predecessor): its R-hat QP with the free load term and its nine hierarchies.
"""
from __future__ import annotations

from .sets import lines_of


def builder_trees(P, Q, Y, V0sq, u=None):
    """The lines (meter groupings) of the nine builder trees: RNJ, NJ, RG, UPGMA, Ward and four correlation clusterings
    on R-hat or on the data (Appendix B)."""
    from rnj_wzzt_dr.identification import combo, final, revised as rv, source as src
    data = rv.Series(P, Q, Y, V0sq)
    config = rv.RevisedConfig()
    if u is None:
        S = final.guarded(combo.setup, data, None, config)
    else:
        S = final.guarded(src.setup_projected, data, None, u[0], u[1], config)
    n = P.shape[1]
    return [lines_of(combo.source_family(S, name), n) for name in combo.SOURCES]
