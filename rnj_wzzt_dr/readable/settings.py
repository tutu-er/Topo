"""Every setting of WZZT-C in one place (readable edition).

Change a value here, then run `run_example.py` to see the effect on one input, or on the development inputs.
The production code (identification/combo.py) uses the same names; PROTOCOL_WZZT_C.md lists the frozen values.
"""
from dataclasses import dataclass


@dataclass
class Settings:
    # ------------------------------------------------------------ the model and its score (shared with WZZT-R)
    # Upper limit of every line's r and x (per unit, with the factor 2 of the linearised equation included).
    bound: float = 2.0
    # A free voltage offset per meter (absorbs a meter's gain bias).
    offsets: bool = True
    # Remove the error of the root (PCC) voltage reading, which every meter's voltage drop shares (GLS).
    gls: bool = True
    # Meter accuracy class, used only to decide whether r and x can be told apart (1 or 2 parameters per line).
    pq_sigma: float = 0.005

    # ------------------------------------------------------------ step 2: the sources that propose lines
    # Distance builders on the fitted matrix: "rnj", "nj", "rg".
    # Clusterings: "volt" (raw V^2 drops), "volt_m" (V^2 after removing offsets and the root error),
    # "resid" (residual after fitting each meter alone), "rrows" (rows of the fitted matrix),
    # "upgma" and "ward" (path distance between meters).
    sources: tuple = ("rnj", "nj", "rg", "volt", "volt_m", "resid", "rrows", "upgma", "ward")
    # Tolerance of RNJ, NJ and RG, as a share of the median root-to-meter path (0 keeps every split).
    tolerance: float = 0.0

    # ------------------------------------------------------------ step 4: which lines are fixed as blocks
    # The one parameter of the block rule: a line becomes a block when the data prefer it to every alternative by
    # at least m half-BIC units (a log Bayes factor). 4.6 = Bayes factor 100.
    m: float = 4.6
    # Lines with more than size_limit * n meters are never blocks (1.0 = no limit, the evidence decides).
    size_limit: float = 1.0

    # ------------------------------------------------------------ step 5: the exact root search
    # The root search tries every union of blocks (2^k of them); it is skipped when there are more than k_max.
    k_max: int = 9

    # ------------------------------------------------------------ step 6: certificate and repair
    # At most this many single lines are added by the repair.
    repairs: int = 10
    # The readable edition checks a node by trying every union of its child blocks; nodes with more child blocks
    # than this are not checked here (the production code uses an MIQP that has no such limit).
    enumerate_limit: int = 12
    # Pair moves (two new lines at once inside one node) are tried on nodes with at most this many child blocks.
    pair_limit: int = 6
    # At most this many pair moves.
    pair_rounds: int = 3
    # After the repair, choose again among all candidates plus the lines the repair added.
    reselect: bool = True

    # ------------------------------------------------------------ solver
    miqp_seconds: float = 240.0      # time limit of each selection MIQP (Gurobi)
    mip_gap: float = 1e-4            # relative optimality gap of the selection MIQP
