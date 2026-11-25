"""Step 1 of WZZT-C (readable edition): the data, the model of the voltage drops, and the score.

The model. For T time slots and n meters, y[t, i] = V0[t]^2 - V_i[t]^2 is meter i's squared-voltage drop. A tree
is a set of lines; line C is the set of meters downstream of it. Line C adds

    r_C * a_C[t] + x_C * b_C[t]        to the drop of every meter in C,

where a_C[t] and b_C[t] are the active and reactive load downstream of C (sums of the meters' P and Q).

Nuisance terms are removed before fitting (the "residual operator" M):
  - a free offset per meter: every meter's series is demeaned in time;
  - the root reading's own error, shared by all meters in a slot: a share c of the slot's mean is removed (GLS).

The score of a tree (lower is better), in the units the code uses:

    objective + alpha * (number of lines),   objective = SSE / (N * sigma2),   alpha = p_s * ln(N) / N,

with SSE the squared fitting error, N = T * n, sigma2 the meters' noise level and p_s the parameters per line.
Multiplied by N/2 this is SSE / (2 sigma2) + (p_s / 2) * ln(N) * lines: half the BIC, i.e. minus the log of the
approximate marginal likelihood. A difference of this "half-BIC" between two trees is a log Bayes factor.

Same mathematics as identification/revised.py (Transform, Design, exact_fit, saturated_sigma2); plainer code.
"""
import math

import numpy as np
import gurobipy as gp


# ------------------------------------------------------------------ small helpers on lines and trees
def size_then_content(line):
    return (len(line), line)


def canonical(lines):
    """Lines as sorted tuples, without repeats, ordered by size then content."""
    unique = set()
    for line in lines:
        unique.add(tuple(sorted(int(i) for i in line)))
    return sorted(unique, key=size_then_content)


def basic_lines(n):
    """The lines every tree has: one per meter (its service cable) and the whole feeder."""
    lines = [(i,) for i in range(n)]
    lines.append(tuple(range(n)))
    return lines


def internal(lines, n):
    """The lines that carry topology information: more than one meter, fewer than all."""
    out = []
    for line in lines:
        if 1 < len(line) < n:
            out.append(tuple(line))
    return out


def compatible(a, b):
    """Two lines can be in the same tree when they are disjoint or one contains the other."""
    a, b = set(a), set(b)
    return len(a & b) == 0 or a <= b or b <= a


def is_tree(lines):
    lines = list(lines)
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if not compatible(lines[i], lines[j]):
                return False
    return True


def child_blocks(tree, n):
    """For every node of a tree (a line, or the whole feeder): its child blocks, the largest lines inside it."""
    nodes = canonical(list(tree)+basic_lines(n))
    out = {}
    for node in nodes:
        inside = [line for line in nodes if set(line) < set(node)]
        kids = []
        for line in inside:
            covered = False
            for other in inside:
                if set(line) < set(other):
                    covered = True
                    break
            if not covered:
                kids.append(line)
        out[node] = kids
    return out


# ------------------------------------------------------------------ the noise-free parts of the input
def gls_strength(v0sq, y):
    """c of the GLS step: the root reading's error appears in every meter's drop with variance ratio rho."""
    n = y.shape[1]
    rho = float(np.mean(v0sq**2)/np.mean((v0sq[:, None]-y)**2))
    return 1.-1./math.sqrt(1.+n*rho)


def parameters_per_line(p, q, pq_sigma, k=2.):
    """1 when (for most meters) Q moves with P within the meter class, so r and x cannot be told apart; else 2."""
    if pq_sigma is None:
        return 2
    p_d = p-p.mean(axis=0)
    q_d = q-q.mean(axis=0)
    votes = 0
    for i in range(p.shape[1]):
        level = float(np.linalg.norm(q[:, i]))
        slope = float(p_d[:, i]@q_d[:, i])/max(float(p_d[:, i]@p_d[:, i]), 1e-300)
        rest = float(np.linalg.norm(q_d[:, i]-slope*p_d[:, i]))
        if level > 0 and rest > k*math.sqrt(2)*pq_sigma*level:
            votes += 1
    if votes > p.shape[1]/2:
        return 2
    return 1


class Problem:
    """One input, ready to fit: the transformed loads and drops, the noise level and the price per line."""

    def __init__(self, p, q, y, v0sq, settings):
        self.settings = settings
        self.raw_y = y                                          # the measured drops, kept for voltage clustering
        self.T, self.n = p.shape
        self.N = self.T*self.n
        self.c = gls_strength(v0sq, y) if settings.gls else 0.
        # the residual operator M
        if settings.offsets:
            self.P = p-p.mean(axis=0)
            self.Q = q-q.mean(axis=0)
        else:
            self.P, self.Q = p, q
        target = y-self.c*y.mean(axis=1, keepdims=True)
        if settings.offsets:
            target = target-target.mean(axis=0)
        self.target = target                                   # T x n, before scaling
        # noise level and price
        self.sigma2 = self.saturated_noise()
        self.scale = math.sqrt(self.sigma2)
        self.y = (self.target/self.scale).ravel()              # the scaled target, length N
        self.p_s = parameters_per_line(p, q, settings.pq_sigma)
        self.alpha = self.p_s*math.log(self.N)/self.N
        self.column_cache = {}

    def slot_step(self, F):
        """The GLS step along the meter axis of a T x n x m array."""
        if self.c == 0:
            return F
        return F-self.c*F.mean(axis=1, keepdims=True)

    def saturated_noise(self):
        """sigma2 from the model with every R_ij and X_ij free (no tree): its residual is noise only."""
        T, n = self.T, self.n
        ii, jj = np.triu_indices(n)
        m = len(ii)
        blocks = []
        for flows in (self.P, self.Q):
            F = np.zeros((T, n, m))
            F[:, jj, np.arange(m)] = flows[:, ii]               # meter j's drop depends on meter i's load ...
            F[:, ii, np.arange(m)] = flows[:, jj]               # ... and meter i's on meter j's (R is symmetric)
            blocks.append(self.slot_step(F).reshape(self.N, m))
        F = np.hstack(blocks)
        target = self.target.ravel()
        theta, _, rank, _ = np.linalg.lstsq(F, target, rcond=1e-10)
        sse = float(np.sum((target-F@theta)**2))
        rows = (T-int(self.settings.offsets))*(n-int(self.c == 1.))
        return sse/(rows-int(rank))

    # -------------------------------------------------------------- columns, fit, score
    def line_columns(self, line):
        """The two columns of one line (its r column and its x column), scaled, length N each."""
        if line in self.column_cache:
            return self.column_cache[line]
        members = list(line)
        a = self.P[:, members].sum(axis=1)                      # downstream active load, length T
        b = self.Q[:, members].sum(axis=1)                      # downstream reactive load
        w = np.zeros(self.n)
        w[members] = 1.
        w = w-self.c*len(members)/self.n                        # the GLS step applied to the line's meters
        col_r = np.outer(a, w).ravel()/self.scale
        col_x = np.outer(b, w).ravel()/self.scale
        self.column_cache[line] = (col_r, col_x)
        return col_r, col_x

    def columns(self, tree):
        """The design matrix of a tree: the r columns of all its lines, then their x columns."""
        r_cols, x_cols = [], []
        for line in tree:
            col_r, col_x = self.line_columns(tuple(line))
            r_cols.append(col_r)
            x_cols.append(col_x)
        return np.column_stack(r_cols+x_cols)

    def fit(self, tree):
        """Best r and x (each between 0 and bound) for a fixed tree; returns (theta, objective = SSE / N)."""
        tree = canonical(list(tree)+basic_lines(self.n))
        F = self.columns(tree)
        Qm, Rm = np.linalg.qr(F)                                 # F = Qm Rm: the same fit in a small square form
        qy = Qm.T@self.y
        rest = float(self.y@self.y-qy@qy)                        # the part of y no tree can explain
        theta = bounded_least_squares(Rm, qy, self.settings.bound)
        sse = float(np.sum((Rm@theta-qy)**2))+rest
        return theta, sse/self.N

    def score(self, tree):
        """objective + alpha * (number of internal lines): the value WZZT-C minimises."""
        _, objective = self.fit(tree)
        return objective+self.alpha*len(internal(canonical(tree), self.n))

    def half_bic(self, score_value):
        """The same score in half-BIC units (log-Bayes-factor scale), as used by the block rule."""
        return .5*self.N*score_value


# ------------------------------------------------------------------ the one quadratic program
_ENV = None


def gurobi_env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


def bounded_least_squares(A, b, bound):
    """min ||A theta - b||^2 subject to 0 <= theta <= bound (Gurobi QP)."""
    k = A.shape[1]
    with gp.Model(env=gurobi_env()) as model:
        model.Params.Threads = 1
        theta = model.addMVar(k, lb=0., ub=bound)
        res = model.addMVar(A.shape[0], lb=-gp.GRB.INFINITY)
        model.addConstr(res == A@theta-b)
        model.setObjective(res@res)
        model.optimize()
        return np.array(theta.X)
