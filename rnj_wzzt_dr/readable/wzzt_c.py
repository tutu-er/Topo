"""WZZT-C, the whole method in six steps (readable edition).

    1. Prepare the input: remove the nuisance terms, estimate the noise level and the price per line (data.py).
    2. Sources propose lines: RNJ, NJ, RG and six clusterings (sources.py).
    3. Joint selection: the exact MIQP chooses T*, the best tree among all proposed lines (selection.py).
    4. Blocks: the lines of T* the data decide by at least m (a log Bayes factor) are fixed (blocks.py).
    5. Root search: the exact MIQP tries every union of the blocks, warm-started at T* (selection.py).
    6. Certificate and repair: at every node, every union of its child blocks is tried; a line that pays its price
       is added (also two at a time); finally the choice is made again with the added lines.

The production code (identification/combo.py) does the same with faster forms: a Gram-matrix fitter instead of
one fit per tree, and an MIQP per node instead of trying every union in step 6 (so it has no size limit there).

Example:
    from rnj_wzzt_dr.readable.settings import Settings
    from rnj_wzzt_dr.readable.wzzt_c import run
    result = run(p, q, y, v0sq, Settings(m=4.6), trace=True)
    result["tree"]          # the internal lines found
"""
import time
from itertools import combinations

from .blocks import choose_blocks, evidence, root_unions
from .data import Problem, canonical, child_blocks, compatible, internal, is_tree
from .selection import best_tree
from .sources import propose


def say(trace, text):
    if trace:
        print(text)


# ------------------------------------------------------------------ step 6 pieces
def unions_of(kids):
    """Every union of 2 or more child blocks of a node, but not all of them (that is the node itself)."""
    out = []
    for size in range(2, len(kids)):
        for picked in combinations(range(len(kids)), size):
            members = []
            for b in picked:
                members += list(kids[b])
            out.append(tuple(sorted(members)))
    return out


def best_single_line(problem, tree, settings):
    """Try every new line at every node. Returns (best new line or None, whether every node could be checked)."""
    n = problem.n
    current = problem.score(tree)
    best_line, best_score, complete = None, current-1e-9, True
    for node, kids in child_blocks(tree, n).items():
        if len(kids) < 3:
            continue
        if len(kids) > settings.enumerate_limit:
            complete = False                    # too many unions to try here (the production MIQP has no limit)
            continue
        for U in unions_of(kids):
            value = problem.score(list(tree)+[U])
            if value < best_score:
                best_line, best_score = U, value
    return best_line, complete


def best_pair(problem, tree, settings):
    """Try every two new lines inside one node (nested or disjoint), on nodes with 4..pair_limit child blocks."""
    n = problem.n
    current = problem.score(tree)
    best, best_score = None, current-1e-9
    for node, kids in child_blocks(tree, n).items():
        if len(kids) < 4 or len(kids) > settings.pair_limit:
            continue
        options = unions_of(kids)
        for i in range(len(options)):
            for j in range(i+1, len(options)):
                if compatible(options[i], options[j]):
                    value = problem.score(list(tree)+[options[i], options[j]])
                    if value < best_score:
                        best, best_score = (options[i], options[j]), value
    return best


def repair(problem, tree, settings, log):
    """Add the best paying line until none pays (at most settings.repairs). Returns (tree, certified)."""
    complete = True
    for _ in range(settings.repairs+1):
        line, complete = best_single_line(problem, tree, settings)
        if line is None:
            return tree, complete
        tree = canonical(list(tree)+[line])
        log.append(("added line", line))
    line, complete = best_single_line(problem, tree, settings)
    return tree, complete and line is None


# ------------------------------------------------------------------ the method
def run(p, q, y, v0sq, settings, trace=False):
    started = time.perf_counter()
    log = []

    # step 1
    problem = Problem(p, q, y, v0sq, settings)
    n = problem.n
    say(trace, f"Step 1  {problem.T} slots x {n} meters; GLS c = {problem.c:.3f}; noise sigma2 = {problem.sigma2:.3g}; "
               f"{problem.p_s} parameter(s) per line; price per line = {problem.half_bic(problem.alpha):.2f} half-BIC units")

    # step 2
    by_source, candidates, R = propose(problem, settings)
    say(trace, f"Step 2  sources proposed {len(candidates)} different lines: "
               + ", ".join(f"{name} {len(lines)}" for name, lines in by_source.items()))

    # step 3
    t_star, info3 = best_tree(problem, candidates, seconds=settings.miqp_seconds, gap=settings.mip_gap)
    t_star_lines = internal(t_star, n)
    say(trace, f"Step 3  joint selection: T* has {len(t_star_lines)} lines (MIQP status {info3['status']}, "
               f"{info3['seconds']:.1f} s, {info3['binaries']} binaries)")

    # step 4
    ev = evidence(problem, t_star_lines, candidates)
    blocks = choose_blocks(t_star_lines, ev, settings.m, n, settings.size_limit)
    if trace:
        print(f"Step 4  evidence of each line of T* (half-BIC units; block when >= m = {settings.m}):")
        for line in sorted(t_star_lines, key=len, reverse=True):
            e = ev[line]
            mark = ""
            if line in blocks:
                mark = "BLOCK"
            elif any(set(line) < set(b) for b in blocks):
                mark = "inside a block (kept as a candidate)"
            print(f"          {len(line):3d} meters  margin {e['margin']:8.2f}  vs {e['alternatives']:3d} alternatives  {mark}")
        print(f"        -> {len(blocks)} blocks: " + " ".join(str(len(b)) for b in blocks) + " meters")

    # step 5
    inside = [C for C in candidates if all(compatible(C, B) for B in blocks)]
    if len(blocks) <= settings.k_max:
        universe = canonical(inside+root_unions(blocks, n))
        tree, info5 = best_tree(problem, universe, warm=t_star, seconds=settings.miqp_seconds, gap=settings.mip_gap)
        if tree is None or problem.score(tree) > problem.score(t_star)+1e-12:
            tree = t_star
        say(trace, f"Step 5  root search over {len(universe)} lines ({len(blocks)} blocks): "
                   f"{len(internal(tree, n))} lines (MIQP status {info5['status']}, {info5['seconds']:.1f} s)")
    else:
        universe = candidates
        tree = t_star
        say(trace, f"Step 5  skipped: {len(blocks)} blocks > k_max = {settings.k_max}; keeping T*")

    # step 6
    tree, certified = repair(problem, tree, settings, log)
    for _ in range(settings.pair_rounds):
        pair = best_pair(problem, tree, settings)
        if pair is None:
            break
        tree = canonical(list(tree)+list(pair))
        log.append(("added pair", pair))
        tree, certified = repair(problem, tree, settings, log)
    added = [line for line in internal(tree, n) if line not in set(universe)]
    if settings.reselect and added:
        again, _ = best_tree(problem, canonical(list(universe)+internal(tree, n)), warm=tree,
                             seconds=settings.miqp_seconds, gap=settings.mip_gap)
        if again is not None and problem.score(again) < problem.score(tree)-1e-9:
            log.append(("re-selection changed the tree", None))
            tree, certified = repair(problem, again, settings, log)
    say(trace, f"Step 6  repair: {len(log)} change(s) {[kind for kind, _ in log]}; certified = {certified}")

    if not is_tree(tree):
        raise RuntimeError("the result is not a valid tree")
    final = internal(tree, n)
    say(trace, f"Result  {len(final)} lines, score {problem.half_bic(problem.score(tree)):.2f} half-BIC units, "
               f"{time.perf_counter()-started:.1f} s")
    return dict(tree=final, t_star=t_star_lines, blocks=blocks, evidence=ev, certified=certified, log=log,
                score=problem.score(tree), problem=problem, seconds=time.perf_counter()-started)
