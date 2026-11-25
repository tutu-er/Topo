"""Checks for the Phase 1 harness: exact fast fits, the compatible-support enumeration, the compressed MIQP."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from itertools import combinations
from time import perf_counter

import numpy as np
import pytest

from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.identification.optimization import QPFitter, StructuralSolver
from rnj_wzzt_dr.identification.trees import basis, canonical, compatible, is_laminar
from rnj_wzzt_dr.physical.backend import Observation

from .fastfit import ExtensionFitter, UniverseFitter, compatible_supports, count_compatible
from . import generators as gen, heuristics as heu

TRUE = [(0, 1), (0, 1, 2), (3, 4), (3, 4, 5, 6)]


def _data(levels, n=7, T=60, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.gamma(2., .5, (T, n)); q = .3*p+.05*rng.standard_normal((T, n))
    fam = canonical([*TRUE, *basis(n)])
    y = np.zeros((T, n))
    for C in fam:
        z = np.zeros(n); z[list(C)] = 1.
        y += np.outer(p@z, z)*.02*len(C)+np.outer(q@z, z)*.01
    y += 1e-3*rng.standard_normal((T, n))
    if levels:
        y += 3.
    return Observation(p, q, y, tuple(range(n)))


def _pool(n):
    return canonical([*TRUE, *combinations(range(n), 2), (0, 1, 2, 3), (2, 3), (4, 5, 6), (1, 2)])


@pytest.mark.parametrize("levels", [False, True])
def test_universe_fit_matches_package(levels):
    data = _data(levels); n = data.n
    ref = 1e-6
    qp = QPFitter(data, reference_mse=ref, bound=2.)
    pool = _pool(n)
    uf = UniverseFitter(data, pool, ref, .003, keep=basis(n))
    rng = np.random.default_rng(1)
    proper = [C for C in pool if 1 < len(C) < n]
    for _ in range(25):
        S = set()
        for i in rng.permutation(len(proper)):
            if rng.random() < .4 and all(compatible(proper[i], D) for D in S):
                S.add(proper[i])
        fam = canonical([*basis(n), *S])
        assert abs(uf.fit(fam)[0]-qp.fit(fam)["objective"]) <= 1e-8*(1+qp.fit(fam)["objective"])


@pytest.mark.parametrize("levels", [False, True])
def test_extension_fit_matches_package(levels):
    data = _data(levels); n = data.n
    ref = 1e-6
    qp = QPFitter(data, reference_mse=ref, bound=2.)
    family = canonical([*basis(n), (0, 1), (3, 4)])
    ext = ExtensionFitter(data, family, ref, .003)
    cands, skipped = compatible_supports(family, n)
    assert not skipped
    for C in cands:
        expected = qp.fit(canonical([*family, C]))["objective"]+.003*3
        assert abs(ext.objective(C)-expected) <= 1e-8*(1+expected)
    assert abs(ext.objective(None)-(qp.fit(family)["objective"]+.003*2)) <= 1e-8


def test_compatible_enumeration_is_complete():
    n = 7
    family = canonical([*basis(n), (0, 1), (0, 1, 2), (4, 5)])
    cands, _ = compatible_supports(family, n)
    brute = [C for k in range(2, n) for C in combinations(range(n), k)
             if C not in family and all(compatible(C, D) for D in family)]
    assert sorted(cands) == sorted(brute)
    assert count_compatible(family, n) == len(brute)


def test_generators_are_compatible():
    data = _data(False); n = data.n
    family = canonical([*basis(n), (0, 1)])
    qp = QPFitter(data, reference_mse=1e-6, bound=2.)
    point = qp.fit(family)
    for S in (gen.g2_sibling_pairs(family, n), gen.g4_residual_clusters(family, n, data, point),
              gen.g5_screen(family, n, data, point, 1e-3), gen.g3_residual_rnj(family, n, data, point)):
        for C in S:
            assert 1 < len(C) < n and C not in family and all(compatible(C, D) for D in family)


def test_g5_screen_finds_true_support():
    data = _data(False); n = data.n
    family = canonical([*basis(n), (0, 1), (0, 1, 2), (3, 4)])
    qp = QPFitter(data, reference_mse=1e-6, bound=2.)
    top = gen.g5_screen(family, n, data, qp.fit(family), 1e-3, top=5)
    assert (3, 4, 5, 6) in top


def test_heuristics_return_laminar_subsets():
    data = _data(False); n = data.n
    pool = _pool(n)
    uf = UniverseFitter(data, pool, 1e-6, .003, keep=basis(n))
    prob = heu.Problem(uf, pool, basis(n))
    H = heu.run_all(prob, e_start=[], rnj_trees=[list(TRUE)], seed=0, cap=20.)
    for k, v in H.items():
        fam = [tuple(C) for C in v["family"]]
        assert is_laminar(fam) and set(fam) <= set(prob.proper), k
    assert set(map(tuple, H["H2_local_base"]["family"])) == set(TRUE)


@pytest.mark.parametrize("levels", [False, True])
def test_compressed_miqp_matches_original(levels):
    pytest.importorskip("gurobipy")
    from .certify import CompressedSolver, SVDSolver
    data = _data(levels); n = data.n
    qp = QPFitter(data, reference_mse=1e-6, bound=2.)
    cfg = SearchConfig()
    pool, keep = _pool(n), basis(n)
    warm = qp.fit(keep)
    results = []
    for cls in (StructuralSolver, CompressedSolver, SVDSolver):
        s = cls(qp, cfg)
        pt, rec = s.select(pool, keep, warm, deadline=perf_counter()+60)
        s.close()
        results.append((canonical(pt["supports"]), rec["objective"]))
    for fam, obj in results[1:]:
        assert fam == results[0][0]
        assert abs(obj-results[0][1]) <= 1e-7*(1+abs(results[0][1]))
