"""Exact arithmetic for this document; no optimizer or project imports."""
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
checks = []


def record(name, value, expected):
    assert value == expected, (name, value, expected)
    checks.append({"name": name, "value": str(value), "expected": str(expected)})


def matrix(rows):
    return [[F(x) for x in row] for row in rows]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def inner(a, b):
    return sum(x * y for ar, br in zip(a, b) for x, y in zip(ar, br))


def atom(n, support):
    return matrix([[int(i in support and j in support) for j in range(n)] for i in range(n)])


def combine(n, terms, diagonal=None):
    out = matrix([[0] * n for _ in range(n)])
    if diagonal is not None:
        for i, v in enumerate(diagonal):
            out[i][i] = F(v)
    for support, weight in terms:
        a = atom(n, support)
        for i in range(n):
            for j in range(n):
                out[i][j] += F(weight) * a[i][j]
    return out


def loss(r, p, y, power=1):
    pred = mul(r, p)
    return sum(abs(y[i][j] - pred[i][j]) ** power
               for i in range(len(y)) for j in range(len(y[0])))


def kernel(off12, off13, off23, diagonal=(10, 10, 10)):
    d1, d2, d3 = diagonal
    return matrix([[d1, off12, off13], [off12, d2, off23], [off13, off23, d3]])


def run():
    p = matrix([[1, 0, 0], [0, 10, 0], [0, 0, 1]])
    y = matrix([[10, 20, 3], [2, 100, 1], [3, 10, 10]])
    r_true = kernel(2, 1, 1)
    r_hat = kernel(2, 3, 1)
    r_base = kernel(2, 2, 2)
    r_13 = kernel(F(3, 2), 3, F(3, 2))
    record("observed_Y_equals_Rhat_P", mul(r_hat, p), y)
    record("true_atom_decomposition", combine(3, [((0, 1, 2), 1), ((0, 1), 1)], [8, 8, 9]), r_true)
    record("L1_reference", loss(r_true, p, y), F(4))
    record("L1_base", loss(r_base, p, y), F(13))
    record("L1_wrong_13_witness", loss(r_13, p, y), F(11))
    record("L1_wrong_23_and_star_witness", loss(r_base, p, y), F(13))
    r2_12 = kernel(2, F(107, 103), F(107, 103))
    r2_star = kernel(F(103, 68), F(103, 68), F(103, 68))
    record("L2_true_family_value", loss(r2_12, p, y, 2), F(808, 103))
    record("L2_wrong_13_value", loss(r_13, p, y, 2), F(101, 2))
    record("L2_star_value", loss(r2_star, p, y, 2), F(3737, 68))
    c = F(107, 103)
    record("L2_true_family_base_stationarity", 2 * (c - 3) + 101 * (c - 1), F(0))
    c = F(103, 68)
    mu = 2 * 101 * (c - 1)
    record("L2_wrong_23_KKT", 2 * 101 * (c - 2) + 4 * (c - 3) + mu, F(0))
    assert mu > 0
    record("first_extension_123", loss(r_base, p, y), F(13))
    record("first_extension_12", loss(kernel(2, 0, 0), p, y), F(17))
    record("first_extension_13", loss(kernel(0, 3, 0), p, y), F(33))
    record("first_extension_23", loss(kernel(0, 0, 1), p, y), F(28))
    distances = {}
    for i in range(3):
        distances[(0, i + 1)] = r_hat[i][i]
        for j in range(i + 1, 3):
            distances[(i + 1, j + 1)] = r_hat[i][i] + r_hat[j][j] - 2 * r_hat[i][j]
    record("NJ_sum_12_03", distances[(1, 2)] + distances[(0, 3)], F(26))
    record("NJ_sum_13_02", distances[(1, 3)] + distances[(0, 2)], F(24))
    record("NJ_sum_23_01", distances[(2, 3)] + distances[(0, 1)], F(28))
    u0 = matrix([[0, F(9, 10), 1], [0, 0, -1], [1, -1, 0]])
    u1 = matrix([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
    old = [(0,), (1,), (2,), (0, 1, 2)]
    candidates = [(0, 1), (0, 2), (1, 2)]
    for index, (u, val, scores) in enumerate([(u0, 13, [9, 2, -11]), (u1, 11, [11, 0, -11])]):
        assert max(abs(x) for row in u for x in row) <= 1
        record(f"u{index}_dual_objective", inner(u, y), F(val))
        for support in old:
            record(f"u{index}_old_column_{support}", inner(u, mul(atom(3, support), p)), F(0))
        for support, expected in zip(candidates, scores):
            score = inner(u, mul(atom(3, support), p))
            record(f"u{index}_candidate_{support}", score, F(expected))
    for mask in range(1, 8):
        support = tuple(i for i in range(3) if mask & (1 << i))
        if support != (0, 1):
            assert inner(u1, mul(atom(3, support), p)) <= 0
    record("complete_omission_dual_witness", inner(u1, y), loss(r_13, p, y))
    record("u0_screening_13_box20", max(F(0), inner(u0, y) - 20 * F(2)), F(0))
    record("u1_screening_13_box20", inner(u1, y), F(11))
    four_true = combine(4, [((0, 1, 2, 3), 1), ((0, 1, 2), 1)], [2] * 4)
    crossing = combine(4, [((0, 1, 2, 3), 1), ((0, 1), 1), ((0, 2), 1), ((1, 2), 1)], [1, 1, 1, 2])
    record("four_terminal_crossing_identity", crossing, four_true)
    four_omit = combine(4, [((0, 1, 2, 3), 1), ((0, 1), 1)], [2, 2, 3, 2])
    identity4 = matrix([[int(i == j) for j in range(4)] for i in range(4)])
    record("four_terminal_legal_omission_upper", loss(four_omit, identity4, four_true), F(4))
    a = F(1, 7)
    weak = combine(3, [((0, 1, 2), 3), ((0, 1), a)], [1, 1, 1])
    weak_omit = combine(3, [((0, 1, 2), 3)], [1 + a, 1 + a, 1])
    identity3 = matrix([[int(i == j) for j in range(3)] for i in range(3)])
    record("weak_block_omission_upper", loss(weak_omit, identity3, weak), 2 * a)
    # Check the exact threshold identities, without sampling or fitting models.
    W = F(101)
    record("L2_threshold_gap_identity", W / 2 - 8 * W / (W + 2), W * (W - 14) / (2 * (W + 2)))
    tex_files = [HERE / "main.tex", *sorted((HERE / "sections").glob("*.tex"))]
    tex = "\n".join(f.read_text(encoding="utf-8") for f in tex_files)
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels)), "duplicate TeX labels"
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
    assert refs <= set(labels), ("unknown labels", refs - set(labels))
    bib = (HERE / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    assert len(keys) == len(set(keys)), "duplicate citation keys"
    cited = set()
    for group in re.findall(r"\\cite\w*(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        cited.update(group.split(","))
    assert cited <= set(keys), ("unknown bibliography keys", cited - set(keys))
    report = {
        "scope": "Exact rational arithmetic and document integrity only; no optimization solver used.",
        "arithmetic_check_count": len(checks), "checks": checks,
        "label_count": len(labels), "bibliography_keys": keys, "cited_keys": sorted(cited),
    }
    verification = HERE / "verification"
    verification.mkdir(exist_ok=True)
    (verification / "arithmetic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sources = [
        "rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py",
        "rnj_wzzt_core/rnj_wzzt/estimation/preprocessing.py",
        "rnj_wzzt_core/rnj_wzzt/pipeline.py",
    ]
    source_files_present = all((ROOT / path).is_file() for path in sources)
    if source_files_present:
        snapshot = {
            "date": "2026-10-28", "baseline_commit": "10b17c164871964f05a5b35a59db45b05870cb1a",
            "scope": "Hashes of files actually read; no assertion that the entire working tree is clean.",
            "files": [{"path": path, "sha256": sha256((ROOT / path).read_bytes()).hexdigest()} for path in sources],
        }
        (verification / "source_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arithmetic_checks_passed": len(checks), "labels": len(labels), "references": len(cited),
                      "source_snapshot": "refreshed" if source_files_present else "project files unavailable; archived snapshot preserved"}, ensure_ascii=False))


if __name__ == "__main__":
    run()
