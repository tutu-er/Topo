"""Independent synthetic tree/data construction and paired study statistics.

The generator never calls RNJ, a tree-fitting routine, or the production
sensitivity builder. Truth is a list of edge descendant sets drawn first.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TreeTruth:
    terminals: list[int]
    supports: list[tuple[int, ...]]
    r_weights: np.ndarray
    x_weights: np.ndarray
    r: np.ndarray
    x: np.ndarray

    @property
    def clades(self):
        n = len(self.terminals)
        return {frozenset(self.terminals[i] for i in s)
                for s in self.supports if 1 < len(s) < n}


def generate_tree(shape: str, n: int, seed: int, internal_scale: float = 1.0):
    """Draw topology before data; no serial hidden degree-two nodes or stem."""
    if shape not in ('balanced', 'caterpillar', 'random', 'multifurcating', 'star'):
        raise ValueError('unknown tree shape')
    if n < 2 or not np.isfinite(internal_scale) or internal_scale <= 0:
        raise ValueError('n >= 2 and a positive internal_scale are required')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()
    supports = []

    def branch(group):
        if len(group) == 1:
            return
        if shape == 'star':
            parts = [[i] for i in group]
        elif shape == 'caterpillar':
            parts = [group[:1], group[1:]]
        elif shape == 'multifurcating':
            parts = [a.tolist() for a in np.array_split(group, min(3, len(group)))]
        else:
            split = len(group) // 2 if shape == 'balanced' else int(rng.integers(1, len(group)))
            parts = [group[:split], group[split:]]
        for part in parts:
            supports.append(tuple(sorted(part)))
            branch(part)

    branch(order)
    rw = rng.uniform(.65, 1.35, len(supports))
    xw = rng.uniform(.65, 1.35, len(supports))
    for j, support in enumerate(supports):
        if len(support) > 1:
            rw[j] *= internal_scale
            xw[j] *= internal_scale
    # Sum individual physical edge contributions without production helpers.
    r, x = np.zeros((n, n)), np.zeros((n, n))
    for support, a, b in zip(supports, rw, xw):
        for i in support:
            for j in support:
                r[i, j] += a
                x[i, j] += b
    return TreeTruth(list(range(1, n + 1)), supports, rw, xw, r, x)


def perturb_matrices(truth, stress: str, seed: int):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(2, len(truth.r), len(truth.r)))
    noise = .5 * (noise + noise.transpose(0, 2, 1))
    # Relative to the same true scale; not retuned using observed performance.
    scale = np.array([np.median(np.diag(truth.r)), np.median(np.diag(truth.x))])
    levels = {'exact': (0, 0), 'weak_internal': (0, 0),
              'noise_02': (.02, .02), 'noise_10': (.10, .10),
              'x_noisy': (.01, .20), 'r_noisy': (.20, .01),
              'diagonal_only': (.20, .20)}
    if stress not in levels:
        raise ValueError('unknown matrix stress')
    if stress == 'diagonal_only':
        noise *= np.eye(len(truth.r))[None, :, :]
    return tuple(matrix + level * sc * err for matrix, level, sc, err
                 in zip((truth.r, truth.x), levels[stress], scale, noise))


def synthetic_splits(truth, stress: str, seed: int, samples: int = 32):
    """Three independent replicas, with persistent scenario intercepts.

    Noise levels refer to squared-voltage drop RMS, not meter percent. This is
    a linear-model mechanism experiment; separate AC experiments test mismatch.
    """
    allowed = ('reference', 'low_sample', 'high_noise', 'pq_collinear',
               'terminal_correlated', 'outliers', 'weak_internal', 'common_noise')
    if stress not in allowed:
        raise ValueError('unknown measurement stress')
    n = len(truth.terminals)
    r, x = .001 * truth.r, .001 * truth.x
    result = {}
    for split_id, split in enumerate(('train', 'validation', 'test')):
        count = 96 if split == 'test' else (8 if stress == 'low_sample' else samples)
        scenarios = []
        for day in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, day]))
            innovation = rng.normal(size=(count, 2 * n))
            # Stationary variance-one AR(1), with correlation shared by methods.
            z = innovation.copy()
            for t in range(1, count):
                z[t] = .5 * z[t - 1] + np.sqrt(.75) * innovation[t]
            if stress == 'terminal_correlated':
                common = rng.normal(size=(count, 2))
                z[:, :n] = .1 * z[:, :n] + np.sqrt(.99) * common[:, :1]
                z[:, n:] = .1 * z[:, n:] + np.sqrt(.99) * common[:, 1:]
            p, q = .20 * z[:, :n], .12 * z[:, n:]
            if stress == 'pq_collinear':
                q = .6 * p  # Same power factor in all scenarios: exact ambiguity.
            p += .25 + .10 * day
            q += .10 + .04 * day
            if stress == 'pq_collinear':
                q = .6 * p
            intercept = np.linspace(-.0001, .0001, n) + day * .00005
            clean_drop = p @ r.T + q @ x.T + intercept
            # Use a fixed population signal scale, not per-split outcome tuning.
            signal_scale = np.sqrt(np.mean(.20**2 * np.sum(r*r, axis=1)
                                            + .12**2 * np.sum(x*x, axis=1)))
            noise_level = .30 if stress == 'high_noise' else .05
            observed_drop = clean_drop + noise_level * signal_scale * rng.normal(size=(count, n))
            if stress == 'outliers':
                mask = rng.random((count, n)) < .03
                observed_drop += mask * signal_scale * 8 * rng.normal(size=(count, n))
            if stress == 'common_noise':
                observed_drop += .5 * signal_scale * rng.normal(size=(count, 1))
            # Only train+validation inputs reach fitting and model selection.
            frames = {key: pd.DataFrame(value, columns=truth.terminals)
                      for key, value in [('P_terminal', p), ('Q_terminal', q),
                                         ('drop_target', observed_drop)]}
            frames['name'] = f'scenario_{day}'
            frames['V_terminal'] = pd.DataFrame(np.sqrt(1.02**2 - observed_drop), columns=truth.terminals)
            frames['root_voltage'] = pd.Series(np.full(count, 1.02))
            scenarios.append(frames)
        result[split] = scenarios
    return result, (r, x)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def source_fingerprint(core):
    """Hash core and primary study implementations, excluding post-result tools.

    Paths are relative to the repository so manifests identify the actual
    research implementations rather than the old compatibility entry points.
    The scripts subdirectory is intentionally excluded: matched RNJ tuning is
    added after the primary experiment, and records its own script hash.
    """
    core = Path(core).resolve()
    research = Path(__file__).resolve().parent
    files = [*sorted((core / 'rnj_wzzt').rglob('*.py')),
             *sorted(research.glob('*.py')), research.parent / '__init__.py']
    return {str(p.relative_to(core.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def paired_summary(rows, reference, *, seed=20260910, bootstrap=2000):
    """Paired F1 differences, clustered by independent tree/AC replicate.

    CIs describe this fixed test bank. Missing methods are retained in counts;
    incomplete solver searches with an available fallback remain in the score.
    There is no row-as-independent-sample significance test or universal claim.
    """
    frame = pd.DataFrame(rows)
    summaries = []
    if frame.empty:
        return summaries
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 1:
        raise ValueError('bootstrap must be a positive integer')
    keys = ['tier', 'stress', 'method', 'job_id']
    if frame.duplicated(keys).any():
        raise ValueError('duplicate method/job rows are not independent pairs')
    if frame.cluster_id.isna().any() or frame.cluster_id.eq('').any():
        raise ValueError('cluster_id must be nonempty')
    if (frame.groupby(['tier', 'stress', 'job_id']).cluster_id.nunique() > 1).any():
        raise ValueError('paired job rows must have the same cluster_id')
    for (tier, stress), group in frame.groupby(['tier', 'stress'], sort=True):
        baseline = group[group.method == reference].set_index('job_id')
        for method, current in group.groupby('method', sort=True):
            if method == reference:
                continue
            pairs = current.set_index('job_id').join(baseline[['clade_f1']], how='outer', rsuffix='_reference')
            valid = pairs[['clade_f1', 'clade_f1_reference']].notna().all(axis=1)
            available = pairs.loc[valid].copy()
            delta = available.clade_f1 - available.clade_f1_reference
            item = dict(tier=tier, stress=stress, method=method, reference=reference,
                        attempted=len(pairs), paired_available=int(valid.sum()),
                        missing=int((~valid).sum()), wins=int((delta > 1e-10).sum()),
                        ties=int((abs(delta) <= 1e-10).sum()), losses=int((delta < -1e-10).sum()))
            if len(delta):
                available['delta'] = delta
                clusters = [a.delta.to_numpy() for _, a in available.groupby('cluster_id', sort=True)]
                rng = np.random.default_rng(seed)
                draws = [np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]).mean()
                         for _ in range(bootstrap)]
                item.update(mean_delta=float(delta.mean()), cluster_count=len(clusters),
                            ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975)))
            summaries.append(item)
    return summaries
