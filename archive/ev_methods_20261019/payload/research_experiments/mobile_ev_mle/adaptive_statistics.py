"""History-unit statistical tools for a frozen adaptive EV research policy.

Calibration is marginal over the declared simulation population, with
independent calibration/evaluation histories and one scalar per full policy
trace. Failed fits cause abstention. Missing the true topology is reported
separately and is not repaired by a conservative score or threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy.stats import beta, binomtest


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identifier(value, name):
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty identifier")
    return str(value)


def _unique_history_ids(values):
    result = tuple(_identifier(value, "history_id") for value in values)
    if len(set(result)) != len(result):
        raise ValueError("history IDs must be unique; future-noise repeats are not new histories")
    return result


@dataclass(frozen=True)
class ScoreRecord:
    """One complete-history maximum; truth is used only by the evaluator."""

    relative_gap: float
    absolute_loss: float
    minimum_absolute_loss: float
    complete: bool
    coverage_ok: bool
    compute_failure: bool
    coverage_failure: bool
    fit_count: int
    failed_fit_count: int


def policy_trial_scores(losses, true_index, *, success=None, coverage_ok=True):
    """Reduce candidate losses to a full-policy truth score.

    losses[M,R,H] covers all declared methods and rounds. [M,H] or [H]
    promotes a missing round or method axis. Each trial may have a different H.
    The driver calls this function once per independent history, so calibration
    receives a list of ScoreRecords rather than a fixed candidate-count tensor.
    A missing truth index (None or -1) marks candidate coverage failure. Any
    required fit failure maps both calibration scores to +inf. No losses are
    clipped or silently omitted. Achieved local losses are allowed: calibration
    describes the frozen numerical policy, not a global-MLE statistic.
    """
    array = np.asarray(losses, dtype=float)
    if array.ndim == 1:
        array = array[None, None, :]
    elif array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3 or not all(array.shape):
        raise ValueError("losses must have nonempty shape [M,R,H], [M,H], or [H]")
    if success is None:
        succeeded = np.ones(array.shape, dtype=bool)
    else:
        succeeded = np.asarray(success, dtype=bool)
        if succeeded.ndim == 1:
            succeeded = succeeded[None, None, :]
        elif succeeded.ndim == 2:
            succeeded = succeeded[:, None, :]
        if succeeded.shape != array.shape:
            raise ValueError("success must have the same shape as losses")
    valid = succeeded & np.isfinite(array) & (array >= 0.0)
    failed_count = int(np.count_nonzero(~valid))
    missing = true_index is None or true_index == -1 or not bool(coverage_ok)
    if not missing:
        true_index = _integer(true_index, "true_index")
        if true_index >= array.shape[-1]:
            raise ValueError("true_index is outside the declared candidate axis")
    complete = failed_count == 0 and not missing
    if not complete:
        return ScoreRecord(math.inf, math.inf, math.inf, False, not missing,
                           failed_count > 0, missing, int(array.size), failed_count)
    best = array.min(axis=-1)
    truth = array[:, :, true_index]
    return ScoreRecord(
        relative_gap=float(np.max(truth - best)),
        absolute_loss=float(np.max(truth)),
        minimum_absolute_loss=float(np.max(best)),
        complete=True, coverage_ok=True, compute_failure=False,
        coverage_failure=False, fit_count=int(array.size), failed_fit_count=0,
    )


@dataclass(frozen=True)
class QuantileResult:
    threshold: float
    rank: int
    sample_count: int
    alpha: float
    rank_tail_bound: float

    def summary(self):
        return dict(threshold=None if math.isinf(self.threshold) else self.threshold,
                    vacuous=math.isinf(self.threshold), rank=self.rank,
                    sample_count=self.sample_count, alpha=self.alpha,
                    rank_tail_bound=self.rank_tail_bound)


def conformal_upper_quantile(scores, alpha):
    """Upper order statistic with strict 'score > threshold' rejection.

    k=ceil((n+1)(1-alpha)); k>n returns +inf. Scores may contain +inf to retain
    failed or missing-coverage calibration histories. This is a marginal rank
    gate for an exchangeable, previously fixed full-policy score distribution.
    """
    alpha = _probability(alpha, "alpha")
    array = np.asarray(scores, dtype=float)
    if array.ndim != 1 or np.any(np.isnan(array)) or np.any(np.isneginf(array)):
        raise ValueError("scores must be a vector of real values or +inf")
    n = len(array)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    threshold = float(np.sort(array)[rank - 1]) if rank <= n else math.inf
    bound = max(0.0, (n + 1 - rank) / (n + 1))
    return QuantileResult(threshold, rank, n, alpha, float(bound))


def _record(value):
    if isinstance(value, ScoreRecord):
        return value
    if isinstance(value, Mapping):
        return ScoreRecord(**value)
    raise TypeError("calibration records must be ScoreRecords or their exact mappings")


@dataclass(frozen=True)
class PolicyGateCalibration:
    relative: QuantileResult
    absolute: QuantileResult
    alpha_total: float
    policy_id: str
    population_id: str
    calibration_history_ids: tuple[str, ...]
    complete_history_count: int
    compute_failure_count: int
    coverage_failure_count: int

    def summary(self):
        return dict(
            relative=self.relative.summary(), absolute=self.absolute.summary(),
            alpha_total=self.alpha_total,
            combined_rank_tail_bound=min(1.0, self.relative.rank_tail_bound
                                          + self.absolute.rank_tail_bound),
            policy_id=self.policy_id, population_id=self.population_id,
            calibration_history_ids=list(self.calibration_history_ids),
            complete_history_count=self.complete_history_count,
            compute_failure_count=self.compute_failure_count,
            coverage_failure_count=self.coverage_failure_count,
            scope="fixed-policy simulation-population marginal gate",
            continuous_parameter_uniform_alpha=False,
            missed_truth_repaired=False,
        )


def calibrate_policy_gates(records: Sequence[ScoreRecord], calibration_history_ids,
                           *, policy_id, population_id, alpha_total=0.1,
                           relative_fraction=0.5):
    """Calibrate two max-method/round statistics on independent histories.

    Both score rules and the policy must be frozen before these histories.
    Relative and absolute risk budgets sum to alpha_total. Failures stay in the
    calibration sample as +inf, rather than conditioning on complete trials.
    """
    alpha_total = _probability(alpha_total, "alpha_total")
    fraction = _probability(relative_fraction, "relative_fraction")
    ids = _unique_history_ids(calibration_history_ids)
    rows = [_record(row) for row in records]
    if len(rows) != len(ids):
        raise ValueError("one ScoreRecord is required per calibration history")
    relative, absolute = [], []
    for row in rows:
        complete = bool(row.complete and row.coverage_ok
                        and not row.compute_failure and not row.coverage_failure)
        if complete:
            if (not math.isfinite(row.relative_gap) or row.relative_gap < 0.0
                    or not math.isfinite(row.absolute_loss) or row.absolute_loss < 0.0):
                raise ValueError("complete truth scores must be finite and nonnegative")
            relative.append(float(row.relative_gap))
            absolute.append(float(row.absolute_loss))
        else:
            relative.append(math.inf)
            absolute.append(math.inf)
    return PolicyGateCalibration(
        relative=conformal_upper_quantile(relative, alpha_total * fraction),
        absolute=conformal_upper_quantile(absolute, alpha_total * (1.0 - fraction)),
        alpha_total=alpha_total,
        policy_id=_identifier(policy_id, "policy_id"),
        population_id=_identifier(population_id, "population_id"),
        calibration_history_ids=ids,
        complete_history_count=sum(bool(row.complete and row.coverage_ok
                                        and not row.compute_failure
                                        and not row.coverage_failure) for row in rows),
        compute_failure_count=sum(bool(row.compute_failure) for row in rows),
        coverage_failure_count=sum(bool(row.coverage_failure or not row.coverage_ok) for row in rows),
    )


def apply_candidate_gates(candidate_losses, calibration, *, success=None,
                          history_id, policy_id, population_id,
                          candidate_names=None):
    """Apply the fixed guard to one method/round, without evaluation truth.

    Any required candidate fit failure retains every candidate and marks
    compute_failure. The relative/absolute masks and their union are returned;
    the driver may use absolute loss only as an adequacy alarm, as documented
    in its frozen policy. Candidate omission requires an external coverage
    audit; this function cannot infer an absent true topology from loss alone.
    """
    if not isinstance(calibration, PolicyGateCalibration):
        raise TypeError("calibration must be returned by calibrate_policy_gates")
    history_id = _identifier(history_id, "history_id")
    if history_id in calibration.calibration_history_ids:
        raise ValueError("evaluation history overlaps calibration histories")
    if (_identifier(policy_id, "policy_id") != calibration.policy_id
            or _identifier(population_id, "population_id") != calibration.population_id):
        raise ValueError("policy/population context differs from calibration")
    array = np.asarray(candidate_losses, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("candidate_losses must be a nonempty vector")
    succeeded = np.ones(len(array), dtype=bool) if success is None else np.asarray(success, dtype=bool)
    if succeeded.shape != array.shape:
        raise ValueError("success must have the same shape as candidate_losses")
    names = list(range(len(array))) if candidate_names is None else list(candidate_names)
    if len(names) != len(array) or len(set(map(str, names))) != len(names):
        raise ValueError("candidate_names must uniquely label the candidate axis")
    complete = bool(np.all(succeeded & np.isfinite(array) & (array >= 0.0)))
    if not complete:
        empty = [False] * len(array)
        return dict(relative_reject=empty.copy(), absolute_reject=empty.copy(),
                    combined_reject=empty.copy(), retained_relative=names.copy(),
                    retained_combined=names.copy(), relative_gaps=None,
                    absolute_alarm=False, compute_failure=True, complete=False,
                    history_id=history_id, coverage_status="requires_external_audit")
    gap = array - array.min()
    relative = gap > calibration.relative.threshold
    absolute = array > calibration.absolute.threshold
    combined = relative | absolute
    return dict(
        relative_reject=relative.tolist(), absolute_reject=absolute.tolist(),
        combined_reject=combined.tolist(),
        retained_relative=[name for name, rejected in zip(names, relative) if not rejected],
        retained_combined=[name for name, rejected in zip(names, combined) if not rejected],
        relative_gaps=gap.tolist(),
        absolute_alarm=bool(array.min() > calibration.absolute.threshold),
        compute_failure=False, complete=True, history_id=history_id,
        coverage_status="requires_external_audit",
    )


def binomial_exact_interval(successes, trials, *, confidence=0.95):
    """Two-sided Clopper--Pearson interval for iid history-unit events."""
    n = _integer(trials, "trials")
    k = _integer(successes, "successes")
    if k > n:
        raise ValueError("successes cannot exceed trials")
    confidence = _probability(confidence, "confidence")
    if n == 0:
        return (0.0, 1.0)
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return lower, upper


def history_event_summary(history_ids, events, *, confidence=0.95):
    """One boolean per independent history, with no dropping or noise pooling."""
    ids = _unique_history_ids(history_ids)
    values = np.asarray(events)
    if values.shape != (len(ids),) or values.dtype != np.dtype(bool):
        raise ValueError("events must be one boolean per history")
    k, n = int(values.sum()), len(values)
    return dict(count=k, history_count=n, rate=None if n == 0 else k / n,
                interval=binomial_exact_interval(k, n, confidence=confidence),
                confidence=float(confidence), unit="independent_history")


def paired_binary_effects(history_ids, treatment, control, *, confidence=0.95):
    """Paired history-unit difference, discordances and exact McNemar test.

    The risk-difference interval uses simultaneous CP intervals for improved
    and worsened event probabilities, with a Bonferroni split. It is exact
    conservative coverage under iid paired histories, and is not a noise-level
    confidence interval. Failed-policy success must be coded False by the
    evaluator and reported separately, rather than silently dropping the pair.
    """
    ids = _unique_history_ids(history_ids)
    a, b = np.asarray(treatment), np.asarray(control)
    if (a.shape != (len(ids),) or b.shape != a.shape
            or a.dtype != np.dtype(bool) or b.dtype != np.dtype(bool)):
        raise ValueError("treatment and control must be paired boolean histories")
    confidence = _probability(confidence, "confidence")
    n = len(ids)
    positive, negative = int(np.sum(a & ~b)), int(np.sum(~a & b))
    discordant = positive + negative
    delta = a.astype(float) - b.astype(float)
    effect = None if n == 0 else float(delta.mean())
    se = None if n < 2 else float(delta.std(ddof=1) / np.sqrt(n))
    simultaneous_confidence = 1.0 - (1.0 - confidence) / 2.0
    plus = binomial_exact_interval(positive, n, confidence=simultaneous_confidence)
    minus = binomial_exact_interval(negative, n, confidence=simultaneous_confidence)
    interval = (max(-1.0, plus[0] - minus[1]), min(1.0, plus[1] - minus[0]))
    pvalue = 1.0 if discordant == 0 else float(binomtest(positive, discordant, 0.5).pvalue)
    return dict(effect=effect, paired_standard_error=se,
                conservative_exact_interval=interval, confidence=confidence,
                improved=positive, worsened=negative, tied=n - discordant,
                history_count=n, exact_mcnemar_pvalue=pvalue,
                unit="independent_paired_history")


def paired_continuous_effects(history_ids, treatment, control):
    """Mean and empirical standard error of paired history-level measurements."""
    ids = _unique_history_ids(history_ids)
    a, b = np.asarray(treatment, dtype=float), np.asarray(control, dtype=float)
    if a.shape != (len(ids),) or b.shape != a.shape or not np.all(np.isfinite(a - b)):
        raise ValueError("paired measurements must be finite and match unique history IDs")
    delta = a - b
    return dict(effect=None if len(ids) == 0 else float(delta.mean()),
                paired_standard_error=None if len(ids) < 2 else
                float(delta.std(ddof=1) / np.sqrt(len(ids))),
                history_count=len(ids), differences=delta.tolist(),
                unit="independent_paired_history", exact_confidence_interval=False)


def _array_hash(array):
    value = np.ascontiguousarray(array, dtype=float)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def replay_eig_shortlist(posterior, planned_actions: Mapping, mean_builder: Callable,
                         *, eligible_names=None, samples_per_topology=2048,
                         selection_seed=0, validation_seeds=(1, 2), batch_size=256,
                         replay_context_id, scorer=None):
    """Score actual planned actions using dynamically replayed proper-bank means.

    mean_builder(action) returns whitened physical means[G,DN] in the exact
    posterior atom order. It must replay that atom's shared line/background/
    same-point source with the planned EV injection once. Realized future data
    and evaluation truth are not inputs. The policy freezes a selected action
    on the first integration stream and validates it on independent streams.
    Physics, feasibility, covariance and atom-order audits belong to the caller.
    """
    if not isinstance(planned_actions, Mapping) or not planned_actions:
        raise ValueError("planned_actions must be a nonempty action mapping")
    if not all(isinstance(name, str) for name in planned_actions):
        raise ValueError("planned action names must be strings")
    if not callable(mean_builder):
        raise TypeError("mean_builder must be callable")
    context = _identifier(replay_context_id, "replay_context_id")
    eligible = sorted(planned_actions) if eligible_names is None else list(eligible_names)
    if not eligible or len(set(eligible)) != len(eligible) or any(name not in planned_actions for name in eligible):
        raise ValueError("eligible_names must uniquely name available planned actions")
    selection_seed = _integer(selection_seed, "selection_seed")
    validation = tuple(_integer(seed, "validation_seed") for seed in validation_seeds)
    if len(set(validation)) != len(validation) or selection_seed in validation:
        raise ValueError("selection and validation integration seeds must be independent")
    means = {name: np.asarray(mean_builder(planned_actions[name]), dtype=float)
             for name in sorted(planned_actions)}
    dimensions = set()
    for array in means.values():
        if (array.ndim != 2 or not all(array.shape) or not np.all(np.isfinite(array))
                or len(array) != len(posterior.log_weights)):
            raise ValueError("replayed means must be finite [G,DN] on the posterior atom rows")
        dimensions.add(array.shape[1])
    if len(dimensions) != 1:
        raise ValueError("all planned actions must replay the same future channels/times")
    if scorer is None:
        try:
            from .bayes_rb import rb_eig_scores
        except ImportError:
            from research_experiments.mobile_ev_mle.bayes_rb import rb_eig_scores
        scorer = rb_eig_scores
    kwargs = dict(samples_per_topology=samples_per_topology, batch_size=batch_size)
    selected_scores = scorer(posterior, means, seed=selection_seed, **kwargs)
    chosen = min(eligible, key=lambda name: (-float(selected_scores[name]["eig"]), name))
    validation_scores = [scorer(posterior, means, seed=seed, **kwargs) for seed in validation]
    comparisons = {}
    for baseline in sorted(planned_actions):
        rows = [result[chosen]["paired_comparisons"][baseline] for result in validation_scores]
        comparisons[baseline] = dict(
            eig_gain=None if not rows else float(np.mean([row["eig_gain"] for row in rows])),
            eig_gain_se=None if not rows else float(np.sqrt(sum(
                row["eig_gain_se"] ** 2 for row in rows)) / len(rows)),
            independent_integration_repeats=len(rows),
        )
    return dict(selected_action=chosen, selection_scores=selected_scores,
                validation_scores=validation_scores,
                fixed_action_validation_comparisons=comparisons,
                mean_hashes={name: _array_hash(array) for name, array in means.items()},
                replay_context_id=context, selection_seed=selection_seed,
                validation_seeds=list(validation),
                validation_action_frozen=True, evaluation_truth_input=False,
                realized_future_noise_input=False)
