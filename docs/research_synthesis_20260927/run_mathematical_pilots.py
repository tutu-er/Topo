"""Small, independent mathematical audits for the research synthesis.

These are counterexamples / statistical toys, not AC power-flow validation.
Only NumPy is needed. Existing Topo modules and result files are not changed.
Run from any directory; outputs go beside this script by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


def wilson(k: int, n: int) -> list[float]:
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def topology_ambiguity() -> dict:
    """Two positive-impedance trees share passive responses on rank-one loads."""
    one = np.ones(3)
    z12, z13 = np.array([1., 1., 0.]), np.array([1., 0., 1.])
    ra = 3 * np.eye(3) + np.outer(one, one) + np.outer(z12, z12)
    rb = np.diag([3., 5., 1.]) + np.outer(one, one) + np.outer(z13, z13)
    p = np.linspace(0.1, 1., 19)[:, None] * one
    passive_gap = np.max(np.abs(p @ ra.T - p @ rb.T))
    actions = {"common_mode": one, "terminal_1": np.eye(3)[0],
               "terminal_2": np.eye(3)[1], "terminal_3": np.eye(3)[2]}
    gaps = {k: float(np.linalg.norm((ra - rb) @ (v / np.linalg.norm(v))))
            for k, v in actions.items()}
    assert passive_gap < 1e-12 and gaps["common_mode"] < 1e-12
    assert all(gaps[k] > 0 for k in actions if k != "common_mode")
    return {"R_A": ra.tolist(), "R_B": rb.tolist(),
            "nontrivial_clade_A": [1, 2], "nontrivial_clade_B": [1, 3],
            "max_passive_response_gap": float(passive_gap),
            "response_gap_for_unit_energy_actions": gaps,
            "limitation": "Known fixed candidate matrices; no nuisance refit or AC model."}


def theft_confounding() -> dict:
    """Parameter/theft confounding and cancellation of fixed hidden load."""
    one = np.ones(2)
    r_normal = np.eye(2) + 2 * np.outer(one, one)
    r_hidden = np.eye(2) + np.outer(one, one)
    p = np.array([[.2, .4], [.5, .1], [.8, .2], [.7, .9]])
    a = p.sum(axis=1)
    y_normal = p @ r_normal.T
    y_hidden = p @ r_hidden.T + a[:, None] * one
    gaps = {"balanced": float(np.linalg.norm((r_normal-r_hidden) @ np.array([1., -1.]))),
            "single_terminal": float(np.linalg.norm((r_normal-r_hidden) @ np.array([1., 0.])))}
    u = np.array([.05, 0.])
    signatures = [np.array([1., 1.]), np.array([2., 1.])]
    responses = []
    for s in signatures:
        before = r_hidden @ p[0] + s * .3
        after = r_hidden @ (p[0] + u) + s * .3
        responses.append(after - before)
    hidden_location_increment_gap = np.linalg.norm(responses[0] - responses[1])
    assert np.allclose(y_normal, y_hidden)
    assert hidden_location_increment_gap < 1e-12
    return {"max_voltage_gap_between_normal_and_hidden": float(np.max(np.abs(y_normal-y_hidden))),
            "lossless_root_balance_normal": [0.] * len(a),
            "lossless_root_balance_hidden": a.tolist(),
            "probe_gaps_if_hidden_load_is_frozen": gaps,
            "same_network_different_hidden_sites_increment_gap": float(hidden_location_increment_gap),
            "limitation": "Same clades with different parameters; root bias/legal unmetered load excluded."}


def nuisance_projection() -> dict:
    """A large response parallel to common-mode nuisance provides no separation."""
    nuisance = np.ones((2, 1))
    proj = np.eye(2) - nuisance @ np.linalg.pinv(nuisance)
    deltas = {"large_common_mode": np.array([10., 10.]),
              "small_differential": np.array([1., -1.])}
    scores = {k: {"raw_squared_gap": float(d @ d),
                  "profiled_squared_gap": float(np.linalg.norm(proj @ d)**2)}
              for k, d in deltas.items()}
    assert scores["large_common_mode"]["profiled_squared_gap"] < 1e-24
    assert scores["small_differential"]["profiled_squared_gap"] > 1.99
    return {"scores": scores,
            "limitation": "Unbounded common linear nuisance and identity covariance; a local illustration."}


def query_value() -> dict:
    """Under an explicit synthetic prior, entropy and decision value disagree."""
    weights = np.full(8, 1 / 8)
    theft = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    queries = {"structure_bit_no_decision_value": np.array([1, 1, 1, 1, 0, 0, 0, 0]),
               "theft_fact": theft.copy()}

    def risk(w, label):
        # False alarm costs ten; missed theft costs one.
        return min(float(np.sum(w * label)), float(10 * np.sum(w * (1-label))))

    base = risk(weights, theft)
    ans = {}
    for name, outcomes in queries.items():
        entropy = 0.
        post_risk = 0.
        for out in np.unique(outcomes):
            keep = outcomes == out
            mass = float(weights[keep].sum())
            entropy -= mass * np.log2(mass)
            # Unnormalized weights directly include the outcome probability.
            post_risk += risk(weights[keep], theft[keep])
        ans[name] = {"information_bits": float(entropy), "expected_risk_after": post_risk,
                     "risk_reduction": base - post_risk}
    assert ans["structure_bit_no_decision_value"]["risk_reduction"] == 0.
    assert ans["theft_fact"]["risk_reduction"] == base
    return {"prior_risk": base, "queries": ans,
            "limitation": "Eight abstract states, explicit uniform prior, perfect queries, equal costs; no feeder data."}


def split_likelihood_bound_audit(seed: int, repetitions: int) -> dict:
    """H0: mean in [-1,0], sigma=1; independent training/testing."""
    rng = np.random.default_rng(seed)
    n = 32
    alpha = .05
    train_noise = rng.normal(size=(repetitions, n))
    test_noise = rng.normal(size=(repetitions, n))
    scenarios = []
    for truth in [-1., -.5, 0., .5]:
        train = train_noise + truth
        test = test_noise + truth
        prediction_mean = train.mean(axis=1)
        test_mean = test.mean(axis=1)
        pred_loss = .5 * np.sum((test-prediction_mean[:, None])**2, axis=1)
        exact_loss = .5 * np.sum((test-np.clip(test_mean, -1, 0)[:, None])**2, axis=1)
        relaxed_lb = .5 * np.sum((test-np.minimum(test_mean, .1)[:, None])**2, axis=1)
        feasible_ub = .5 * np.sum((test+1.)**2, axis=1)
        assert np.all(relaxed_lb <= exact_loss+1e-10)
        assert np.all(exact_loss <= feasible_ub+1e-10)
        reports = {}
        for name, loss in {"exact_composite_null": exact_loss,
                           "valid_relaxation_lower_bound": relaxed_lb,
                           "UNSAFE_feasible_incumbent_as_lower_bound": feasible_ub}.items():
            rejected = loss - pred_loss >= np.log(1/alpha)
            count = int(rejected.sum())
            reports[name] = {"rejections": count, "trials": repetitions,
                             "rejection_rate": count/repetitions,
                             "wilson_95_interval": wilson(count, repetitions)}
        scenarios.append({"true_mean": truth, "is_null": -1 <= truth <= 0,
                          "results": reports})
    return {"seed": seed, "n_train": n, "n_test": n, "alpha": alpha,
            "null_mean_interval": [-1., 0.], "relaxed_interval": ["-infinity", .1],
            "incumbent_mean": -1., "known_sigma": 1., "scenarios": scenarios,
            "limitation": "Three null parameters and one alternative, not supremum-null validation; paired noise across scenarios."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("mathematical_pilot_results.json"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    results = {"evidence_level": "independent mathematical counterexamples; not feeder/AC or end-to-end validation",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "numpy_version": np.__version__,
               "topology_ambiguity": topology_ambiguity(),
               "theft_confounding": theft_confounding(),
               "nuisance_projection": nuisance_projection(),
               "query_value": query_value(),
               "split_likelihood_bounds": split_likelihood_bound_audit(args.seed, args.repetitions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
