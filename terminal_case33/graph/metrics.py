"""Evaluation metrics for sensitivity and topology recovery."""

from __future__ import annotations

import numpy as np
from terminal_case33.graph.latent_tree import terminal_splits


def rmse_R(R_hat, R_true) -> float:
    """RMSE for R."""

    return float(np.sqrt(np.mean((np.asarray(R_hat) - np.asarray(R_true)) ** 2)))


def rmse_X(X_hat, X_true) -> float:
    """RMSE for X."""

    return float(np.sqrt(np.mean((np.asarray(X_hat) - np.asarray(X_true)) ** 2)))


def relative_error_R(R_hat, R_true) -> float:
    """Relative Frobenius error for R."""

    return float(np.linalg.norm(np.asarray(R_hat) - np.asarray(R_true)) / max(np.linalg.norm(R_true), 1e-12))


def relative_error_X(X_hat, X_true) -> float:
    """Relative Frobenius error for X."""

    return float(np.linalg.norm(np.asarray(X_hat) - np.asarray(X_true)) / max(np.linalg.norm(X_true), 1e-12))


def distance_matrix_rmse(d_hat, d_true) -> float:
    """RMSE for distance matrices."""

    return float(np.sqrt(np.mean((np.asarray(d_hat) - np.asarray(d_true)) ** 2)))


def edge_precision_recall_f1(pred_edges, true_edges) -> dict:
    """Precision, recall, and F1 for undirected edge sets."""

    pred = {tuple(sorted(map(int, e))) for e in pred_edges}
    true = {tuple(sorted(map(int, e))) for e in true_edges}
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def terminal_split_precision_recall_f1(pred_edges, true_edges, terminals) -> dict:
    """Precision/recall/F1 for hidden-tree topology using terminal splits.

    Hidden node labels are arbitrary across reconstruction algorithms, so raw
    hidden-node edge comparison is not meaningful. Split comparison evaluates
    whether internal tree edges induce the same terminal bipartitions.
    """

    pred = terminal_splits(pred_edges, terminals)
    true = terminal_splits(true_edges, terminals)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_split_count": len(pred),
        "true_split_count": len(true),
        "matched_split_count": tp,
    }


def map_tree_accuracy_on_terminal_equivalent_graph(pred_edges, true_edges) -> dict:
    """Accuracy on the terminal-equivalent graph level."""

    out = edge_precision_recall_f1(pred_edges, true_edges)
    out["level"] = "terminal_equivalent_topology"
    return out


def edge_marginal_calibration(edge_marginals, true_edges) -> dict:
    """Simple calibration summary for posterior edge marginals."""

    true = {tuple(sorted(map(int, e))) for e in true_edges}
    probs = []
    labels = []
    for row in edge_marginals.itertuples():
        probs.append(float(row.edge_marginal))
        labels.append(tuple(sorted((int(row.u), int(row.v)))) in true)
    if not probs:
        return {"brier": 0.0, "mean_marginal": 0.0}
    probs_arr = np.asarray(probs)
    labels_arr = np.asarray(labels, dtype=float)
    return {"brier": float(np.mean((probs_arr - labels_arr) ** 2)), "mean_marginal": float(probs_arr.mean())}


def hidden_degree_summary(net) -> dict:
    """Summarize hidden-node degrees."""

    hidden = net.buses[net.buses["bus_type"].eq("hidden_internal")]
    return hidden["hidden_degree"].astype(int).value_counts().sort_index().to_dict()


def strict_scenario_violations(summary: dict) -> int:
    """Count strict validation violations."""

    return len(summary.get("violations", []))


def voltage_reconstruction_rmse(V_hat, V_true) -> float:
    """RMSE for reconstructed voltage magnitudes."""

    return float(np.sqrt(np.mean((np.asarray(V_hat) - np.asarray(V_true)) ** 2)))
