"""Conditional adequacy audit for a pre-history finite bank of fixed means.

This module does not inspect truth, run AC, refit parameters, select EV actions,
or change calibration. The caller must explicitly declare the fixed-mean and
known-covariance scope; an EIG summary alone cannot verify those assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def _scope(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("an explicit fixed_mean_scope declaration is required")
    return value.strip()


def _level(alpha):
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be a number strictly between zero and one")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    return alpha


def fixed_mean_statistic(history_white, means_white):
    """Minimum squared residual over a finite, already fixed whitened bank."""
    y, means = np.asarray(history_white, float), np.asarray(means_white, float)
    if (y.ndim != 1 or not len(y) or means.ndim != 2 or not len(means)
            or means.shape[1] != len(y)
            or not np.isfinite(y).all() or not np.isfinite(means).all()):
        raise ValueError("finite whitened vector and nonempty compatible mean bank required")
    residuals = means - y[None, :]
    losses = np.einsum("ij,ij->i", residuals, residuals)
    if not np.isfinite(losses).all():
        raise ValueError("squared residual overflow")
    return float(losses.min())


def audit_fixed_mean_loss(best_history_loss, df, *, alpha=0.01, fixed_mean_scope):
    """Test a fixed-mean family; its tail bound is not topology correctness."""
    scope, alpha = _scope(fixed_mean_scope), _level(alpha)
    if isinstance(df, (bool, np.bool_)) or not isinstance(df, (int, np.integer)) or df <= 0:
        raise ValueError("df must be a positive integer measurement dimension")
    df = int(df)
    try:
        dimension = float(df)
    except OverflowError as exc:
        raise ValueError("df is not numerically representable") from exc
    if not math.isfinite(dimension):
        raise ValueError("df is not numerically representable")
    if isinstance(best_history_loss, (bool, np.bool_)):
        raise ValueError("best_history_loss must be finite and nonnegative")
    loss = float(best_history_loss)
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("best_history_loss must be finite and nonnegative")
    threshold = float(chi2.ppf(1.0 - alpha, df))
    if not math.isfinite(threshold):
        # isf is numerically preferable when 1-alpha rounds to one.
        threshold = float(chi2.isf(alpha, df))
    if not math.isfinite(threshold):
        raise ValueError("chi-square threshold could not be evaluated finitely")
    sf_display = float(chi2.sf(loss, df))
    if not math.isfinite(sf_display):
        raise ValueError("chi-square tail could not be evaluated")
    # For x>d, log Chernoff = -(x-d-d*log(x/d))/2.
    # log1p avoids an unnecessary loss of precision close to x=d.
    if loss > dimension:
        h = (loss - dimension) / dimension
        log_tail = min(0.0, -0.5 * dimension * (h - math.log1p(h)))
    else:
        log_tail = 0.0  # the trivial, always valid probability bound one
    return {
        "df": df, "alpha": alpha, "best_history_loss": loss,
        "threshold": threshold, "reject": bool(loss > threshold),
        "sf_display": sf_display, "underflow": bool(sf_display == 0.0),
        "log_tail_upper_bound": float(log_tail),
        "log10_tail_upper_bound": float(log_tail / math.log(10.0)),
        "fixed_mean_scope": scope,
        "scope_verified_from_summary": False,
        "tail_meaning": "upper bound under any true pre-H fixed atom; not a topology probability",
        "zero_sf_meaning": "floating display underflow; the finite-x Gaussian tail is positive",
        "free_parameter_fit_chi_square_claim": False,
        "topology_correctness_inferred": False,
    }


def audit_saved_eig(summary, out=None, *, alpha=0.01, fixed_mean_scope,
                    prior_config_hash=None):
    """Read only summary history loss/dimension and labels; never evaluator truth.

    summary is a JSON path or mapping. A provided prior_config_hash is retained
    as caller provenance, not as proof that atoms/root/covariance were fixed.
    """
    scope, alpha = _scope(fixed_mean_scope), _level(alpha)
    if prior_config_hash is not None and not isinstance(prior_config_hash, str):
        raise ValueError("prior_config_hash must be a string or None")
    source_sha256 = None
    if isinstance(summary, (str, Path)):
        raw = Path(summary).read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        source = json.loads(raw.decode("utf-8-sig"))
    else:
        source = summary
    if not isinstance(source, dict) or not isinstance(source.get("histories"), list):
        raise ValueError("saved EIG summary must contain a histories list")
    records = []
    for index, row in enumerate(source["histories"]):
        label = row.get("input_identifier", f"history_{index}") if isinstance(row, dict) else f"history_{index}"
        record = {"input_identifier": label}
        try:
            if not isinstance(row, dict):
                raise ValueError("history record is not a mapping")
            record.update(audit_fixed_mean_loss(row["best_history_loss"], row["history_dimension"],
                alpha=alpha, fixed_mean_scope=scope))
            record["status"] = "audited"
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            record.update(status="unavailable", error=str(exc), reject=None)
        records.append(record)
    available = [row for row in records if row["status"] == "audited"]
    result = {
        "version": "fixed_finite_prior_adequacy_v1", "alpha": alpha,
        "alpha_scope": "per history; no across-history multiplicity correction",
        "history_count": len(records), "audited_history_count": len(available),
        "unavailable_history_count": len(records) - len(available),
        "reject_count": sum(row["reject"] for row in available), "histories": records,
        "source_summary_sha256": source_sha256, "provided_prior_config_hash": prior_config_hash,
        "fixed_mean_scope": scope, "scope_verified_from_summary": False,
        "used_history_fields": ["input_identifier", "best_history_loss", "history_dimension"],
        "truth_future_or_selected_action_used": False,
        "free_parameter_fit_chi_square_claim": False, "topology_correctness_inferred": False,
        "interpretation": "rejection challenges this fixed parameter/process prior representation; Gaussian observation densities retain full support",
    }
    if out is not None:
        destination = Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--fixed-mean-scope", required=True)
    parser.add_argument("--prior-config-hash")
    args = parser.parse_args()
    audit_saved_eig(args.summary, args.out, alpha=args.alpha,
        fixed_mean_scope=args.fixed_mean_scope, prior_config_hash=args.prior_config_hash)


if __name__ == "__main__":
    main()
