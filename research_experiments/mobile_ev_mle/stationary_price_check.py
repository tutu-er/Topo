"""Independent audit of stationary affine EV profile-design majorization.

Only the audit enumerates a tiny four-slot grid. The candidate action is
constructed analytically; no production selector imports this module.
Full AC, noisy SAA, and heterogeneous schedules are outside this theorem.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
from hashlib import sha256
import json
import numpy as np


def extreme_power_vector(total_power, pmax, slots):
    """Sorted saturated allocation of a fixed sum; no profile enumeration."""
    if not isinstance(slots, (int, np.integer)) or slots < 1:
        raise ValueError("positive integer slots required")
    if not np.isfinite([total_power, pmax]).all() or pmax <= 0:
        raise ValueError("finite total and positive pmax required")
    if total_power < 0 or total_power > slots*pmax+1e-12:
        raise ValueError("total outside uniform charging capacity")
    count = min(slots, int(np.floor(total_power/pmax+1e-12)))
    result = np.zeros(slots)
    result[:count] = pmax
    remaining = total_power-count*pmax
    if count < slots and remaining > 1e-12:
        result[count] = remaining
    if abs(result.sum()-total_power) > 1e-10:
        raise ValueError("saturation rounding exceeded tolerance")
    return result


def slot_half_cost(power, slope, reference_slope, intercept,
                   sigma_p=.08, sigma_v=.03):
    """Open true-P compensation: HALF of the profiled P/V squared error."""
    power = np.asarray(power, float)
    if not np.isfinite(power).all() or not np.isfinite(
            [slope, reference_slope, intercept, sigma_p, sigma_v]).all():
        raise ValueError("finite values required")
    if sigma_p < 0 or sigma_v <= 0:
        raise ValueError("nonnegative P sigma and positive V sigma required")
    mismatch = (slope-reference_slope)*power+intercept
    return .5*mismatch*mismatch/(sigma_v*sigma_v+slope*slope*sigma_p*sigma_p)


def family_half_score(power, slopes, intercepts, historical_half_cost,
                      reference_slope=1.4, sigma_p=.08, sigma_v=.03):
    """Shared slope across all slots plus its historical explanation cost."""
    power = np.asarray(power, float)
    slopes, intercepts, history = (np.asarray(a, float) for a in
                                  (slopes, intercepts, historical_half_cost))
    if slopes.ndim != 1 or not len(slopes) or intercepts.shape != slopes.shape or history.shape != slopes.shape:
        raise ValueError("matching nonempty one-dimensional competitor arrays")
    if not np.isfinite(history).all():
        raise ValueError("finite historical costs required")
    costs = np.asarray([h+slot_half_cost(power, s, reference_slope, c,
                         sigma_p, sigma_v).sum()
                        for s, c, h in zip(slopes, intercepts, history)])
    return float(costs.min()-history.min()), costs


def continuous_slope_half_distance(power, kappa=2., a0=.7,
                                   reference_slope=1.4, sigma_v=.03):
    """Unknown shared slope, exact P, quadratic history cost; closed minimum."""
    power = np.asarray(power, float)
    if not np.isfinite(power).all() or not np.isfinite(
            [kappa, a0, reference_slope, sigma_v]).all() or kappa <= 0 or sigma_v <= 0:
        raise ValueError("finite power, positive history curvature and V sigma required")
    s2 = float(power@power)
    slope = (kappa*sigma_v*sigma_v*a0+s2*reference_slope)/(kappa*sigma_v*sigma_v+s2)
    value = .5*kappa*s2*(reference_slope-a0)**2/(kappa*sigma_v*sigma_v+s2)
    return value, float(slope)


def run_audit():
    """Tiny independent grid and least-squares audits; no AC calls."""
    config = dict(slots=4, pmax=.30, total_power=.45, grid=[0., .15, .30],
                  reference_slope=1.4, slopes=[.7, 1.2, 1.8],
                  intercepts=[.02, -.03, .01],
                  historical_half_cost=[.12, .03, .08],
                  sigma_p=.08, sigma_v=.03,
                  continuous_kappa=2., continuous_a0=.7)
    extreme = extreme_power_vector(config["total_power"], config["pmax"], config["slots"])
    profiles = [np.asarray(u, float) for u in product(config["grid"], repeat=config["slots"])
                if abs(sum(u)-config["total_power"]) < 1e-12]
    score, costs = family_half_score(extreme, config["slopes"], config["intercepts"],
        config["historical_half_cost"], config["reference_slope"], config["sigma_p"], config["sigma_v"])
    rows = []
    max_pq_error = max_stationarity = max_continuous_error = 0.
    for u in profiles:
        candidate_score, candidate_costs = family_half_score(u, config["slopes"],
            config["intercepts"], config["historical_half_cost"], config["reference_slope"],
            config["sigma_p"], config["sigma_v"])
        prefixes = np.cumsum(np.sort(u)[::-1])
        majorized = bool(np.all(np.cumsum(extreme) >= prefixes-1e-12))
        per_theta_order = bool(np.all(costs >= candidate_costs-1e-12))
        # Numerical scalar least squares for true-P adjustment, independent of
        # the rational profiling formula used by slot_half_cost.
        for slope, intercept in zip(config["slopes"], config["intercepts"]):
            for power in u:
                mismatch = (slope-config["reference_slope"])*power+intercept
                matrix = np.array([[1/config["sigma_p"]], [slope/config["sigma_v"]]])
                rhs = np.array([0., -mismatch/config["sigma_v"]])
                adjustment = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
                residual = matrix@adjustment-rhs
                numerical = .5*float(residual@residual)
                closed = float(slot_half_cost(power, slope, config["reference_slope"], intercept,
                                               config["sigma_p"], config["sigma_v"]))
                max_pq_error = max(max_pq_error, abs(numerical-closed))
                max_stationarity = max(max_stationarity, float(np.max(abs(matrix.T@residual))))
        closed_continuous, fitted_slope = continuous_slope_half_distance(
            u, config["continuous_kappa"], config["continuous_a0"],
            config["reference_slope"], config["sigma_v"])
        design = np.r_[np.sqrt(config["continuous_kappa"]), u/config["sigma_v"]][:, None]
        target = np.r_[np.sqrt(config["continuous_kappa"])*config["continuous_a0"],
                       u*config["reference_slope"]/config["sigma_v"]]
        slope_ls = np.linalg.lstsq(design, target, rcond=None)[0]
        residual_ls = design@slope_ls-target
        max_continuous_error = max(max_continuous_error,
            abs(.5*float(residual_ls@residual_ls)-closed_continuous),
            abs(float(slope_ls[0])-fitted_slope))
        rows.append(dict(power=u.tolist(), power_squared_sum=float(u@u),
                         majorized_by_extreme=majorized, per_theta_order=per_theta_order,
                         family_half_score=candidate_score,
                         competitor_half_costs=candidate_costs.tolist(),
                         continuous_half_score=closed_continuous))
    continuous_extreme, slope_extreme = continuous_slope_half_distance(
        extreme, config["continuous_kappa"], config["continuous_a0"],
        config["reference_slope"], config["sigma_v"])
    checks = dict(
        feasible_grid_denominator=len(rows), expected_grid_denominator=16,
        majorization=all(row["majorized_by_extreme"] for row in rows),
        per_fixed_theta_order=all(row["per_theta_order"] for row in rows),
        min_theta_order=all(score >= row["family_half_score"]-1e-12 for row in rows),
        open_P_compensation_max_error=max_pq_error,
        open_P_compensation_KKT_max_residual=max_stationarity,
        continuous_shared_slope_max_error=max_continuous_error,
        continuous_extreme_order=all(continuous_extreme >= row["continuous_half_score"]-1e-12
                                     for row in rows),
        analytic_extreme_generated_without_grid=True,
    )
    passed = bool(len(rows) == 16 and checks["majorization"] and checks["per_fixed_theta_order"]
        and checks["min_theta_order"] and checks["continuous_extreme_order"]
        and max_pq_error < 1e-10 and max_stationarity < 1e-9 and max_continuous_error < 1e-10)
    return dict(config=config, status="passed" if passed else "failed", checks=checks,
                extreme_power=extreme.tolist(), extreme_family_half_score=score,
                extreme_competitor_half_costs=costs.tolist(),
                extreme_continuous_half_score=continuous_extreme,
                extreme_fitted_continuous_slope=slope_extreme, grid_rows=rows,
                proof_scope="stationary affine mean-response models with independent convex future state domains",
                grid_scope="independent audit only; no production action library",
                full_AC_global_optimality_certified=False,
                selector_or_threshold_changed=False)


def main():
    evidence = run_audit()
    path = Path(__file__).resolve()
    evidence["source_sha256"] = sha256(path.read_bytes()).hexdigest()
    output = path.parent/"evidence"/"gradient_price"/"stationary_price_check.json"
    if output.exists():
        raise FileExistsError("Preserve existing stationary audit evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "checks": evidence["checks"], "output": str(output)}))
    return 0 if evidence["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
