"""Exact rational constants for a conditional full-AC weak-cherry bound.

This is an isolated mathematical audit. It does not change EV policies,
fit candidates, calibrate gates, or estimate a topology probability.
All decimal proof constants enter Fraction through strings, not floats.
"""
from __future__ import annotations

from fractions import Fraction as F


def rational_constants():
    vmin, radius = F(".9"), F(".1")
    mb, db = F(".145"), F(".012")
    zleaf, dzleaf, zroot, zweak = map(F, (".045", ".022", ".008", ".000044"))
    msource, umax = F(".15"), F(".3")
    mtot, mmax = msource + 3 * mb + umax, mb + umax
    zstar_leaf = zleaf + 2 * zweak
    c = zroot * mtot + zstar_leaf * mmax
    k = c / vmin**2
    l_den = 1 - mb * zleaf / vmin**2
    l0 = (db / vmin + mb**2 * dzleaf / vmin**3) / l_den
    l1 = (1 / vmin) / l_den
    a, b = zweak * l0 / (1 - k), zweak * l1 / (1 - k)
    sigma_v, sigma_master = F(".00012"), F(".001")
    history_count, future_count = 12, 8
    base_kw, dt_hours, grid_energy_kwh = F("20"), F(".25"), F("6")
    sum_u = grid_energy_kwh / (base_kw * dt_hours)
    sum_u2 = umax * sum_u
    master_gain = mtot / vmin**2
    gaussian_weight = F(3) / sigma_v**2 + master_gain**2 / sigma_master**2
    sum_delta_v2 = ((history_count + future_count) * a**2
                    + 2 * a * b * sum_u + b**2 * sum_u2)
    kl_exact = gaussian_weight * sum_delta_v2 / 2
    return dict(vmin=vmin, radius=radius, mb=mb, db=db, zleaf=zleaf,
                dzleaf=dzleaf, zroot=zroot, zweak=zweak, msource=msource,
                umax=umax, mtot=mtot, mmax=mmax, zstar_leaf=zstar_leaf,
                C=c, K=k, L_den=l_den, L0=l0, L1=l1, a=a, b=b,
                sigma_v=sigma_v, sigma_master=sigma_master,
                history_count=history_count, future_count=future_count,
                base_kw=base_kw, dt_hours=dt_hours,
                grid_energy_kwh=grid_energy_kwh, sum_u=sum_u,
                sum_u2=sum_u2, master_gain=master_gain,
                gaussian_weight=gaussian_weight, sum_delta_v2=sum_delta_v2,
                kl_exact=kl_exact, kl_upper=F(".11"), alpha=F(".05"))


def nominal_domain_checks():
    """Squared, exact checks from simulation.generate and nominal POP scales.

    n=3 only; common load variation cancels in a baseline terminal contrast.
    These checks exclude declared source_drift/load_trend/bias stresses.
    """
    c = rational_constants()
    scale_lo, scale_hi = F(".8"), F("1.2")
    leaf_r = tuple(map(F, (".028", ".031", ".034")))
    leaf_x = tuple(map(F, (".011", ".013", ".015")))
    leaf_r_hi, leaf_x_hi = max(leaf_r) * scale_hi, max(leaf_x) * scale_hi
    leaf_r_lo, leaf_x_lo = min(leaf_r) * scale_lo, min(leaf_x) * scale_lo
    return dict(
        baseline_load_sq=F(".136")**2 + F(".047")**2,
        baseline_load_cap_sq=c["mb"]**2,
        baseline_pair_contrast_sq=2 * F(".008")**2,
        baseline_pair_contrast_cap_sq=c["db"]**2,
        source_load_sq=F(".135")**2 + F(".05")**2,
        source_load_cap_sq=c["msource"]**2,
        root_impedance_sq=(F(".005") * scale_hi)**2 + (F(".0035") * scale_hi)**2,
        root_impedance_cap_sq=c["zroot"]**2,
        weak_impedance_sq=(F(".00003") * scale_hi)**2 + (F(".00002") * scale_hi)**2,
        weak_impedance_cap_sq=c["zweak"]**2,
        leaf_impedance_sq=leaf_r_hi**2 + leaf_x_hi**2,
        leaf_impedance_cap_sq=c["zleaf"]**2,
        leaf_difference_sq=(leaf_r_hi - leaf_r_lo)**2 + (leaf_x_hi - leaf_x_lo)**2,
        leaf_difference_cap_sq=c["dzleaf"]**2,
    )


def exact_certificate():
    c, checks = rational_constants(), nominal_domain_checks()
    margins = {key.removesuffix("_sq"): checks[key.replace("_sq", "_cap_sq")] - value
               for key, value in checks.items() if not key.endswith("_cap_sq")}
    margins.update(ball=c["radius"] - c["C"] / c["vmin"],
                   contraction=1 - c["K"], current_contrast=c["L_den"],
                   kl=c["kl_upper"] - c["kl_exact"])
    if any(value <= 0 for value in margins.values()):
        raise AssertionError("A required rational proof margin is not strictly positive")
    return c, checks, margins


def serialise_rationals(values):
    return {key: ({"fraction": str(value), "decimal": float(value)}
                  if isinstance(value, F) else value)
            for key, value in values.items()}

