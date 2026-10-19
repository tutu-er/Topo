"""Balanced LV simulation parameters from pinned public cable matrices.

Lengths are explicit user-declared simulation inputs, never inferred from the
undocumented ``length`` field of the public network. This module supplies no
asset-derived fitted impedance floor and generates no observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PUBLIC_REVISION = "a710c7212324f5085e1c6536714dc838f13ed81b"
PUBLIC_SOURCE = ("https://github.com/rubencarmona/lvnetworkdataset/blob/"
                 + PUBLIC_REVISION + "/cables/cableparameters.json")
SOURCE_RELATIVE = "data_external/lvnetworkdataset_20260920/cables/cableparameters.json"


def public_lv_parameter_family(length_m, cable_codes=("C201", "C250"), *,
                               base_kv=.4, base_mva=.02):
    """Return one line per explicit (length, cable code), in the same order.

    ``length_m`` and ``cable_codes`` broadcast to one vector. The four-wire
    matrix must preserve balanced currents exactly to tolerance; otherwise
    use a four-wire model instead of quietly scalarizing that cable.
    """
    if not np.isfinite([base_kv, base_mva]).all() or base_kv <= 0 or base_mva <= 0:
        raise ValueError("Voltage and power bases must be finite and positive")
    length, codes = np.broadcast_arrays(np.asarray(length_m, float),
                                         np.asarray(cable_codes, str))
    if length.ndim != 1 or not len(length) or not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError("Explicit positive length_m must broadcast to a nonempty vector")
    source = Path(__file__).resolve().parents[2] / SOURCE_RELATIVE
    payload = source.read_bytes()
    library = json.loads(payload)
    a = np.r_[np.exp(-2j * np.pi * np.arange(3) / 3), 0j]
    per_km, errors = [], []
    for code in codes:
        if str(code) not in library:
            raise ValueError(f"Cable code absent from pinned library: {code}")
        record = library[str(code)]
        Z = np.asarray(record["real"], float) + 1j * np.asarray(record["image"], float)
        if Z.shape != (4, 4) or not np.isfinite(Z).all():
            raise ValueError(f"Invalid four-wire matrix: {code}")
        z = complex(a.conj() @ Z @ a / 3)
        error = float(np.linalg.norm(Z @ a - z * a) / max(np.linalg.norm(Z @ a), 1e-30))
        if error > 1e-10:
            raise ValueError(f"Cable {code} does not exactly preserve balanced current; "
                             "retain the four-wire model")
        if z.real <= 0 or z.imag < 0:
            raise ValueError(f"Unexpected projected impedance signs for {code}")
        per_km.append(z)
        errors.append(error)
    per_km = np.asarray(per_km, complex)
    ohm = per_km * length / 1000.
    zbase = float(base_kv) ** 2 / float(base_mva)
    return dict(r=ohm.real / zbase, x=ohm.imag / zbase,
                r_ohm=ohm.real, x_ohm=ohm.imag,
                r_ohm_per_km=per_km.real, x_ohm_per_km=per_km.imag,
                length_m=length.copy(), cable_codes=codes.tolist(),
                base_kv=float(base_kv), base_mva=float(base_mva),
                base_kw=float(base_mva) * 1000., zbase_ohm=zbase,
                provenance=dict(classification="public_cable_parameter_derived_simulation",
                    source_url=PUBLIC_SOURCE, source_revision=PUBLIC_REVISION,
                    source_relative_path=SOURCE_RELATIVE,
                    source_sha256=hashlib.sha256(payload).hexdigest(),
                    source_lf_sha256=hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest(),
                    source_units="four-wire impedance matrix in ohm/km; author README",
                    conductor_order_assumption="first three phase conductors, fourth neutral",
                    balanced_relative_errors=errors,
                    length_source="explicit declared simulation length_m, not measured assets",
                    scalar_table_used=False, field_pqv=False,
                    fitted_impedance_floor_derived=False))
