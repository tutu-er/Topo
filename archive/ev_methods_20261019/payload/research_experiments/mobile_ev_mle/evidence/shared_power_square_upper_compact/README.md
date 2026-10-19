# Shared power-square upper compact evidence

This independent archive preserves two completed H-only search pilots, the prior
H1 same-pool single-upper comparison, the synthetic toy, and the C optimization
diagnosis that motivated the compact representation. No new selector or large
experiment was run during integration. Original C/D policies and statistics are
unchanged. The source drivers retain their original paths as provenance.

Run the portable saved-data audit from the repository root:

    python -B -m research_experiments.mobile_ev_mle.shared_power_square_audit

The package adapter is opt-in and copies the original adaptive_search globals;
the search body and defaults remain identical. The saved audit checks archive
bytes, 15 frozen sources, pair fingerprints, 696 QP records, 16 MILP snapshots,
and 96 saved-state full-AC witnesses without a fit optimizer or selector. The
same-pool H1 comparison independently replays 72 QP records and two snapshots.

Large original JSON files are losslessly gzip-compressed. archive_manifest.json
records both stored SHA256 and the SHA256/size of decompressed original bytes.
Existing gzip snapshots/trace files and NPZ files preserve their original bytes.
The pilot's original evidence_manifest.json also validates after transparent
decompression. All failure/timeout records, raw controls, QP intervals, solver
gap values and AC witnesses are retained.

The shared backend obtains a global numerical bound of the frozen affine proxy
in both fixed H pilots. Its selected-action gain intervals versus constant
charging both cross zero. One shared upper reports status 0 with a nonzero
mip_gap, which is preserved. The original upper has no incumbent or best bound
in all three pilot calls; the separate C diagnosis records 515 such timeouts.
AC rollback establishes feasible witnesses for fixed action/pair nuisance
minima. It does not establish actual topology elimination or a global AC design
certificate. The full pair Taylor errors and history-dominated stopping scale
remain part of the evidence. Formal statistical evaluation of the new backend
requires a separately frozen protocol.
