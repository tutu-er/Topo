# Compact EV evidence delivery

These .slim.json.gz files are explicit slim traces, not byte-equivalent raw files. The complete original raw is retained locally; its exact compressed-byte SHA-256, size and path are in delivery_manifest.json.

Preserved: historical measurements/fits, truth and measurement sigmas, every actual noisy/noiseless observation and fit, selected physical route/resource ledger, all H-only task scores/failures, all solver bounds/gaps/scales/timing, each exchange evaluation pair interval and status, and complete physical parameters/residuals of every round best QP.

Omitted: repeated LP vectors, full H-only prediction fit parameter/residual arrays, nonbest QP parameter/residual vectors, full witness intercept/slope banks, repeated predicted observation arrays and AC gate residual vectors. A complete numerical replay of those omitted stages requires the original raw plus the byte-preserved frozen source snapshot.

Method failures remain in the planned method denominator. Missing batch histories are listed separately. Development, evaluation and calibration exports retain their original phases.
