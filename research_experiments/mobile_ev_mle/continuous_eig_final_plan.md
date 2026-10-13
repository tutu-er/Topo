# Final frozen independent H-only diagnostic and paired latent doubling

Parent freezes the normalized proposals from
`continuous_eig_tempered_pilot_v1/final_proposals_unvalidated_for_posterior.json`.
No mean/covariance/prior or regularization changes are allowed after this
freeze. The pilot .001..1 schedule is engineering only; none of its weights
enter the final H estimator.

Each of three fresh independent repeats uses seeds28301301,28301302,28301303,
4096 theta draws per mode, and one independent-time white-normal cloud
Z(T,128,8) plus an independent mixture-uniform cloud U(T,128). The same fresh
theta/logq and each time's first32 nuisance samples produce paired L32 and
L128 likelihood estimates. Each H likelihood replaces the other; they are
never multiplied, and no old H particle weight is reused.

Both levels target the original pi(beta)*L_H(beta) once, with full beta q and
restored meter/latent q densities. Weighted pilot fitting does not make the
final target tempered. Seven modes remain present even when displayed masses
underflow. Source/prior-domain zeros, supported-domain numerical failures,
conditional mode ESS, global ESS, maximum weight, time/joint sample-moment
relative SE and independent-repeat evidence spread remain separate.

Predeclared gates remain globalESS>=100, every conditional modeESS>=20,
maximum weight<=.05 and repeat/integration stability. No rare-mass exception
or model deletion is allowed. A numerical failure preserves the seven-mode
ledger and stops. No predictive EIG/action ranking is authorized.

To bound memory, evaluate beta in fixed chunks of256. All chunks use the same
time-normal/uniform clouds, so this changes memory use only. Prefix likelihoods
are reduced from the exact same evaluated per-time kernel, which avoids a
second physical solve or a changed-prefix stream. Source/code/configuration
and frozen-q hashes are saved before MC. BelowNormal and one numerical thread
are used. The soft budget is30min, checked after a complete repeat, alongside
the parent outer deadline2026-10-13 01:00UTC; checkpoint without omitting a
mode. Raw particle banks remain staging/local artifacts.
