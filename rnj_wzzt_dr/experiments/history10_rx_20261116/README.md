# Ten-day history: resolve the R/X estimation question first

This isolated experiment checks the origin of the current P/Q and compares
two versus eight training days, with two common validation days and day eleven
held out. It does not change the identifier, regularization, resource offers,
production defaults, or run flexibility procurement.

See [PROTOCOL.md](PROTOCOL.md) for the predeclared scope and
[SOURCE_AUDIT.md](SOURCE_AUDIT.md) for provenance. Results will distinguish
native topology identification from offline fixed-support parameter diagnostics.
Private source data and numerical arrays stay under ignored `outputs/`.

The experiment and independent audit are complete: see [RESULTS.md](RESULTS.md).
All16 native cells and240 unique offline fits passed audit. More data lowers
the selected X matrix error to15--19%, rather than eliminating it. The separate
known-topology per-line X MAPE is still55--66%; both metrics are retained.
See [RX_LITERATURE.md](RX_LITERATURE.md) for source-grounded paper comparisons.

The locked comparison uses January 21--February 1, 2024, because the original
January 4 eleven-day extension has missing source observations. The short fit
uses January 27--29, the long fit January 21--29; both validate January 29--31
and evaluate January 31--February 1. P/Q scales remain those known on January 4.

```powershell
python -m pytest rnj_wzzt_dr/experiments/history10_rx_20261116 -q
python -m rnj_wzzt_dr.experiments.history10_rx_20261116.run --prepare-only --workers 3
python -m rnj_wzzt_dr.experiments.history10_rx_20261116.run --resume
python -m rnj_wzzt_dr.experiments.history10_rx_20261116.run --offline-only
python -m rnj_wzzt_dr.experiments.history10_rx_20261116.audit_report --run outputs/history10_rx_20261116_v1 --out outputs/history10_rx_audit_20261116_v2
python -m rnj_wzzt_dr.experiments.history10_rx_20261116.line_metrics --run outputs/history10_rx_20261116_v1 --audit outputs/history10_rx_audit_20261116_v2/report/audit.json --out outputs/history10_rx_literature_metric_20261116_v2
```

Default output is `outputs/history10_rx_20261116_v1`. Preparation refuses an
existing directory. For a fresh reproduction choose a new `--out`; completed
fit attempts can be resumed after hash checks, while partial fits refuse an
automatic budget reset. The main native bank is newly generated from observed
training/validation only. Old candidate supports and the physical support are
used exclusively in the separately labeled offline parameter diagnostics.

`data.py` extracts profiles and checks AC, `run.py` seals inputs and candidates,
and `metrics.py` independently measures matrix, device-column, and excitation
errors. Source preparation and native fitting are separately traceable from
later `audit_report.py` post-processing, which records its own source hash.
