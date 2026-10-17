# Compact frozen G-precision evidence

All ten raw records remain unchanged in the original D-stage output. Their exact SHA256 and sizes, and each compact export SHA256, are in raw_and_export_sha256.json.

The compact records preserve observations, meter models, noise, configuration, candidate structures, trained history, response coefficients, plans, parents, complete MILP logs/bounds, final FIX/JNT states, block/channel SSE, decisions, costs and failures. Repeated numerical arrays inside optimization starts and final residual vectors are replaced by explicit hash/size receipts; no endpoint is replaced or refitted.

Raw gzip records: 33,299,189 bytes. Compact gzip records: 8,004,114 bytes.

source_snapshot.zip is the frozen scientific source. The analysis postprocessor is intentionally outside the science hash closure and is archived separately with its own SHA256. original_analysis preserves the full-batch original summary and audit; development_audit_passed preserves the first successful development audit.

To replay the independent forward-AC and SSE audit after restoring the source:

```powershell
Expand-Archive -LiteralPath '<evidence>/source_snapshot.zip' -DestinationPath '<restore-root>'
Copy-Item -LiteralPath '<evidence>/analysis_postprocessor.py' -Destination '<restore-root>/research_experiments/mobile_ev_mle/g_precision/analysis.py'
Set-Location -LiteralPath '<restore-root>'
python -m research_experiments.mobile_ev_mle.g_precision.analysis --output '<absolute-evidence-path>'
```

This requires the recorded Python/numerical environment but no MILP solver or new fitting. Complete simulation and fitting cost remains the original experiment's cost; replay calls are recorded separately.

Replay audit passed: True; issues: 0; forward AC calls: 1920.
Scientific summary, audit and history-pair content identical: True.
