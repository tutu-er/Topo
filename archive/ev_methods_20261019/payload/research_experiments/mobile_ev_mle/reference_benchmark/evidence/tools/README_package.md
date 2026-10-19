# Frozen reference-benchmark evidence packaging

This tool lives outside the 893-file scientific freeze. It does not run or tune
planning, fitting, thresholds, forecasts, or a new scientific batch.

After controller `statistical_complete`, primary analysis, and diagnostics:

```powershell
D:/apps/miniconda3/envs/Topo/python.exe outputs/reference_tools/package_evidence.py --output outputs/reference_v1 --destination outputs/reference_delivery_v1
```

The destination must not already exist. The tool refuses to read actual
calibration/evaluation records while the formal controller is still running.
A passed `audit.json`, matching freeze/controller/summary identity, complete
phase condition manifest and byte-verified source snapshot are required.

Outputs:

- `bundle/`: compact statistical inputs for every declared condition; all
  failures and unresolved denominators; original raw registry; exact prepared
  V2 windows; complete frozen source ZIP; licences and source receipts; reports,
  original thresholds/audits/summary; losslessly shared physical fixtures.
- `raw_representatives.zip`: full byte-exact original files for each network ×
  source at the minimum declared development index. Selection uses the calendar
  index only. This separate large attachment stays local by default.
- `<run>_evidence.zip`: duplicate archive of `bundle/`, target at most 20 MiB.
- `delivery_manifest.json`: `bundle_files` maps relative names to SHA256/bytes;
  `core_archive`, `full_attachment`, `restore_audit` provide checksums and scope;
  `ready_for_review` requires size, numerical replay and restore checks to pass.
- `restore_check/`: verified extraction of the core; the complete scientific
  source is restored here. Independent Python processes regenerate numerical
  statistics and physically replay all fixed representatives.
- `restore_audit.json`, `package_receipt.json`, `restored_*.log`: evidence of the
  actual restore/replay. Failure leaves files and logs for inspection.

The core physical projection removes only requested/initial/attained states
inside optimizer restart logs. It retains final line/source states, all meters,
failures, local solver diagnostics and bounds. Every omitted block has its
original JSON path, semantic SHA256 and canonical byte count. Duplicate blocks
are losslessly content addressed. Original and projected numerical AC audits
must be identical; restored fixtures receive the same audit in a fresh process.

Compact/calibration bytes differ from original raw bytes. Reproduction compares
pure threshold payloads and all numerical statistics; it never regenerates,
overwrites, or rebrands the original threshold identity. The original thresholds
are copied byte-for-byte. Original full raw SHA256s are in the registry. The
raw attachment preserves exact original gzip bytes; core shared blocks restore
the projected JSON object, including every retained physical observation/state.

Publisher should commit `bundle/` and `delivery_manifest.json` only, without the
large duplicate archives. No Git operation is performed by this tool.

If Windows status-file recovery was used, its launcher, tests, receipts and
per-process retry logs are saved separately under `provenance/runtime_tools/`
and `provenance/runtime_history/`. These retries call the original serializer;
the frozen scientific source, configuration, samples and failed results remain
unchanged. For a compute continuation, place runtime tools back under the
execution source's `outputs/reference_tools/`; archive locations are provenance
copies, not alternate source roots.

Development-only validation:

```powershell
D:/apps/miniconda3/envs/Topo/python.exe outputs/reference_tools/package_evidence.py self-test --source-root . --sample-dev outputs/pilot_v1/development/history_000_cigre_residential_none.json.gz
```

This tests saved development data, tiny artificial three-phase dictionaries,
path safety, byte tampering, original/projected AC replay and statistical
sufficiency. It reads no real calibration/evaluation records.
