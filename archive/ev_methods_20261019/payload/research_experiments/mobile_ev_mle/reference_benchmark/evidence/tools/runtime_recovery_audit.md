# Frozen V1 controller status and safe recovery

Read-only inspection on 2026-10-18, about 12:49–12:51 Asia/Shanghai.

## What is observed

- Original controller PID 29528 and all 16 recorded worker PIDs are alive. The
  controller used about 3.50 CPU seconds; workers used thousands of CPU seconds.
  Controller threads are waiting. No existing process was terminated or debugged.
- `controller.json` last records 10 completed development conditions at
  03:00:24 UTC. `controller.json.tmp` contains the same state with a later
  03:00:29 UTC heartbeat. The `.tmp` has not replaced the old file.
- There were 21 complete raw result files when inspected. Worker stdout still
  showed completed conditions and active evaluations after the controller
  heartbeat stopped. `controller.stderr.log` was empty.
- `io.write` writes a temporary file and then calls `tmp.replace(path)` without a
  retry. An exception inside the executor `with` scope first calls
  `ProcessPoolExecutor.__exit__`, whose local Python implementation waits for its
  manager thread and all already submitted work. The outer controller exception
  handler only runs after that wait.

These facts match a temporary-file replacement error followed by executor
shutdown waiting. This remains an inference until the old controller exits and
records its actual traceback. The state-file age alone does not show that the
workers stopped, and it is unsafe to launch another copy while they are alive.

## Independent mechanism test

`runtime_resume.py --self-test` passed. In a temporary output file:

1. An ordinary Windows Python read handle held open on the target made the
   original `io.write` raise `PermissionError` with `winerror=5` during replacement.
2. Closing the handle after 0.12 seconds allowed the bounded-retry wrapper to
   succeed, with unchanged record content.
3. Injected permission failures on the first two calls recovered on call three.
   Non-permission `ValueError` and `OSError` propagated on their first calls.
4. Original and wrapped JSON and gzip bytes were identical.
5. Two Windows-spawn children installed the wrapper before importing scientific
   study functions. No scientific fitting or planning was called by the tests.

The receipt is `runtime_resume_selftest/receipt.json`. The test proves the
Windows locking mechanism; it does not identify which reader, if any, caused the
old controller's original error.

## Minimal runtime repair

`runtime_resume.py` is outside the frozen source directories. At module import,
including in Windows-spawn workers, it wraps only
`reference_benchmark.io.write`. It calls the original serializer, gzip encoder,
temporary-file write and replacement. It retries only `PermissionError`, at
most eight times (delays 0.05, 0.1, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5 seconds), then
propagates it. Other exceptions propagate immediately. Logs are separate by PID
under `outputs/reference_v1/runtime_io/`, with runtime/source hashes and retries.

It also exposes `read_shared_json(path)`, whose Windows handle permits
`FILE_SHARE_DELETE`, so a status reader does not itself prevent atomic replace.
Use that helper for live controller reads in the recovery supervisor.

## Safe continuation

Do not start the following command until the original controller **and every
recorded worker** has exited and the original `controller.lock` has been removed
by its owner. The tool refuses immediately if any old recorded PID or lock
remains; it neither waits internally, removes a lock nor terminates a process.

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' `
  'outputs/reference_tools/runtime_resume.py' --resume `
  --output 'outputs/reference_v1' `
  --prepared 'outputs/inputs_v2/prepared.json.gz' --workers 16
```

Before calling the original controller it verifies the original freeze, every
scientific-source hash, prepared input hash, configuration, and all stored raw
condition identities. Already saved failures remain complete computational
results and are not resampled. The controller retains its original development,
calibration and evaluation phase order and only submits missing conditions.

A separate supervisor may wait for the old run to finish, then execute this
command and finally the existing postprocessor. If the eventual traceback has a
different cause, retain it and reassess; do not call this mechanism confirmed
from the present inference alone.
