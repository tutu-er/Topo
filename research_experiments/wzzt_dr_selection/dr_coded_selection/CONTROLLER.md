# Frozen actual controller

This isolated controller is prepared only after price locks and physical source
are fixed. Root starts the actual run. Importing the module performs no AC or LP.

`controller.py` accepts the price agent's `case_plans.json` with the complete
source-inventory `history_row`, and two frozen `locks` (`local`/`common`). Each
lock has eight `plans` as aligned 2x15 price arrays. The physical worker reads
waveform templates/directions directly from the same lock; this controller
does not reinterpret them.

Fixed pilot: paper15 b8/.001, eight policies x two laws x early/final = 32
paired records, one common four-day natural path. Early uses all original
288 training rows plus only the first event's first 84 quarter-hour rows;
validation remains the original 288 rows. Final uses 480/480 rows. A physical
worker may generate all four days, but later values do not enter the early
identifier or input hash. Truth is read only after identifier output returns.

Before preparation, finish all producer sources; subsequent source drift is
rejected. The actual root must be new. Commands (root supplies absolute paths):

```powershell
python -B scratch/dr_coded_runner_stage/controller.py prepare --root <new-actual-root> --case-plans <frozen-case_plans.json> --physical-worker <coded-actual_worker.py> --physical-module <coded_physical.py> --source-root D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source --replicate-base 20000 --participation-seed 90410000
python -B scratch/dr_coded_runner_stage/controller.py run --root <new-actual-root> --max-records 32
```

No AC or LP has been run by the controller author. Five pure tests check the
fixed checkpoint slices, full-history retention, overlapping cost accounting,
future failure/unknown costs, failure/unique-fit denominators, and mandatory
price completion plus source/lock freeze drift rejection:

```powershell
python -B scratch/dr_coded_runner_stage/test_controller.py
```

Full observed-data fitting rebuilds the original pool/RNJ/starts with 25s per
search path and a 180s identifier subprocess limit. Exact identical observed
inputs share one fit, including retaining failed attempts without a fresh solve
budget under another policy label. Partial directories are retained for review;
the controller never automatically restarts them.

Early and final fees overlap. Each record reports fees paid by its checkpoint,
worst remaining payment commitments, post-hoc realized future fees, full-contract
realized/worst fees, and accepted rebound obligations. A successful early result
never cancels a scheduled rebound or future event. Unknown costs stay null.
The price rule is planned for the final endpoint. Early is a prelocked mechanism
diagnostic and must not be described as an early-optimized price policy.
