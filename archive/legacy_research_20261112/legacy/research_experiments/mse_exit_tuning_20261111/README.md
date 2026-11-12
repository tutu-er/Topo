# Final-E pure relative-MSE gate

This isolated experiment changes only the acceptance of the already-solved
final E proposal. Forward WZZT, T, the proposal count penalty and .24 output
mapping remain frozen. It does not change the Gurobi stopping parameters.

The reusable function is pure_mse_gate in gate.py:

    decision = pure_mse_gate(old_mse, raw_proposal_mse, K=K, tau=.001)

Accept only if decision["accepted"], after checking the proposal's feasibility
and construction/time guards. As in the original algorithm, acceptance is
followed by fixed-family QP fitting and numerical-zero cleanup.

run.py freezes the controller files and replays 48 existing T003 E proposals.
Each input uses a fresh process; only previously unfitted, newly admitted
families need a QP. No structural MIQP is rerun. analyze.py freezes a validation-
only selection.json before topology/test evaluation, using grouped leave-seed-
out calibration of the mapped endpoint MSE ratio.

Run from the repository:

    python -B research_experiments/mse_exit_tuning_20261111/run.py freeze outputs/mse_exit_tuning_20261111
    python -B research_experiments/mse_exit_tuning_20261111/run.py run outputs/mse_exit_tuning_20261111 --workers 8
    python -B -m research_experiments.mse_exit_tuning_20261111.analyze outputs/mse_exit_tuning_20261111
    python -B -m research_experiments.mse_exit_tuning_20261111.figures outputs/mse_exit_tuning_20261111

Do not overwrite an existing protocol, result or calibration decision. This
historical calibration is not a fresh-seed prospective experiment.
