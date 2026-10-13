Raw run artifacts copied from pop-os `/ml2/altgan/runs/`.

- `gpu-run-2/`: decision-label-fix GPU run. Contains `launch.sh` (exact command), `command.json`, `hardware.json`, `split.json`, `results.json`, `local_results.json`, `baseline_results.json`, `cell_eval_results.json`, and the full `main.log` (75 KB, untrimmed). It does not include the checkpoint, predictions, or h5ad files, which total about 450 MB.
- `gpu-run-1/command.json`: the saved command of the unpublished 20,000-step GPU run 1 (`/ml2/altgan/runs/gpu-run-1`), which RESULTS_run2.md compares against. It is a different run from `results/real_data_run_1` (2,000 steps on CPU, batch size 64), which RESULTS.md documents.
