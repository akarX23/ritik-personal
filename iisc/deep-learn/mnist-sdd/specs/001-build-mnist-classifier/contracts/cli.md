# CLI Contract: MNIST Digit Classifier Pipeline

## `python -m src.train`

Purpose: train the MNIST model, run scheduled validation/testing, and write metrics CSV files.

Arguments:
- `-e` or `--epochs`: number of training epochs, default `10`
- `-r` or `--results`: results directory, default `./results`
- `-d` or `--device`: required device selection, `cpu` or `xpu`
- `-b` or `--batch`: batch size, default `64`
- `-lr` or `--lr`: learning rate, default `0.001`
- `-m` or `--data`: MNIST data directory, default `./data`

Behavior:
- Trains the fixed `784 -> 256 -> 128 -> 10` PyTorch model.
- Records per-epoch loss, accuracy, elapsed time, and device in CSV files.
- Runs validation and test checks every 5 epochs.
- Logs total run time as `training_time_seconds` in `run_summary.csv`.
- Emits plain-text progress logs to console and to `<results_dir>/run_<run_id>.log`.
- Emits concise per-epoch log lines with `epoch`, `elapsed_seconds`, `loss`, and `accuracy`.
- Fails if the requested device is unavailable (no automatic fallback).

Outputs:
- `metrics_train.csv`
- `metrics_validation.csv`
- `metrics_test.csv`
- `run_summary.csv`
- `predictions.csv` (final test predictions and labels)
- model checkpoint file (`model.pt`)
- per-run plain-text log file (`run_<run_id>.log`)

## `python -m src.analyze`

Purpose: generate curves and comparisons from saved CSV outputs.

Arguments:
- `-r` or `--results`: results directory, required

Behavior:
- Reads CSV files from the results directory.
- Generates training, validation, and testing curves.
- Produces classification metrics visualizations.
- Produces CPU-vs-XPU comparison plots when both device runs are available.
- Produces CPU-vs-XPU training time comparison from `run_summary.csv` entries.
- Emits lifecycle progress logs to console and per-run plain-text log file in the results directory.

Outputs:
- learning curve images
- classification report plots
- confusion matrix image
- device quality comparison image when both device datasets exist
- device time comparison image when both device summaries exist
- per-run plain-text log file (`run_<run_id>.log`)