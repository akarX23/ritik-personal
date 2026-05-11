# CLI Contract: MNIST Digit Classifier Pipeline

## `train.py`

Purpose: train the MNIST model, run scheduled validation/testing, and write metrics CSV files.

Arguments:
- `-e` or `--epochs`: number of training epochs, default `10`
- `-r` or `--results`: results directory, default `./results`
- `-d` or `--device`: required device selection, `cpu` or `gpu`
- `-b` or `--batch`: batch size, default `64`
- `-lr` or `--lr`: learning rate, default `0.001`
- `-m` or `--data`: MNIST data directory, default `./data`

Behavior:
- Trains the fixed `784 -> 256 -> 128 -> 10` PyTorch model.
- Records per-epoch loss, accuracy, elapsed time, and device in CSV files.
- Runs validation and test checks every 5 epochs.
- Fails if the requested device is unavailable.

Outputs:
- `metrics_train.csv`
- `metrics_validation.csv`
- `metrics_test.csv`
- `run_summary.csv`
- model checkpoint file

## `analyze.py`

Purpose: generate curves and comparisons from saved CSV outputs.

Arguments:
- `-r` or `--results`: results directory, required

Behavior:
- Reads CSV files from the results directory.
- Generates training, validation, and testing curves.
- Produces classification metrics visualizations.
- Produces CPU-vs-GPU comparison plots when both device runs are available.

Outputs:
- learning curve images
- classification report plots
- confusion matrix image
- device comparison image when both device datasets exist