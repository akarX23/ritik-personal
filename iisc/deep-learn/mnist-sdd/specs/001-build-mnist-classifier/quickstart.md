# Quickstart: MNIST Digit Classifier Pipeline

## Install

Install the minimal dependency set:

```bash
python -m pip install -r requirements.txt
```

## Train

Run training with explicit device selection:

```bash
python src/train.py -d cpu
```

Example with custom options:

```bash
python src/train.py -d gpu -e 20 -r ./results -b 64 -lr 0.001 -m ./data
```

## Analyze

Generate curves and metrics from saved CSV files:

```bash
python src/analyze.py -r ./results
```

## Expected Outputs

- CSV files with per-epoch metrics and device metadata
- model checkpoint file
- learning curve plots
- classification metrics visualizations
- CPU-vs-GPU comparison plot when both runs are available

## Notes

- Device selection is explicit; the scripts must not switch devices automatically.
- The default dataset directory is `./data`.
- Keep results in separate directories per run to avoid overwriting CSV outputs.