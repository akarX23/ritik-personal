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
python src/train.py -d xpu -e 20 -r ./results/xpu-run -b 64 -lr 0.001 -m ./data
```

Run a paired CPU and XPU experiment for comparison:

```bash
python src/train.py -d cpu -e 10 -r ./results/cpu-run -m ./data
python src/train.py -d xpu -e 10 -r ./results/xpu-run -m ./data
```

## Analyze

Generate curves and metrics from saved CSV files:

```bash
python src/analyze.py -r ./results
```

## Expected Outputs

- CSV files with per-epoch metrics and device metadata
- CSV files include epoch-level elapsed time and run-level training time
- model checkpoint file
- learning curve plots
- classification metrics visualizations
- CPU-vs-XPU quality comparison plot when both runs are available
- CPU-vs-XPU training-time comparison plot when both runs are available

## Notes

- Device selection is explicit; the scripts must not switch devices automatically.
- In this environment, `xpu` is the integrated laptop GPU target.
- If `xpu` is requested and unavailable, execution fails with a clear error.
- The default dataset directory is `./data`.
- Keep results in separate directories per run to avoid overwriting CSV outputs.