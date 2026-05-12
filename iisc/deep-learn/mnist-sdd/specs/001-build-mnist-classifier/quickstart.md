# Quickstart: MNIST Digit Classifier Pipeline

## Install

Install the dependency set:

```bash
python -m pip install -r requirements.txt
```

## Train

Run training with explicit device selection:

```bash
python -m src.train -d cpu
```

Example with custom options:

```bash
python -m src.train -d xpu -e 20 -r ./results/xpu-run -b 64 -lr 0.001 -m ./data
```

Run multi-batch experiments in one command:

```bash
python -m src.train -d cpu -e 20 -r ./results/multi-batch -lr 0.001 -m ./data --batches 32,64,128
```

The command executes batch sizes sequentially and appends rows to shared CSV files with a `batch_size` column.

Run paired CPU and XPU experiments for comparison:

```bash
python -m src.train -d cpu -e 10 -r ./results/cpu-run -m ./data
python -m src.train -d xpu -e 10 -r ./results/xpu-run -m ./data
```

## Analyze

Generate curves and metrics from saved CSV files:

```bash
python -m src.analyze -r ./results
```

`analyze.py` also writes `results.md` that compares matching historical rows in the selected results directory.

### Filter learning curves and epoch-comparison to one batch size

When results contain multiple batch sizes, use `--filter-batch` (`-b`) to scope curves and the epoch-comparison table to a single batch:

```bash
python -m src.analyze -r ./results --filter-batch 64
# or using the short form:
python -m src.analyze -r ./results -b 64
```

The Final Metrics by Batch Size table in `results.md` is always shown for all batch sizes regardless of `--filter-batch`.

If the specified batch size is not found in the CSV files, the command exits with a non-zero status and prints all available batch sizes.

## Validate Standard Accuracy Target (SC-002)

Check the most recent run summary row and confirm `test_accuracy >= 0.97`:

```bash
tail -n 1 ./results/run_summary.csv
```

## Validate Batch-Comparison Report

```bash
grep -n "Final Metrics by Batch Size\|Epoch Comparison" ./results/results.md
```

Confirm both sections exist and configuration details are present near the top of `results.md`.

## Expected Outputs

- CSV files with per-epoch metrics and device metadata
- CSV metric rows include `batch_size` for shared-file batch comparisons
- CSV files include epoch-level elapsed time and run-level training time
- model checkpoint file (`model.pt`)
- per-run plain-text log file named `run_<run_id>.log`
- learning curve plots
- classification metrics visualizations
- CPU-vs-XPU quality comparison plot when both runs are available
- CPU-vs-XPU training-time comparison plot when both runs are available
- markdown comparison report (`results.md`) with:
  - final metrics table by batch size (train/validation/test loss+accuracy)
  - epoch-sampled comparison table (multiples of 10 plus final epoch)
  - compared-run configuration summary (device, epochs, learning rate, batch-size list)

## Validate Code Quality (CAR-002)

Run lint and type checks on changed modules before committing:

```bash
python -m flake8 src/ tests/ --max-line-length 120
python -m mypy src/ --ignore-missing-imports
```

## Validate SC-001 Reliability (≥95% success)

Run the primary workflow 3 times and confirm all succeed:

```bash
for i in 1 2 3; do
  python -m src.train -d cpu -e 10 -r ./results/reliability-run-$i -m ./data
done
```

All three should exit 0 and produce model.pt and metrics CSVs.

## Notes

- Device selection is explicit; scripts do not switch devices automatically.
- In this environment, `xpu` is the integrated GPU target; `cpu` is always available.
- If `xpu` is requested and unavailable, execution fails with a clear error.
- Default dataset directory is `./data`.
- Keep results in separate directories per run to avoid CSV overwrite.

---

## Docker (CPU-only, portable)

### Build

```bash
docker build -t mnist-classifier .
```

### Train (default)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-classifier
```

MNIST data is downloaded automatically if `/app/data` is empty or not mounted.

### Train with custom options

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-classifier --epochs 20 --batch 128 --lr 0.001
```

### Run analysis

```bash
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  --entrypoint python mnist-classifier \
  -m src.analyze --results /app/results
```

### Run tests inside container

```bash
docker run --rm mnist-classifier pytest tests/ -v
```

### Docker Notes

- The container is CPU-only; `--device xpu` fails fast inside Docker.
- Results in `/app/results` persist only when a host volume is mounted.
- XPU experiments must be run directly on host using `requirements.txt`.
