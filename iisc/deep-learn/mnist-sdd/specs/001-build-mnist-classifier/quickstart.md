# Quickstart: MNIST Digit Classifier Pipeline

## Install

Install the minimal dependency set:

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

Run a paired CPU and XPU experiment for comparison:

```bash
python -m src.train -d cpu -e 10 -r ./results/cpu-run -m ./data
python -m src.train -d xpu -e 10 -r ./results/xpu-run -m ./data
```

## Analyze

Generate curves and metrics from saved CSV files:

```bash
python -m src.analyze -r ./results
```

## Expected Outputs

- CSV files with per-epoch metrics and device metadata
- CSV files include epoch-level elapsed time and run-level training time
- model checkpoint file
- per-run plain-text log file named `run_<run_id>.log`
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

---

## Docker (CPU-only, portable)

### Build

```bash
docker build -t mnist-classifier .
```

### Train (default — CPU, volumes mounted from host)

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

### Run full test suite inside container

```bash
docker run --rm mnist-classifier pytest tests/ -v
```

### Notes

- The container runs CPU only; `--device xpu` will fail with a clear error inside Docker.
- Results written to `/app/results` are only persisted if a host volume is mounted.
- XPU experiments must be run directly on the host using `requirements.txt`.