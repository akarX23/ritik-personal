# Contract: Docker CLI

**Type**: Container interface  
**Scope**: `Dockerfile`, `requirements-docker.txt`, `.dockerignore`  
**Status**: Defined (Phase 1)

---

## Build Contract

```
docker build -t mnist-classifier .
```

**Preconditions**:
- Docker daemon is running
- `Dockerfile`, `requirements-docker.txt`, `src/`, `tests/` are present at build context root

**Postconditions**:
- Image tagged `mnist-classifier` is available locally
- Image runs on CPU only; no XPU/GPU driver dependencies are installed
- `python -m src.train --help` succeeds inside the image
- `pytest tests/ -v` succeeds inside the image

---

## Run Contracts

### Train (default)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-classifier
```

- Device: `cpu` (hardcoded in default `CMD`)
- Data: reads from `/app/data`; downloads MNIST if directory is empty or absent
- Results: writes CSV metrics, PNGs, and `model.pt` to `/app/results`
- Exit code 0 on success; non-zero on any failure

### Train (custom epochs)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-classifier --epochs 20 --batch 128 --lr 0.001
```

- All `src.train` CLI flags are forwarded through the entrypoint
- `--device` flag accepted; only `cpu` is valid inside the container

### Run Analysis

```bash
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  --entrypoint python mnist-classifier \
  -m src.analyze --results /app/results
```

- Reads CSVs from `/app/results`
- Writes PNG visualisations to `/app/results`
- Fails with clear message if required CSVs are absent

### Run Tests

```bash
docker run --rm mnist-classifier pytest tests/ -v
```

- Full test suite runs inside the container
- No data volume required for unit and contract tests
- Integration tests may download MNIST if no volume is mounted

---

## Entrypoint / CMD Schema

| Layer | Value |
|-------|-------|
| `ENTRYPOINT` | `["python", "-m", "src.train"]` |
| `CMD` | `["--device", "cpu", "--data", "/app/data", "--results", "/app/results"]` |

`CMD` is overridable; `ENTRYPOINT` is fixed to `src.train` unless overridden with `--entrypoint`.

---

## Volume Schema

| Volume | Container Path | Purpose |
|--------|----------------|---------|
| data | `/app/data` | MNIST dataset (optional; downloaded if absent) |
| results | `/app/results` | CSV metrics, PNGs, `model.pt` (persist beyond container) |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General runtime error (invalid args, missing files) |
| 2 | Device error (unsupported device; inside container only `cpu` is valid) |
