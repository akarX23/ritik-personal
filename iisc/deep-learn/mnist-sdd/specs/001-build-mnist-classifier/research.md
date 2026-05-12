# Research: MNIST Digit Classifier Pipeline

## Decision 1: Runtime devices are `cpu` and `xpu` only

- Decision: Expose device contract as `cpu` or `xpu`, where XPU represents the integrated laptop GPU.
- Rationale: The runtime environment does not provide CUDA GPU detection, and user intent is explicit CPU/XPU testing.
- Alternatives considered:
	- `cpu|gpu` generic naming: rejected because it does not match runtime behavior.
	- Automatic fallback from `xpu` to `cpu`: rejected by explicit no-fallback requirement.

## Decision 2: Use PyTorch XPU availability checks, not CUDA checks

- Decision: Device validation logic will use XPU capability checks through PyTorch XPU APIs.
- Rationale: CUDA checks are irrelevant in this environment and would produce misleading behavior.
- Alternatives considered:
	- `torch.cuda.is_available()`: rejected because CUDA/GPU is not detected here.
	- Silent fallback if XPU unavailable: rejected due to fail-fast policy.

## Decision 3: Persist both per-epoch and run-level timing in CSV

- Decision: Record `elapsed_seconds` per epoch and `training_time_seconds` per run in CSV artifacts.
- Rationale: The feature requires reproducible, quantitative CPU vs XPU time comparison.
- Alternatives considered:
	- Logging only per-epoch time: rejected because total-run comparison becomes indirect.
	- Logging only total time: rejected because epoch-level diagnostics are needed for curve interpretation.

## Decision 4: Analysis must include CPU vs XPU time comparison

- Decision: `analyze.py` generates a dedicated timing comparison output when both CPU and XPU run summaries are available.
- Rationale: Time comparison is now a first-class requirement alongside quality metrics.
- Alternatives considered:
	- Manual spreadsheet comparison: rejected as non-reproducible workflow.
	- Console-only timing output: rejected because visual outputs are required for reporting.

## Decision 5: Keep dependencies minimal

- Decision: Use `torch`, `torchvision`, `matplotlib`, and stdlib `csv`.
- Rationale: Minimal dependencies simplify setup and keep the project aligned with the existing plan.
- Alternatives considered:
	- Pandas/seaborn: rejected as unnecessary for required outputs.
	- Additional experiment trackers: rejected as out of local-scope workflow.

## Decision 6: Containerise with `python:3.11-slim` base image

- Decision: Use `python:3.11-slim` as the Docker base image.
- Rationale: `slim` removes non-essential OS packages (docs, man pages, apt cache) while keeping glibc and pip, which are required for PyTorch wheel installation. Alpine would require building PyTorch from source due to musl libc incompatibility.
- Alternatives considered:
	- `python:3.11-alpine`: rejected — musl libc incompatible with pre-built PyTorch wheels; requires full build toolchain, inflating image more than `slim`.
	- `python:3.11` (full Debian): rejected — unnecessarily large; `slim` is sufficient.
	- `ubuntu:22.04` + manual Python install: rejected — more Dockerfile complexity; no benefit over official Python image.

## Decision 7: Install CPU-only PyTorch wheel inside Docker; keep XPU wheel for host only

- Decision: Inside the Dockerfile, install `torch` and `torchvision` from `https://download.pytorch.org/whl/cpu`. On the host, `requirements.txt` installs `torch` and `torchvision` from the XPU nightly index, scoped per-package.
- Rationale: XPU passthrough inside Docker is not supported (requires kernel-level device access and Intel GPU userspace drivers that inflate image and break portability). CPU-only wheel is significantly smaller (~700 MB vs ~1.4 GB for XPU nightly). `requirements.txt` must not apply `--index-url` globally or it would route `matplotlib` and `pytest` through the XPU nightly index unnecessarily.
- Alternatives considered:
	- Single `requirements.txt` with global `--index-url`: rejected — applies XPU index to all packages including `matplotlib` and `pytest` which have no XPU-specific versions and may fail or silently install wrong wheels.
	- Separate `requirements-host.txt` and `requirements-docker.txt`: evaluated — opted for per-package `--index-url` in `requirements.txt` for host, and a separate `requirements-docker.txt` (containing only `matplotlib` and `pytest`) used by the Dockerfile.

## Decision 8: Data provisioning — volume mount with download fallback

- Decision: The Dockerfile declares `/app/data` as a `VOLUME`. When the user mounts `./data:/app/data`, torchvision uses pre-existing files. When no volume is mounted, `torchvision.datasets.MNIST(download=True)` fetches the dataset automatically at runtime.
- Rationale: Baking the 11 MB dataset into the image adds unnecessary layer size and requires the data to exist at build time. A volume mount is the standard Docker pattern for mutable data. The download fallback keeps the image self-contained for CI and first-run scenarios.
- Alternatives considered:
	- Bake dataset into image via `COPY data/`: rejected — inflates image; fails if data not present at build time.
	- Require volume mount with no fallback: rejected — breaks CI and first-run user experience.