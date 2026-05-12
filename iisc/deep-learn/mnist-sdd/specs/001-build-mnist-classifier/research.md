# Research: MNIST Digit Classifier Pipeline

## Decision 1: Runtime devices are `cpu` and `xpu` only

- Decision: Expose device contract as `cpu` or `xpu`, where XPU represents the integrated laptop GPU.
- Rationale: The runtime environment does not provide CUDA GPU detection, and user intent is explicit CPU/XPU testing.
- Alternatives considered:
  - `cpu|gpu` generic naming: rejected because it does not match runtime behavior.
  - Automatic fallback from `xpu` to `cpu`: rejected by explicit no-fallback requirement.

## Decision 2: Use PyTorch XPU availability checks, not CUDA checks

- Decision: Device validation logic uses XPU capability checks through PyTorch XPU APIs.
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

## Decision 4: Analysis includes CPU vs XPU time comparison

- Decision: `analyze.py` generates a dedicated timing comparison output when both CPU and XPU run summaries are available.
- Rationale: Time comparison is a first-class requirement alongside quality metrics.
- Alternatives considered:
  - Manual spreadsheet comparison: rejected as non-reproducible workflow.
  - Console-only timing output: rejected because visual outputs are required for reporting.

## Decision 5: Keep dependencies minimal

- Decision: Use `torch`, `torchvision`, `matplotlib`, and stdlib `csv`/`logging`.
- Rationale: Minimal dependencies simplify setup and keep the project aligned with required outputs.
- Alternatives considered:
  - Pandas/seaborn: rejected as unnecessary for required outputs.
  - External experiment trackers: rejected as out of scope.

## Decision 6: Containerize with `python:3.11-slim`

- Decision: Use `python:3.11-slim` as the Docker base image.
- Rationale: `slim` minimizes image footprint while keeping glibc and pip needed for PyTorch wheels.
- Alternatives considered:
  - `python:3.11-alpine`: rejected due to musl/PyTorch wheel incompatibility.
  - `python:3.11` full image: rejected as larger than needed.

## Decision 7: Install CPU-only PyTorch in Docker; keep XPU wheel for host

- Decision: Docker installs `torch`/`torchvision` from the CPU index; host installs from XPU index with per-package scoping.
- Rationale: Docker is CPU-only by requirement; per-package index scoping avoids misrouting non-PyTorch dependencies.
- Alternatives considered:
  - Global `--index-url` in `requirements.txt`: rejected because it can route unrelated packages incorrectly.
  - Single host+docker requirements file: rejected to keep environment contracts explicit.

## Decision 8: Data provisioning uses mount-first plus download fallback

- Decision: Use `/app/data` volume mount when provided, otherwise allow torchvision download fallback.
- Rationale: Avoids baking dataset into image and supports first-run/CI usage.
- Alternatives considered:
  - Bake dataset into image: rejected due to image bloat and build-time coupling.
  - Require volume always: rejected due to weaker first-run ergonomics.

## Decision 9: Progress logging destination is console plus per-run file

- Decision: Emit progress logs to both console and a per-run log file in `results_dir`.
- Rationale: Console gives live feedback; files preserve history for reproducibility.
- Alternatives considered:
  - Console-only: rejected (no durable logs).
  - File-only: rejected (weak real-time feedback).

## Decision 10: Use plain-text concise per-epoch logs

- Decision: Use plain-text log lines with concise per-epoch entries plus lifecycle events.
- Rationale: Human-readable logs suit CLI workflows and keep noise low.
- Alternatives considered:
  - JSON logs: rejected as unnecessary complexity for current scope.
  - Per-batch logs: rejected as too verbose.

## Decision 11: Mandatory per-epoch log fields

- Decision: Each per-epoch log line includes `epoch`, `elapsed_seconds`, `loss`, and `accuracy`.
- Rationale: These are the minimal fields needed for progress and quality visibility.
- Alternatives considered:
  - Epoch-only logs: rejected as insufficient.
  - Expanded full-context logs each epoch: rejected as noisy.

## Decision 12: Per-run log file naming

- Decision: Name each run log file `run_<run_id>.log` in `results_dir`.
- Rationale: Deterministic naming maps logs directly to run artifacts.
- Alternatives considered:
  - Single shared `training.log`: rejected due to mixed-run ambiguity.
  - Timestamp-only naming: rejected due to weaker run linkage.

## Decision 13: Standard run quality threshold is 97% test accuracy

- Decision: Set SC-002 baseline to `>= 97%` test accuracy for standard training runs.
- Rationale: Updated clarification defines the target and aligns acceptance criteria with expected baseline.
- Alternatives considered:
  - Keep `>= 98%`: rejected by latest clarification.
  - Lower than `>= 97%`: rejected because it weakens quality expectations without need.


## Decision 14: FR-014 data provisioning requires two explicit runtime validation paths

- Decision: Add contract/integration tests that verify (a) mounted-data path where torchvision reads pre-existing MNIST files and (b) no-mount/download-fallback path where torchvision fetches MNIST automatically.
- Rationale: Volume mount is the primary expected usage; download fallback is a required safety contract. Neither can be verified purely via static code review.
- Alternatives considered:
  - Single test covering only the download path: rejected — leaves mount path and fallback-only behavior unvalidated.
  - Manual verification step only: rejected — not reproducible and not aligned with test-first requirements.

## Decision 15: Multi-batch CLI uses comma-separated `--batches`

- Decision: Extend `src.train` CLI with `--batches` accepting comma-separated integers (for example `--batches 32,64,128`).
- Rationale: Compact syntax scales well for experimentation and avoids repeated flag parsing ambiguity.
- Alternatives considered:
  - Repeated `--batch` flags: rejected to keep parser behavior simple and explicit.
  - Range-based flags: rejected because arbitrary non-uniform batch sets are common.

## Decision 16: Execute batch sizes sequentially with distinct run IDs

- Decision: For one multi-batch command, run each batch-size configuration sequentially and assign each execution its own `run_id`.
- Rationale: Distinct run IDs preserve per-run traceability while allowing shared CSV storage.
- Alternatives considered:
  - Single `run_id` for the whole command: rejected due to ambiguous per-batch lineage.
  - Parallel batch execution: rejected because it complicates deterministic resource use and reproducibility.

## Decision 17: Add `batch_size` column to persisted metric rows

- Decision: Include `batch_size` in train/validation/test metric rows (and summary/report contexts) so cross-batch comparisons can be reconstructed from shared files.
- Rationale: Required for unambiguous historical comparisons when multiple runs share one directory.
- Alternatives considered:
  - Per-batch subdirectories only: rejected because requirement is shared CSV files.
  - Infer batch size from command logs: rejected as brittle and non-relational.

## Decision 18: `results.md` includes two tables plus configuration block

- Decision: Generate `results.md` with (a) final train/validation/test loss+accuracy per batch size and (b) epoch comparison sampled every 10 epochs plus final epoch; include compared-run configuration metadata.
- Rationale: Provides both concise outcomes and trend inspection while keeping report readable.
- Alternatives considered:
  - Final metrics only: rejected because trend behavior would be hidden.
  - All epochs table: rejected as too verbose for larger runs.

## Decision 19: `results.md` compares historical matching rows in results directory

- Decision: Comparison scope includes all matching historical rows already present in selected results-directory CSVs.
- Rationale: Enables cumulative experiment tracking without extra orchestration.
- Alternatives considered:
  - Current command rows only: rejected by clarification.
  - Optional scope flag: deferred as non-required complexity.

## Decision 20: `--filter-batch` / `-b` optional flag scopes curves and epoch-comparison; final-metrics table unfiltered

- Decision: Add optional `--filter-batch INT` (short form `-b`) to `src/analyze.py`. When provided: learning-curve plots and the epoch-comparison table in `results.md` include only rows matching that batch size. The Final Metrics by Batch Size table always shows all batch sizes. When the flag is omitted, all existing all-batches behavior is preserved. When the specified batch size is absent from the CSV files, execution fails fast with a clear error that lists all available batch sizes found in the files.
- Rationale: Users running multi-batch experiments need per-batch curve inspection without losing the cross-batch summary. Fail-fast with an available-list error follows FR-009 (actionable failure feedback) and avoids silent empty output.
- Alternatives considered:
  - Require `--filter-batch` always: rejected — breaks backward compatibility and forces users to look up a batch size before every analysis invocation.
  - Silent empty output on missing batch: rejected — produces confusing blank plots with no remediation path.
  - Separate `--filter-epoch-table` flag for just the epoch-comparison: rejected — over-engineering; users who filter curves want consistent scoping in the report.
