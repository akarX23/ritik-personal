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