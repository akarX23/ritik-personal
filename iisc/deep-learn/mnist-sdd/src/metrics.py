"""CSV metric-writing helpers for the MNIST training pipeline.

All writers use the stdlib ``csv`` module only — no pandas dependency.
Files are created on first write and appended on subsequent calls.
Column schemas match the entities defined in data-model.md.
"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Column definitions (kept in sync with data-model.md)
# ---------------------------------------------------------------------------

TRAIN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "epoch",
    "split",
    "loss",
    "accuracy",
    "elapsed_seconds",
    "device",
)

EVAL_COLUMNS: tuple[str, ...] = (
    "run_id",
    "epoch",
    "split",
    "loss",
    "accuracy",
    "elapsed_seconds",
    "device",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "device",
    "epochs",
    "start_time",
    "end_time",
    "training_time_seconds",
    "final_train_loss",
    "final_train_accuracy",
    "final_val_loss",
    "final_val_accuracy",
    "final_test_loss",
    "final_test_accuracy",
    "status",
    "results_dir",
)

PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "sample_index",
    "true_label",
    "predicted_label",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_row(path: Path, columns: tuple[str, ...], row: dict) -> None:
    """Append *row* to the CSV at *path*, writing a header if the file is new."""
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_train_row(results_dir: Path, row: dict) -> None:
    """Append one training-epoch row to ``metrics_train.csv``."""
    _append_row(results_dir / "metrics_train.csv", TRAIN_COLUMNS, row)


def write_eval_row(results_dir: Path, split: str, row: dict) -> None:
    """Append one evaluation row to the appropriate CSV.

    Parameters
    ----------
    split:
        ``"validation"`` or ``"test"``
    """
    filename = f"metrics_{split}.csv"
    _append_row(results_dir / filename, EVAL_COLUMNS, row)


def write_summary_row(results_dir: Path, row: dict) -> None:
    """Write (or overwrite) the run summary CSV."""
    path = results_dir / "run_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})


def write_prediction_row(results_dir: Path, row: dict) -> None:
    """Append one prediction row to ``predictions.csv``."""
    _append_row(results_dir / "predictions.csv", PREDICTIONS_COLUMNS, row)


def new_run_id() -> str:
    """Return a fresh UUID4 string to identify a training run."""
    return str(uuid.uuid4())


def run_log_path(results_dir: Path, run_id: str) -> Path:
    """Return the canonical per-run log path for *run_id*."""
    return results_dir / f"run_{run_id}.log"
