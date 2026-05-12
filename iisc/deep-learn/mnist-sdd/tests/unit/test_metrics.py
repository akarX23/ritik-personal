"""T010 – Unit tests for metrics.py CSV writers.

Validates column schemas, append behaviour, and file creation.
Tests must FAIL before src/metrics.py is implemented.
"""

import csv
import pytest
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# write_train_row
# ---------------------------------------------------------------------------


def test_train_row_creates_file(tmp_path):
    from src.metrics import write_train_row, TRAIN_COLUMNS

    row = dict(run_id="r1", epoch=1, split="train", loss=0.5, accuracy=0.9,
               elapsed_seconds=2.3, device="cpu")
    write_train_row(tmp_path, row)

    assert (tmp_path / "metrics_train.csv").exists()


def test_train_row_correct_columns(tmp_path):
    from src.metrics import write_train_row, TRAIN_COLUMNS

    row = dict(run_id="r1", epoch=1, split="train", loss=0.5, accuracy=0.9,
               elapsed_seconds=2.3, device="cpu")
    write_train_row(tmp_path, row)

    rows = _read_csv(tmp_path / "metrics_train.csv")
    assert set(rows[0].keys()) == set(TRAIN_COLUMNS)


def test_train_row_appends(tmp_path):
    from src.metrics import write_train_row

    row = dict(run_id="r1", epoch=1, split="train", loss=0.5, accuracy=0.9,
               elapsed_seconds=2.3, device="cpu")
    write_train_row(tmp_path, row)
    write_train_row(tmp_path, {**row, "epoch": 2})

    rows = _read_csv(tmp_path / "metrics_train.csv")
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# write_eval_row
# ---------------------------------------------------------------------------


def test_eval_row_validation_split(tmp_path):
    from src.metrics import write_eval_row

    row = dict(run_id="r1", epoch=5, split="validation", loss=0.3, accuracy=0.95,
               elapsed_seconds=0.8, device="cpu")
    write_eval_row(tmp_path, "validation", row)

    assert (tmp_path / "metrics_validation.csv").exists()


def test_eval_row_test_split(tmp_path):
    from src.metrics import write_eval_row

    row = dict(run_id="r1", epoch=5, split="test", loss=0.2, accuracy=0.98,
               elapsed_seconds=0.5, device="cpu")
    write_eval_row(tmp_path, "test", row)

    assert (tmp_path / "metrics_test.csv").exists()


# ---------------------------------------------------------------------------
# write_summary_row
# ---------------------------------------------------------------------------


def test_summary_row_has_timing_columns(tmp_path):
    from src.metrics import write_summary_row, SUMMARY_COLUMNS

    row = {col: "x" for col in SUMMARY_COLUMNS}
    write_summary_row(tmp_path, row)

    rows = _read_csv(tmp_path / "run_summary.csv")
    assert "training_time_seconds" in rows[0]


# ---------------------------------------------------------------------------
# new_run_id
# ---------------------------------------------------------------------------


def test_new_run_id_is_unique():
    from src.metrics import new_run_id

    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100
