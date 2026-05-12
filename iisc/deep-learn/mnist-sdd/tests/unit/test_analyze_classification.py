"""T027/T028 – Unit tests for classification analysis in analyze.py.

Uses synthetic prediction data; no real training run required.
Tests must FAIL before src/analyze.py is implemented.
"""

import csv
import pytest
from pathlib import Path


def _write_predictions(path: Path, n: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": "r1",
            "sample_index": i,
            "true_label": i % 10,
            "predicted_label": i % 10,  # perfect predictions
        }
        for i in range(n)
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_plot_confusion_matrix_creates_file(tmp_path):
    from src.analyze import plot_confusion_matrix

    _write_predictions(tmp_path / "predictions.csv")
    plot_confusion_matrix(tmp_path)

    assert (tmp_path / "confusion_matrix.png").exists()


def test_plot_classification_report_creates_file(tmp_path):
    from src.analyze import plot_classification_report

    _write_predictions(tmp_path / "predictions.csv")
    plot_classification_report(tmp_path)

    assert (tmp_path / "classification_report.png").exists()


def test_missing_predictions_raises(tmp_path):
    from src.analyze import plot_confusion_matrix

    with pytest.raises((FileNotFoundError, SystemExit, ValueError)):
        plot_confusion_matrix(tmp_path)  # no predictions.csv
