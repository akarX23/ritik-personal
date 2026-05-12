"""T029 – Integration test for classification analysis.

Uses a fixture predictions.csv; verifies confusion_matrix.png and
classification_report.png are produced.
Tests must FAIL before src/analyze.py is implemented.
"""

import csv
import pytest
from pathlib import Path


def _write_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": "r1",
            "sample_index": i,
            "true_label": i % 10,
            "predicted_label": (i + (1 if i % 7 == 0 else 0)) % 10,
        }
        for i in range(200)
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("clf_results")
    _write_predictions(d / "predictions.csv")
    return d


def test_confusion_matrix_png(fixture_dir):
    from src.analyze import plot_confusion_matrix

    plot_confusion_matrix(fixture_dir)
    assert (fixture_dir / "confusion_matrix.png").exists()


def test_classification_report_png(fixture_dir):
    from src.analyze import plot_classification_report

    plot_classification_report(fixture_dir)
    assert (fixture_dir / "classification_report.png").exists()
