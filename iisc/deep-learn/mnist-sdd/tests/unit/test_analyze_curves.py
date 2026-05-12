"""T020 – Unit tests for learning-curve plotting in analyze.py.

Uses synthetic CSV data; no real training run required.
Tests must FAIL before src/analyze.py is implemented.
"""

import csv
import pytest
from pathlib import Path


def _write_metric_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _make_metrics(results_dir: Path) -> None:
    cols = ["run_id", "epoch", "split", "loss", "accuracy", "elapsed_seconds", "device"]
    for split, fname in [
        ("train", "metrics_train.csv"),
        ("validation", "metrics_validation.csv"),
        ("test", "metrics_test.csv"),
    ]:
        rows = [
            {
                "run_id": "r1",
                "epoch": i,
                "split": split,
                "loss": round(1.0 / i, 4),
                "accuracy": round(0.5 + 0.04 * i, 4),
                "elapsed_seconds": 1.0,
                "device": "cpu",
            }
            for i in range(1, 6)
        ]
        _write_metric_csv(results_dir / fname, rows)


def test_plot_learning_curves_creates_files(tmp_path):
    from src.analyze import plot_learning_curves

    _make_metrics(tmp_path)
    plot_learning_curves(tmp_path)

    assert (tmp_path / "learning_curves_loss.png").exists()
    assert (tmp_path / "learning_curves_accuracy.png").exists()


def test_plot_learning_curves_missing_file_raises(tmp_path):
    from src.analyze import plot_learning_curves

    # only write train CSV – others missing
    _make_metrics(tmp_path)
    (tmp_path / "metrics_validation.csv").unlink()

    with pytest.raises((FileNotFoundError, SystemExit, ValueError)):
        plot_learning_curves(tmp_path)
