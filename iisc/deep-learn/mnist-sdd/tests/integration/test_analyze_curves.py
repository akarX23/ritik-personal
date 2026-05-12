"""T021 – Integration test for learning curves and device comparison.

Uses fixture CSV data to verify full analysis pipeline outputs.
Tests must FAIL before src/analyze.py is implemented.
"""

import csv
import pytest
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(device: str, results_dir: str) -> dict:
    return {
        "run_id": f"r-{device}",
        "device": device,
        "epochs": 5,
        "start_time": "2026-05-11T00:00:00+00:00",
        "end_time": "2026-05-11T00:01:00+00:00",
        "training_time_seconds": 60.0 if device == "cpu" else 30.0,
        "final_train_loss": "",
        "final_train_accuracy": "",
        "final_val_loss": 0.1,
        "final_val_accuracy": 0.97,
        "final_test_loss": 0.09,
        "final_test_accuracy": 0.98,
        "status": "completed",
        "results_dir": results_dir,
    }


@pytest.fixture(scope="module")
def fixture_results(tmp_path_factory):
    results = tmp_path_factory.mktemp("results")

    metric_cols = ["run_id", "epoch", "split", "loss", "accuracy",
                   "elapsed_seconds", "device"]
    for split, fname in [
        ("train", "metrics_train.csv"),
        ("validation", "metrics_validation.csv"),
        ("test", "metrics_test.csv"),
    ]:
        rows = [
            {"run_id": "r1", "epoch": i, "split": split,
             "loss": round(1.0 / i, 4), "accuracy": round(0.5 + 0.04 * i, 4),
             "elapsed_seconds": 1.0, "device": "cpu"}
            for i in range(1, 6)
        ]
        _write_csv(results / fname, rows)

    return results


def test_learning_curves_loss_png(fixture_results):
    from src.analyze import plot_learning_curves

    plot_learning_curves(fixture_results)
    assert (fixture_results / "learning_curves_loss.png").exists()


def test_learning_curves_accuracy_png(fixture_results):
    from src.analyze import plot_learning_curves

    plot_learning_curves(fixture_results)
    assert (fixture_results / "learning_curves_accuracy.png").exists()
