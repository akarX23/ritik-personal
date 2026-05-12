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


def test_results_md_generation_from_multibatch_metrics(tmp_path):
    from src.analyze import run_analysis

    results = Path(tmp_path) / "results"
    metric_cols = [
        "run_id",
        "batch_size",
        "epoch",
        "split",
        "loss",
        "accuracy",
        "elapsed_seconds",
        "device",
    ]
    summary_cols = [
        "run_id",
        "batch_size",
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
        "learning_rate",
    ]

    train_rows = [
        {
            "run_id": "r-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "train",
            "loss": 0.20,
            "accuracy": 0.95,
            "elapsed_seconds": 1.0,
            "device": "cpu",
        },
        {
            "run_id": "r-64",
            "batch_size": 64,
            "epoch": 10,
            "split": "train",
            "loss": 0.25,
            "accuracy": 0.94,
            "elapsed_seconds": 1.0,
            "device": "cpu",
        },
    ]
    val_rows = [
        {
            "run_id": "r-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "validation",
            "loss": 0.22,
            "accuracy": 0.94,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
        {
            "run_id": "r-64",
            "batch_size": 64,
            "epoch": 10,
            "split": "validation",
            "loss": 0.27,
            "accuracy": 0.92,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
    ]
    test_rows = [
        {
            "run_id": "r-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "test",
            "loss": 0.23,
            "accuracy": 0.93,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
        {
            "run_id": "r-64",
            "batch_size": 64,
            "epoch": 10,
            "split": "test",
            "loss": 0.28,
            "accuracy": 0.91,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
    ]
    summaries = [
        {
            "run_id": "r-32",
            "batch_size": 32,
            "device": "cpu",
            "epochs": 10,
            "start_time": "2026-05-12T10:00:00+00:00",
            "end_time": "2026-05-12T10:00:30+00:00",
            "training_time_seconds": 30.0,
            "final_train_loss": 0.20,
            "final_train_accuracy": 0.95,
            "final_val_loss": 0.22,
            "final_val_accuracy": 0.94,
            "final_test_loss": 0.23,
            "final_test_accuracy": 0.93,
            "status": "completed",
            "results_dir": str(results),
            "learning_rate": 0.001,
        },
        {
            "run_id": "r-64",
            "batch_size": 64,
            "device": "cpu",
            "epochs": 10,
            "start_time": "2026-05-12T10:01:00+00:00",
            "end_time": "2026-05-12T10:01:35+00:00",
            "training_time_seconds": 35.0,
            "final_train_loss": 0.25,
            "final_train_accuracy": 0.94,
            "final_val_loss": 0.27,
            "final_val_accuracy": 0.92,
            "final_test_loss": 0.28,
            "final_test_accuracy": 0.91,
            "status": "completed",
            "results_dir": str(results),
            "learning_rate": 0.001,
        },
    ]

    _write_csv(results / "metrics_train.csv", train_rows)
    _write_csv(results / "metrics_validation.csv", val_rows)
    _write_csv(results / "metrics_test.csv", test_rows)
    _write_csv(results / "run_summary.csv", summaries)

    run_analysis(str(results))

    report_path = results / "results.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Final Metrics by Batch Size" in content
    assert "| 32 |" in content
    assert "| 64 |" in content


def test_results_md_includes_all_historical_matching_rows(tmp_path):
    from src.analyze import run_analysis

    results = Path(tmp_path) / "results"

    train_rows = [
        {
            "run_id": "old-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "train",
            "loss": 0.30,
            "accuracy": 0.90,
            "elapsed_seconds": 1.0,
            "device": "cpu",
        },
        {
            "run_id": "new-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "train",
            "loss": 0.20,
            "accuracy": 0.95,
            "elapsed_seconds": 1.0,
            "device": "cpu",
        },
    ]
    val_rows = [
        {
            "run_id": "old-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "validation",
            "loss": 0.31,
            "accuracy": 0.89,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
        {
            "run_id": "new-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "validation",
            "loss": 0.22,
            "accuracy": 0.94,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
    ]
    test_rows = [
        {
            "run_id": "old-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "test",
            "loss": 0.32,
            "accuracy": 0.88,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
        {
            "run_id": "new-32",
            "batch_size": 32,
            "epoch": 10,
            "split": "test",
            "loss": 0.23,
            "accuracy": 0.93,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
    ]

    _write_csv(results / "metrics_train.csv", train_rows)
    _write_csv(results / "metrics_validation.csv", val_rows)
    _write_csv(results / "metrics_test.csv", test_rows)

    run_analysis(str(results))

    content = (results / "results.md").read_text(encoding="utf-8")
    assert "old-32" in content
    assert "new-32" in content
