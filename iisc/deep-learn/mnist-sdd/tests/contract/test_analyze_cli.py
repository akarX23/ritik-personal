"""T019 – Contract tests for analyze.py CLI.

Tests must FAIL before src/analyze.py is implemented.
"""

import pytest
import csv
from pathlib import Path


def _get_parser():
    from src.analyze import build_parser
    return build_parser()


def test_results_required():
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # no -r provided


def test_results_short_flag():
    parser = _get_parser()
    args = parser.parse_args(["-r", "/tmp/results"])
    assert args.results == "/tmp/results"


def test_results_long_flag():
    parser = _get_parser()
    args = parser.parse_args(["--results", "/tmp/results"])
    assert args.results == "/tmp/results"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_emits_lifecycle_logs_to_console_and_file(tmp_path, capsys):
    from src.analyze import run_analysis

    results_dir = Path(tmp_path) / "results"
    metric_cols = [
        "run_id",
        "epoch",
        "split",
        "loss",
        "accuracy",
        "elapsed_seconds",
        "device",
    ]

    train_rows = [
        {
            "run_id": "r1",
            "epoch": 1,
            "split": "train",
            "loss": 0.8,
            "accuracy": 0.8,
            "elapsed_seconds": 1.0,
            "device": "cpu",
        },
        {
            "run_id": "r1",
            "epoch": 2,
            "split": "train",
            "loss": 0.6,
            "accuracy": 0.86,
            "elapsed_seconds": 1.0,
            "device": "cpu",
        },
    ]
    eval_rows = [
        {
            "run_id": "r1",
            "epoch": 1,
            "split": "validation",
            "loss": 0.7,
            "accuracy": 0.82,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
        {
            "run_id": "r1",
            "epoch": 2,
            "split": "validation",
            "loss": 0.5,
            "accuracy": 0.88,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
    ]
    test_rows = [
        {
            "run_id": "r1",
            "epoch": 1,
            "split": "test",
            "loss": 0.72,
            "accuracy": 0.81,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
        {
            "run_id": "r1",
            "epoch": 2,
            "split": "test",
            "loss": 0.55,
            "accuracy": 0.87,
            "elapsed_seconds": 0.5,
            "device": "cpu",
        },
    ]
    pred_cols = ["run_id", "sample_index", "true_label", "predicted_label"]
    pred_rows = [
        {"run_id": "r1", "sample_index": i, "true_label": i % 10, "predicted_label": i % 10}
        for i in range(30)
    ]

    _write_csv(results_dir / "metrics_train.csv", metric_cols, train_rows)
    _write_csv(results_dir / "metrics_validation.csv", metric_cols, eval_rows)
    _write_csv(results_dir / "metrics_test.csv", metric_cols, test_rows)
    _write_csv(results_dir / "predictions.csv", pred_cols, pred_rows)

    run_analysis(str(results_dir))

    stdout = capsys.readouterr().out
    assert "event=lifecycle stage=start" in stdout
    assert "event=lifecycle stage=complete" in stdout

    run_logs = list(results_dir.glob("run_*.log"))
    assert len(run_logs) == 1
    content = run_logs[0].read_text()
    assert "event=lifecycle stage=start" in content
    assert "event=lifecycle stage=complete" in content


def test_analyze_generates_results_md_in_results_dir(tmp_path):
    from src.analyze import run_analysis

    results_dir = Path(tmp_path) / "results"
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

    _write_csv(
        results_dir / "metrics_train.csv",
        metric_cols,
        [
            {
                "run_id": "r1",
                "batch_size": 32,
                "epoch": 1,
                "split": "train",
                "loss": 0.9,
                "accuracy": 0.70,
                "elapsed_seconds": 1.0,
                "device": "cpu",
            },
            {
                "run_id": "r1",
                "batch_size": 32,
                "epoch": 10,
                "split": "train",
                "loss": 0.2,
                "accuracy": 0.95,
                "elapsed_seconds": 1.0,
                "device": "cpu",
            },
        ],
    )
    _write_csv(
        results_dir / "metrics_validation.csv",
        metric_cols,
        [
            {
                "run_id": "r1",
                "batch_size": 32,
                "epoch": 10,
                "split": "validation",
                "loss": 0.25,
                "accuracy": 0.93,
                "elapsed_seconds": 0.5,
                "device": "cpu",
            }
        ],
    )
    _write_csv(
        results_dir / "metrics_test.csv",
        metric_cols,
        [
            {
                "run_id": "r1",
                "batch_size": 32,
                "epoch": 10,
                "split": "test",
                "loss": 0.26,
                "accuracy": 0.92,
                "elapsed_seconds": 0.5,
                "device": "cpu",
            }
        ],
    )
    _write_csv(
        results_dir / "run_summary.csv",
        summary_cols,
        [
            {
                "run_id": "r1",
                "batch_size": 32,
                "device": "cpu",
                "epochs": 10,
                "start_time": "2026-05-12T10:00:00+00:00",
                "end_time": "2026-05-12T10:00:30+00:00",
                "training_time_seconds": 30.0,
                "final_train_loss": 0.2,
                "final_train_accuracy": 0.95,
                "final_val_loss": 0.25,
                "final_val_accuracy": 0.93,
                "final_test_loss": 0.26,
                "final_test_accuracy": 0.92,
                "status": "completed",
                "results_dir": str(results_dir),
                "learning_rate": 0.001,
            }
        ],
    )

    run_analysis(str(results_dir))

    report_path = results_dir / "results.md"
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Final Metrics by Batch Size" in report
    assert "Epoch Comparison" in report


# ---------------------------------------------------------------------------
# T089 – --filter-batch / -b flag parsing (FR-026)
# ---------------------------------------------------------------------------


def test_filter_batch_long_flag_parses_as_int():
    """--filter-batch INT must be accepted and stored as an int."""
    parser = _get_parser()
    args = parser.parse_args(["-r", "/tmp/r", "--filter-batch", "64"])
    assert args.filter_batch == 64


def test_filter_batch_short_flag_parses_as_int():
    """-b INT must be the short form for --filter-batch."""
    parser = _get_parser()
    args = parser.parse_args(["-r", "/tmp/r", "-b", "32"])
    assert args.filter_batch == 32


def test_filter_batch_defaults_to_none():
    """Omitting --filter-batch must leave the attribute as None (all-batches behavior)."""
    parser = _get_parser()
    args = parser.parse_args(["-r", "/tmp/r"])
    assert args.filter_batch is None


# ---------------------------------------------------------------------------
# T090 – fail-fast when filter-batch has no matching rows (FR-028)
# ---------------------------------------------------------------------------


def _write_multibatch_csv_fixture(results_dir: Path) -> None:
    """Write minimal 3-batch CSV fixture for filter-batch error tests."""
    metric_cols = [
        "run_id", "batch_size", "epoch", "split",
        "loss", "accuracy", "elapsed_seconds", "device",
    ]
    summary_cols = [
        "run_id", "batch_size", "device", "epochs",
        "start_time", "end_time", "training_time_seconds",
        "final_train_loss", "final_train_accuracy",
        "final_val_loss", "final_val_accuracy",
        "final_test_loss", "final_test_accuracy",
        "status", "results_dir", "learning_rate",
    ]
    for split_name, fname in [
        ("train", "metrics_train.csv"),
        ("validation", "metrics_validation.csv"),
        ("test", "metrics_test.csv"),
    ]:
        rows = []
        for bs in (32, 64):
            rows.append({
                "run_id": f"r-{bs}", "batch_size": bs, "epoch": 1,
                "split": split_name, "loss": 0.5, "accuracy": 0.8,
                "elapsed_seconds": 1.0, "device": "cpu",
            })
        _write_csv(results_dir / fname, metric_cols, rows)
    _write_csv(
        results_dir / "run_summary.csv",
        summary_cols,
        [{
            "run_id": "r-32", "batch_size": 32, "device": "cpu", "epochs": 1,
            "start_time": "", "end_time": "", "training_time_seconds": 1.0,
            "final_train_loss": 0.5, "final_train_accuracy": 0.8,
            "final_val_loss": 0.5, "final_val_accuracy": 0.8,
            "final_test_loss": 0.5, "final_test_accuracy": 0.8,
            "status": "completed", "results_dir": str(results_dir), "learning_rate": 0.001,
        }],
    )


def test_run_analysis_exits_nonzero_when_filter_batch_not_found(tmp_path, capsys):
    """run_analysis must sys.exit(1) when --filter-batch value has no CSV rows (FR-028)."""
    from src.analyze import run_analysis

    results_dir = Path(tmp_path) / "results"
    _write_multibatch_csv_fixture(results_dir)

    with pytest.raises(SystemExit) as exc_info:
        run_analysis(str(results_dir), filter_batch=999)

    assert exc_info.value.code == 1


def test_run_analysis_error_message_lists_available_batches(tmp_path, capsys):
    """Error output must list all available batch sizes when filter-batch not found (FR-028)."""
    from src.analyze import run_analysis

    results_dir = Path(tmp_path) / "results"
    _write_multibatch_csv_fixture(results_dir)

    with pytest.raises(SystemExit):
        run_analysis(str(results_dir), filter_batch=999)

    stderr = capsys.readouterr().err
    assert "32" in stderr
    assert "64" in stderr
