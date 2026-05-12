"""T011 – Contract tests for train.py CLI.

Validates all required arguments exist, defaults are correct, and that
--device is required (no default).

Tests must FAIL before src/train.py is implemented.
"""

import pytest
import re
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def _get_parser():
    from src.train import build_parser
    return build_parser()


def test_epochs_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.epochs == 10


def test_results_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.results == "./results"


def test_batch_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.batch == 64


def test_lr_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.lr == pytest.approx(0.001)


def test_data_default():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.data == "./data"


def test_device_required():
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # no -d provided


def test_device_cpu_accepted():
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert args.device == "cpu"


def test_device_xpu_accepted():
    parser = _get_parser()
    args = parser.parse_args(["-d", "xpu"])
    assert args.device == "xpu"


def test_short_flags_work():
    parser = _get_parser()
    args = parser.parse_args(
        ["-d", "cpu", "-e", "5", "-r", "/tmp/r", "-b", "32", "-lr", "0.01", "-m", "/tmp/d"]
    )
    assert args.epochs == 5
    assert args.results == "/tmp/r"
    assert args.batch == 32
    assert args.lr == pytest.approx(0.01)
    assert args.data == "/tmp/d"


def test_long_flags_work():
    parser = _get_parser()
    args = parser.parse_args(["--device", "cpu", "--epochs", "3"])
    assert args.device == "cpu"
    assert args.epochs == 3


def test_run_id_and_log_file_pattern(tmp_path):
    from src.metrics import new_run_id, run_log_path

    run_id = new_run_id()
    assert isinstance(run_id, str)
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", run_id
    )

    log_path = run_log_path(Path(tmp_path), run_id)
    assert log_path.name == f"run_{run_id}.log"


def test_train_logs_console_and_file_with_epoch_fields(tmp_path, capsys, monkeypatch):
    from src.train import run_training

    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = Path(tmp_path) / "results"
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=8,
        learning_rate=0.001,
        data_dir=str(Path(tmp_path) / "data"),
    )

    stdout = capsys.readouterr().out
    assert "event=epoch" in stdout
    assert "elapsed_seconds=" in stdout
    assert "loss=" in stdout
    assert "accuracy=" in stdout

    run_logs = list(results_dir.glob("run_*.log"))
    assert len(run_logs) == 1
    log_text = run_logs[0].read_text()
    assert "event=epoch" in log_text
    assert "elapsed_seconds=" in log_text
    assert "loss=" in log_text
    assert "accuracy=" in log_text


# --- T054: Multi-batch CLI flag tests ---


def test_batches_flag_optional():
    """T054: --batches flag should be optional."""
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu"])
    assert hasattr(args, "batches")
    assert args.batches is None


def test_batches_flag_single_value():
    """T054: --batches accepts single comma-separated value."""
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu", "--batches", "32"])
    assert args.batches == "32"


def test_batches_flag_multiple_values():
    """T054: --batches accepts comma-separated list of values."""
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu", "--batches", "32,64,128"])
    assert args.batches == "32,64,128"


def test_batches_flag_with_spaces_in_list():
    """T054: --batches parsing handles comma-separated values correctly."""
    parser = _get_parser()
    args = parser.parse_args(["-d", "cpu", "--batches", "32, 64, 128"])
    assert args.batches == "32, 64, 128"


# --- T057: Batch size column in CSV tests ---


def test_train_csv_includes_batch_size_column(tmp_path, monkeypatch):
    """T057: metrics_train.csv should include batch_size column when multi-batch runs."""
    from src.train import run_training
    import csv

    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = Path(tmp_path) / "results"
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=32,
        learning_rate=0.001,
        data_dir=str(Path(tmp_path) / "data"),
    )

    train_csv = results_dir / "metrics_train.csv"
    assert train_csv.exists()
    
    with open(train_csv, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert "batch_size" in headers, f"Expected 'batch_size' in CSV headers, got: {headers}"


def test_eval_csv_includes_batch_size_column(tmp_path, monkeypatch):
    """T057: metrics_validation.csv and metrics_test.csv should include batch_size."""
    from src.train import run_training
    import csv

    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = Path(tmp_path) / "results"
    run_training(
        epochs=5,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=32,
        learning_rate=0.001,
        data_dir=str(Path(tmp_path) / "data"),
    )

    val_csv = results_dir / "metrics_validation.csv"
    assert val_csv.exists()
    
    with open(val_csv, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert "batch_size" in headers, f"Expected 'batch_size' in validation CSV, got: {headers}"

    test_csv = results_dir / "metrics_test.csv"
    assert test_csv.exists()
    
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert "batch_size" in headers, f"Expected 'batch_size' in test CSV, got: {headers}"


def test_summary_csv_includes_batch_size_column(tmp_path, monkeypatch):
    """T057: run_summary.csv should include batch_size column."""
    from src.train import run_training
    import csv

    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = Path(tmp_path) / "results"
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=64,
        learning_rate=0.001,
        data_dir=str(Path(tmp_path) / "data"),
    )

    summary_csv = results_dir / "run_summary.csv"
    assert summary_csv.exists()
    
    with open(summary_csv, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert "batch_size" in headers, f"Expected 'batch_size' in summary CSV, got: {headers}"
