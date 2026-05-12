"""T012 – Integration test for one full training run.

Verifies that a one-epoch CPU training run produces all required CSV output
files with correct columns and at least one data row each, and that the
model checkpoint is saved.

Tests must FAIL before src/train.py is implemented.
"""

import csv
import pytest
from pathlib import Path
from torch.utils.data import TensorDataset
import torch


REQUIRED_CSV_FILES = [
    "metrics_train.csv",
    "metrics_validation.csv",
    "metrics_test.csv",
    "run_summary.csv",
]

EXPECTED_TRAIN_COLUMNS = {
    "run_id", "epoch", "split", "loss", "accuracy", "elapsed_seconds", "device"
}
EXPECTED_SUMMARY_COLUMNS = {
    "run_id", "device", "epochs", "training_time_seconds", "status", "results_dir"
}


def _run_training(results_dir: Path, data_dir: Path) -> None:
    from src.train import run_training

    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=64,
        learning_rate=0.001,
        data_dir=str(data_dir),
    )


@pytest.fixture(scope="module")
def training_outputs(tmp_path_factory):
    results_dir = tmp_path_factory.mktemp("results")
    data_dir = tmp_path_factory.mktemp("data")
    _run_training(results_dir, data_dir)
    return results_dir


def test_metrics_train_csv_exists(training_outputs):
    assert (training_outputs / "metrics_train.csv").exists()


def test_metrics_train_has_rows(training_outputs):
    with (training_outputs / "metrics_train.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 1


def test_metrics_train_columns(training_outputs):
    with (training_outputs / "metrics_train.csv").open() as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
    assert EXPECTED_TRAIN_COLUMNS.issubset(cols)


def test_run_summary_exists(training_outputs):
    assert (training_outputs / "run_summary.csv").exists()


def test_run_summary_has_timing(training_outputs):
    with (training_outputs / "run_summary.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert "training_time_seconds" in rows[0]
    assert float(rows[0]["training_time_seconds"]) >= 0


def test_model_checkpoint_saved(training_outputs):
    assert (training_outputs / "model.pt").exists()


def test_fr014_mount_path_skips_download_when_data_present(tmp_path, monkeypatch):
    from src.data import load_mnist

    data_dir = tmp_path / "mounted-data"
    data_dir.mkdir(parents=True)
    (data_dir / "placeholder.bin").write_bytes(b"mounted")

    calls = []

    class FakeMNIST(TensorDataset):
        def __init__(self, root, train, download, transform):
            calls.append({"root": root, "train": train, "download": download})
            features = torch.randn(20, 1, 28, 28)
            labels = torch.randint(0, 10, (20,))
            super().__init__(features, labels)

    monkeypatch.setattr("src.data.datasets.MNIST", FakeMNIST)

    load_mnist(data_dir, batch_size=8)

    assert calls, "Expected MNIST dataset constructor to be called"
    assert all(c["download"] is False for c in calls)


def test_fr014_download_fallback_when_data_absent(tmp_path, monkeypatch):
    from src.data import load_mnist

    data_dir = tmp_path / "no-data-dir"
    calls = []

    class FakeMNIST(TensorDataset):
        def __init__(self, root, train, download, transform):
            calls.append({"root": root, "train": train, "download": download})
            features = torch.randn(20, 1, 28, 28)
            labels = torch.randint(0, 10, (20,))
            super().__init__(features, labels)

    monkeypatch.setattr("src.data.datasets.MNIST", FakeMNIST)

    load_mnist(data_dir, batch_size=8)

    assert calls, "Expected MNIST dataset constructor to be called"
    assert all(c["download"] is True for c in calls)
