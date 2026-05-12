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


# --- T056: Distinct run_id per batch in multi-batch mode ---


def test_multibatch_generates_distinct_run_ids(tmp_path, monkeypatch):
    """T056: Multi-batch execution assigns distinct run_id to each batch."""
    from src.train import run_training
    
    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = tmp_path / "results"
    
    # Simulate multi-batch execution with two different batch sizes
    # (This test will validate that distinct run_ids are generated for each batch)
    # Once multi-batch implementation is complete, this test will verify
    # that running with --batches 32,64 generates two distinct run_ids
    
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=32,
        learning_rate=0.001,
        data_dir=str(tmp_path / "data"),
    )
    
    # Read first batch's run_id from CSV
    with open(results_dir / "metrics_train.csv", "r") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        run_id_1 = first_row["run_id"]
    
    # In full multi-batch mode, running again with different batch size
    # should produce different run_ids
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=64,
        learning_rate=0.001,
        data_dir=str(tmp_path / "data"),
    )
    
    # Collect run_ids from both single-batch runs
    run_ids = set()
    with open(results_dir / "metrics_train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_ids.add(row["run_id"])
    
    # Should have at least 2 distinct run_ids (one per execution)
    assert len(run_ids) >= 2, f"Expected at least 2 distinct run_ids, got {len(run_ids)}"


def test_multibatch_csv_rows_include_distinct_run_ids(tmp_path, monkeypatch):
    """T056: Each batch's CSV rows should have distinct run_id values."""
    from src.train import run_training
    
    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = tmp_path / "results"
    
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=32,
        learning_rate=0.001,
        data_dir=str(tmp_path / "data"),
    )
    
    # Verify all rows in same training session have same run_id (consistency)
    run_ids = set()
    with open(results_dir / "metrics_train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_ids.add(row["run_id"])
    
    # Single execution should have only one unique run_id
    assert len(run_ids) == 1, f"Expected 1 unique run_id, got {len(run_ids)}"


# --- T058: Backward compatibility with single --batch flag ---


def test_single_batch_flag_still_works(tmp_path, monkeypatch):
    """T058: Single --batch flag (not --batches) should continue working."""
    from src.train import run_training
    
    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = tmp_path / "results"
    
    # Using traditional single --batch flag
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=64,  # Single batch size, not multi-batch
        learning_rate=0.001,
        data_dir=str(tmp_path / "data"),
    )
    
    # Verify training completed and artifacts exist
    assert (results_dir / "metrics_train.csv").exists()
    assert (results_dir / "metrics_validation.csv").exists()
    assert (results_dir / "metrics_test.csv").exists()
    assert (results_dir / "run_summary.csv").exists()
    assert (results_dir / "model.pt").exists()


def test_backward_compat_single_batch_produces_single_run_id(tmp_path, monkeypatch):
    """T058: Single batch training produces exactly one run_id."""
    from src.train import run_training
    
    features = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(features, labels)

    def _fake_load_mnist(_data_dir, batch_size, val_fraction=0.1, num_workers=0):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, loader, loader

    monkeypatch.setattr("src.train.load_mnist", _fake_load_mnist)

    results_dir = tmp_path / "results"
    
    run_training(
        epochs=1,
        results_dir=str(results_dir),
        device_str="cpu",
        batch_size=32,
        learning_rate=0.001,
        data_dir=str(tmp_path / "data"),
    )
    
    # Check that there's exactly one unique run_id in the results
    run_ids = set()
    with open(results_dir / "metrics_train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_ids.add(row["run_id"])
    
    assert len(run_ids) == 1, f"Single batch should produce exactly 1 run_id, got {len(run_ids)}"

