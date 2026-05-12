"""Phase 9 unit tests for results.md report generation helpers in analyze.py."""


def test_sample_epoch_rows_every_10_plus_final():
    from src.analyze import sample_epoch_rows

    rows = [{"epoch": str(i), "loss": str(1.0 / i)} for i in range(1, 26)]
    sampled = sample_epoch_rows(rows)

    sampled_epochs = [int(r["epoch"]) for r in sampled]
    assert sampled_epochs == [10, 20, 25]


def test_build_final_metrics_table_uses_final_epoch_per_batch():
    from src.analyze import build_final_metrics_table

    train_rows = [
        {"run_id": "r1", "batch_size": "32", "epoch": "9", "loss": "0.30", "accuracy": "0.90"},
        {"run_id": "r1", "batch_size": "32", "epoch": "10", "loss": "0.20", "accuracy": "0.95"},
        {"run_id": "r2", "batch_size": "64", "epoch": "10", "loss": "0.25", "accuracy": "0.94"},
    ]
    val_rows = [
        {"run_id": "r1", "batch_size": "32", "epoch": "10", "loss": "0.22", "accuracy": "0.94"},
        {"run_id": "r2", "batch_size": "64", "epoch": "10", "loss": "0.27", "accuracy": "0.92"},
    ]
    test_rows = [
        {"run_id": "r1", "batch_size": "32", "epoch": "10", "loss": "0.23", "accuracy": "0.93"},
        {"run_id": "r2", "batch_size": "64", "epoch": "10", "loss": "0.28", "accuracy": "0.91"},
    ]

    table = build_final_metrics_table(train_rows, val_rows, test_rows)

    assert "| 32 | 0.200000 | 0.950000 | 0.220000 | 0.940000 | 0.230000 | 0.930000 |" in table
    assert "| 64 | 0.250000 | 0.940000 | 0.270000 | 0.920000 | 0.280000 | 0.910000 |" in table


# ---------------------------------------------------------------------------
# T091 – filter_rows_by_batch helper unit tests (FR-026)
# ---------------------------------------------------------------------------


def test_filter_rows_by_batch_returns_matching_rows():
    """filter_rows_by_batch must return only rows whose batch_size equals the given int."""
    from src.analyze import filter_rows_by_batch

    rows = [
        {"batch_size": "32", "epoch": "1", "loss": "0.5"},
        {"batch_size": "64", "epoch": "1", "loss": "0.4"},
        {"batch_size": "32", "epoch": "2", "loss": "0.3"},
        {"batch_size": "128", "epoch": "1", "loss": "0.6"},
    ]
    result = filter_rows_by_batch(rows, 32)
    assert len(result) == 2
    assert all(r["batch_size"] == "32" for r in result)


def test_filter_rows_by_batch_returns_empty_for_missing_batch():
    """filter_rows_by_batch must return [] when no rows match."""
    from src.analyze import filter_rows_by_batch

    rows = [
        {"batch_size": "32", "epoch": "1", "loss": "0.5"},
        {"batch_size": "64", "epoch": "1", "loss": "0.4"},
    ]
    result = filter_rows_by_batch(rows, 128)
    assert result == []


def test_filter_rows_by_batch_empty_input():
    """filter_rows_by_batch must return [] for empty row list."""
    from src.analyze import filter_rows_by_batch

    assert filter_rows_by_batch([], 64) == []


# ---------------------------------------------------------------------------
# T092 – _build_epoch_comparison_table applies filter_batch (FR-026, FR-027)
# ---------------------------------------------------------------------------


def _make_multibatch_rows() -> tuple[list[dict], list[dict], list[dict]]:
    """Minimal synthetic multi-batch CSV rows for epoch-comparison tests."""
    base = {"loss": "0.5", "accuracy": "0.9", "elapsed_seconds": "1.0", "device": "cpu"}
    train = [
        {**base, "run_id": "r-32", "batch_size": "32", "epoch": "10", "split": "train"},
        {**base, "run_id": "r-64", "batch_size": "64", "epoch": "10", "split": "train"},
    ]
    val = [
        {**base, "run_id": "r-32", "batch_size": "32", "epoch": "10", "split": "validation"},
        {**base, "run_id": "r-64", "batch_size": "64", "epoch": "10", "split": "validation"},
    ]
    test = [
        {**base, "run_id": "r-32", "batch_size": "32", "epoch": "10", "split": "test"},
        {**base, "run_id": "r-64", "batch_size": "64", "epoch": "10", "split": "test"},
    ]
    return train, val, test


def test_epoch_comparison_table_filtered_when_filter_batch_set():
    """_build_epoch_comparison_table must include only the specified batch_size rows when filter_batch is set."""
    from src.analyze import _build_epoch_comparison_table

    train, val, test = _make_multibatch_rows()
    table = _build_epoch_comparison_table(train, val, test, filter_batch=32)

    assert "| 32 |" in table
    assert "| 64 |" not in table


def test_epoch_comparison_table_unfiltered_when_filter_batch_is_none():
    """_build_epoch_comparison_table must include all batch sizes when filter_batch is None."""
    from src.analyze import _build_epoch_comparison_table

    train, val, test = _make_multibatch_rows()
    table = _build_epoch_comparison_table(train, val, test, filter_batch=None)

    assert "| 32 |" in table
    assert "| 64 |" in table


def test_final_metrics_table_not_filtered_when_filter_batch_set():
    """build_final_metrics_table must always show all batch sizes regardless of filter (FR-027)."""
    from src.analyze import build_final_metrics_table

    train_rows = [
        {"run_id": "r1", "batch_size": "32", "epoch": "10", "loss": "0.20", "accuracy": "0.95"},
        {"run_id": "r2", "batch_size": "64", "epoch": "10", "loss": "0.25", "accuracy": "0.94"},
    ]
    val_rows = [
        {"run_id": "r1", "batch_size": "32", "epoch": "10", "loss": "0.22", "accuracy": "0.94"},
        {"run_id": "r2", "batch_size": "64", "epoch": "10", "loss": "0.27", "accuracy": "0.92"},
    ]
    test_rows = [
        {"run_id": "r1", "batch_size": "32", "epoch": "10", "loss": "0.23", "accuracy": "0.93"},
        {"run_id": "r2", "batch_size": "64", "epoch": "10", "loss": "0.28", "accuracy": "0.91"},
    ]
    # build_final_metrics_table has no filter_batch parameter — it must never be filtered
    table = build_final_metrics_table(train_rows, val_rows, test_rows)
    assert "| 32 |" in table
    assert "| 64 |" in table
