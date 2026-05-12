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
