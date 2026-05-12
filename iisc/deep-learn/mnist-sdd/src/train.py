"""train.py – MNIST training pipeline entry point.

Usage:
    python src/train.py -d cpu
    python src/train.py -d xpu -e 20 -r ./results/xpu-run -b 64 -lr 0.001 -m ./data

All CSV outputs follow the schema defined in data-model.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from src.device import get_device
from src.data import load_mnist
from src.metrics import (
    new_run_id,
    write_train_row,
    write_eval_row,
    write_summary_row,
    write_prediction_row,
)
from src.model import MnistClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for train.py."""
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train the MNIST digit classifier.",
    )
    p.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of training epochs (default: 10)",
    )
    p.add_argument(
        "-r", "--results",
        default="./results",
        metavar="DIR",
        help="Results directory (default: ./results)",
    )
    p.add_argument(
        "-d", "--device",
        required=True,
        choices=["cpu", "xpu"],
        metavar="DEVICE",
        help="Compute device: 'cpu' or 'xpu' (required)",
    )
    p.add_argument(
        "-b", "--batch",
        type=int,
        default=64,
        metavar="N",
        help="Mini-batch size (default: 64)",
    )
    p.add_argument(
        "-lr", "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate (default: 0.001)",
    )
    p.add_argument(
        "-m", "--data",
        default="./data",
        metavar="DIR",
        help="MNIST data directory (default: ./data)",
    )
    return p


# ---------------------------------------------------------------------------
# Core training function (callable from tests without subprocess)
# ---------------------------------------------------------------------------


def run_training(
    epochs: int,
    results_dir: str,
    device_str: str,
    batch_size: int,
    learning_rate: float,
    data_dir: str,
) -> None:
    """Execute the full MNIST training pipeline.

    Parameters
    ----------
    epochs:
        Number of training epochs.
    results_dir:
        Directory to write CSV outputs and model checkpoint.
    device_str:
        ``"cpu"`` or ``"xpu"``.
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam learning rate.
    data_dir:
        Path to directory containing MNIST data (downloaded if absent).

    Raises
    ------
    RuntimeError
        If the selected device is unavailable.
    FileNotFoundError
        If *data_dir* does not exist and MNIST cannot be downloaded.
    """
    # --- resolve device (fail-fast if unavailable) -------------------------
    device = get_device(device_str)

    # --- verify data directory ---------------------------------------------
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # --- build run metadata ------------------------------------------------
    run_id = new_run_id()
    start_dt = datetime.now(timezone.utc)
    run_start = time.perf_counter()

    status = "failed"
    summary: dict = {}

    try:
        # --- data -----------------------------------------------------------
        train_loader, val_loader, test_loader = load_mnist(
            data_path, batch_size
        )

        # --- model ----------------------------------------------------------
        model = MnistClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        last_val_loss = last_val_acc = last_test_loss = last_test_acc = 0.0

        # --- epoch loop -----------------------------------------------------
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            epoch_elapsed = time.perf_counter() - epoch_start

            write_train_row(
                results_path,
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "split": "train",
                    "loss": round(train_loss, 6),
                    "accuracy": round(train_acc, 6),
                    "elapsed_seconds": round(epoch_elapsed, 4),
                    "device": device_str,
                },
            )

            # validation + test every 5 epochs (and on final epoch)
            if epoch % 5 == 0 or epoch == epochs:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                eval_start = time.perf_counter()
                write_eval_row(
                    results_path,
                    "validation",
                    {
                        "run_id": run_id,
                        "epoch": epoch,
                        "split": "validation",
                        "loss": round(val_loss, 6),
                        "accuracy": round(val_acc, 6),
                        "elapsed_seconds": round(time.perf_counter() - eval_start, 4),
                        "device": device_str,
                    },
                )

                test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
                test_eval_start = time.perf_counter()
                write_eval_row(
                    results_path,
                    "test",
                    {
                        "run_id": run_id,
                        "epoch": epoch,
                        "split": "test",
                        "loss": round(test_loss, 6),
                        "accuracy": round(test_acc, 6),
                        "elapsed_seconds": round(time.perf_counter() - test_eval_start, 4),
                        "device": device_str,
                    },
                )

                last_val_loss, last_val_acc = val_loss, val_acc
                last_test_loss, last_test_acc = test_loss, test_acc

        # --- save checkpoint ------------------------------------------------
        torch.save(model.state_dict(), results_path / "model.pt")

        # --- write final predictions ----------------------------------------
        _write_predictions(model, test_loader, device, run_id, results_path)

        status = "completed"

    except KeyboardInterrupt:
        status = "interrupted"
        print(
            "\n[train] Run interrupted. Partial CSV logs are available in "
            f"{results_dir}",
            file=sys.stderr,
        )
        raise

    finally:
        training_time = time.perf_counter() - run_start
        end_dt = datetime.now(timezone.utc)

        write_summary_row(
            results_path,
            {
                "run_id": run_id,
                "device": device_str,
                "epochs": epochs,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "training_time_seconds": round(training_time, 4),
                "final_train_loss": "",
                "final_train_accuracy": "",
                "final_val_loss": round(last_val_loss, 6) if status != "failed" else "",
                "final_val_accuracy": round(last_val_acc, 6) if status != "failed" else "",
                "final_test_loss": round(last_test_loss, 6) if status != "failed" else "",
                "final_test_accuracy": round(last_test_acc, 6) if status != "failed" else "",
                "status": status,
                "results_dir": str(results_path.resolve()),
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _train_epoch(
    model: MnistClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch; return (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def _evaluate(
    model: MnistClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on *loader*; return (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def _write_predictions(
    model: MnistClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    run_id: str,
    results_path: Path,
) -> None:
    """Write per-sample true/predicted labels to predictions.csv."""
    model.eval()
    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            true_labels = labels.tolist()

            for true_lbl, pred_lbl in zip(true_labels, preds):
                write_prediction_row(
                    results_path,
                    {
                        "run_id": run_id,
                        "sample_index": sample_index,
                        "true_label": true_lbl,
                        "predicted_label": pred_lbl,
                    },
                )
                sample_index += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_training(
            epochs=args.epochs,
            results_dir=args.results,
            device_str=args.device,
            batch_size=args.batch,
            learning_rate=args.lr,
            data_dir=args.data,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[train] Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(
            f"[train] Data error: {exc}\n"
            "Ensure the MNIST data directory exists or allow download by "
            "using the default ./data path.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
