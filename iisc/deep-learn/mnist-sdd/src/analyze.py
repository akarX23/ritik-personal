"""analyze.py – Offline analysis and visualisation for MNIST training runs.

Usage:
    python src/analyze.py -r ./results

Generates from saved CSV files:
  - learning_curves_loss.png / learning_curves_accuracy.png
  - device_comparison.png  (if cpu/ and xpu/ sub-dirs are present)
  - time_comparison.png    (if run_summary.csv in both device sub-dirs)
  - confusion_matrix.png
  - classification_report.png

All artefacts are written into the results directory provided via -r.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="Generate visualisations from MNIST training CSV outputs.",
    )
    p.add_argument(
        "-r", "--results",
        required=True,
        metavar="DIR",
        help="Results directory containing CSV files (required)",
    )
    return p


# ---------------------------------------------------------------------------
# Internal CSV helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run a training pass first with: python src/train.py -d <device>"
        )
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _epochs(rows: list[dict]) -> list[int]:
    return [int(r["epoch"]) for r in rows]


def _values(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


# ---------------------------------------------------------------------------
# Learning curves  (US2 – T023)
# ---------------------------------------------------------------------------


def plot_learning_curves(results_dir: Path) -> None:
    """Plot train/val/test loss and accuracy curves into *results_dir*."""
    train_rows = _read_csv(results_dir / "metrics_train.csv")
    val_rows = _read_csv(results_dir / "metrics_validation.csv")
    test_rows = _read_csv(results_dir / "metrics_test.csv")

    # ---- loss curve -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_epochs(train_rows), _values(train_rows, "loss"), label="train")
    ax.plot(_epochs(val_rows), _values(val_rows, "loss"), label="validation", marker="o")
    ax.plot(_epochs(test_rows), _values(test_rows, "loss"), label="test", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves – Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curves_loss.png", dpi=120)
    plt.close(fig)

    # ---- accuracy curve ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_epochs(train_rows), _values(train_rows, "accuracy"), label="train")
    ax.plot(_epochs(val_rows), _values(val_rows, "accuracy"), label="validation", marker="o")
    ax.plot(_epochs(test_rows), _values(test_rows, "accuracy"), label="test", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curves – Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curves_accuracy.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CPU vs XPU quality comparison  (US2 – T024)
# ---------------------------------------------------------------------------


def plot_device_comparison(results_dir: Path) -> None:
    """Overlay cpu and xpu training-loss curves if both sub-dirs exist."""
    cpu_dir = results_dir / "cpu"
    xpu_dir = results_dir / "xpu"

    if not cpu_dir.exists() or not xpu_dir.exists():
        return  # not enough data – skip silently

    cpu_train = _read_csv(cpu_dir / "metrics_train.csv")
    xpu_train = _read_csv(xpu_dir / "metrics_train.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(_epochs(cpu_train), _values(cpu_train, "loss"), label="cpu")
    axes[0].plot(_epochs(xpu_train), _values(xpu_train, "loss"), label="xpu")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CPU vs XPU – Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(_epochs(cpu_train), _values(cpu_train, "accuracy"), label="cpu")
    axes[1].plot(_epochs(xpu_train), _values(xpu_train, "accuracy"), label="xpu")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CPU vs XPU – Training Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "device_comparison.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CPU vs XPU timing comparison  (US2 – T026)
# ---------------------------------------------------------------------------


def plot_time_comparison(results_dir: Path) -> None:
    """Bar chart of total training time for cpu vs xpu runs."""
    cpu_summary = results_dir / "cpu" / "run_summary.csv"
    xpu_summary = results_dir / "xpu" / "run_summary.csv"

    if not cpu_summary.exists() or not xpu_summary.exists():
        return  # not enough data – skip silently

    cpu_time = float(_read_csv(cpu_summary)[0]["training_time_seconds"])
    xpu_time = float(_read_csv(xpu_summary)[0]["training_time_seconds"])

    devices = ["cpu", "xpu"]
    times = [cpu_time, xpu_time]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(devices, times, color=colors, width=0.4)
    ax.bar_label(bars, labels=[f"{t:.1f}s" for t in times], padding=3)
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("CPU vs XPU – Total Training Time")
    ax.grid(axis="y", alpha=0.3)

    if xpu_time > 0:
        ratio = cpu_time / xpu_time
        ax.set_xlabel(f"XPU speedup: {ratio:.2f}×")

    fig.tight_layout()
    fig.savefig(results_dir / "time_comparison.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix  (US3 – T030)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(results_dir: Path) -> None:
    """Read predictions.csv and render a 10×10 annotated confusion matrix."""
    pred_path = results_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.csv not found in {results_dir}.\n"
            "Run a training pass that includes the test evaluation step."
        )

    rows = _read_csv(pred_path)
    n_classes = 10
    matrix = [[0] * n_classes for _ in range(n_classes)]

    for row in rows:
        true_lbl = int(row["true_label"])
        pred_lbl = int(row["predicted_label"])
        matrix[true_lbl][pred_lbl] += 1

    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > max(matrix[i]) * 0.6 else "black",
                    fontsize=8)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(range(n_classes))
    ax.set_yticklabels(range(n_classes))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Classification report bar chart  (US3 – T031)
# ---------------------------------------------------------------------------


def plot_classification_report(results_dir: Path) -> None:
    """Compute per-class precision/recall/F1 and render a grouped bar chart."""
    pred_path = results_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.csv not found in {results_dir}.\n"
            "Run a training pass that includes the test evaluation step."
        )

    rows = _read_csv(pred_path)
    n_classes = 10

    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for row in rows:
        true_lbl = int(row["true_label"])
        pred_lbl = int(row["predicted_label"])
        if true_lbl == pred_lbl:
            tp[true_lbl] += 1
        else:
            fp[pred_lbl] += 1
            fn[true_lbl] += 1

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []

    for c in range(n_classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    x = list(range(n_classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width for i in x], precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar([i + width for i in x], f1, width=width, label="F1")

    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Score")
    ax.set_title("Classification Report – Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "classification_report.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline  (US2 T025 – wire everything together)
# ---------------------------------------------------------------------------


def run_analysis(results_dir: str) -> None:
    """Execute the full analysis pipeline from *results_dir*.

    Parameters
    ----------
    results_dir:
        Path to a directory containing CSV outputs from train.py.

    Raises
    ------
    SystemExit
        If the directory does not exist.
    """
    path = Path(results_dir)
    if not path.exists():
        print(
            f"[analyze] Error: results directory not found: {results_dir}\n"
            "Run a training pass first: python src/train.py -d <device> -r <DIR>",
            file=sys.stderr,
        )
        sys.exit(1)

    errors: list[str] = []

    # Learning curves
    try:
        plot_learning_curves(path)
        print(f"[analyze] Saved learning_curves_*.png → {path}")
    except FileNotFoundError as exc:
        errors.append(f"Learning curves skipped: {exc}")

    # Device quality comparison
    try:
        plot_device_comparison(path)
        if (path / "device_comparison.png").exists():
            print(f"[analyze] Saved device_comparison.png → {path}")
    except FileNotFoundError as exc:
        errors.append(f"Device comparison skipped: {exc}")

    # Timing comparison
    try:
        plot_time_comparison(path)
        if (path / "time_comparison.png").exists():
            print(f"[analyze] Saved time_comparison.png → {path}")
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"Time comparison skipped: {exc}")

    # Classification outputs
    try:
        plot_confusion_matrix(path)
        print(f"[analyze] Saved confusion_matrix.png → {path}")
    except FileNotFoundError as exc:
        errors.append(
            f"Confusion matrix skipped – predictions.csv not found in {path}.\n"
            "  Remediation: run train.py to generate predictions.csv."
        )

    try:
        plot_classification_report(path)
        print(f"[analyze] Saved classification_report.png → {path}")
    except FileNotFoundError as exc:
        errors.append(
            f"Classification report skipped – predictions.csv not found in {path}.\n"
            "  Remediation: run train.py to generate predictions.csv."
        )

    if errors:
        print("\n[analyze] Warnings:", file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args.results)


if __name__ == "__main__":
    main()
