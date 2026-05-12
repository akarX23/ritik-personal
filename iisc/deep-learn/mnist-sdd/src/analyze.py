"""analyze.py – Offline analysis and visualisation for MNIST training runs.

Usage:
    python src/analyze.py -r ./results

Generates from saved CSV files:
  - learning_curves_loss.png / learning_curves_accuracy.png
  - device_comparison.png  (if cpu/ and xpu/ sub-dirs are present)
  - time_comparison.png    (if run_summary.csv in both device sub-dirs)
  - confusion_matrix.png
  - classification_report.png
    - results.md

All artefacts are written into the results directory provided via -r.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from src.metrics import run_log_path

import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402


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


def sample_epoch_rows(rows: list[dict]) -> list[dict]:
    """Return rows sampled at epochs multiple of 10 plus final epoch."""
    if not rows:
        return []

    final_epoch = max(int(r["epoch"]) for r in rows)
    sampled = [
        r for r in rows
        if int(r["epoch"]) % 10 == 0 or int(r["epoch"]) == final_epoch
    ]
    return sorted(
        sampled,
        key=lambda r: (int(r["epoch"]), str(r.get("batch_size", "")), str(r.get("run_id", ""))),
    )


def _final_row_by_run_batch(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Pick the final-epoch row for each (run_id, batch_size) pair."""
    final_rows: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("run_id", "")), str(row.get("batch_size", "")))
        epoch = int(row["epoch"])
        prev = final_rows.get(key)
        if prev is None or epoch >= int(prev["epoch"]):
            final_rows[key] = row
    return final_rows


def build_final_metrics_table(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> str:
    """Build markdown table for final split metrics by batch size."""
    train_final = _final_row_by_run_batch(train_rows)
    val_final = _final_row_by_run_batch(val_rows)
    test_final = _final_row_by_run_batch(test_rows)

    batches = sorted(
        {
            str(r.get("batch_size", ""))
            for r in train_rows + val_rows + test_rows
            if str(r.get("batch_size", ""))
        },
        key=lambda b: int(b),
    )

    lines = [
        "## Final Metrics by Batch Size",
        "",
        (
            "| Batch Size | Train Loss | Train Accuracy | Validation Loss | "
            "Validation Accuracy | Test Loss | Test Accuracy |"
        ),
        "|---|---|---|---|---|---|---|",
    ]

    def pick_best_for_batch(rows_by_key: dict[tuple[str, str], dict], batch: str) -> dict | None:
        candidates = [r for (_, b), r in rows_by_key.items() if b == batch]
        if not candidates:
            return None
        return sorted(candidates, key=lambda r: (int(r["epoch"]), str(r.get("run_id", ""))))[-1]

    for batch in batches:
        train = pick_best_for_batch(train_final, batch)
        val = pick_best_for_batch(val_final, batch)
        test = pick_best_for_batch(test_final, batch)

        def fmt(row: dict | None, key: str) -> str:
            return f"{float(row[key]):.6f}" if row and row.get(key, "") != "" else "n/a"

        lines.append(
            f"| {batch} | {fmt(train, 'loss')} | {fmt(train, 'accuracy')} | "
            f"{fmt(val, 'loss')} | {fmt(val, 'accuracy')} | {fmt(test, 'loss')} | {fmt(test, 'accuracy')} |"
        )

    return "\n".join(lines)


def _build_epoch_comparison_table(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> str:
    """Build markdown table with sampled historical rows across splits."""
    merged_rows = []
    for split, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        sampled = sample_epoch_rows(rows)
        for row in sampled:
            merged = dict(row)
            merged["split"] = split
            merged_rows.append(merged)

    merged_rows.sort(
        key=lambda r: (
            int(r["epoch"]),
            str(r.get("batch_size", "")),
            str(r.get("run_id", "")),
            str(r.get("split", "")),
        )
    )

    lines = [
        "## Epoch Comparison",
        "",
        "| Epoch | Batch Size | Run ID | Split | Loss | Accuracy | Device |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in merged_rows:
        lines.append(
            "| {epoch} | {batch_size} | {run_id} | {split} | {loss:.6f} | {accuracy:.6f} | {device} |".format(
                epoch=int(row["epoch"]),
                batch_size=row.get("batch_size", "n/a"),
                run_id=row.get("run_id", "n/a"),
                split=row.get("split", "n/a"),
                loss=float(row.get("loss", 0.0)),
                accuracy=float(row.get("accuracy", 0.0)),
                device=row.get("device", "n/a"),
            )
        )

    return "\n".join(lines)


def _build_config_section(results_dir: Path, train_rows: list[dict], run_summary_rows: list[dict]) -> str:
    """Build configuration section for compared runs."""
    devices = sorted({str(r.get("device", "")) for r in run_summary_rows if str(r.get("device", ""))})
    epochs = sorted({str(r.get("epochs", "")) for r in run_summary_rows if str(r.get("epochs", ""))})
    learning_rates = sorted(
        {str(r.get("learning_rate", "")) for r in run_summary_rows if str(r.get("learning_rate", ""))}
    )
    summary_batches = {
        str(r.get("batch_size", "")) for r in run_summary_rows if str(r.get("batch_size", ""))
    }
    metric_batches = {
        str(r.get("batch_size", "")) for r in train_rows if str(r.get("batch_size", ""))
    }
    batch_sizes = sorted(summary_batches | metric_batches, key=lambda b: int(b))

    return "\n".join(
        [
            "## Configuration",
            "",
            f"- results_dir: {results_dir}",
            f"- device: {', '.join(devices) if devices else 'n/a'}",
            f"- epochs: {', '.join(epochs) if epochs else 'n/a'}",
            f"- learning_rate: {', '.join(learning_rates) if learning_rates else 'n/a'}",
            f"- batch_sizes: {', '.join(batch_sizes) if batch_sizes else 'n/a'}",
            "",
            "This report includes all matching historical rows found in the selected results directory.",
        ]
    )


def generate_results_report(results_dir: Path) -> Path:
    """Generate results.md with config, final metrics, and epoch comparison."""
    train_rows = _read_csv(results_dir / "metrics_train.csv")
    val_rows = _read_csv(results_dir / "metrics_validation.csv")
    test_rows = _read_csv(results_dir / "metrics_test.csv")

    summary_path = results_dir / "run_summary.csv"
    run_summary_rows = _read_csv(summary_path) if summary_path.exists() else []

    parts = [
        "# Results Report",
        "",
        _build_config_section(results_dir, train_rows, run_summary_rows),
        "",
        build_final_metrics_table(train_rows, val_rows, test_rows),
        "",
        _build_epoch_comparison_table(train_rows, val_rows, test_rows),
        "",
    ]

    report_path = results_dir / "results.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path


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
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.set_yticklabels([str(i) for i in range(n_classes)])
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
    run_id = str(uuid.uuid4())
    logger = _configure_analysis_logger(run_id, path)
    if not path.exists():
        print(
            f"[analyze] Error: results directory not found: {results_dir}\n"
            "Run a training pass first: python src/train.py -d <device> -r <DIR>",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("event=lifecycle stage=start run_id=%s results_dir=%s", run_id, path)

    errors: list[str] = []

    # Learning curves
    try:
        plot_learning_curves(path)
        print(f"[analyze] Saved learning_curves_*.png → {path}")
        logger.info("event=lifecycle stage=learning_curves status=ok")
    except FileNotFoundError as exc:
        errors.append(f"Learning curves skipped: {exc}")
        logger.warning("event=lifecycle stage=learning_curves status=skipped reason=%s", exc)

    # Device quality comparison
    try:
        plot_device_comparison(path)
        if (path / "device_comparison.png").exists():
            print(f"[analyze] Saved device_comparison.png → {path}")
            logger.info("event=lifecycle stage=device_comparison status=ok")
    except FileNotFoundError as exc:
        errors.append(f"Device comparison skipped: {exc}")
        logger.warning("event=lifecycle stage=device_comparison status=skipped reason=%s", exc)

    # Timing comparison
    try:
        plot_time_comparison(path)
        if (path / "time_comparison.png").exists():
            print(f"[analyze] Saved time_comparison.png → {path}")
            logger.info("event=lifecycle stage=time_comparison status=ok")
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"Time comparison skipped: {exc}")
        logger.warning("event=lifecycle stage=time_comparison status=skipped reason=%s", exc)

    # Classification outputs
    try:
        plot_confusion_matrix(path)
        print(f"[analyze] Saved confusion_matrix.png → {path}")
        logger.info("event=lifecycle stage=confusion_matrix status=ok")
    except FileNotFoundError as exc:
        errors.append(
            f"Confusion matrix skipped – predictions.csv not found in {path}.\n"
            "  Remediation: run train.py to generate predictions.csv."
        )
        logger.warning("event=lifecycle stage=confusion_matrix status=skipped reason=%s", exc)

    try:
        plot_classification_report(path)
        print(f"[analyze] Saved classification_report.png → {path}")
        logger.info("event=lifecycle stage=classification_report status=ok")
    except FileNotFoundError as exc:
        errors.append(
            f"Classification report skipped – predictions.csv not found in {path}.\n"
            "  Remediation: run train.py to generate predictions.csv."
        )
        logger.warning("event=lifecycle stage=classification_report status=skipped reason=%s", exc)

    # Markdown report generation
    try:
        logger.info("event=lifecycle stage=results_report status=start")
        report_path = generate_results_report(path)
        print(f"[analyze] Saved results.md → {report_path}")
        logger.info("event=lifecycle stage=results_report status=ok path=%s", report_path)
    except FileNotFoundError as exc:
        errors.append(f"Results report skipped: {exc}")
        logger.warning("event=lifecycle stage=results_report status=skipped reason=%s", exc)

    if errors:
        print("\n[analyze] Warnings:", file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)

    logger.info("event=lifecycle stage=complete run_id=%s warning_count=%d", run_id, len(errors))


def _configure_analysis_logger(run_id: str, results_path: Path) -> logging.Logger:
    """Create a plain-text logger that writes to console and run log file."""
    logger = logging.getLogger(f"mnist.analyze.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    results_path.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(run_log_path(results_path, run_id), encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args.results)


if __name__ == "__main__":
    main()
