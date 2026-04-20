import csv
import sys
import matplotlib.pyplot as plt

LABELS = {
    "A_sync": "Synchronous",
    "B_persist": "Persist Pipeline",
    "C_snap_persist": "Snapshot+Persist Pipeline",
}

def generate_charts(csv_path="checkpoint_results.csv"):
    # Read per-epoch times from CSV
    all_epoch_times = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        strategies = header[1:]  # skip 'epoch'
        for s in strategies:
            all_epoch_times[s] = []
        for row in reader:
            for i, s in enumerate(strategies):
                all_epoch_times[s].append(float(row[i + 1]))

    num_epochs = len(all_epoch_times[strategies[0]])
    epochs_range = list(range(1, num_epochs + 1))

    # ── Chart 1: Bar chart of total training time ──────────────────────────────
    totals = {s: sum(times) for s, times in all_epoch_times.items()}
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    names = list(totals.keys())
    bars = ax1.bar(
        [LABELS.get(n, n) for n in names],
        [totals[n] for n in names],
        color=['#e74c3c', '#3498db', '#2ecc71'],
        edgecolor='black',
    )
    for bar, n in zip(bars, names):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{totals[n]:.2f}s", ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel("Total Training Time (s)")
    ax1.set_title("Total Training Time by Checkpointing Strategy")
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_total_time.png", dpi=150)
    print("Saved chart_total_time.png")

    # ── Chart 2: Cumulative time per epoch ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for name, times in all_epoch_times.items():
        cumulative = [sum(times[:i+1]) for i in range(len(times))]
        ax2.plot(epochs_range, cumulative, label=LABELS.get(name, name), linewidth=2, alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cumulative Time (s)")
    ax2.set_title("Cumulative Training Time per Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_cumulative_time.png", dpi=150)
    print("Saved chart_cumulative_time.png")

    plt.show()

    # Print summary
    print("\n" + "=" * 50)
    print(f"Summary ({num_epochs} epochs):")
    print("=" * 50)
    for s in strategies:
        print(f"  {LABELS.get(s, s):35s}: {totals[s]:.2f}s")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_results.csv"
    generate_charts(csv_path)
