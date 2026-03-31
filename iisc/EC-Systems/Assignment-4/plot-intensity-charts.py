import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def arithmetic_intensity(N, d, w_bytes=2):
    return (4 * N * d - N - d) / (4 * (N + d) * w_bytes)


def plot_ai_curves(d=128, w_bytes=2):
    prefill_seq_lens = [128, 256, 512, 1024, 2048, 3072, 4096]
    decode_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    prefill_ai = [arithmetic_intensity(N, d, w_bytes) for N in prefill_seq_lens]
    decode_ai = [arithmetic_intensity(B, d, w_bytes) for B in decode_batch_sizes]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Prefill plot
    axes[0].plot(prefill_seq_lens, prefill_ai, "o-", color="tab:blue", linewidth=2, markersize=6)
    axes[0].set_xlabel("Context Length (N)")
    axes[0].set_ylabel("Arithmetic Intensity (FLOPs/Byte)")
    axes[0].set_title("Prefill Phase (Batch Size = 1)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(prefill_seq_lens)
    axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[0].grid(True, alpha=0.3)

    # Decode plot
    axes[1].plot(decode_batch_sizes, decode_ai, "s-", color="tab:orange", linewidth=2, markersize=6)
    axes[1].set_xlabel("Batch Size (B)")
    axes[1].set_ylabel("Arithmetic Intensity (FLOPs/Byte)")
    axes[1].set_title("Decode Phase (1 token per request)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(decode_batch_sizes)
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Arithmetic Intensity (d={d}, w_bytes={w_bytes})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("ai_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_ai_curves()
