import torch
import time
import sys
import os
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer_skeleton import DecoderOnlyTransformer

# Model config
vocab_size = 1000
d_model = 50
d_ff = 100
num_layers = 2
max_seq_len = 1200

seq_lengths = [10, 50, 100, 500, 1000]
decode_len = 40
num_runs = 5

def drop_caches():
    subprocess.run(["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"], check=True)

def benchmark(seq_lengths, decode_len, num_runs, use_cache):
    label = "kv_cache" if use_cache else "no_cache"
    print(f"=== Benchmarking {'WITH' if use_cache else 'WITHOUT'} KV Cache ===")
    times = []
    for seq_len in seq_lengths:
        model = DecoderOnlyTransformer(vocab_size, d_model, d_ff, num_layers, max_seq_len)
        model.eval()
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        # Warmup
        with torch.no_grad():
            model.generate(input_ids, decode_len, use_cache=use_cache)
        t = []
        for _ in range(num_runs):
            drop_caches()
            start = time.time()
            with torch.no_grad():
                model.generate(input_ids, decode_len, use_cache=use_cache)
            t.append(time.time() - start)
        avg = sum(t) / num_runs
        times.append(avg)
        print(f"  seq_len={seq_len:5d} | {label}: {avg:.4f}s")
        del model
    return times

times_with_cache = benchmark(seq_lengths, decode_len, num_runs, use_cache=True)
print()
times_no_cache = benchmark(seq_lengths, decode_len, num_runs, use_cache=False)

# --- Report ---
print("\n=== Benchmark Report ===")
print(f"{'seq_len':>10} | {'no_cache (s)':>12} | {'kv_cache (s)':>12} | {'speedup':>8}")
print("-" * 52)
for i, seq_len in enumerate(seq_lengths):
    speedup = times_no_cache[i] / times_with_cache[i]
    print(f"{seq_len:>10} | {times_no_cache[i]:>12.4f} | {times_with_cache[i]:>12.4f} | {speedup:>7.2f}x")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(seq_lengths, times_no_cache, "o-", label="Without KV Cache")
plt.plot(seq_lengths, times_with_cache, "s-", label="With KV Cache")
plt.xlabel("Input Sequence Length")
plt.xticks(seq_lengths)
plt.ylabel("Latency (seconds)")
plt.title(f"Generation Latency (decode_len={decode_len})")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = os.path.join("./", "kv_cache_benchmark.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
