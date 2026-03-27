#!/usr/bin/env python3
"""
Analyze OpenVINO benchmark results and generate charts + markdown tables.

Usage:
    python analyze_results.py <results_directory>

Outputs are saved into the same results directory:
    - PNG chart files
    - layer_tables.md  (markdown tables)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_benchmark_results(results_dir):
    """Load benchmark_results.csv into a list of dicts."""
    path = os.path.join(results_dir, "benchmark_results.csv")
    if not os.path.isfile(path):
        print(f"[WARN] {path} not found – skipping throughput charts.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_layer_metrics(results_dir):
    """Load all *_layer_metrics.csv files. Returns {(model, batch_size): [rows]}."""
    metrics = {}
    for fname in os.listdir(results_dir):
        m = re.match(r"(.+)_bs(\d+)_layer_metrics\.csv$", fname)
        if not m:
            continue
        model, bs = m.group(1), int(m.group(2))
        path = os.path.join(results_dir, fname)
        with open(path, newline="") as f:
            metrics[(model, bs)] = list(csv.DictReader(f))
    return metrics


def _detect_detailed_report_folders(results_dir):
    """
    Return list of (model, batch_size, device, json_path) for folders matching
    report_{model}_bs{batch_size}_{device}/.
    Returns empty list if no such folders exist (per-layer analysis skipped).
    """
    pattern = re.compile(r"^report_(.+)_bs(\d+)_(\w+)$")
    entries = []
    for name in os.listdir(results_dir):
        full = os.path.join(results_dir, name)
        if not os.path.isdir(full):
            continue
        m = pattern.match(name)
        if not m:
            continue
        model, bs, device = m.group(1), int(m.group(2)), m.group(3)
        jpath = os.path.join(full, "benchmark_detailed_counters_report.json")
        if os.path.isfile(jpath):
            entries.append((model, bs, device, jpath))
    return entries


def load_detailed_reports(results_dir):
    """
    Load per-layer latency data.
    Returns {(model, bs, device): [node_dicts]} or empty dict.
    """
    entries = _detect_detailed_report_folders(results_dir)
    if not entries:
        return {}
    data = {}
    for model, bs, device, jpath in entries:
        with open(jpath) as f:
            report = json.load(f)
        nodes = report["detailed_performance"][0]["nodes"]
        data[(model, bs, device)] = nodes
    return data


# ---------------------------------------------------------------------------
# Table 1: Total MACs comparison across models (markdown)
# ---------------------------------------------------------------------------

def generate_macs_comparison_md(layer_metrics, results_dir):
    """Write a markdown table comparing total MACs across models and batch sizes."""
    if not layer_metrics:
        return

    # Aggregate total MACs
    agg = defaultdict(lambda: defaultdict(float))  # model -> bs -> total_macs
    for (model, bs), rows in layer_metrics.items():
        total = sum(float(r["macs"]) for r in rows)
        agg[model][bs] = total

    models = sorted(agg.keys())
    batch_sizes = sorted({bs for m in agg for bs in agg[m]})

    lines = ["# MACs Comparison Across Models\n"]
    header = "| Model | " + " | ".join(f"bs={bs} (GMACs)" for bs in batch_sizes) + " |"
    sep = "|-------|" + "|".join("--------:" for _ in batch_sizes) + "|"
    lines.append(header)
    lines.append(sep)
    for m in models:
        cols = []
        for bs in batch_sizes:
            val = agg[m].get(bs, 0) / 1e9
            cols.append(f"{val:,.2f}")
        lines.append(f"| {m} | " + " | ".join(cols) + " |")
    lines.append("")

    md_path = os.path.join(results_dir, "macs_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {md_path}")


# ---------------------------------------------------------------------------
# Table: Per-layer MACs breakdown + device compatibility (markdown)
# ---------------------------------------------------------------------------

def generate_layer_tables_md(layer_metrics, results_dir):
    """Write per-layer MACs & compatibility tables as markdown."""
    if not layer_metrics:
        return

    lines = ["# Per-Layer MACs Breakdown and Device Compatibility\n"]

    for (model, bs) in sorted(layer_metrics.keys()):
        rows = layer_metrics[(model, bs)]
        # Filter to layers with nonzero MACs or interesting types
        compute_rows = [r for r in rows if float(r["macs"]) > 0]
        if not compute_rows:
            continue

        lines.append(f"## {model} (batch_size={bs})\n")
        lines.append("| Layer | Type | MACs (ops) | Memory Traffic (bytes) | Arith Intensity (ops/byte) | Energy (J) | Supported On |")
        lines.append("|-------|------|-----:|---------------:|----------------:|-------:|:-------------|")
        total_macs = 0
        for r in compute_rows:
            macs = int(float(r["macs"]))
            mem = int(float(r["memory_traffic"]))
            ai = float(r["arith_intensity"])
            energy = float(r["energy_consumption"])
            name = r["name"]
            # Shorten long layer names for readability
            short = name.split("/")[-1] if "/" in name else name
            full_label = name
            total_macs += macs
            lines.append(
                f"| {full_label} | {r['type']} | {macs:,} | {mem:,} | {ai:.2f} | {energy:.6f} | {r['supported_on']} |"
            )
        lines.append(f"| **Total** | | **{total_macs:,}** | | | | |")
        lines.append("")

    md_path = os.path.join(results_dir, "layer_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {md_path}")


# ---------------------------------------------------------------------------
# Chart 2: Normalized throughput comparison
# ---------------------------------------------------------------------------

def plot_normalized_throughput(bench_rows, results_dir):
    """Vertically stacked subplots: one row per model, normalized to CPU."""
    if not bench_rows:
        return

    # Filter to benchmark_app method only
    rows = [r for r in bench_rows if r.get("method", "benchmark_app") == "benchmark_app"]
    if not rows:
        rows = bench_rows

    # Group by (model, batch_size)
    groups = defaultdict(dict)
    for r in rows:
        key = (r["model"], int(r["batch_size"]))
        groups[key][r["device"]] = float(r["throughput_fps"])

    all_devices = sorted({d for g in groups.values() for d in g})
    models = sorted({k[0] for k in groups})
    batch_sizes = sorted({k[1] for k in groups})

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(4.5, 3 * n_models),
                             sharex=False, squeeze=False)

    for row_idx, model in enumerate(models):
        ax = axes[row_idx, 0]
        model_bs = sorted([bs for bs in batch_sizes if (model, bs) in groups])
        x = np.arange(len(model_bs))
        width = 0.8 / max(len(all_devices), 1)

        for i, dev in enumerate(all_devices):
            norm_vals = []
            for bs in model_bs:
                devmap = groups[(model, bs)]
                cpu_val = devmap.get("CPU", 0)
                if dev in devmap and cpu_val > 0:
                    norm_vals.append(devmap[dev] / cpu_val)
                else:
                    norm_vals.append(0)
            bars = ax.bar(x + i * width, norm_vals, width, label=dev)
            for bar, v in zip(bars, norm_vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.2f}x", ha="center", va="bottom", fontsize=6)

        ax.set_ylabel("Norm. Throughput\n(CPU = 1.0x)", fontsize=8)
        ax.set_title(f"{model}", fontsize=9, fontweight="bold")
        ax.set_xticks(x + width * (len(all_devices) - 1) / 2)
        ax.set_xticklabels([f"bs={bs}" for bs in model_bs], fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Normalized Throughput Comparison Across Devices", fontsize=10, y=1.0)
    plt.tight_layout()
    out = os.path.join(results_dir, "chart_normalized_throughput.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _extract_executed_layers(nodes):
    """Return {layer_name: (real_time_ms, node_type)} for EXECUTED layers."""
    out = {}
    for n in nodes:
        if n["status"] == "Status.EXECUTED" and float(n["real_time"]) > 0:
            out[n["name"]] = (float(n["real_time"]), n["node_type"])
    return out


# ---------------------------------------------------------------------------
# Chart 5: Layer speedup heatmap (device vs device, per layer)
# ---------------------------------------------------------------------------

def _build_macs_lookup(layer_metrics, model, bs):
    """Build {layer_name: macs} from layer_metrics for a given model/bs."""
    rows = layer_metrics.get((model, bs), [])
    lookup = {}
    for r in rows:
        macs = float(r["macs"])
        if macs > 0:
            lookup[r["name"]] = macs
    return lookup


def _match_macs(layer_name, macs_lookup):
    """Find MACs for a layer by exact or substring match against the lookup."""
    if layer_name in macs_lookup:
        return macs_lookup[layer_name]
    for metrics_name, macs in macs_lookup.items():
        if layer_name in metrics_name or metrics_name in layer_name:
            return macs
    return 0


def _format_macs(macs):
    """Format MACs into a human-readable string."""
    if macs >= 1e9:
        return f"{macs / 1e9:.1f}G"
    if macs >= 1e6:
        return f"{macs / 1e6:.1f}M"
    if macs >= 1e3:
        return f"{macs / 1e3:.0f}K"
    return str(int(macs))


_COMPUTE_TYPES = {"Convolution", "MatMul", "FullyConnected"}


def plot_speedup_heatmap(detailed, layer_metrics, results_dir):
    """Heatmap: for each Convolution/MatMul layer show speedup of GPU and NPU over CPU."""
    if not detailed:
        return

    combos = defaultdict(dict)
    for (model, bs, dev), nodes in detailed.items():
        combos[(model, bs)][dev] = _extract_executed_layers(nodes)

    for (model, bs), dev_layers in sorted(combos.items()):
        if "CPU" not in dev_layers:
            continue
        cpu_layers = dev_layers["CPU"]
        other_devices = sorted(d for d in dev_layers if d != "CPU")
        if not other_devices:
            continue

        # Find layers present in CPU and at least one other device
        common = set(cpu_layers.keys())
        for d in other_devices:
            common &= set(dev_layers[d].keys())
        if not common:
            common = set(cpu_layers.keys())
            for d in other_devices:
                common = common.union(set(dev_layers[d].keys()) & set(cpu_layers.keys()))

        # Filter to Convolution / MatMul / FullyConnected with nonzero CPU time
        common = [
            l for l in common
            if cpu_layers.get(l, (0, ""))[0] > 0
            and cpu_layers[l][1] in _COMPUTE_TYPES
        ]
        if not common:
            continue

        macs_lookup = _build_macs_lookup(layer_metrics, model, bs)

        # Sort by CPU latency descending
        common = sorted(common, key=lambda l: cpu_layers[l][0], reverse=True)[:15]

        # Build speedup matrix
        matrix = []
        for l in common:
            row = []
            cpu_t = cpu_layers[l][0]
            for d in other_devices:
                entry = dev_layers[d].get(l)
                d_t = entry[0] if entry else 0
                if d_t > 0:
                    row.append(cpu_t / d_t)
                else:
                    row.append(float("nan"))
            matrix.append(row)

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(max(5, len(other_devices) * 2.5), max(4, len(common) * 0.55)))
        vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0.1
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 10
        norm = mcolors.LogNorm(vmin=max(vmin, 0.01), vmax=max(vmax, 0.02))

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Speedup over CPU (log scale)")

        # Annotate cells
        for i in range(len(common)):
            for j in range(len(other_devices)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}x", ha="center", va="center", fontsize=8,
                            color="black" if 0.3 < val < 3 else "white")

        # Build y-axis labels: short layer name + MACs
        short_labels = []
        for l in common:
            parts = l.split("/")
            short = parts[-1] if len(parts) > 1 else l
            macs = _match_macs(l, macs_lookup)
            if macs > 0:
                short = f"{short}  [{_format_macs(macs)} MACs]"
            short_labels.append(short)

        ax.set_xticks(range(len(other_devices)))
        ax.set_xticklabels(other_devices)
        ax.set_yticks(range(len(common)))
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(f"Layer Speedup over CPU \u2013 {model} bs={bs}")
        plt.tight_layout()
        out = os.path.join(results_dir, f"chart_speedup_heatmap_{model}_bs{bs}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Table: Fused / optimized layer analysis (markdown)
# ---------------------------------------------------------------------------

def generate_fusion_table(detailed, results_dir):
    """Write compact fusion analysis: summary counts by op-type per device."""
    if not detailed:
        return

    # Group by model (fusion patterns are the same across batch sizes)
    model_data = defaultdict(lambda: defaultdict(list))  # model -> device -> [nodes]
    seen = set()
    for (model, bs, dev), nodes in detailed.items():
        if (model, dev) not in seen:
            model_data[model][dev] = nodes
            seen.add((model, dev))

    lines = ["# Fused / Optimized Layer Analysis\n",
             "Layers marked **OPTIMIZED_OUT** or **NOT_RUN** by the runtime "
             "were fused into adjacent layers or eliminated during graph "
             "compilation. The tables below summarise counts by operation type.\n"]

    for model in sorted(model_data.keys()):
        dev_nodes = model_data[model]
        devices = sorted(dev_nodes.keys())

        lines.append(f"## {model}\n")

        # --- Per-device summary counts ---
        lines.append("### Node Status Summary\n")
        lines.append("| Device | Executed | Fused / Optimized-Out | Not Run | Total |")
        lines.append("|--------|--------:|-----------:|--------:|------:|")
        for dev in devices:
            total = len(dev_nodes[dev])
            exe = sum(1 for n in dev_nodes[dev] if n["status"] == "Status.EXECUTED")
            opt = sum(1 for n in dev_nodes[dev] if n["status"] == "Status.OPTIMIZED_OUT")
            nr = sum(1 for n in dev_nodes[dev] if n["status"] == "Status.NOT_RUN")
            lines.append(f"| {dev} | {exe} | {opt} | {nr} | {total} |")
        lines.append("")

        # --- Breakdown by operation type ---
        lines.append("### Fusion Breakdown by Operation Type\n")
        # Collect op-type -> device -> {executed, fused, not_run}
        op_stats = defaultdict(lambda: defaultdict(lambda: {"Executed": 0, "Fused": 0, "Not Run": 0}))
        for dev in devices:
            for n in dev_nodes[dev]:
                ntype = n["node_type"]
                st = n["status"]
                if st == "Status.EXECUTED":
                    op_stats[ntype][dev]["Executed"] += 1
                elif st == "Status.OPTIMIZED_OUT":
                    op_stats[ntype][dev]["Fused"] += 1
                elif st == "Status.NOT_RUN":
                    op_stats[ntype][dev]["Not Run"] += 1

        # Build a compact table: Op Type | CPU status | GPU status | NPU status
        header = "| Op Type | " + " | ".join(devices) + " |"
        sep = "|---------|" + "|".join(":---------:" for _ in devices) + "|"
        lines.append(header)
        lines.append(sep)

        for op in sorted(op_stats.keys()):
            cols = []
            for dev in devices:
                s = op_stats[op][dev]
                parts = []
                if s["Executed"]:
                    parts.append(f"{s['Executed']} exec")
                if s["Fused"]:
                    parts.append(f"**{s['Fused']} fused**")
                if s["Not Run"]:
                    parts.append(f"*{s['Not Run']} NR*")
                cols.append(", ".join(parts) if parts else "\u2014")
            lines.append(f"| {op} | " + " | ".join(cols) + " |")
        lines.append("")

    md_path = os.path.join(results_dir, "fusion_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {md_path}")


# ---------------------------------------------------------------------------
# Chart: Latency vs Throughput scatter
# ---------------------------------------------------------------------------

_DEVICE_MARKERS = {"CPU": "o", "GPU": "s", "NPU": "D"}
_DEVICE_COLORS = {"CPU": "#1f77b4", "GPU": "#ff7f0e", "NPU": "#2ca02c"}


def plot_latency_vs_throughput(bench_rows, results_dir):
    """Scatter plot: median latency vs throughput, colored by device, shaped by model."""
    if not bench_rows:
        return

    rows = [r for r in bench_rows if r.get("method", "benchmark_app") == "benchmark_app"]
    if not rows:
        rows = bench_rows

    fig, ax = plt.subplots(figsize=(9, 6))

    # Track what we've added to legend
    device_legend = {}
    model_legend = {}
    models = sorted({r["model"] for r in rows})
    # Assign distinct markers per model if many; fallback cycle
    model_markers = {}
    marker_cycle = ["o", "s", "D", "^" , "v", "P", "X", "*"]
    for i, m in enumerate(models):
        model_markers[m] = marker_cycle[i % len(marker_cycle)]

    for r in rows:
        dev = r["device"]
        model = r["model"]
        bs = int(r["batch_size"])
        lat = float(r["median_latency_ms"])
        fps = float(r["throughput_fps"])
        color = _DEVICE_COLORS.get(dev, "gray")
        marker = model_markers[model]
        size = 60 + bs / 10  # slightly bigger for larger batch

        ax.scatter(lat, fps, c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(f"bs={bs}", (lat, fps), textcoords="offset points",
                    xytext=(6, 4), fontsize=6, color=color)

        # Collect legend handles
        if dev not in device_legend:
            device_legend[dev] = ax.scatter([], [], c=color, marker="o", s=50,
                                            edgecolors="black", linewidths=0.5, label=dev)
        if model not in model_legend:
            model_legend[model] = ax.scatter([], [], c="gray", marker=marker, s=50,
                                              edgecolors="black", linewidths=0.5, label=model)

    ax.set_xlabel("Median Latency (ms)")
    ax.set_ylabel("Throughput (FPS)")
    ax.set_title("Latency vs Throughput")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # Build two-part legend
    handles1 = list(device_legend.values())
    handles2 = list(model_legend.values())
    leg = ax.legend(handles=handles1 + handles2, loc="best", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(results_dir, "chart_latency_vs_throughput.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Chart: Roofline plot
# ---------------------------------------------------------------------------

# Peak specs (TOPS) and estimated memory bandwidth (GB/s)
_PEAK_TOPS = {"CPU": 5, "GPU": 66, "NPU": 48}
# Intel Core Ultra 7 268V – LPDDR5x shared memory ~67 GB/s
_PEAK_BW_GBs = {"CPU": 67, "GPU": 67, "NPU": 67}


def _build_arith_intensity_lookup(layer_metrics, model, bs):
    """Build {layer_name: (macs, arith_intensity)} from layer_metrics."""
    rows = layer_metrics.get((model, bs), [])
    lookup = {}
    for r in rows:
        macs = float(r["macs"])
        ai = float(r["arith_intensity"])
        if macs > 0 and ai > 0:
            lookup[r["name"]] = (macs, ai)
    return lookup


def _match_layer_metrics(layer_name, lookup):
    """Match a detailed-report layer name to a layer_metrics entry."""
    if layer_name in lookup:
        return lookup[layer_name]
    for metrics_name, val in lookup.items():
        if layer_name in metrics_name or metrics_name in layer_name:
            return val
    return None


def plot_roofline(detailed, layer_metrics, results_dir):
    """Roofline plot: arithmetic intensity vs attained performance per device."""
    if not detailed or not layer_metrics:
        return

    # Group by (model, bs)
    combos = defaultdict(dict)
    for (model, bs, dev), nodes in detailed.items():
        combos[(model, bs)][dev] = _extract_executed_layers(nodes)

    for (model, bs), dev_layers in sorted(combos.items()):
        ai_lookup = _build_arith_intensity_lookup(layer_metrics, model, bs)
        if not ai_lookup:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine x range for roofline ceiling lines
        ai_min, ai_max = 0.1, 1000

        # Draw roofline ceilings per device
        for dev in sorted(_PEAK_TOPS.keys()):
            peak_tops = _PEAK_TOPS[dev]  # TOPS = 1e12 ops/s
            peak_gops = peak_tops * 1e3  # GOps/s
            bw = _PEAK_BW_GBs[dev]       # GB/s
            color = _DEVICE_COLORS.get(dev, "gray")

            # Ridge point: where memory ceiling meets compute ceiling
            ridge_ai = peak_gops / bw  # ops/byte

            # Memory-bound region (diagonal): perf = bw * ai
            ai_range_mem = np.logspace(np.log10(ai_min), np.log10(ridge_ai), 200)
            perf_mem = bw * ai_range_mem  # GOps/s

            # Compute-bound region (horizontal)
            ai_range_comp = np.logspace(np.log10(ridge_ai), np.log10(ai_max), 200)
            perf_comp = np.full_like(ai_range_comp, peak_gops)

            ax.plot(ai_range_mem, perf_mem, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
            ax.plot(ai_range_comp, perf_comp, color=color, linewidth=1.5, linestyle="--", alpha=0.7,
                    label=f"{dev} roof ({peak_tops} TOPS, {bw} GB/s)")

        # Plot data points per device
        for dev, layers in sorted(dev_layers.items()):
            color = _DEVICE_COLORS.get(dev, "gray")
            marker = _DEVICE_MARKERS.get(dev, "o")

            for lname, (real_time_ms, ntype) in layers.items():
                match = _match_layer_metrics(lname, ai_lookup)
                if match is None:
                    continue
                macs, ai = match
                # Attained performance: GOps/s = MACs / (time_ms * 1e-3) / 1e9
                attained_gops = macs / (real_time_ms * 1e-3) / 1e9

                ax.scatter(ai, attained_gops, c=color, marker=marker, s=50,
                           edgecolors="black", linewidths=0.4, zorder=4)

                # Label the point with short layer name
                parts = lname.split("/")
                short = parts[-1] if len(parts) > 1 else lname
                ax.annotate(short, (ai, attained_gops), textcoords="offset points",
                            xytext=(5, 4), fontsize=5, color=color, alpha=0.85)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (ops/byte)")
        ax.set_ylabel("Attained Performance (GOps/s)")
        ax.set_title(f"Roofline Plot \u2013 {model} bs={bs}")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        out = os.path.join(results_dir, f"chart_roofline_{model}_bs{bs}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze OpenVINO benchmark results.")
    parser.add_argument("results_dir", help="Path to the results directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' is not a directory.")
        sys.exit(1)

    print(f"Analyzing results in: {results_dir}\n")

    # Load data
    bench_rows = load_benchmark_results(results_dir)
    layer_metrics = load_layer_metrics(results_dir)
    detailed = load_detailed_reports(results_dir)

    has_per_layer = len(detailed) > 0
    if not has_per_layer:
        print("[INFO] No report_{model}_bs{batch_size}_{device} folders found – "
              "skipping per-layer latency charts.\n")

    # 1. MACs comparison across models (table)
    print("[1/7] MACs comparison across models...")
    generate_macs_comparison_md(layer_metrics, results_dir)

    # 2. Per-layer MACs table (markdown)
    print("[2/7] Per-layer MACs breakdown tables...")
    generate_layer_tables_md(layer_metrics, results_dir)

    # 3. Normalized throughput
    print("[3/7] Normalized throughput comparison...")
    plot_normalized_throughput(bench_rows, results_dir)

    # 4. Layer speedup heatmap
    print("[4/7] Layer speedup heatmap...")
    if has_per_layer:
        plot_speedup_heatmap(detailed, layer_metrics, results_dir)
    else:
        print("  Skipped (no per-layer data).")

    # 5. Fused / optimized layer analysis
    print("[5/7] Fused/optimized layer analysis...")
    if has_per_layer:
        generate_fusion_table(detailed, results_dir)
    else:
        print("  Skipped (no per-layer data).")

    # 6. Latency vs Throughput scatter
    print("[6/7] Latency vs Throughput scatter...")
    plot_latency_vs_throughput(bench_rows, results_dir)

    # 7. Roofline plot
    print("[7/7] Roofline plot...")
    if has_per_layer and layer_metrics:
        plot_roofline(detailed, layer_metrics, results_dir)
    else:
        print("  Skipped (need both per-layer reports and layer metrics).")

    print("\nDone.")


if __name__ == "__main__":
    main()
