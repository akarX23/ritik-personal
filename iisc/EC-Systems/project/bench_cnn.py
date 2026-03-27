import argparse
import csv
import logging
import os
import subprocess
import sys
import re
import time
from pathlib import Path

import openvino as ov
import numpy as np
from torchvision import datasets

from cnn_aipc import LeNet_AIPC, AlexNet_AIPC

SUPPORTED_MODELS = {"lenet", "alexnet"}
SUPPORTED_DEVICES = {"CPU", "GPU", "NPU"}

def setup_logger(results_dir: str) -> logging.Logger:
    logger = logging.getLogger("bench_cnn")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG level)
    fh = logging.FileHandler(os.path.join(results_dir, "benchmark.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def build_model(model_name: str, logger: logging.Logger, batch_size: int = 1):
    if model_name == "lenet":
        logger.info("Initializing LeNet model via LeNet_AIPC")
        aipc = LeNet_AIPC(batch_size=batch_size)
    elif model_name == "alexnet":
        logger.info("Initializing AlexNet model via AlexNet_AIPC")
        aipc = AlexNet_AIPC(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    logger.debug(
        "Model analytics extracted: %d layer entries", len(aipc.analytics)
    )
    return aipc


def save_model_xml(aipc, results_dir: str, model_name: str, logger: logging.Logger) -> str:
    xml_path = os.path.join(results_dir, f"{model_name}.xml")
    bin_path = os.path.join(results_dir, f"{model_name}.bin")
    if not os.path.exists(xml_path):
        logger.info("Serializing OpenVINO IR to %s", xml_path)
        ov.save_model(aipc.ov_model, xml_path)
        logger.debug("Model files written: %s, %s", xml_path, bin_path)
    else:
        logger.debug("Model IR already exists at %s, skipping serialization", xml_path)
    return xml_path


def load_dataset(total_images: int, data_dir: str, logger: logging.Logger):
    """Download MNIST test set and return raw images as numpy arrays."""
    dataset = datasets.MNIST(root=data_dir, train=False, download=True)

    num_images = min(total_images, len(dataset))
    images = []
    for i in range(num_images):
        img, _ = dataset[i]
        images.append(np.array(img))  # raw PIL -> numpy, uint8

    logger.info("Loaded %d images from MNIST test set", len(images))
    return images


def run_real_data_benchmark(aipc, raw_images, device, batch_size, logger):
    """Benchmark inference using real images with preprocess_raw_images and predict_batch."""
    preprocessed = aipc.preprocess_raw_images(raw_images)

    load_start = time.perf_counter()
    aipc.init_model_infer_object(device=device)
    load_elapsed = (time.perf_counter() - load_start) * 1000
    logger.info("Model load time on %s: %.4f ms", device, load_elapsed)

    total = len(raw_images)
    num_batches = max(total // batch_size, 1)

    # Warmup
    logger.info("Running warmup inference on %s...", device)
    aipc.predict_batch(preprocessed[:batch_size])

    # Timed runs
    latencies = []
    logger.info("Benchmarking %d batches on %s (batch_size=%d)", num_batches, device, batch_size)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = preprocessed[start_idx:end_idx]

        start = time.perf_counter()
        aipc.predict_batch(batch)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

    latencies_arr = np.array(latencies)
    total_time_s = latencies_arr.sum() / 1000.0
    total_images_processed = num_batches * batch_size

    metrics = {
        "throughput_fps": total_images_processed / total_time_s if total_time_s > 0 else 0,
        "median_latency_ms": float(np.median(latencies_arr)),
        "avg_latency_ms": float(np.mean(latencies_arr)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
    }

    logger.info("Throughput: %.2f FPS", metrics["throughput_fps"])
    logger.info("Median latency: %.4f ms", metrics["median_latency_ms"])
    logger.info("Average latency: %.4f ms", metrics["avg_latency_ms"])
    logger.info("Min latency: %.4f ms", metrics["min_latency_ms"])
    logger.info("Max latency: %.4f ms", metrics["max_latency_ms"])

    return metrics


def run_benchmark_app(
    xml_path: str,
    device: str,
    batch_size: int,
    total_images: int,
    results_dir: str,
    logger: logging.Logger,
    model_name: str = "model",
):
    """Invoke OpenVINO benchmark_app and return parsed metrics."""
    report_folder = os.path.join(results_dir, f"report_{model_name}_bs{batch_size}_{device}")
    os.makedirs(report_folder, exist_ok=True)

    # Determine benchmark_app executable next to the python executable
    scripts_dir = os.path.dirname(sys.executable)
    benchmark_exe = os.path.join(scripts_dir, "benchmark_app")
    if sys.platform == "win32" and not benchmark_exe.endswith(".exe"):
        benchmark_exe += ".exe"

    niter = max(total_images // batch_size, 10)

    cmd = [
        benchmark_exe,
        "-m", xml_path,
        "-d", device,
        "-b", str(batch_size),
        "-niter", str(niter),
        "-hint", "latency",
        "-report_type", "detailed_counters",
        "-report_folder", report_folder,
        "-json_stats",
    ]

    logger.info("Running benchmark_app for device=%s batch_size=%d niter=%d", device, batch_size, niter)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    logger.debug("benchmark_app stdout:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error("benchmark_app stderr:\n%s", result.stderr)
        logger.error("benchmark_app failed for device=%s (exit code %d)", device, result.returncode)
        return None

    metrics = parse_benchmark_output(result.stdout, logger)

    # Also try to read JSON stats produced by benchmark_app
    json_stats_path = os.path.join(report_folder, "benchmark_report.json")
    if os.path.exists(json_stats_path):
        logger.debug("Found JSON report at %s", json_stats_path)

    return metrics


def parse_benchmark_output(stdout: str, logger: logging.Logger) -> dict:
    """Extract latency and throughput from benchmark_app console output."""
    metrics = {}

    # Throughput line: e.g. "Throughput: 1234.56 FPS"
    tp_match = re.search(r"Throughput:\s+([\d.]+)\s+FPS", stdout)
    if tp_match:
        metrics["throughput_fps"] = float(tp_match.group(1))
        logger.info("Throughput: %.2f FPS", metrics["throughput_fps"])

    # Median latency: e.g. "[ INFO ]    Median:     1.23 ms"
    lat_match = re.search(r"Median:\s+([\d.]+)\s+ms", stdout)
    if lat_match:
        metrics["median_latency_ms"] = float(lat_match.group(1))
        logger.info("Median latency: %.2f ms", metrics["median_latency_ms"])

    # Average latency
    avg_match = re.search(r"Average:\s+([\d.]+)\s+ms", stdout)
    if avg_match:
        metrics["avg_latency_ms"] = float(avg_match.group(1))
        logger.info("Average latency: %.2f ms", metrics["avg_latency_ms"])

    # Min latency
    min_match = re.search(r"Min:\s+([\d.]+)\s+ms", stdout)
    if min_match:
        metrics["min_latency_ms"] = float(min_match.group(1))

    # Max latency
    max_match = re.search(r"Max:\s+([\d.]+)\s+ms", stdout)
    if max_match:
        metrics["max_latency_ms"] = float(max_match.group(1))

    if not metrics:
        logger.warning("Could not parse any metrics from benchmark_app output")

    return metrics


def write_layer_metrics_csv(analytics: list, results_dir: str, model_name: str, batch_size: int, logger: logging.Logger):
    csv_path = os.path.join(results_dir, f"{model_name}_bs{batch_size}_layer_metrics.csv")
    fieldnames = ["name", "type", "macs", "memory_traffic", "arith_intensity", "energy_consumption", "supported_on"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in analytics:
            writer.writerow(row)
    logger.info("Layer metrics written to %s", csv_path)


def write_results_csv(all_results: list, results_dir: str, logger: logging.Logger):
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    fieldnames = [
        "model", "device", "method", "batch_size", "total_images",
        "throughput_fps", "median_latency_ms", "avg_latency_ms",
        "min_latency_ms", "max_latency_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    logger.info("Benchmark results written to %s", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="CNN Benchmark using OpenVINO benchmark_app")
    parser.add_argument(
        "--devices", nargs="+", required=True,
        choices=sorted(SUPPORTED_DEVICES),
        help="Device(s) to benchmark on (CPU, GPU, NPU)",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Model(s) to benchmark",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1], help="Batch size(s) for inference")
    parser.add_argument("--total-images", type=int, default=100, help="Number of images to use from the MNIST test set")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store results, logs, and reports")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to download/cache the MNIST dataset")
    parser.add_argument(
        "--methods", nargs="+", choices=["real_data", "benchmark_app"], default=["real_data"],
        help="Benchmark method(s) to run (default: real_data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    logger = setup_logger(args.results_dir)

    logger.info("=== Benchmark Configuration ===")
    logger.info("Models     : %s", ", ".join(args.models))
    logger.info("Devices    : %s", ", ".join(args.devices))
    logger.info("Batch sizes: %s", ", ".join(str(b) for b in args.batch_sizes))
    logger.info("Total imgs : %d", args.total_images)
    logger.info("Results dir: %s", args.results_dir)
    logger.info("Data dir   : %s", args.data_dir)
    logger.info("Methods    : %s", ", ".join(args.methods))

    # Load real images once if using real_data method
    raw_images = None
    if "real_data" in args.methods:
        raw_images = load_dataset(args.total_images, args.data_dir, logger)

    all_results = []
    for device in args.devices:
        logger.info("============================== Device: %s ==============================", device)
        for model_name in args.models:
            logger.info("---------------------------- Model: %s ----------------------------", model_name)
            for batch_size in args.batch_sizes:
                logger.info("  Batch size: %d", batch_size)

                # Build model with current batch size
                aipc = build_model(model_name, logger, batch_size=batch_size)
                write_layer_metrics_csv(aipc.analytics, args.results_dir, model_name, batch_size, logger)

                for method in args.methods:
                    logger.info("  Method: %s", method)

                    row = {
                        "model": model_name,
                        "device": device,
                        "method": method,
                        "batch_size": batch_size,
                        "total_images": args.total_images,
                        "throughput_fps": "",
                        "median_latency_ms": "",
                        "avg_latency_ms": "",
                        "min_latency_ms": "",
                        "max_latency_ms": "",
                    }

                    if method == "real_data":
                        logger.info("Running real-data benchmark on %s", device)
                        metrics = run_real_data_benchmark(
                            aipc, raw_images, device, batch_size, logger,
                        )
                    else:
                        xml_path = save_model_xml(aipc, args.results_dir, model_name, logger)
                        logger.info("Running benchmark_app on %s", device)
                        metrics = run_benchmark_app(
                            xml_path=xml_path,
                            device=device,
                            batch_size=batch_size,
                            total_images=args.total_images,
                            results_dir=args.results_dir,
                            logger=logger,
                            model_name=model_name,
                        )

                    if metrics:
                        row.update(metrics)
                    all_results.append(row)

    # Write CSV summary
    write_results_csv(all_results, args.results_dir, logger)

    logger.info("=== Benchmark complete ===")


if __name__ == "__main__":
    main()
