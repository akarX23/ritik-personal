import logging
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import itertools
import matplotlib.pyplot as plt

# Built-in multiprocessing library for parallelism
from multiprocessing import Process, Pipe, Queue, Barrier

from lenet import DenseNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_mnist_subset(n_samples=100):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    return Subset(dataset, indices)

# All reduce implementation, where each worker sends its local gradients and receives the local gradients from all other workers.
# The number of communications required reach O(num_workers^2) since each worker needs to send to and receive from every other worker.
def all_reduce(rank, gradients, pipes, _):
    worker_grads = [] # List to store gradients from other workers.
    
    # We first send the local gradients to all other workers using the writer pipes.
    # This avoids deadlock when the training starts. If we receive first, then it can happen that in the first round there are no gradients to be shared and all workers halt.
    for writer in pipes["writers"]:
        writer.send(gradients)
        
    # Small sleep to ensure all workers have sent their gradients before we start receiving.
    # While not necessary, the script threw error sometimes. This helps increase reliability of the script.
    time.sleep(0.05) 
    
    # Receiving gradients from all the other workers using the reader pipes.
    for reader in pipes["readers"]:
        worker_grads.append(reader.recv())

    # Concatenating all gradients and averaging them for each parameter.
    all_gradients = [gradients] + worker_grads
    averaged_gradients = [torch.mean(torch.stack([g[i] for g in all_gradients]), dim=0) for i in range(len(gradients))]
    logger.debug(f"Worker {rank}: averaged gradients from {len(all_gradients)} workers")
    return averaged_gradients 

# The logic here is to arrange the worker in a ring. Each worker chunks its gradients and communicates one chunk with its neighbour until it has all the chunks for a particular grad_index from all workers.
# In the second phase, the summed chunks are again communicated between the workers until all workers have all the gradients.
# Finally, the summed gradients are averaged, giving the required gradients for the next training step.
def ring_all_reduce(rank, gradients, pipes, num_workers):
    # Split the gradients into num_workers chunks. Chunking is done to better utilize the bandwidth since gradients can be huge tensors.
    # Flattening is done here to ensure that the chunk sizes are more or less equal, since the original gradients can have different shapes and sizes.
    grad_chunks = list(torch.chunk(torch.cat([g.reshape(-1) for g in gradients]), num_workers))
    
    # Scatter Reduce phase: Each worker sends its chunk to the next worker and receives a chunk from the previous worker in a ring manner.
    for i in range(num_workers - 1):
        send_idx = (rank - i) % num_workers
        pipes["writer"].send({"chunk": grad_chunks[send_idx], "index": send_idx})
        data = pipes["reader"].recv()
        
        # Add the chunk received in the correct position in the grad_chunks list.
        grad_chunks[data["index"]] += data["chunk"]
        
    # All-Gather phase: Each worker sends its copy of summed chunks, and receives the summed chunks from the other workers.
    # After num_workers - 1 iterations, each worker has all the required gradients.    
    for i in range(num_workers - 1):
        send_idx = (rank - i + 1) % num_workers
        pipes["writer"].send({"chunk": grad_chunks[send_idx], "index": send_idx})
        data = pipes["reader"].recv()
        grad_chunks[data["index"]] = data["chunk"]
        
    # Concatenate all the gradients and average them for each parameter.
    summed_flat_grad = torch.cat(grad_chunks)
    averaged_flat_grad = summed_flat_grad / num_workers
    
    # Un-flatten the averaged_gradients to match the original shapes of the gradients for each parameter.
    averaged_gradients = []
    offset = 0
    for g in gradients:
        num_elements = g.numel()
        # Slice out the correct number of elements and reshape to match the original gradient
        grad_tensor = averaged_flat_grad[offset : offset + num_elements].view_as(g)
        averaged_gradients.append(grad_tensor)
        offset += num_elements
        
    logger.debug(f"Worker {rank}: successfully completed ring all-reduce")
    return averaged_gradients

# Function which runs indepdently on each worker process. 
# Each worker computes gradients on its local data and then participates in the gradient reduction step using the provided reduce_fn and communication pipes.
def train_worker(rank, dataset, model_state, epochs, reduce_fn, pipes, result_queue, barrier, num_workers):
    # Set one thread for the pytorch operations to avoid contentions between workers, else workers throw error.
    torch.set_num_threads(1)
    # Setting the process to a particular CPU core.
    os.sched_setaffinity(0, {rank % os.cpu_count()})
    logger.debug(f"Worker {rank}: initialized on CPU {rank % os.cpu_count()}, dataset size={len(dataset)}")
    
    # Model, loss, and loader initialization inside the worker for its local state.
    model = DenseNet()
    model.load_state_dict(model_state)
    criterion = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for _, (images, labels) in enumerate(tqdm(loader)):
            outputs = model(images)
            targets = nn.functional.one_hot(labels, num_classes=10).float()
            loss = criterion(outputs, targets)

            # Backpropagation to compute local gradients
            model.zero_grad()
            loss.backward()
            grads = [p.grad.clone() for p in model.parameters()]
            
            # Calling the reduction function, providing the local gradients and the reader and writer pipes for this worker as arguments.
            # The expected return value is the averaged gradients across all workers for this batch, which the worker can then use to update its model parameters.
            logger.debug(f"Worker {rank}: epoch {epoch+1}, computed local gradients, entering reduce")
            averaged_grads = reduce_fn(rank, grads, pipes, num_workers)

            # Normal model update step using the averaged gradients. We use a fixed learning rate of 0.01 for simplicity.
            if averaged_grads is not None:
                with torch.no_grad():
                    for param, grad in zip(model.parameters(), averaged_grads):
                        param -= 0.01 * grad
                logger.debug(f"Worker {rank}: epoch {epoch+1}, applied averaged gradients")

        logger.debug(f"Worker {rank} | Epoch {epoch + 1} | Loss: {loss.item():.4f}")

    # After all epochs, we wait for all workers to reach this point using the barrier, and then we let worker 0 put the final model state in the result queue for the main process to read.
    # We assume worker 0 to be a coordinator, while in reality this role might be taken by a separate process completely.
    barrier.wait()
    if rank == 0:
        result_queue.put(model.state_dict())
        time.sleep(0.1)

# Function to create a dictionary of communication pipes for the all-reduce strategy. Each worker has a list of reader and writer connections to communicate with every other worker.
# For num_workers, each worker will have num_workers-1 writers and num_workers-1 readers, since it needs to send gradients to all other workers and receive from all other workers.
def setup_all_reduce_pipes(num_workers):
    # Intialize the dictionary
    worker_conn_map = {i: {"readers": [], "writers": []} for i in range(num_workers)}
    
    for worker_i in range(num_workers):
        for worker_j in range(num_workers):
            # No pipe requried for the same worker
            if worker_i == worker_j:
                continue
            # Creating a pipe with one writer end and one receiver end.
            reader_conn, writer_conn = Pipe(duplex=False)
            
            # We treat worker_i as the sender and worker_j as the receiver for this pipe, so we add the writer connection to worker_i's list of writers and the reader connection to worker_j's list of readers.
            worker_conn_map[worker_i]["writers"].append(writer_conn)
            worker_conn_map[worker_j]["readers"].append(reader_conn)
    return worker_conn_map

def setup_ring_pipes(num_workers):
    # Intialize the dictionary
    worker_conn_map = {i: {"reader": None, "writer": None} for i in range(num_workers)}
    
    for worker_i in range(num_workers):
        reader_conn, writer_conn = Pipe(duplex=False)
        worker_conn_map[(worker_i + 1) % num_workers]["writer"] = writer_conn
        worker_conn_map[(worker_i - 1) % num_workers]["reader"] = reader_conn
    
    return worker_conn_map

# Re-usable function for testing multiple scenarios
def train_parallel(n_samples=100, num_workers=4, epochs=10, use_ring=False):
    dataset = load_mnist_subset(n_samples)

    # Creating a dataset split for each worker
    dataset_size_per_worker = len(dataset) // num_workers
    data_splits = []
    for i in range(num_workers):
        data_splits.append(Subset(dataset, list(range(i * dataset_size_per_worker, (i + 1) * dataset_size_per_worker))))

    # Loading the LeNet-5 from Assignment-2
    model = DenseNet()
    model_state = model.state_dict()
    
    # A multiprocessing queue to share the final model state from the workers to the main process. 
    result_queue = Queue()
    # The barrier is used to let workers synchronize at the end of training before sending the final model state.
    # Without this, the workers might communicate even after the final model state is shared, leading to exceptions.
    barrier = Barrier(num_workers)

    # Choosing the reduction function and setting up communication pipes
    if use_ring:
        reduce_fn = ring_all_reduce
        all_pipes = setup_ring_pipes(num_workers)
    else:
        reduce_fn = all_reduce
        all_pipes = setup_all_reduce_pipes(num_workers)

    # Creating worker processes for computing gradients in parallel. Each worker stores the complete model.
    processes = []
    for rank in range(num_workers):
        # We pass the model state dict to each worker so they can initialize their local model with the same parameters, a result queue to which the final model state will be written
        # and the communication pipes depending on the reduction strategy. 
        p = Process(target=train_worker,
                    args=(rank, data_splits[rank], model_state, epochs,
                          reduce_fn, all_pipes[rank], result_queue, barrier, num_workers))
        p.start()
        logger.debug(f"Spawned worker {rank} (pid={p.pid})")
        processes.append(p)

    # Getting the final model weights from Worker 0
    final_state = result_queue.get()

    for p in processes:
        p.join()
    logger.debug("All workers finished")

    # Loading final model weights and returning for evaluation.
    model.load_state_dict(final_state)
    logger.debug("Final model state loaded")
    return model

# Function to evaluate the accuracy of the trained model on the MNIST test set. This is used to benchmark the performance of the parallel training approaches.
def evaluate_accuracy(model):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Helper function to benchmark the training reduction approaches across different configurations of sample sizes, worker counts, and epoch counts. 
# It runs the training for each configuration, evaluates the accuracy, and saves the results in a CSV file for later analysis and chart generation.
def run_benchmarks(sample_sizes, worker_counts, epoch_counts, approaches, csv_path="benchmark_results.csv"):
    results = []
    
    # Using itertools to generate all combinations
    configs = list(itertools.product(sample_sizes, worker_counts, epoch_counts, approaches))
    print(f"Running {len(configs)} benchmark configurations...")

    for n_samples, num_workers, epochs, approach in configs:
        use_ring = (approach == "ring_all_reduce")
        label = approach
        print(f"\n{label} | samples={n_samples}, workers={num_workers}, epochs={epochs}")

        start = time.time()
        model = train_parallel(n_samples=n_samples, num_workers=num_workers, epochs=epochs, use_ring=use_ring)
        train_time = time.time() - start

        accuracy = evaluate_accuracy(model)

        results.append({
            "approach": label,
            "n_samples": n_samples,
            "num_workers": num_workers,
            "epochs": epochs,
            "accuracy": round(accuracy, 4),
            "train_time_s": round(train_time, 2),
        })
        
        time.sleep(3)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")
    return results

# Generate two bar charts from the benchmark results: one for accuracy and one for training time, both plotted against the combined x-axis of "samples_workers" for each approach.
def generate_charts(csv_path="benchmark_results.csv"):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n_samples"] = int(row["n_samples"])
            row["num_workers"] = int(row["num_workers"])
            row["epochs"] = int(row["epochs"])
            row["accuracy"] = float(row["accuracy"])
            row["train_time_s"] = float(row["train_time_s"])
            rows.append(row)

    approaches = sorted(set(r["approach"] for r in rows))

    # Build x-axis labels as "samples_workers" and group by approach
    # Use only the highest epoch count for each (approach, samples, workers) combo
    def grouped_data(rows, approaches, metric):
        data = {}
        max_epoch = max(r["epochs"] for r in rows)
        for approach in approaches:
            subset = [r for r in rows if r["approach"] == approach and r["epochs"] == max_epoch]
            keys = sorted(set((r["n_samples"], r["num_workers"]) for r in subset))
            labels = [f"{s}_{w}" for s, w in keys]
            values = []
            for s, w in keys:
                matching = [r[metric] for r in subset if r["n_samples"] == s and r["num_workers"] == w]
                values.append(sum(matching) / len(matching))
            data[approach] = (labels, values)
        return data

    import numpy as np

    acc_colors = ['#0072B2', '#E69F00'] 
    time_colors = ['#D55E00', '#56B4E9']

    # Chart 1: Accuracy vs samples_workers
    acc_data = grouped_data(rows, approaches, "accuracy")
    labels = list(acc_data.values())[0][0]
    x = np.arange(len(labels))
    width = 0.8 / len(approaches)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    for i, approach in enumerate(approaches):
        vals = acc_data[approach][1]
        bars = ax.bar(x + i * width, vals, width, label=approach, color=acc_colors[i % len(acc_colors)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xlabel("Samples_Workers")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Samples_Workers")
    ax.set_xticks(x + width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig("benchmark_accuracy.png", dpi=150)
    print("Saved benchmark_accuracy.png")

    # Chart 2: Training time vs samples_workers
    time_data = grouped_data(rows, approaches, "train_time_s")
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    for i, approach in enumerate(approaches):
        vals = time_data[approach][1]
        bars = ax.bar(x + i * width, vals, width, label=approach, color=time_colors[i % len(time_colors)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.1f}s',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xlabel("Samples_Workers")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Training Time vs Samples_Workers")
    ax.set_xticks(x + width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig("benchmark_time.png", dpi=150)
    print("Saved benchmark_time.png")


if __name__ == "__main__":
    # Define parameter grid for benchmarks
    SAMPLE_SIZES = [100, 5000]
    WORKER_COUNTS = [4, 8]
    EPOCH_COUNTS = [30]
    APPROACHES = ["all_reduce", "ring_all_reduce"]
    
    CSV_PATH = "results_laptop.csv"
    results = run_benchmarks(SAMPLE_SIZES, WORKER_COUNTS, EPOCH_COUNTS, APPROACHES, CSV_PATH)
    generate_charts(CSV_PATH)

# if __name__ == "__main__":
#     # Quick test run with a single configuration to verify everything is working before running the full benchmarks.
#     model = train_parallel(n_samples=100, num_workers=4, epochs=10, use_ring=True)
#     accuracy = evaluate_accuracy(model)
#     print(f"Test run accuracy: {accuracy:.4f}")