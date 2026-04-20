import time
import copy
import csv
import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# LeNet Torch module
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load Dataset
def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Re-usable function to train for one epoch and return average loss
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    return running_loss / len(loader.dataset)


# Synchronous Checkpointing
# After each epoch, save the model state dict to disk and wait for it to finish.
def train_synchronous(model, loader, criterion, optimizer, device, epochs, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    epoch_times = []
    start = time.time()
    
    # Train loop
    for epoch in range(1, epochs + 1):
        # Counter for epoch
        ep_start = time.time()
        loss = train_epoch(model, loader, criterion, optimizer, device)
        # Synchronously save a checkpoint and block until done
        path = os.path.join(ckpt_dir, f"epoch_count_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        epoch_times.append(time.time() - ep_start)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} done, loss={loss:.4f}")
    total = time.time() - start
    print(f"Synchronous Time: {total:.2f}s")
    return total, epoch_times


# Only persist() Pipelining
# After each epoch, hand off the disk write to a background thread so training
# can continue. But we still do the (potentially expensive) state_dict copy on
# the main thread, since the model may be modified during the next epoch.
# Writing data to disk using a background thread. We use previous thread join to ensure we don't have multiple concurrent writes.
def persist_in_background(state, path, prev_thread):
    if prev_thread is not None:
        prev_thread.join()  # ensure previous write finished
    torch.save(state, path)

# Training while offloading the torch.save to a bg thread
def train_persist_pipeline(model, loader, criterion, optimizer, device, epochs, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    bg_thread = None
    epoch_times = []
    start = time.time()
    
    # Loop over epochs
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        loss = train_epoch(model, loader, criterion, optimizer, device)
        # Snapshot is done synchronously (deep copy), persist is async. This will block the main thread.
        state = {
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        }
        path = os.path.join(ckpt_dir, f"epoch_count_{epoch}.pt")
        
        # Hand-off the the actual saving to a background thread so it doesn't block training. We also ensure that only one persist happens at a time by joining the previous thread.
        bg_thread = threading.Thread(target=persist_in_background, args=(state, path, bg_thread))
        bg_thread.start()
        epoch_times.append(time.time() - ep_start)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} done, loss={loss:.4f}")
    
    # Wait for the last thread to finish
    if bg_thread is not None:
        bg_thread.join()
    total = time.time() - start
    print(f"Strategy B (Persist pipeline): {total:.2f}s")
    return total, epoch_times


# Snapshot() and persist() Pipelining
# Both snapshot (deep copy) and persist (disk write) happen in a background
# thread, so the main training thread is not blocked at all.
def snapshot_and_persist(model, optimizer, epoch, path, prev_thread):
    # Snapshot
    state = {
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
    }
    # Wait for previous persist to finish before writing
    if prev_thread is not None:
        prev_thread.join()
    # Persist
    torch.save(state, path)

# Training while offloading both snapshot and persist to a bg thread
def train_snapshot_persist_pipeline(model, loader, criterion, optimizer, device, epochs, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    bg_thread = None
    epoch_times = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        loss = train_epoch(model, loader, criterion, optimizer, device)
        path = os.path.join(ckpt_dir, f"epoch_count_{epoch}.pt")
        
        # Both snapshot and persist are offloaded to a background thread. We also ensure that only one persist happens at a time by joining the previous thread. 
        bg_thread = threading.Thread(
            target=snapshot_and_persist,
            args=(model, optimizer, epoch, path, bg_thread)
        )
        bg_thread.start()
        epoch_times.append(time.time() - ep_start)
        if epoch % 10 == 0:
            print(f" Epoch {epoch} done, loss={loss:.4f}")
    
    # Wait for the last thread to finish
    if bg_thread is not None:
        bg_thread.join()
    total = time.time() - start
    print(f"Strategy C (Snapshot+Persist pipeline): {total:.2f}s")
    return total, epoch_times


if __name__ == "__main__":
    EPOCHS = 100
    CSV_PATH = "checkpoint_results.csv"
    DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Setup MNIST dataloader
    loader = get_dataloader()

    results = {}
    all_epoch_times = {}

    # Run all three strategies in a loop and collect metrics
    for name, train_fn, ckpt_dir in [
        ("A_sync",            train_synchronous,              "ckpts_sync"),
        ("B_persist",         train_persist_pipeline,          "ckpts_persist"),
        ("C_snap_persist",    train_snapshot_persist_pipeline,  "ckpts_snap_persist"),
    ]:
        print(f"\nStrategy: {name} ")
        model = LeNet5().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        t, epoch_times = train_fn(model, loader, criterion, optimizer, DEVICE, EPOCHS, ckpt_dir)
        results[name] = t
        all_epoch_times[name] = epoch_times

    # Save per-epoch times to CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        strategies = list(all_epoch_times.keys())
        writer.writerow(["epoch"] + strategies)
        for i in range(EPOCHS):
            row = [i + 1] + [f"{all_epoch_times[s][i]:.4f}" for s in strategies]
            writer.writerow(row)
    print(f"\nPer-epoch times saved to {CSV_PATH}")

    print("\n" + "=" * 50)
    print(f"Summary ({EPOCHS} epochs):")
    print("=" * 50)
    for name, t in results.items():
        print(f"  {name:25s}: {t:.2f}s")
