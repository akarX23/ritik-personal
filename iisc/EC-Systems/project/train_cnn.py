import argparse

import torch
from cnn_aipc import LeNet_AIPC, AlexNet_AIPC

MODEL_CLASSES = {
    "lenet": LeNet_AIPC,
    "alexnet": AlexNet_AIPC,
}


def resolve_device(preferred="GPU"):
    preferred = preferred.upper()
    if preferred == "GPU" and torch.cuda.is_available():
        return "cuda"
    if preferred == "XPU" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if preferred == "CPU":
        return "cpu"

    # fallback chain: GPU -> XPU -> CPU
    print(f"WARNING: Preferred device '{preferred}' is not available.")
    if torch.cuda.is_available():
        print("Falling back to GPU (cuda).")
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Falling back to XPU.")
        return "xpu"
    print("Falling back to CPU.")
    return "cpu"


def reduce_model_with_training(model_aipc, num_epochs=5, target_accuracy=0.98, device="cpu"):
    print("Starting model reduction with training...")
    
    for epoch in range(num_epochs):
        avg_loss = model_aipc.finetune_epoch()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        val_accuracy = model_aipc.evaluate_accuracy()
        print(f"Validation accuracy after epoch {epoch+1}: {val_accuracy:.4f}")
        
        if val_accuracy >= 0.97 * target_accuracy:
            print(f"Target accuracy reached: {val_accuracy:.4f} is within 3% of {target_accuracy:.4f}. Stopping reduction.")
            break
    print("Model reduction with training completed.")

    
def train_model_for_init_acc(model_aipc, num_epochs=30, device="cpu"):
    for epoch in range(num_epochs):
        avg_loss = model_aipc.finetune_epoch()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    initial_accuracy = model_aipc.evaluate_accuracy()
    print(f"Initial validation accuracy: {initial_accuracy:.4f}")
    return initial_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train and reduce CNN models (LeNet / AlexNet)")
    parser.add_argument("--model", type=str, default="lenet", choices=MODEL_CLASSES.keys(),
                        help="Model architecture to use (default: lenet)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and evaluation (default: 64)")
    parser.add_argument("--train-dense", action="store_true",
                        help="Train the dense model from scratch to get initial accuracy")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Number of epochs for initial dense training (default: 30)")
    parser.add_argument("--reduce-epochs", type=int, default=5,
                        help="Number of epochs for model reduction fine-tuning (default: 5)")
    parser.add_argument("--device", type=str, default="GPU",
                        help="Preferred training device: GPU, XPU, or CPU (default: GPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Using model: {args.model}, batch_size: {args.batch_size}, device: {device}")
    
    ModelClass = MODEL_CLASSES[args.model]
    model_aipc = ModelClass(batch_size=args.batch_size)
    model_aipc.load_train_val_datasets(batch_size=args.batch_size)
    print("Datasets loaded successfully.")

    model_aipc.setup_training(device=device)
    print("Model setup for training completed.")

    if args.train_dense:
        print("Training the dense model to get initial accuracy...")
        initial_accuracy = train_model_for_init_acc(model_aipc, num_epochs=args.train_epochs, device=device)
        print(f"Initial accuracy before reduction: {initial_accuracy:.4f}")
    else:
        print("Evaluating initial accuracy without training...")
        initial_accuracy = model_aipc.evaluate_accuracy()
        print(f"Initial accuracy (without training): {initial_accuracy:.4f}")

    print("Starting model reduction with training...")
    reduce_model_with_training(model_aipc, num_epochs=args.reduce_epochs, target_accuracy=initial_accuracy, device=device)


if __name__ == "__main__":
    main()