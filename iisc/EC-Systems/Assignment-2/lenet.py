import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

# For progress bar
from tqdm import tqdm

# For plotting graphs
import matplotlib.pyplot as plt

# Creating the LeNet architecture using Torch CNN module
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        
        # First convolution layer. Input = 32 x 32 image, Output = 6 x 28 x 28 feature maps
        # Add a bias to the kernel as in the original LeNet paper
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=0, bias=True)
        
        # Pooling layer to reduce dimensions. Input is 6 x 28 x 28, Output is 6 x 14 x 14
        # LeNet had average pooling, we are using max pooling here since it's more common in modern architectures
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second Convolution layer. Input = 6 x 14 x 14, Output = 16 x 10 x 10
        # Origional paper had partial connections between the 16 kernels and 6 Input channels
        # For simplicity, we implement a dense convolution here
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0, bias=True)
        
        # Same Pooling as above
        # Input = 16 x 10 x 10, Output = 16 x 5 x 5
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # First fully connected layer with 120 neurons
        # This was originally implemented as a convolution layer in the LeNet paper 
        # However, since the kernel dimensions are same as the input feature map dimensions, we have implemented it as a fully connected 
        # Input = 16 x 5 x 5, Output = 120
        self.linear1 = nn.Linear(16 * 5 * 5, 120, bias=True)
        
        # Second fully connected layer, 84 Neurons
        # Input = 120 neurons, Output = 84 neurons
        self.linear2 = nn.Linear(120, 84)
        
        # Final output layer with 10 neurons for 10 classes (digits 0-9)
        self.linear3 = nn.Linear(84, 10)
        
    def forward(self, input):
        
        # Define ReLU activation and Softmax operations
        relu = nn.ReLU()
        softmax = nn.Softmax(dim=1)
        
        # We implement the neural network operation for images with a 32 x 32 dimension
        
        # First Convolution + Pooling operation
        # Input = 32 x 32, Output = 6 x 14 x 14
        conv_out_1 = self.conv1(input)
        pool_out_1 = self.pool1(conv_out_1)
        
        # Here we use ReLU activation to bring non-linearity in the CNN
        # Origional paper uses tanh, but ReLU provides for faster training and removes the vanishing gradient problem
        act_out_1 = relu(pool_out_1)
        
        # Second Convolution + Pooling Operation
        # Input = 6 x 14 x 14, Output = 16 x 5 x 5
        conv_out_2 = self.conv2(act_out_1)
        pool_out_2 = self.pool2(conv_out_2)
        act_out_2 = relu(pool_out_2)
        
        # Flatten the activation layer from second convolution
        # Input = 16 x 5 x 5, Output = N x 400
        flat_out = torch.flatten(act_out_2, start_dim=1)
        
        # Execute first fully connected layer, with activation
        # Input = 16 x 5 x 5, Output = N x 120
        linear_out_1 = self.linear1(flat_out)
        linear_act_out_1 = relu(linear_out_1)
        
        # Execute second fully connected layer, with activation
        # Input = N x 120, Output = N x 84
        linear_out_2 = self.linear2(linear_act_out_1)
        linear_act_out_2 = relu(linear_out_2)
        
        # Final output layer which classifies the result into 10 classes using softmax
        # In the LeNet paper, this was done using a RBF network, but we are using a fully connected layer followed by softmax for simplicity
        # Input = N x 84, Output = N x 10
        final_out = self.linear3(linear_act_out_2)
        output = softmax(final_out)
        
        return output

class DepthWiseSeparableNet(nn.Module):
    def __init__(self):
        super(DepthWiseSeparableNet, self).__init__()
        
        # To perform depth_wise convolution, we set the 'groups' parameter equal to the number of input channels
        # The result is that each input channel has a filter for itself, due to which only spatial correlation is captured
        # Point wise convolution is then applied to multiply and add values across channels
        
        # Depth Wise convolution for each input channel. Input = 1 x 32 x 32, Output = 1x 28 x 28 
        self.dep_conv_1 = nn.Conv2d(in_channels=1, out_channels=1, groups=1, kernel_size=5, stride=1, padding=0, bias=True)
        # Point wise convolution to get the correct number of output channels
        # Input = 1 x 28 x 28, Output = 6 x 28 x 28
        self.pnt_conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Pooling layer to reduce dimensions. Input is 6 x 28 x 28, Output is 6 x 14 x 14
        # LeNet had average pooling, we are using max pooling here since it's more common in modern architectures
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second Depth Wise Convolution. Input = 6 x 14 x 14, Output = 6 x 10 x 10
        self.dep_conv_2 = nn.Conv2d(in_channels=6, out_channels=6, groups=6, kernel_size=5, stride=1, padding=0, bias=True)
        # Point Wise Convolution. Input = 6 x 10 x 10, Output = 16 x 10 x 10
        self.pnt_conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Same Pooling as above
        # Input = 16 x 10 x 10, Output = 16 x 5 x 5
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third Convolution Layer
        # We implemented this as a fully connected layer in the DenseNet, but here we implement it as a convolution layer to separate depth wise and point wise convolutions
        # Depth Wise Convolution, Input = 16 x 5 x 5, Output = 16 x 1 x 1
        self.dep_conv_3 = nn.Conv2d(in_channels=16, out_channels=16, groups=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Point Wise Convolution, Input = 16 x 1 x 1, Output = 120 x 1 x 1
        self.pnt_conv_3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=1, stride=1, padding=0, bias=True)
        
        # First fully connected layer, 84 Neurons
        # Input = 120 neurons, Output = 84 neurons
        self.linear1 = nn.Linear(120, 84)
        
        # Final output layer with 10 neurons for 10 classes (digits 0-9)
        self.linear2 = nn.Linear(84, 10)
    
    def forward(self, input):
        
        # Define ReLU activation and Softmax operations
        relu = nn.ReLU()
        softmax = nn.Softmax(dim=1)
        
        # We implement the neural network operation for images with a 32 x 32 dimension
        
        # First Convolution + Pooling operation
        # Input = 32 x 32, Output = 6 x 14 x 14
        dep_conv_out_1 = self.dep_conv_1(input)
        conv_out_1 = self.pnt_conv_1(dep_conv_out_1)
        pool_out_1 = self.pool1(conv_out_1)
        
        # Here we use ReLU activation to bring non-linearity in the CNN
        # Origional paper uses tanh, but ReLU provides for faster training and removes the vanishing gradient problem
        act_out_1 = relu(pool_out_1)
        
        # Second Convolution + Pooling Operation
        # Input = 6 x 14 x 14, Output = 16 x 5 x 5
        dep_conv_out_2 = self.dep_conv_2(act_out_1)
        conv_out_2 = self.pnt_conv_2(dep_conv_out_2)
        pool_out_2 = self.pool2(conv_out_2)
        act_out_2 = relu(pool_out_2)
        
        # Third Convolution with Depth Wise + Point Wise Convolution
        # Input = 16 x 5 x 5, Output = 120 x 1 x 1
        dep_conv_out_3 = self.dep_conv_3(act_out_2)
        conv_out_3 = self.pnt_conv_3(dep_conv_out_3)
        act_out_3 = relu(conv_out_3)
        
        # Flatten the output from the convolution layer to feed into the fully connected layer
        # Input = 120 x 1 x 1, Output = N x 120
        act_out_3 = torch.flatten(act_out_3, start_dim=1)
        
        # Execute first fully connected layer, with activation
        # Input = N x 120, Output = N x 84
        linear_out_1 = self.linear1(act_out_3)
        linear_act_out_1 = relu(linear_out_1)
        
        # Final output layer which classifies the result into 10 classes using softmax
        # In the LeNet paper, this was done using a RBF network, but we are using a fully connected layer followed by softmax for simplicity
        # Input = N x 84, Output = N x 10
        final_out = self.linear2(linear_act_out_1)
        output = softmax(final_out)
        
        return output

def load_minist_train():
    # Create a transform object to convert images to the required tensor shape
    transformations = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    
    # Get the train datasets with the transformations applied
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transformations)
    
    # We use a dataloader for batch processing
    # Batch Size is set to 100 for faster convergence
    # Shuffle is set to true to randomize the data so the model does not learn any order in the data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    
    return train_dataloader
    

# Function to train the Dense CNN on the MNIST Dataset
def train_CNN(net: nn.Module, graph_prefix: str, epochs: int = 50):
    train_dataloader = load_minist_train()
    
    # Create the loss function. We use the MSE loss which was the origional loss function in the LeNet paper
    criterion = nn.MSELoss()
    
    # Create the optimizer. We use SGD here to update the learning rate as the training progresses for faster convergence, which was also used in the LeNet paper
    # Momentum is added to help the optimizer converge faster
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    # Lists to store metrics for plotting
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        
        # Variables to track loss and accuracy for this epoch
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Iterate over each batch of training images
        for _, (images, labels) in enumerate(tqdm(train_dataloader)):
            # Forward pass through the network
            outputs = net(images)
            
            # Calculate the loss between CNN output and label
            # We also need to one-hot encode the labels for MSE loss calculation, since MSE requires same dimensions for NN output (N x 10) and target (which is just a list of integers in the raw form)
            loss = criterion(outputs, nn.functional.one_hot(labels, num_classes=10).float())
            
            # Zero the gradients before running the backward pass
            optimizer.zero_grad()
            
            # Calculate the gradients with backpropagation
            loss.backward()
            
            # Update the weights using the optimizer
            optimizer.step()
            
            # Add the loss for this batch
            running_loss += loss.item()
            
            # Get the predicted output class (index with the highest probability after softmax)
            _, preds = torch.max(outputs.data, 1)
            
            # Get the correct predictions for this batch
            correct_predictions += (preds == labels).sum().item()
            
            # Add the samples processed in this batch
            total_samples += labels.size(0)
            
        # Calculate average loss and the accuracy for this epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100 * (correct_predictions / total_samples)
        
        # Append to the respective lists
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
    # Plot Loss vs Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.grid(True)
    loss_plot_path = f"{graph_prefix}loss_vs_epochs.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss vs Epochs graph saved as '{loss_plot_path}'")
    
    # Plot Accuracy vs Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), epoch_accuracies, marker='o', linestyle='-', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs Epochs')
    plt.grid(True)
    accuracy_plot_path = f"{graph_prefix}accuracy_vs_epochs.png"
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy vs Epochs graph saved as '{accuracy_plot_path}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LeNet variants on MNIST")
    parser.add_argument(
        "--model",
        choices=["dense", "depthwise", "both"],
        default="depthwise",
        help="Model to train: dense, depthwise separable, or both",
    )
    parser.add_argument(
        "--graph-prefix",
        default="",
        help="Prefix to add before graph filenames",
    )
    return parser.parse_args()

# Define separate epochs for dense and depthwise models since they have different convergence rates
DENSE_EPOCHS = 25
DEPTHWISE_EPOCHS = 70

def build_model(model_name: str) -> tuple[nn.Module, int]:
    if model_name == "dense":
        return DenseNet(), DENSE_EPOCHS
    return DepthWiseSeparableNet(), DEPTHWISE_EPOCHS


def train_selected_models(model_name: str, graph_prefix: str):
    if model_name == "both":
        train_CNN(DenseNet(), "dense_", epochs=DENSE_EPOCHS)
        train_CNN(DepthWiseSeparableNet(), "depthwise_", epochs=DEPTHWISE_EPOCHS)
        return
    model, epochs = build_model(model_name)
    train_CNN(model, graph_prefix, epochs=epochs)


if __name__ == "__main__":
    args = parse_args()
    if not args.graph_prefix:
        args.graph_prefix = "" if args.model == "both" else f"{args.model}_"
    train_selected_models(args.model, args.graph_prefix)
