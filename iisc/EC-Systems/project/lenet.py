import torch
import torch.nn as nn

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