import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Net(nn.Module):
    # Define the structure of the neural network
    def __init__(self):
        super(Net, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        # Define the fully connected layers
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Define the forward pass of the network
        x = F.relu(self.conv1(x), 2)  # Apply convolution and ReLU activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Apply convolution, max pooling, and ReLU activation
        x = F.relu(self.conv3(x), 2)  # Apply convolution and ReLU activation
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # Apply convolution, max pooling, and ReLU activation
        x = x.view(-1, 4096)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply fully connected layer and ReLU activation
        x = self.fc2(x)  # Apply fully connected layer
        return F.log_softmax(x, dim=1)  # Apply logarithmic softmax activation

def model_summary(model):
    # Generate a summary of the model architecture
    return summary(model, input_size=(1, 28, 28))
