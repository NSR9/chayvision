import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataProcessor():
    def __init__(self):
        pass

    def define_data_transforms(self):
        # Define data transformations for training and test data
        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),  # Randomly apply center crop
            transforms.Resize((28, 28)),  # Resize images to (28, 28)
            transforms.RandomRotation((-15., 15.), fill=0),  # Randomly rotate images
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize images
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize images
        ])
        return [train_transforms, test_transforms]

    def download_dataset(self, dataset_name, data_transforms):
        # Download the specified dataset and apply data transformations
        train_transformations = data_transforms[0]
        test_transformations = data_transforms[1]

        if dataset_name == "MNIST":
            train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transformations)  # Download MNIST dataset for training
            test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transformations)  # Download MNIST dataset for testing
        elif dataset_name == "FashionMNIST":
            train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=train_transformations)  # Download FashionMNIST dataset for training
            test_data = datasets.FashionMNIST('../data', train=False, download=True, transform=test_transformations)  # Download FashionMNIST dataset for testing
        elif dataset_name == "CIFAR10":
            train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transformations)  # Download CIFAR10 dataset for training
            test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transformations)  # Download CIFAR10 dataset for testing
        
        return train_data, test_data

    def define_data_loaders(self, batch_size, train_data, test_data):
        # Create data loaders for the training and test data
        kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(test_data, **kwargs)  # Test data loader
        train_loader = torch.utils.data.DataLoader(train_data, **kwargs)  # Train data loader

        return train_loader, test_loader

class DataViewer():
    def __init__(self):
        pass

    def plot_train_data(self, train_loader):
        # Plot a batch of training data samples
        batch_data, batch_label = next(iter(train_loader)) 

        fig = plt.figure()

        for i in range(12):
            plt.subplot(3, 4, i+1)
            plt.tight_layout()
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
            plt.title(batch_label[i].item())
            plt.xticks([])
            plt.yticks([])

    def plot_test_data(self, test_loader):
        # Plot a batch of test data samples
        batch_data, batch_label = next(iter(test_loader)) 

        fig = plt.figure()

        for i in range(12):
            plt.subplot(3, 4, i+1)
            plt.tight_layout()
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
            plt.title(batch_label[i].item())
            plt.xticks([])
            plt.yticks([])

def check_cuda():
    # Check if CUDA is available
    is_cuda = torch.cuda.is_available()
    print("CUDA Available?", is_cuda)
    return is_cuda
