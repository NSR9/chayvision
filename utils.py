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


class TrainLoop():
    def __init__(self):
        # Initialize lists to store training and test metrics
        self.train_acc = []  # Training accuracy
        self.train_losses = []  # Training loss
        self.test_acc = []  # Test accuracy
        self.test_losses = []  # Test loss

    def train_model(self, model, device, train_loader, optimizer):
        """
        Method for training the model.

        Args:
            model (nn.Module): The model to be trained.
            device (torch.device): The device to be used for training (e.g., 'cuda' or 'cpu').
            train_loader (DataLoader): The data loader for training data.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
        """
        model.train()  # Set the model in training mode
        pbar = tqdm(train_loader)  # Create a progress bar for training iterations

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict
            pred = model(data)

            # Calculate loss
            loss = F.nll_loss(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += GetCorrectPredCount(pred, target)
            processed += len(data)

            # Update the progress bar with training metrics
            pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        # Store the training metrics
        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def test_model(self, model, device, test_loader):
        """
        Method for testing the model.

        Args:
            model (nn.Module): The model to be tested.
            device (torch.device): The device to be used for testing (e.g., 'cuda' or 'cpu').
            test_loader (DataLoader): The data loader for test data.
        """
        model.eval()  # Set the model in evaluation mode

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss

                correct += GetCorrectPredCount(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        # Print the test metrics
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def plot_graphs(self):
        """
        Method for plotting the training and test metrics.
        """
        # Plot the training and test metrics
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")




def check_cuda():
    # Check if CUDA is available
    is_cuda = torch.cuda.is_available()
    print("CUDA Available?", is_cuda)
    return is_cuda

def GetCorrectPredCount(pPrediction, pLabels):
           return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
