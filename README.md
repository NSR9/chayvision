# Model.py

This Python script contains the definition of a Convolutional Neural Network (CNN) model in PyTorch and a function to generate a summary of the model.

## Components

### Net Class

The `Net` class defines the structure of the neural network model. This class is a subclass of PyTorch's `nn.Module`.

The constructor (`__init__`) of the `Net` class defines the layers of the model:
- Four convolutional layers (`self.conv1`, `self.conv2`, `self.conv3`, `self.conv4`) with progressively increasing output channels.
- Two fully connected layers (`self.fc1`, `self.fc2`).

The `forward` method defines the forward pass of the network. The layers are applied to the input tensor `x` in order:
- The convolutional layers are applied with a ReLU activation function.
- Max pooling is applied after the second and fourth convolutional layers.
- After the last convolutional layer, the tensor is flattened.
- The fully connected layers are applied with a ReLU activation function after the first fully connected layer.
- The final layer's output is passed through a logarithmic softmax function to generate probabilities for each class.

### model_summary Function

The `model_summary` function generates and prints a summary of the model architecture. This function uses the `summary` function from the `torchsummary` library to provide a layer-by-layer description of the model, including the input and output shapes of each layer and the number of parameters.

## Usage

To use this script, you need to instantiate the `Net` class and call `model_summary` function. For example:

```python
# Instantiate the model
model = Net()

# Generate and print the model summary
model_summary(model)
```


# Utils.py

This file contains all the data processing, data visualiation and training utils.

I used Object oriented programming concepts in designing the components. 
## Components

### DataProcessor

The `DataProcessor` class handles the data transformations, downloading of the dataset, and setting up the data loaders.

- `define_data_transforms`: This method defines two sets of transformations, one for the training data and another for the test data. For the training data, the transformations include random cropping, resizing, rotation, and normalization. The test data is only transformed to tensors and normalized.
- `download_dataset`: This method downloads the dataset (MNIST, FashionMNIST, or CIFAR10) and applies the transformations defined in `define_data_transforms`.
- `define_data_loaders`: This method creates data loaders for the training and test data with the specified batch size.


### DataViewer

The `DataViewer` class is used for visualizing the training and test data.

- `plot_train_data`: This method plots a batch of training data samples.
- `plot_test_data`: This method plots a batch of test data samples.

### TrainLoop

The `TrainLoop` class handles the training and testing of the model, and plotting the training and testing results.

- `train_model`: This method sets the model in training mode, initializes the progress bar for training iterations, and loops over the training data. During each iteration, the model makes predictions, calculates the loss, performs backpropagation, and the optimizer updates the model parameters.
- `test_model`: This method sets the model in evaluation mode, loops over the test data, and computes the test loss and accuracy.
- `plot_graphs`: This method plots the training loss, training accuracy, test loss, and test accuracy as a function of epochs.

### Functions

- `check_cuda`: This function checks if CUDA is available on the device.
- `GetCorrectPredCount`: This function calculates the number of correct predictions by comparing the predicted labels with the true labels.
- `get_device`: This function returns the device that will be used for computation (either 'cuda' if available, or 'cpu').

## Usage

To use this script, you need to instantiate the classes and call their methods in a sequence. For example:

```python
# Define hyperparameters, model, and optimizer
# ...

# Initialize data processor and get data loaders
data_processor = DataProcessor()
data_transforms = data_processor.define_data_transforms()
train_data, test_data = data_processor.download_dataset("MNIST", data_transforms)
train_loader, test_loader = data_processor.define_data_loaders(batch_size, train_data, test_data)

# View data
data_viewer = DataViewer()
data_viewer.plot_train_data(train_loader)

# Train and test the model
train_loop = TrainLoop()
for epoch in range(epochs):
    train_loop.train_model(model, device, train_loader, optimizer)
    train_loop.test_model(model, device, test_loader)

# Plot training and testing results
train_loop.plot_graphs()
```

Please make sure you have PyTorch, torchvision, tqdm, and matplotlib installed in your Python environment.


##Dependencies
1. torch
2. torchvision
3. matplotlib
4. tqdm
5. Torch Summary

This code was written in Python 3 and uses PyTorch for model definition, training, and testing. Tqdm is used for displaying progress bars, and matplotlib is used for plotting the training and test metrics.