import torch
from torch import nn

class CNN(nn.Module):
    '''
    Baseline Convolutional Neural Network

    Attributes
    -------
    conv1, conv2
        First, second convolutional layers
    fc1, fc2, fc3
        First, second and third fully-connected layers
    classifier
        Final fully-connected layer, rendering the binary prediction
    drop
        Dropout layer, applied after each linear layer before classification (excepting fc3)
    pool
        Max-Pooling layer
    relu
        ReLU activation
    sigmoid
        Sigmoid activation (for classification layer)
    '''

    def __init__(self, verbose=True):
        '''
        Initialize the CNN

        Parameters
        -------
        verbose
            If true, prints number of parameters in the model
        '''

        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 49, kernel_size=3)

        # fully connected layers
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)
        self.classifier = nn.Linear(10, 1)

        # Regularizers
        self.drop = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2,2)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if verbose:
            print(f'Parameters: {self.count_params()}')

    def count_params(self):
        '''
        Counts number of parameters in model
        '''

        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        '''
        Forward pass

        Parameters
        -------
        x
            Input to the model, dimension: Nx2x14x14

        Returns
        -------
        tensor
            Binary classification output, dimension: Nx1
        tensor
            Auxiliary classification (None as the network is not Siamese)
            (Siamese networks cannot benefit from auxiliary classes)
        '''

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)

        x = self.relu(self.fc2(x))
        x = self.drop(x)

        x = self.relu(self.fc3(x.flatten(start_dim=1)))

        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), None
