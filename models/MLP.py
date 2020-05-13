import torch
from torch import nn

class MLP(nn.Module):
    '''
    Baseline Multi-Layer Perceptron

    Attributes
    -------
    fc1, fc2, fc3
        First, second and third fully-connected layers
    classifier
        Final fully-connected layer, rendering the binary prediction
    drop
        Dropout layer, applied after each linear layer before classification (excepting fc3)
    relu
        ReLU activation
    sigmoid
        Sigmoid activation (for classification layer)
    '''

    def __init__(self, hidden=128, verbose=True):
        '''
        Initialize the MLP

        Parameters
        -------
        hidden
            Number of neurons in the hidden layers
            (Though not reported, we have not found much difference when
            choosing the number of hidden neurons)
        verbose
            If true, prints number of parameters in the model
        '''

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2*14*14, hidden)
        self.fc2 = nn.Linear(hidden, 20)
        self.fc3 = nn.Linear(20, 10)
        self.classifier = nn.Linear(10, 1)

        # Regularizers
        self.drop = nn.Dropout(0.2)

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

        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), None
