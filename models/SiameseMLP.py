import torch
from torch import nn

class SiameseMLP(nn.Module):
    '''
    Siamese Multi-Layer Perceptron

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
        Initialize the S-MLP

        Parameters
        -------
        hidden
            Number of neurons in the hidden layers
            (Though not reported, we have not found much difference when
            choosing the number of hidden neurons)
        verbose
            If true, prints number of parameters in the model
        '''

        super(SiameseMLP, self).__init__()

        # Siamese block
        self.fc1 = nn.Linear(14*14, hidden)
        self.fc2 = nn.Linear(hidden, 10)

        # Decision block
        self.fc3 = nn.Linear(20, 10)
        self.classifier = nn.Linear(10, 1)

        # Regularizers
        self.drop = nn.Dropout(0.2)

        # Activation fcns
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
        x
            Binary classification output, dimension: Nx1
        aux
            Auxiliary classification output, dimension: Nx2x10
        '''

        x = self.relu(self.fc1(x.flatten(start_dim=2)))
        x = self.drop(x)
        x = self.relu(self.fc2(x))

        # Collect auxiliary classification output
        # Dimension of aux: Nx2x10
        aux = x

        x = self.drop(x)
        # Concatenate Siamese branches and enter decision block
        x = self.relu(self.fc3(x.flatten(start_dim=1)))
        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), aux
