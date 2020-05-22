import torch
from torch import nn


class SiameseCNN(nn.Module):
    '''
    Siamese Convolutional Neural Network

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
        Initialize the S-CNN

        Parameters
        -------
        verbose
            If true, prints number of parameters in the model
        '''
        super(SiameseCNN, self).__init__()
        
        # Siamese block
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 49, kernel_size=3)
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Decision block
        self.fc3 = nn.Linear(20, 10)
        self.classifier = nn.Linear(10, 1)
        
        # Regularizers: Dropout, Max-Pooling
        self.drop = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2,2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        if verbose:
            print(f'{self._get_name()} - Number of parameters: {self.count_params()}  \n')

    def count_params(self):
        '''
        Counts number of parameters in model
        '''
        return sum(p.numel() for p in self.parameters())
    
    def siamese_block(self, x):
        '''
        Pass a single image through Siamese block

        Parameters
        -------
        x
            Single image input, dimension: Nx14x14

        Returns
        -------
        x
            Siamese block output, dimension: Nx1x10
        '''
        x = self.pool(self.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return x
    
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
        
        x1, x2 = x.unbind(1)
        
        x1, x2 = self.siamese_block(x1), self.siamese_block(x2)
        
        # Collect auxiliary classification output
        # Dimension of aux: Nx2x10
        aux = torch.stack([x1, x2], dim=1)
        
        # Concatenate outputs from Siamese blocks
        # Dimension of x: Nx20
        x = torch.cat([x1, x2], dim=1)
        
        x = self.drop(x)
        x = self.relu(self.fc3(x.flatten(start_dim=1)))
        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), aux