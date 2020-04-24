import torch
from torch import nn

class SiameseCNN(nn.Module):
    def __init__(self, mode='baseline', verbose=True):
        super(SiameseCNN, self).__init__()
        
        ## Siamese block
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 49, kernel_size=3)
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Decision block
        self.fc3 = nn.Linear(20, 10)
        self.classifier = nn.Linear(10, 1)
        
        # Regularizers: Dropout, Max-Pooling
        if mode == 'baseline':
            self.drop = nn.Dropout(0)
        elif mode == 'dropout':
            self.drop = nn.Dropout(0.2)
        else:
            raise ValueError('Unknown mode. Try "baseline" or "dropout".')

        self.pool = nn.MaxPool2d(2,2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        if verbose:
            print(f'Parameters: {self.count_params()}')

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def siamese_block(self, x):
        x = self.pool(self.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return x
    
    def forward(self, x):
        x1, x2 = x.unbind(1)
        
        x1, x2 = self.siamese_block(x1), self.siamese_block(x2)
        
        # Dim x1: Nx1x10
        aux = torch.stack([x1, x2], dim=1)
        
        # Dim aux: Nx2x10
        x = torch.cat([x1, x2], dim=1)
        
        # Dim x: Nx20
        x = self.drop(x)
        x = self.relu(self.fc3(x.flatten(start_dim=1)))
        #x = self.drop(x)
        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), aux