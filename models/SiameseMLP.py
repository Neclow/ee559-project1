import torch
from torch import nn

class SiameseMLP(nn.Module):
    def __init__(self, hidden=128, mode='baseline', verbose=True):
        super(SiameseMLP, self).__init__()
        self.fc1 = nn.Linear(14*14, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.fc3 = nn.Linear(20, 10)
        self.classifier = nn.Linear(10, 1)
        
        if mode == 'baseline':
            self.drop = nn.Dropout(0)
        elif mode == 'dropout':
            self.drop = nn.Dropout(0.2)
        else:
            raise ValueError('Unknown mode. Try "baseline" or "dropout"')

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        if verbose:
            print(f'Parameters: {self.count_params()}')

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):   
        x = self.relu(self.fc1(x.flatten(start_dim=2)))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        
        aux = x
        
        x = self.drop(x)
        x = self.relu(self.fc3(x.flatten(start_dim=1)))
        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), aux