import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden=128, mode='baseline', sharing=False, verbose=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2*14*14, hidden)
        self.fc2 = nn.Linear(hidden, 20)
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
        
        self.sharing = sharing
        self.shared_idxs = torch.randperm(self.fc2.in_features)[:2*self.fc2.out_features]
        
        if verbose:
            print(f'Parameters: {self.count_params()}')

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):   
        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        x = self.relu(self.fc2(x)) 
        x = self.drop(x)
        x = self.fc3(x)
        x = self.sigmoid(self.classifier(x))
        return x.squeeze(), None