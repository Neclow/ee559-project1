import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, nLayers, K = 3, nParams = 70000, D = 1, T = 10, hidden=100, verbose=True):
        super(CNN, self).__init__()
        self.nLayers = nLayers
        self.nParams = nParams
        self.kernel_size = K
        self.in_ = D
        self.out_ = T
        self.hidden = hidden
        
        self.nch = self.get_params(self.nLayers, self.nParams, self.in_, self.out_, self.kernel_size, self.hidden)
        
        self.conv = self.create_conv_layers()
        
        self.fc1 = nn.Linear(self.nch*18, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 10)
        
        if verbose:
            print(f'Width for {self.nParams} parameters in a {self.nLayers}-layer CNN: {self.nch}')
            print(f'Parameters: {self.count_params()}')
    
    def create_conv_layers(self):
        '''return nn.Sequential(nn.Conv2d(1,32, kernel_size=3),
                             nn.MaxPool2d(3,3),
                             nn.ReLU(),
                             nn.Conv2d(32, 64, kernel_size=3, padding=1),
                             nn.MaxPool2d(2,2),
                             nn.ReLU()
                             )'''
        layers = []
        
        in_ = self.in_
        out_ = self.nch
        pad = 0
        
        for i in range(self.nLayers):
            layers.extend([nn.Conv2d(in_, out_, self.kernel_size, stride=1, padding=pad),
                         nn.ReLU()])
            
            if i < 2:
                layers.extend([nn.MaxPool2d(2,2)])
            in_ = out_
            if i == 0:
                out_ = 2*self.nch
            pad = 1
        return nn.Sequential(*layers)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x1, x2):
        x1, x2 = self.conv(x1.unsqueeze(1)), self.conv(x2.unsqueeze(1))
        #print(x1.shape)
        #x1, x2 = self.avgpool(x1), self.avgpool(x2)
        x1, x2 = self.fc1(x1.flatten(start_dim=1)), self.fc1(x2.flatten(start_dim=1))
        return self.fc2(x1), self.fc2(x2)
    
    @staticmethod
    def get_params(L, N, D, T, K, H):
        K2 = pow(K,2)
        a = K2*(4*L-6)
        b = D*K2 + 18*H + 4*L - 1
        c = H+H*T+T-N

        s = pow(b**2-4*a*c, 0.5)
        return round((-b+s)/(2*a))
