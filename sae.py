import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, in_, out_):
        super(AE, self).__init__()
        self.in_ = in_
        self.out_ = out_
        
        self.encoder = nn.Sequential(nn.Linear(in_, out_), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(out_, in_), nn.ReLU())
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def encode(self, x1, x2):
        return self.encoder(x1), self.encoder(x2)
    
    def decode(self, x1, x2):
        return self.decoder(x1), self.decoder(x2)
        
    def forward(self, x1, x2):
        e1, e2 = self.encode(x1, x2)
        return self.decode(e1, e2)


class StackedAE(nn.Module):
    def __init__(self, nLayers, N=70000, D=14*14, T=10, verbose=True):
        super(StackedAE, self).__init__() 
        self.nParams = N
        self.nLayers = nLayers
        self.in_ = D
        self.out_ = T
        self.hidden = self.get_params(nLayers, N, D, T)
        
        self.aes = [AE(self.in_, self.hidden)] + (self.nLayers-1)*[AE(self.hidden, self.hidden)]
        
        self.encoder = nn.Sequential(*[ae.encoder for ae in self.aes])
        
        self.classifier = nn.Linear(self.hidden, self.out_)
        
        if verbose:
            print(f'Width for {self.nParams} parameters in a {self.nLayers}-layer FCN: {self.hidden}')
            print(f'Autoencoder Parameters: {sum([ae.count_params() for ae in self.aes])}')
            print(f'SAE Parameters: {self.count_params()}')

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x1, x2):
        e1, e2 = self.encoder(x1.flatten(start_dim=1)), self.encoder(x2.flatten(start_dim=1))
        return self.classifier(e1), self.classifier(e2)
    
    @staticmethod
    def get_params(L, N, D, T):
        # Solve: N = (L-1)*H^2 + (D+T+L)*H+T    
        a = 3*(L-1)
        b = 3*(D+L)+T-1
        c = D+T-N

        s = pow(b**2-4*a*c, 0.5)
        return round((-b+s)/(2*a))