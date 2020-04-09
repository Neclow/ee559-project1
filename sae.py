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
        
        self.encoder = self.create_encoder_layers()
        
        self.classifier = nn.Linear(self.hidden, self.out_)
        
        if verbose:
            print(f'Width for {self.nParams} parameters in a {self.nLayers}-layer stacked AE: {self.hidden}')
            #print(f'Autoencoder Parameters: {sum([ae.count_params() for ae in self.aes])}')
            print(f'SAE Parameters: {self.count_params()}')
    
    def create_encoder_layers(self):
        layers = []
        in_= self.in_
        
        for i in range(self.nLayers):
            layers.extend(self.buildLayer(in_, self.hidden))
            in_ = self.hidden
        
        return nn.Sequential(*layers)

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
    
    @staticmethod
    def buildLayer(in_, out_):
        return nn.Sequential(nn.Linear(in_, out_), nn.ReLU())
    
    def pretrain(self, train_input, eta=1e-3, n_epochs=25, batch_size=50, verbose=False):
        criterion = nn.MSELoss()
        
        in_, out_ = self.in_, self.hidden
        
        aes = []
        
        for i in range(self.nLayers):
            ae = AE(in_, out_)
            in_ = self.hidden
            
            optimizer = torch.optim.Adam(ae.parameters(), lr=eta, weight_decay=1e-5)
            print(f'Pretraining Autoencoder {i+1}/{self.nLayers}...')
            for e in range(n_epochs):
                ae.train()   
                sum_loss = 0
                for b in range(0, train_input.size(0), batch_size):
                    trainX = train_input.narrow(0, b, batch_size)
                    x1, x2 = trainX[:,0,:,:].flatten(start_dim=1), trainX[:,1,:,:].flatten(start_dim=1)
                    
                    #print(x1.shape)

                    for j in range(i):
                        x1, x2 = aes[j].encode(x1, x2)

                    out1, out2 = ae(x1, x2)
                    
                    #print(out1.shape)

                    loss1 = criterion(out1, x1)
                    loss2 = criterion(out2, x2)

                    loss = loss1 + loss2
                    sum_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if verbose:
                    print(f'Epoch {e+1}/{n_epochs}: loss = {sum_loss}')
                
            aes.append(ae)            
            self.transferWeights(ae, 2*i)
        
        ae_params = sum([ae.count_params() for ae in aes])
        print(f'Number of parameters required to pretrain autoencoders: {ae_params}')
        print(f'Total number of parameters for stacked AE: {ae_params + self.count_params()}')

    def transferWeights(self, autoencoder, layer_idx):
        ae_params = autoencoder.encoder.named_parameters()
        sae_params = self.encoder[layer_idx].named_parameters()

        dict_sae_params = dict(sae_params)

        for name1, param1 in ae_params:
            if name1 in dict_sae_params:
                dict_sae_params[name1].data.copy_(param1.data)