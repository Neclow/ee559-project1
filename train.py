import torch
from torch import nn, optim

def pretrain_autoencoders(aes, train_input, eta=1e-3, n_epochs=25, batch_size=50):
    criterion = nn.MSELoss()
    
    for i, ae in enumerate(aes):
        optimizer = torch.optim.Adam(ae.parameters(), lr=eta, weight_decay=1e-5)
        print(f'Pretraining AE {i+1}/{len(aes)}...')
        for e in range(n_epochs):
            ae.train()   
            sum_loss = 0
            for b in range(0, train_input.size(0), batch_size):
                trainX = train_input.narrow(0, b, batch_size)
                x1, x2 = trainX[:,0,:,:].flatten(start_dim=1), trainX[:,1,:,:].flatten(start_dim=1)
                
                for j in range(i):
                    x1, x2 = aes[j].encode(x1, x2)
                
                out1, out2 = ae(x1, x2)
                
                loss1 = criterion(out1, x1)
                loss2 = criterion(out2, x2)

                loss = loss1 + loss2
                sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {e+1}/{n_epochs}: loss = {sum_loss}')       

def train(net, train_input, train_classes, eta=1e-3, n_epochs=25, batch_size=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=eta, weight_decay=1e-5)
    
    for e in range(n_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            trainX = train_input.narrow(0, b, batch_size)
            trainC = train_classes.narrow(0, b, batch_size)
            
            x1, x2 = trainX[:,0,:,:], trainX[:,1,:,:]
            c1, c2 = trainC[:,0], trainC[:,1]

            out1, out2 = net(x1, x2)

            loss1 = criterion(out1, c1)
            loss2 = criterion(out2, c2)
            
            loss = loss1 + loss2
            sum_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {e+1}/{n_epochs}: loss = {sum_loss}')