import torch
import time
from torch import nn, optim
from utils import split_data, shuffle, weight_initialization
from metrics import compute_accuracy

def train(net, train_loader, valid_loader, eta=1e-3, decay=1e-5, n_epochs=25, alpha=1, alpha_decay=1, verbose=False):
    aux_crit = nn.CrossEntropyLoss()
    binary_crit = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=eta, weight_decay=decay)

    tr_losses, val_losses = torch.zeros(n_epochs), torch.zeros(n_epochs)
    
    for e in range(n_epochs):
        # Reset training/validation loss
        tr_loss, val_loss = 0, 0
        
        # Training mode
        net.train()
        
        for (trainX, trainY, trainC) in train_loader:
            # Forward pass
            out, aux = net(trainX)
            
            # Separate outputs and target classes for each image
            aux1, aux2 = aux.unbind(1)
            c1, c2 = trainC.unbind(1)
            
            # Auxiliary loss
            aux_loss = aux_crit(aux1, c1) + aux_crit(aux2, c2)
            
            # Binary classification loss
            binary_loss = binary_crit(out, trainY.float())
            
            # Total loss
            total_loss = binary_loss + alpha*aux_loss
            
            tr_loss += total_loss.item()
           
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Validation
        net.eval() # Dropout layers will work in eval mode
        with torch.no_grad():
            for (valX, valY, trainC) in valid_loader:
                out, _ = net(valX)
                loss = binary_crit(out, valY.float())
                val_loss += loss.item()
        
        # Collect data
        tr_losses[e] = tr_loss
        val_losses[e] = val_loss
        
        # Update auxiliary loss coefficient
        alpha *= alpha_decay
        
        if verbose:
            print('Epoch %d/%d, Binary loss: %.3f, Auxiliary loss: %.3f, Validation loss: %.3f, ' % 
                  (e+1, n_epochs, binary_loss, aux_loss, val_loss))
    
    return tr_losses


def trial(net, train_data, test_data, n_epochs=25, n_trials=30, alpha=1, alpha_decay=1, verbose=True):
    all_losses = torch.zeros((n_trials, n_epochs))
    tr_accuracies = torch.zeros(n_trials)
    te_accuracies = torch.zeros(n_trials)
    for i in range(n_trials):
        # Shuffle data
        train_loader, valid_loader, test_loader = split_data(train_data, test_data, seed=i)
        
        # Reset weights
        net.train()
        net.apply(weight_initialization)
        
        # Train
        start = time.time()
        train_loss = train(net, train_loader, valid_loader, alpha=alpha, alpha_decay=alpha_decay)
        print('Trial %d/%d... Training time: %.2f s' % (i+1, n_trials, time.time()-start))
        
        # Collect data
        all_losses[i] = train_loss
        
        # Compute train and test accuracy
        net.eval() # Dropout layers will work in eval mode
        with torch.no_grad():
            tr_accuracies[i] = compute_accuracy(net, train_loader)
            te_accuracies[i] = compute_accuracy(net, test_loader)
        
        if verbose:
            print('Loss: %.4f, Train acc: %.4f, Test acc: %.4f' % 
                  (train_loss[-1], tr_accuracies[i], te_accuracies[i]))
    
    return all_losses, tr_accuracies, te_accuracies