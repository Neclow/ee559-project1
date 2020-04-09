import torch
from torch import nn, optim
from metrics import compute_accuracy  
from utils import train_visualization, weight_initialization, shuffle


def train(net, train_input, train_target, train_classes, eta=1e-3, n_epochs=25, batch_size=100, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=eta, weight_decay=1e-5)
    
    losses, accuracies = [], []
    
    for e in range(n_epochs):
        sum_loss = 0
        net.train()
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
        
        net.eval()
        with torch.no_grad():
            acc = compute_accuracy(net, train_input, train_target)
        losses.append(sum_loss)
        accuracies.append(acc)
        if verbose:
            print('Epoch %d/%d, Loss: %.3f, Accuracy: %.3f' % (e+1, n_epochs, sum_loss, acc))
    
    return losses, accuracies

def trial(net, train_input, train_classes, train_target, test_input, test_target, n_trials=10):
    all_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    for i in range(n_trials):
        print(f'Trial {i+1}/{n_trials}...')
        # Shuffle data
        train_input, train_target, train_classes = shuffle(train_input, train_target, train_classes)
        
        # Reset weights
        net.train()
        net.apply(weight_initialization)
        
        # Train
        train_loss, train_acc = train(net, train_input, train_target, train_classes, verbose=False)
        
        # Collect data
        all_losses.append(train_loss)
        all_train_accuracies.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_acc = compute_accuracy(net, test_input, test_target)
            all_test_accuracies.append(test_acc)
        
        print('Loss: %.3f, Train accuracy: %.3f, Test_accuracy: %.3f' % (train_loss[-1], train_acc[-1], test_acc))
    
    return torch.FloatTensor(all_losses), torch.FloatTensor(all_train_accuracies), torch.FloatTensor(all_test_accuracies)
    