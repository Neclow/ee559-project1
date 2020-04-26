import torch
import time
from torch import nn, optim
from torch.optim import lr_scheduler
from utils import load_data, weight_initialization
from metrics import compute_accuracy

def train(net, train_loader, test_loader=None, eta=1e-3, decay=1e-5,
          n_epochs=25, alpha=1, alpha_decay=1, verbose=False, plotting=False):
    aux_crit = nn.CrossEntropyLoss()
    binary_crit = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=eta, weight_decay=decay)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    tr_losses = torch.zeros(n_epochs)
    tr_accuracies = torch.zeros(n_epochs)
    te_accuracies = torch.zeros(n_epochs)

    for e in range(n_epochs):
        # Reset training/validation loss
        tr_loss = 0

        # Training mode
        net.train()

        for (trainX, trainY, trainC) in train_loader:
            # Forward pass
            out, aux = net(trainX)

            # Binary classification loss
            binary_loss = binary_crit(out, trainY.float())

            # Separate outputs and target classes for each image
            if aux is not None:
                aux1, aux2 = aux.unbind(1)
                c1, c2 = trainC.unbind(1)

                # Auxiliary loss
                aux_loss = aux_crit(aux1, c1) + aux_crit(aux2, c2)
            else:
                # Total loss
                aux_loss = 0
            total_loss = binary_loss + alpha*aux_loss
            tr_loss += total_loss.item()

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Collect accuracy data for later plotting
        if plotting:
            tr_accuracies[e] = compute_accuracy(net, train_loader)
            te_accuracies[e] = compute_accuracy(net, test_loader)

        # Collect loss data
        tr_losses[e] = tr_loss

        # Update auxiliary loss coefficient
        alpha *= alpha_decay

        if verbose:
            print('Epoch %d/%d, Binary loss: %.3f, Auxiliary loss: %.3f' %
                  (e+1, n_epochs, binary_loss, aux_loss))

    return tr_losses, tr_accuracies, te_accuracies


def trial(net, n_trials=30, n_epochs=25, alpha=0, alpha_decay=1, verbose=False):
    all_losses = torch.zeros((n_trials, n_epochs))
    tr_accuracies = torch.zeros(n_trials)
    te_accuracies = torch.zeros(n_trials)
    for i in range(n_trials):
        # Shuffle data
        torch.manual_seed(i)
        train_loader, test_loader = load_data(seed=i)

        # Reset weights
        net.train()
        net.apply(weight_initialization)

        # Train
        start = time.time()
        tr_loss = train(net, train_loader, n_epochs=n_epochs, alpha=alpha, alpha_decay=alpha_decay)[0]
        print('Trial %d/%d... Training time: %.2f s' % (i+1, n_trials, time.time()-start))

        # Collect data
        all_losses[i] = tr_loss

        # Compute train and test accuracy
        net.eval() # Dropout layers will work in eval mode
        with torch.no_grad():
            tr_accuracies[i] = compute_accuracy(net, train_loader)
            te_accuracies[i] = compute_accuracy(net, test_loader)

        if verbose:
            print('Loss: %.4f, Train acc: %.4f, Test acc: %.4f' %
                  (train_loss[-1], tr_accuracies[i], te_accuracies[i]))

    # Print trial results
    print('Train accuracy - mean: %.4f, std: %.4f, median: %.4f' %
         (tr_accuracies.mean(), tr_accuracies.std(), tr_accuracies.median()))
    print('Test accuracy - mean: %.4f, std: %.4f, median: %.4f' %
         (te_accuracies.mean(), te_accuracies.std(), te_accuracies.median()))

    return all_losses, tr_accuracies, te_accuracies
