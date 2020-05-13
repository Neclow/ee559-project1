import torch
import time
from torch import nn, optim
from torch.optim import lr_scheduler
from utils import load_data, weight_initialization
from metrics import compute_accuracy

def train(net, train_loader, n_epochs=25, eta=1e-3, decay=1e-5,
          alpha=1, verbose=False, plotting=False):
    '''
    Train a neural network

    Parameters
    -------
    model
        The neural network
    train_loader
        The training set (DataLoader)
    n_epochs
        Number of epochs
    eta
        Learning rate
    decay
        L2-regularization coefficient
    alpha
        Auxiliary loss coefficient for Siamese networks (0, 0.5 or 1s)
    verbose
        If true, print loss at each epoch
    plotting
        If true, collects training accuracies at each epoch for future plotting

    Returns
    -------
    tr_losses (tensor)
        Training losses collected at each epoch
    tr_accuracies (tensor)
        Training accuracies collected at each epoch
        If plotting is False, tr_accuracies will only consist of zeros.
    '''

    aux_crit = nn.CrossEntropyLoss()
    binary_crit = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=eta, weight_decay=decay)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    tr_losses = torch.zeros(n_epochs)
    tr_accuracies = torch.zeros(n_epochs)

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

            # Calculate auxiliary loss for Siamese netwoks
            if aux is not None:
                # Separate outputs and target classes for each image
                aux1, aux2 = aux.unbind(1)
                c1, c2 = trainC.unbind(1)

                # Auxiliary loss
                aux_loss = aux_crit(aux1, c1) + aux_crit(aux2, c2)
            else:
                aux_loss = 0
            # Total loss = Binary loss + alpha*auxiliary loss
            total_loss = binary_loss + alpha*aux_loss
            tr_loss += total_loss.item()

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Collect accuracy data for later plotting
        if plotting:
            tr_accuracies[e] = compute_accuracy(net, train_loader)

        # Collect loss data
        tr_losses[e] = tr_loss

        if verbose:
            print('Epoch %d/%d, Binary loss: %.3f, Auxiliary loss: %.3f' %
                  (e+1, n_epochs, binary_loss, aux_loss))

    return tr_losses, tr_accuracies


def hyperparam_opt(net):
    '''
    Hyperparameter optimization by grid search, performed with three trials.

    Optimized hyperparameters
        eta (learning rate)
        lambda (L2-regularization coefficient)

    Parameters
    -------
    net
        The neural network

    Returns
    -------
    HP_res
        Dictionary containing the tested configuration,
        the reported train and test accuracies during three trials
    '''
    # Hyperparameters: eta, decay
    etas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    decays = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    HP_res = {'config': [], 'tr_accuracy': [], 'te_accuracy':[]}

    # Alpha is only 0 for non-Siamese networks (no auxiliary loss)
    if str.find(net._get_name(), 'Siamese') < 0:
        alphas = [0]
    else:
        alphas = [0, 0.5, 1]

    for eta in etas:
        for decay in decays:
            for alpha in alphas:
                print(f'Hyperparameters: eta={eta}, decay={decay}, alpha={alpha} \n')
                # Run three trials
                _, tr_accuracies, te_accuracies = trial(net, n_trials=3, eta=eta, decay=decay,
                                                                alpha=alpha, start_seed=1000, verbose=False)

                # Collect results
                HP_res['config'].append([eta, decay, alpha])
                HP_res['tr_accuracy'].append(tr_accuracies)
                HP_res['te_accuracy'].append(te_accuracies)

                # Skip a line
                print()
    return HP_res


def trial(net, n_trials=30, n_epochs=25, eta=1e-3, decay=1e-5, alpha=0, start_seed=0, verbose=False):
    '''
    Perform a trial on a network, i.e. several rounds of training.

    Parameters
    -------
    net
        The neural network
    n_trials
        Number of trainings to perform (Default: 30)
    n_epochs
        Number of training epochs (Default: 25)
    eta
        Learning rate
    decay
        L2-regularization coefficient
    alpha
        Auxiliary loss coefficient for Siamese networks (0, 0.5 or 1s)
    start_seed
        Indicates from where seeds are generated.
        start_seed = 0 with 20 trials means that seeds will be 0, ..., 19

        (This is useful to ensure that different datasets were used for
        hyperparameter optimization and trials)
    verbose
        If true, prints final loss, training accuracy and test accuracy for each trial

    Returns
    -------
    all_losses
        Training losses accumulated at each epoch for each trial
    tr_accuracies
        Final train accuracy reported at the end of each trial
    te_accuracies
        Final test accuracy reported at the end of each trial
    '''

    all_losses = torch.zeros((n_trials, n_epochs))
    tr_accuracies = torch.zeros(n_trials)
    te_accuracies = torch.zeros(n_trials)
    for i in range(n_trials):
        # Shuffle data
        torch.manual_seed(start_seed+i)
        train_loader, test_loader = load_data(seed=i)

        # Reset weights
        net.train()
        net.apply(weight_initialization)

        # Train
        start = time.time()
        tr_loss, _ = train(net, train_loader, n_epochs=n_epochs, eta=eta, decay=decay,
                           alpha=alpha)
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
                  (tr_loss[-1], tr_accuracies[i], te_accuracies[i]))

    # Print trial results
    print('Train accuracy - mean: %.4f, std: %.4f, median: %.4f' %
         (tr_accuracies.mean(), tr_accuracies.std(), tr_accuracies.median()))
    print('Test accuracy - mean: %.4f, std: %.4f, median: %.4f' %
         (te_accuracies.mean(), te_accuracies.std(), te_accuracies.median()))

    return all_losses, tr_accuracies, te_accuracies
