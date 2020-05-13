import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import warnings
from models import *
from train import train, trial
from utils import load_data, weight_initialization, train_visualization
from metrics import compute_accuracy

def run_train(model, alpha, eta, decay, plotting=False, verbose=True, seed=14):
    '''
    Run a single training.

    Parameters
    -------
    model
        The neural network
    alpha
        The auxiliary loss coefficient
    eta
        Learning rate for training
    decay
        L2-regularization coefficient
    plotting
        If true, plots training loss and training accuracy at each epoch
    verbose
        If true, gives additional information during training (loss at each epoch)
    seed
        Random seed (for reproducibility)
    '''

    # Generate data
    torch.manual_seed(seed) # For reproducbility
    train_loader, test_loader = load_data(seed=seed)

    # Apply training mode and weight initialization
    model.train()
    model.apply(weight_initialization)

    # Train model
    start = time.time()
    tr_loss, tr_acc = train(model, train_loader, alpha=alpha,
                            eta=eta, decay=decay,
                            verbose=verbose, plotting=plotting)
    print('\n Training ended. Training time: %.2f s \n' % (time.time()-start))

    # Visualize data if plotting
    # Else, compute final train and test accuracy
    if plotting:
        train_visualization(model, tr_loss, tr_acc)
    final_train_accuracy = compute_accuracy(model, train_loader)
    final_test_accuracy = compute_accuracy(model, test_loader)

    print('Train accuracy: %.4f // Test accuracy: %.4f' %
         (final_train_accuracy, final_test_accuracy))

def run(model, alpha, mode='train', plotting=False, n_trials=10):
    '''
    Main run of the framework. Run a single training or a full trial.

    Parameters
    -------
    model
        The neural network
    alpha
        Auxiliary loss coefficient for Siamese networks
    mode
        If 'train', runs a single training
        If 'trial', runs a full trial (with n_trials trials)
    plotting
        If true, plots training loss and training accuracy at each epoch
        (Only for training mode)
    n_trials
        Number of trials to perform (if mode = 'trial')
    '''

    print('Default run: single training with best model')
    print('In default run: for plotting, change flag "plotting" to True.')
    print('For full trial, select mode="trial". Default number of trials: 10.')

    print(f'Model: {model._get_name()}')

    # Auxiliary loss implementation is not possible for non-Siamese networks
    if str.find(model._get_name(), 'Siamese') < 0:
        warnings.warn(f'Auxiliary loss is not implemented for model: {model._get_name()}', stacklevel=2)

    print(f'Auxiliary loss coefficient: {alpha}')

    time.sleep(2)

    # Load optimized hyperparameters for given framework
    config = load_hyperparam_config(model._get_name(), alpha)

    if mode == 'train':
        print('Running: single training.')
        print(f'Plotting mode: {plotting}')
        time.sleep(2)
        run_train(model, alpha, eta=config['eta'], decay=config['decay'], plotting=plotting)
    elif mode == "trial":
        print(f'Running: {n_trials}-round trial.')
        trial(model, n_trials=n_trials, eta=config['eta'], decay=config['decay'], alpha=alpha)
    else:
        raise ValueError('Running mode not found. Try "train" for simple train, "trial" for full trial.')

def load_hyperparam_config(name, alpha):
    '''
    Load optimized hyperparameters for given configuration.

    Parameters
    -------
    name
        Name of neural network
    alpha
        Auxiliary loss coefficient (for Siamese networks)

    Returns
    -------
    dict
        Contains optimized learning rate (eta) and L2-regularization coefficient (decay)
    '''

    if name == 'MLP':
        return {'eta': 5e-3, 'decay': 1e-5}
    elif name == 'CNN':
        return {'eta': 1e-3, 'decay': 1e-2}
    elif name == 'SiameseCNN':
        if alpha == 0:
            return {'eta': 1e-3, 'decay': 1e-2}
        elif alpha == 0.5:
            return {'eta': 5e-3, 'decay': 1e-4}
        elif alpha == 1:
            return {'eta': 2.5e-3, 'decay': 1e-3}
        else:
            raise ValueError('Alpha value unknown. Alpha can be 0, 0.5 or 1.')
    elif name == 'SiameseMLP':
        if alpha == 0:
            return {'eta': 5e-3, 'decay': 1e-3}
        elif alpha == 0.5:
            return {'eta': 5e-3, 'decay': 1e-5}
        elif alpha == 1:
            return {'eta': 5e-3, 'decay': 1e-4}
        else:
            raise ValueError('Alpha value unknown. Alpha can be 0, 0.5 or 1.')
    else:
        raise ValueError('Model name not found. Available: "MLP", "CNN",\
                         "SiameseCNN", "SiameseMLP".')

def main():
    best_model = SiameseCNN(verbose=True)
    best_alpha = 1
    run(model=best_model, alpha=best_alpha, mode='train', plotting=False)

if __name__ == "__main__":
    main()
