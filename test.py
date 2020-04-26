import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import warnings
from models import *
from train import train, trial
from utils import load_data, weight_initialization, train_visualization
from metrics import compute_accuracy

def run_train(model, alpha, alpha_decay, plotting=False, verbose=True, seed=14):
    # Generate data
    torch.manual_seed(seed) # For reproducbility
    train_loader, test_loader = load_data(seed=seed)

    # Apply training mode and weight initialization
    model.train()
    model.apply(weight_initialization)

    # Train model
    start = time.time()
    tr_loss, tr_acc, te_acc = train(model, train_loader, test_loader, alpha=alpha,
                                    alpha_decay=alpha_decay, verbose=verbose, plotting=plotting)
    print('\n Training ended. Training time: %.2f s \n' % (time.time()-start))

    # Visualize data if plotting
    # Else, compute final train and test accuracy
    if plotting:
        final_train_accuracy = tr_acc[-1]
        final_test_accuracy = te_acc[-1]
        train_visualization(model, tr_loss, tr_acc, te_acc)
    else:
        final_train_accuracy = compute_accuracy(model, train_loader)
        final_test_accuracy = compute_accuracy(model, test_loader)

    print('Train accuracy: %.4f // Test accuracy: %.4f' %
         (final_train_accuracy, final_test_accuracy))

def run(model, alpha, alpha_decay, mode='train', plotting=False, n_trials=10):
    print('Default run: single training with best model')
    print('In default run: for plotting, change flag "plotting" to True.')
    print('For full trial, select mode="trial". Default number of trials: 10.')

    print(f'Model: {model._get_name()}')

    if str.find(model._get_name(), 'Siamese') < 0:
        warnings.warn(f'Auxiliary loss is not implemented for model: {model._get_name()}', stacklevel=2)

    print(f'Auxiliary loss coefficient: {alpha}')
    print(f'Auxiliary loss coefficient decay per epoch: {alpha_decay}')

    time.sleep(3)

    if mode == 'train':
        print('Running: single training.')
        print(f'Plotting mode: {plotting}')
        time.sleep(2)
        run_train(model, alpha, alpha_decay, plotting=plotting)
    elif mode == "trial":
        print(f'Running: {n_trials}-round trial.')
        trial(model, alpha, alpha_decay, n_trials=n_trials)
    else:
        raise ValueError('Running mode not found. Try "train" for simple train, "trial" for full trial.')

def main():
    best_model = SiameseCNN(mode='dropout', verbose=False)
    best_alpha = 1
    best_alpha_decay = 1
    run(model=best_model, alpha=best_alpha, alpha_decay=best_alpha_decay, mode='train', plotting=False)

if __name__ == "__main__":
    main()
