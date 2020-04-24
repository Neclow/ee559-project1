import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import init
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader


def standardize(x, mu, std):
    return x.sub_(mu).div_(std)


def load_data(N=1000, batch_size=50, seed=42):
    trainX, trainY, trainC, testX, testY, testC = prologue.generate_pair_sets(N, seed)

    mu, std = trainX.mean(), trainX.std()

    trainX, testX = [standardize(x, mu, std) for x in [trainX, testX]]

    # Assemble all data
    train_data = TensorDataset(trainX, trainY, trainC)
    test_data = TensorDataset(testX, testY, testC)

    torch.manual_seed(seed) # For reproducibility
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader

def weight_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_visualization(net, tr_losses, tr_accuracies, te_accuracies):
    fig, axs = plt.subplots(1,2, figsize=(8,4))

    n_epochs = len(tr_losses)

    axs[0].plot(range(n_epochs), tr_losses, 'k--')
    #axs[0].set_label(['Train loss'])
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid()
    #axs[0].set_title('%s' % (net._get_name()))

    axs[1].plot(range(n_epochs), tr_accuracies, 'k--', label='train')
    axs[1].plot(range(n_epochs), te_accuracies, 'r--', label='test')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid()
    axs[1].legend()

    fig.suptitle(f'Model: {net._get_name()}')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
