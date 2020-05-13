import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import init
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader


def standardize(x, mu, std):
    '''
    Standardize data to zero-mean and unit variance.

    Parameters
    -------
    x
        Data to be standardized
    mu
        Mean
    std
        Standard deviation

    Returns
    -------
    tensor
        Standardized data
    '''
    return x.sub_(mu).div_(std)


def load_data(N=1000, batch_size=50, seed=42):
    '''
    Load training and test data from MNIST
    Data: pairs of MNIST images
    Label: 1 if first digit is lesser or equal than the second, 0 otherwise

    Parameters
    -------
    N
        Number of examples to generate for each set
    batch_size
        Batch size (for loading datasets into DataLoader type)
    seed
        Random seed (for reproducibility)

    Returns
    -------
    train_loader
        DataLoader containing training examples, binary labels and true image classes
    test_loader
        DataLoader containing test examples, binary labels and true image classes
    '''

    # Generate pairs
    trainX, trainY, trainC, testX, testY, testC = prologue.generate_pair_sets(N, seed)

    # Retrieve mean and standard deviation of training set
    mu, std = trainX.mean(), trainX.std()

    # Standardize data
    trainX, testX = [standardize(x, mu, std) for x in [trainX, testX]]

    # Assemble all data
    train_data = TensorDataset(trainX, trainY, trainC)
    test_data = TensorDataset(testX, testY, testC)

    # Load data in DataLoader and shuffle training set
    torch.manual_seed(seed) # For reproducibility
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader

def weight_initialization(m):
    '''
    Weight initialization of neural networks.

    Xavier uniform initialization was applied to all weights
    (Xavier initialization was chosen as the output layer is Sigmoid-activated)

    Biases were simply initialized to 0.01 (to ensure that ReLU units fire)

    Parameters
    -------
    m
        Layers of neural network
    '''

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_visualization(net, tr_losses, tr_accuracies):
    '''
    Visualize training and accuracy

    Parameters
    -------
    net
        The trained neural network
    tr_losses
        Training loss collected at each epoch
    tr_accuracies
        Training accuracy collected at eah epoch
    '''
    
    fig, axs = plt.subplots(1,2, figsize=(8,4))

    n_epochs = len(tr_losses)
    xdata = range(1, n_epochs+1)

    axs[0].plot(xdata, tr_losses, 'k--')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Training loss')
    axs[0].grid()

    axs[1].plot(xdata, tr_accuracies, 'k--', label='train')
    #axs[1].plot(xdata, te_accuracies, 'r--', label='test')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid()
    axs[1].legend()

    fig.suptitle(f'Model: {net._get_name()}')

    fname = 'img/train_visualization.png'
    plt.savefig(fname)
    print(f'Plot saved under {fname}')
