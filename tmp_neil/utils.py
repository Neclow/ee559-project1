import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import init
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader


def standardize(x, mu, std):
    return x.sub_(mu).div_(std)


def load_data(N=1000):
    trainX, trainY, trainC, testX, testY, testC = prologue.generate_pair_sets(N)

    mu, std = trainX.mean(), trainX.std()
    
    trainX, testX = [standardize(x, mu, std) for x in [trainX, testX]]
    
    # Assemble all data
    train_data = TensorDataset(trainX, trainY, trainC)
    test_data = TensorDataset(testX, testY, testC)    

    return train_data, test_data

def split_data(train_data, test_data, seed=42, batch_size=50, split=800): 
    torch.manual_seed(seed)
    idxs = torch.randperm(len(train_data))
    
    train_idxs = idxs[:split]
    valid_idxs = idxs[split:]
    
    # Prepare sampler for train-validation splitting
    train_sampler = SubsetRandomSampler(train_idxs)
    valid_sampler = SubsetRandomSampler(valid_idxs)
    
    # Load data in DataLoaders, split train set into train and valdiation sets
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


def shuffle(inputs, targets, classes):
    idxs = torch.randperm(inputs.shape[0])
    return inputs[idxs,:,:,:], targets[idxs], classes[idxs,:]


def weight_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)


def train_visualization(net, losses, accuracies, n_epochs):
    fig, axs = plt.subplots(1,2)
    axs[0].plot(range(n_epochs), losses, 'k')
    axs[0].set_label(['Cross-entropy loss'])
    axs[0].set_xlabel('Epoch')
    axs[0].set_title('%s, depth: %d, loss: %.3f' % (net._get_name(), net.nLayers, losses[-1]))
    
    axs[1].plot(range(n_epochs), accuracies, 'k')
    axs[1].set_label(['Train acc'])
    axs[1].set_xlabel('Epoch')
    axs[1].set_title('%s, depth: %d, train accuracy: %.3f' % (net._get_name(), net.nLayers, accuracies[-1]))
    
    plt.tight_layout()