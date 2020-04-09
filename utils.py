import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import torch
from torch import nn


def standardize(train, test):
    m, s = train.mean(), train.std()
    return (train-m)/s, (test-m)/s

def load_data(N=1000):
    trainX, trainY, trainC, testX, testY, testC = prologue.generate_pair_sets(N)

    trainX, testX = standardize(trainX, testX)

    return trainX, trainY, trainC, testX, testY, testC

def shuffle(inputs, targets, classes):
    idxs = torch.randperm(inputs.shape[0])
    return inputs[idxs,:,:,:], targets[idxs], classes[idxs,:]

def weight_initialization(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)

def width_visualization(net, N=70000, D=14*14, T=10, min_depth=2, max_depth=20):
    depth_range = range(min_depth, max_depth+1)
    
    h = [net(L, verbose=False).get_params(L, N, D, T) for L in depth_range]

    plt.plot(x,h)
    plt.ylabel('width')
    plt.xlabel('depth')
    plt.title(net._get_name())
    plt.show()

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

def reconstruct_image(ae, train_input):
    fig, axs = plt.subplots(1,2)
    
    idx = torch.LongTensor(1).random_(0, train_input.shape[0])
    
    x1, x2 = train_input[idx,0,:,:].flatten(start_dim=1), train_input[idx,1,:,:].flatten(start_dim=1)
    
    out1, _ = ae(x1, x2)
    
    axs[0].imshow(out1[0,:].reshape((14, 14)).detach().numpy())
    axs[0].set_title('AE')
    axs[1].imshow(x1[0].reshape(14, 14))
    axs[1].set_title('Ground truth')