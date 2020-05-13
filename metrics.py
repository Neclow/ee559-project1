import torch

def compute_accuracy(net, data_loader):
    '''
    Compute accuracy of the network on a dataset.

    Accuracy = (1/N) * sum(predicted_label_i == true_label_i), i = 1, ..., N

    Parameters
    -------
    net
        The model/network.
    data_loader
        The training/test set.

    Returns
    -------
    tensor
         The accuracy of the model.
    '''

    acc = 0
    total = 0.
    net.eval()
    with torch.no_grad():
        for (X, y, _) in data_loader:
            out, _ = net(X)
            acc += ((out > 0.5) == y).float().sum().item()
            total += len(y)
    return acc/total
